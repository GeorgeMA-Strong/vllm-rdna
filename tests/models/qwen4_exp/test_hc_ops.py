# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if current_platform.is_rocm():
    from vllm.models.qwen4_exp.amd.ops.hc import (
        grouped_gemma_rmsnorm,
        hc_combine,
        hc_combine_norm,
        hc_gate_mix,
    )
else:
    from vllm.models.qwen4_exp.nvidia.ops.hc import (
        grouped_gemma_rmsnorm,
        hc_combine,
        hc_combine_norm,
        hc_gate_mix,
    )

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not HAS_TRITON,
    reason="HC kernels require a GPU and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
EPS = 1e-6


def test_grouped_gemma_rmsnorm() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = grouped_gemma_rmsnorm(x, weight, EPS, HC)

    grouped = x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected = grouped * torch.rsqrt(variance + EPS)
    expected = expected.flatten(-2) * (1.0 + weight.float())
    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_gate_mix() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual = hc_gate_mix(x, gate, HC)
    expected = (
        torch.sigmoid(gate.float().unflatten(-1, (HC, HIDDEN_SIZE)))
        * x.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_combine() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")

    actual = hc_combine(residual, block_output, injection, HC)
    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)

    torch.testing.assert_close(actual, expected.flatten(-2).to(torch.bfloat16))


def test_hc_combine_norm() -> None:
    torch.manual_seed(0)
    block_output = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(2, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    injection = torch.randn(2, HC, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    actual, actual_norm = hc_combine_norm(
        residual, block_output, injection, weight, EPS, HC
    )

    injection_weight = 2.0 * torch.sigmoid(injection.float() / HC)
    expected = residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
    expected = expected + block_output.float().unsqueeze(
        -2
    ) * injection_weight.unsqueeze(-1)
    expected = expected.flatten(-2).to(residual.dtype)
    grouped = expected.float().unflatten(-1, (HC, HIDDEN_SIZE))
    variance = grouped.square().mean(-1, keepdim=True)
    expected_norm = grouped * torch.rsqrt(variance + EPS)
    expected_norm = expected_norm.flatten(-2) * (1.0 + weight.float())

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_norm, expected_norm.to(torch.bfloat16))


@pytest.mark.skipif(not current_platform.is_rocm(), reason="RDNA2 only")
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_hc_prefill_sharding_computes_only_local_rows(monkeypatch, rank):
    """Each TP rank computes one quarter, preserving the full public output."""
    import torch.nn.functional as functional

    import vllm.distributed as distributed
    from vllm.model_executor.layers import rdna_ops  # noqa: F401
    from vllm.models.qwen4_exp.amd.ops.hc import hc_silu

    torch.manual_seed(620)
    x = torch.randn(1024, 128, device="cuda", dtype=torch.float16)
    down = torch.randn(32, 128, device="cuda", dtype=torch.float16) * 0.05
    up = torch.randn(128, 16, device="cuda", dtype=torch.float16) * 0.05
    expected_down = functional.linear(x, down)
    gate = functional.linear(hc_silu(expected_down[:, :16].contiguous(), 4), up)
    expected_block = hc_gate_mix(x, gate, 4)
    expected = [expected_block, expected_down]
    gather_count = 0

    class Group:
        world_size = 4
        rank_in_group = rank

        def all_gather(self, value, dim):
            nonlocal gather_count
            full = expected[gather_count]
            torch.testing.assert_close(value, full[rank * 256 : (rank + 1) * 256])
            assert dim == 0
            gather_count += 1
            return full.clone()

    monkeypatch.setattr(distributed, "get_tp_group", lambda: Group())
    monkeypatch.setenv("VLLM_RDNA_HC_PREFILL_SP", "1")
    original_linear = functional.linear
    projection_rows = []

    def measured_linear(value, weight, bias=None):
        projection_rows.append(value.shape[0])
        return original_linear(value, weight, bias)

    monkeypatch.setattr(functional, "linear", measured_linear)
    actual_block, actual_down = torch.ops.vllm.rdna_hc_mix(
        x, down, None, None, up, None, None, 16, 4
    )
    torch.testing.assert_close(actual_block, expected_block)
    torch.testing.assert_close(actual_down, expected_down)
    assert projection_rows == [256, 256]
    assert gather_count == 2
