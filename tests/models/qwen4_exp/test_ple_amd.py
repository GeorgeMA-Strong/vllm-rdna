# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

from ...utils import create_new_process_for_each_test


@pytest.mark.parametrize("rows", [9, 257])
def test_amd_ple_forward_releases_gate_temporaries_before_convolution(
    monkeypatch, rows
):
    """Consumed projection buffers must not inflate convolution's live memory."""
    import math
    import weakref
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer

    hc, hidden = 4, 32
    references = {}

    def embedding(x, *_):
        value = x.clone()
        references["embedding"] = weakref.ref(value)
        return value

    def projection(name, width):
        def call(x):
            value = x.new_ones((rows, width))
            references[name] = weakref.ref(value)
            return value, None

        return call

    def norm(name, value):
        result = value.clone()
        if name != "conv":
            references[name] = weakref.ref(result)
        return result

    def convolution(inputs, output, prefix):
        assert prefix == "memory-fixture"
        assert all(reference() is None for reference in references.values())
        output.copy_(inputs)

    monkeypatch.setattr(torch.ops.vllm, "qwen4_exp_ple_short_conv", convolution)
    layer = SimpleNamespace(
        ple_embedding=embedding,
        key_proj=projection("key_projection", hc * hidden),
        value_proj=projection("value_projection", hidden),
        hc_count=hc,
        hidden_size=hidden,
        norm_key="key_norm",
        norm_query="query_norm",
        norm_conv="conv",
        _apply_norm=norm,
        prefix="memory-fixture",
    )
    torch.manual_seed(620)
    with torch.inference_mode():
        x = torch.randn(rows, hc * hidden, dtype=torch.float16)
        query = x.reshape(rows, hc, hidden)
        gate = query.sum(-1, keepdim=True) / math.sqrt(hidden)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = gate * x.new_ones((rows, hidden)).unsqueeze(-2)
        expected = gated.flatten(-2) + gated.flatten(-2)
        actual = Qwen4ExpPLELayer.forward(layer, x, torch.arange(rows), None, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _ple_grouped_norm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    if group_size is None:
        grouped = hidden_states.unsqueeze(-2)
    else:
        grouped = hidden_states.unflatten(
            -1, (hidden_states.shape[-1] // group_size, group_size)
        )
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    return (normalized.flatten(-2) * (1.0 + weight.float())).to(input_dtype)


@pytest.mark.parametrize("lengths", [[17], [9, 0, 4, 7]])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_amd_ple_prefill_packs_once_and_preserves_output_and_state(
    monkeypatch, lengths, device
):
    """Ragged prefill packs directly into history without duplicate outputs."""
    from types import SimpleNamespace

    import torch.nn.functional as F

    import vllm.models.qwen4_exp.amd.ple_layer as ple

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    monkeypatch.setattr(ple, "ple_conv_use_rdna2", lambda: False)
    torch.manual_seed(620)
    count, hidden, state_len = len(lengths), 8, 4
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    slots = [1] if count == 1 else [1, 2, 0, 3]
    initial = [False] if count == 1 else [True, False, True, False]
    x = torch.randn(starts[-1], hidden, dtype=torch.float16, device=device)
    weights = torch.randn(hidden, 3, dtype=x.dtype, device=device)
    state = torch.randn(count + 1, hidden, state_len + 2, dtype=x.dtype, device=device)
    expected_state = state.clone()
    packed = x.new_zeros((count, hidden, max(lengths)))
    histories = []
    for row, (length, slot, has_initial) in enumerate(zip(lengths, slots, initial)):
        packed[row, :, :length] = x[starts[row] : starts[row + 1]].T
        previous = (
            state[slot, :, :state_len]
            if slot and has_initial
            else state.new_zeros((hidden, state_len))
        )
        history = torch.cat((previous, packed[row]), dim=-1)
        histories.append(history)
        if slot and length:
            expected_state[slot, :, :state_len] = history[
                :, length : length + state_len
            ]
    expected = F.silu(
        F.conv1d(
            torch.stack(histories), weights.unsqueeze(1), groups=hidden, dilation=2
        )
    ).transpose(1, 2)
    expected = torch.cat(
        [
            expected[row, :length] if slot else torch.zeros_like(expected[row, :length])
            for row, (length, slot) in enumerate(zip(lengths, slots))
        ]
    )
    allocated_packs, empty_outputs = [], []
    original_zeros, original_empty = torch.Tensor.new_zeros, torch.empty_like

    def new_zeros(tensor, size, **kwargs):
        allocated_packs.append(tuple(size))
        return original_zeros(tensor, size, **kwargs)

    def empty_like(tensor, **kwargs):
        empty_outputs.append(tuple(tensor.shape))
        return original_empty(tensor, **kwargs)

    with monkeypatch.context() as allocation_hooks, torch.inference_mode():
        allocation_hooks.setattr(torch.Tensor, "new_zeros", new_zeros)
        allocation_hooks.setattr(torch, "empty_like", empty_like)
        actual = ple.Qwen4ExpPLELayer._short_conv_dilated_prefill_batched(
            SimpleNamespace(conv_state_len=state_len, short_conv_dilation=2),
            x,
            SimpleNamespace(
                non_spec_query_start_loc=torch.tensor(starts, device=device),
                has_initial_states_p=torch.tensor(initial, device=device),
                max_prefill_query_len=max(lengths),
            ),
            state,
            weights,
            torch.tensor(slots, device=device),
            count,
            0,
            starts[-1],
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
    assert allocated_packs == [(count, hidden, max(lengths) + state_len)]
    assert not empty_outputs


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("group_size", [None, 8])
def test_amd_ple_grouped_norm_cpu_fallback_matches_reference(
    group_size: int | None,
) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 32
    eps = 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps, group_size, dtype=torch.float16)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size))
    hidden_states = torch.randn(3, hidden_size, dtype=torch.float16)

    actual = norm(hidden_states)
    expected = _ple_grouped_norm_reference(hidden_states, norm.weight, eps, group_size)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AMD PLE fused norm requires a ROCm GPU",
)
@pytest.mark.parametrize(
    ("dtype", "group_size"),
    [
        (torch.float16, None),
        (torch.float16, 2560),
        (torch.bfloat16, None),
        (torch.bfloat16, 2560),
    ],
)
def test_amd_ple_grouped_norm_fused_matches_reference(
    dtype: torch.dtype, group_size: int | None
) -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 10240
    eps = 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps, group_size, dtype=dtype).cuda()
    weight = torch.linspace(-0.25, 0.25, hidden_size, device="cuda")
    weight[:3] = torch.tensor([-1.0, -0.999, -1.001], device="cuda")
    with torch.no_grad():
        norm.weight.copy_(weight)
    hidden_states = torch.randn(1, 3, hidden_size, dtype=dtype, device="cuda")

    actual = norm(hidden_states)
    expected = _ple_grouped_norm_reference(hidden_states, norm.weight, eps, group_size)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AMD PLE fused norm requires a ROCm GPU",
)
def test_amd_ple_grouped_norm_strided_input_uses_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.models.qwen4_exp.amd.ple_layer as amd_ple_layer_module
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 32
    eps = 1e-6
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps, 8, dtype=torch.float16).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size, device="cuda"))
    hidden_states = torch.randn(3, hidden_size * 2, dtype=torch.float16, device="cuda")[
        :, ::2
    ]
    assert hidden_states.stride(-1) != 1

    def fail_if_called(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("fused grouped norm called for strided input")

    monkeypatch.setattr(amd_ple_layer_module, "grouped_gemma_rmsnorm", fail_if_called)
    actual = norm(hidden_states)
    expected = _ple_grouped_norm_reference(
        hidden_states, norm.weight, eps, norm.group_size
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="AMD PLE fused norm requires a ROCm GPU",
)
def test_amd_ple_grouped_norm_graph_replay() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEGroupedNorm

    hidden_size = 128
    eps = 1e-6
    group_size = 32
    norm = Qwen4ExpPLEGroupedNorm(
        hidden_size, eps, group_size, dtype=torch.float16
    ).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size, device="cuda"))
    hidden_states = torch.randn(3, hidden_size, dtype=torch.float16, device="cuda")
    norm(hidden_states)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = norm(hidden_states)

    hidden_states.copy_(torch.randn_like(hidden_states))
    graph.replay()
    torch.accelerator.synchronize()
    expected = _ple_grouped_norm_reference(hidden_states, norm.weight, eps, group_size)
    torch.testing.assert_close(graph_output, expected, atol=2e-3, rtol=2e-3)
