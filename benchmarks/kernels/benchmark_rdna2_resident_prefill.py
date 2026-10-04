# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic same-geometry resident MoE tile8/tile16/tile8 microbenchmark."""

import json
import os
import statistics
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
    CompressedTensorsWNA16RDNA2MoEMethod,
    _rdna2_fused_moe,
)


def main():
    torch.manual_seed(620)
    native = SimpleNamespace()
    for prefix, k, n in (("w13", 2560, 1280), ("w2", 640, 2560)):
        setattr(
            native,
            prefix + "_weight_packed",
            torch.randint(
                -(2**31), 2**31 - 1, (128, k // 8, n), dtype=torch.int32, device="cuda"
            ),
        )
        setattr(
            native,
            prefix + "_weight_scale",
            torch.rand(128, k // 128, n, dtype=torch.float16, device="cuda") * 0.01,
        )
    method = object.__new__(CompressedTensorsWNA16RDNA2MoEMethod)
    method.group_size = 128
    method.process_weights_after_loading(native)
    x = torch.randn(4096, 2560, dtype=torch.float16, device="cuda") * 0.1
    ids = torch.randint(0, 512, (4096, 10), dtype=torch.int32, device="cuda")
    routing = torch.softmax(torch.randn(4096, 10, device="cuda"), dim=-1)
    mapping = torch.full((512,), -1, dtype=torch.int32, device="cuda")
    mapping[:128] = torch.arange(128, dtype=torch.int32, device="cuda")

    def call():
        return _rdna2_fused_moe(
            x, routing, ids, native, MoEActivation.SILU, False, 512, mapping
        )

    os.environ["VLLM_RDNA_MOE_PREFILL_TILE16"] = "0"
    reference = call().clone()
    for enabled in ("0", "1", "0"):
        os.environ["VLLM_RDNA_MOE_PREFILL_TILE16"] = enabled
        actual = call().clone()
        torch.testing.assert_close(actual, reference, atol=0.1, rtol=0.01)
        for _ in range(5):
            call()
        samples = []
        for _ in range(5):
            start = torch.Event(device=x.device, enable_timing=True)
            end = torch.Event(device=x.device, enable_timing=True)
            start.record()
            for _ in range(20):
                call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 20)
        print(
            json.dumps(
                {
                    "tile": 16 if enabled == "1" else 8,
                    "median_ms": statistics.median(samples),
                    "samples_ms": samples,
                    "max_abs": (actual - reference).abs().max().item(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
