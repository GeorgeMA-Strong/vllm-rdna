# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare resident tiled and skinny MoE with warm and cache-flushed timing."""

import argparse
import json
import runpy
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_rdna2 import (  # noqa: E501
    _rdna2_fused_moe,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fixtures = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "tests/kernels/quantization/test_rdna2_moe_w4a16.py"
        )
    )
    rows = []
    # Exceed Navi21's 128 MiB Infinity Cache; exclude flushing from event time.
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for m in (1, 3, 4):
        layer, x, ids, weights, emap, _, _ = fixtures["_resident_skinny_case"](
            m, 2560, 640
        )
        # TP4 + EP4 owns one quarter of the global experts. Top-k routes
        # are unique within each row, unlike the duplicate-route stress test.
        emap = torch.full((16,), -1, device="cuda", dtype=torch.int32)
        emap[::4] = torch.arange(4, device="cuda", dtype=torch.int32)
        ids = torch.arange(m * 10, device="cuda").reshape(m, 10) % 16
        act = torch.empty(m, 10, 640, device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        funcs = {
            "tiled": partial(
                _rdna2_fused_moe,
                x,
                weights,
                ids,
                layer,
                MoEActivation.SILU,
                False,
                16,
                emap,
            ),
            "skinny": partial(
                fixtures["_run_resident_skinny"], layer, x, ids, weights, emap, act, out
            ),
        }
        for name, fn in funcs.items():
            for _ in range(5):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            # Let clocks settle before comparing these short graph replays.
            for _ in range(1000):
                graph.replay()
            torch.accelerator.synchronize()
            for cold in (False, True):
                timings = []
                for _ in range(30):
                    if cold:
                        flush.zero_()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end))
                row = dict(
                    tokens=m,
                    path=name,
                    cold=cold,
                    median_ms=sorted(timings)[len(timings) // 2],
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
