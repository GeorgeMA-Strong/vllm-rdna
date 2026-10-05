# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in graph census and actual-weight comparisons for isolated experiments."""

import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch


class PerfQualificationWorker:
    def begin_perf_phase(self, name: str, gpu_timing: bool = False):
        if getattr(self, "_perf_dispatch_original", None) is not None:
            raise RuntimeError("End the previous diagnostic phase first")
        manager = self.model_runner.cudagraph_manager
        from vllm.distributed.parallel_state import get_tp_group

        native_ar = getattr(get_tp_group().device_communicator, "rdna_ar_comm", None)
        ar_active = native_ar is not None and not native_ar.disabled
        if os.environ.get("VLLM_RDNA_AR") == "1" and not ar_active:
            raise RuntimeError("Expected native all-reduce; refusing RCCL fallback")
        original = manager.dispatch
        self._perf_counts = Counter()
        self._perf_dispatch_original = original
        self._perf_phase = name

        def dispatch(*args, **kwargs):
            desc = original(*args, **kwargs)
            self._perf_counts[(desc.cg_mode.name, desc.num_tokens, desc.num_reqs)] += 1
            return desc

        manager.dispatch = dispatch
        self._perf_gpu_events = []
        self._perf_graph_original = None
        if gpu_timing:
            graph_original = manager.run_fullgraph
            self._perf_graph_original = graph_original

            def run_fullgraph(desc):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = graph_original(desc)
                end.record()
                self._perf_gpu_events.append((desc, start, end))
                return result

            manager.run_fullgraph = run_fullgraph
        torch.cuda.nvtx.mark("qualification.begin." + name)
        config = self.model_runner.vllm_config
        return {
            "rank": self.rank,
            "phase": name,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            "rdna_ar_active": ar_active,
            "gpu_timing": gpu_timing,
        }

    def end_perf_phase(self):
        original = getattr(self, "_perf_dispatch_original", None)
        if original is None:
            raise RuntimeError("No diagnostic phase is active")
        self.model_runner.cudagraph_manager.dispatch = original
        self._perf_dispatch_original = None
        graph_original = self._perf_graph_original
        timings = defaultdict(list)
        if graph_original is not None:
            self.model_runner.cudagraph_manager.run_fullgraph = graph_original
            self._perf_graph_original = None
            torch.accelerator.synchronize()
            for desc, start, end in self._perf_gpu_events:
                timings[(desc.num_tokens, desc.num_reqs)].append(
                    start.elapsed_time(end)
                )
        self._perf_gpu_events = []
        torch.cuda.nvtx.mark("qualification.end." + self._perf_phase)
        return {
            "rank": self.rank,
            "phase": self._perf_phase,
            "dispatches": [
                {"mode": mode, "rows": rows, "requests": reqs, "count": count}
                for (mode, rows, reqs), count in sorted(
                    self._perf_counts.items(), key=lambda pair: str(pair[0])
                )
            ],
            "target_graph_gpu_ms": [
                {
                    "rows": rows,
                    "requests": requests,
                    "count": len(values),
                    "median": statistics.median(values),
                    "p95": sorted(values)[int(0.95 * (len(values) - 1))],
                    "total": sum(values),
                }
                for (rows, requests), values in sorted(timings.items())
            ],
        }

    @torch.inference_mode()
    def check_actual_moe_weights(
        self, directory: str, save: bool = False, num_rows: int = 4096
    ):
        """Compare the loaded resident experts with a saved baseline output."""
        from vllm.model_executor.layers.quantization.rdna2_moe_resident import (
            apply_resident,
        )

        if not 1 <= num_rows <= 4096:
            raise ValueError("Actual-weight probe rows must be in 1..4096")
        layer = next(
            module
            for module in self.model_runner.model.modules()
            if hasattr(module, "_rdna2_resident")
        )
        native = layer._rdna2_resident
        weight = native.w13_weight_scale
        generator = torch.Generator(device=weight.device).manual_seed(620)
        x = torch.randn(
            num_rows,
            2560,
            dtype=weight.dtype,
            device=weight.device,
            generator=generator,
        )
        ids = torch.randint(
            0,
            512,
            (num_rows, 10),
            device=weight.device,
            dtype=torch.int32,
            generator=generator,
        )
        routing = torch.softmax(
            torch.randn(num_rows, 10, device=weight.device, generator=generator), dim=-1
        )
        actual = apply_resident(layer, x, routing, ids).cpu()
        suffix = "" if num_rows == 4096 else f"-m{num_rows}"
        path = Path(directory) / f"resident{suffix}-rank{self.rank}.pt"
        if save:
            if path.exists():
                raise RuntimeError(f"Refusing to replace {path}")
            torch.save(actual, path)
            maximum = 0.0
        else:
            expected = torch.load(path, weights_only=True)
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.01)
            maximum = (actual - expected).abs().max().item()
        return {
            "rank": self.rank,
            "rows": num_rows,
            "saved": save,
            "max_abs": maximum,
        }
