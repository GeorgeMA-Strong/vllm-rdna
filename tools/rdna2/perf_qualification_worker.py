# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in graph census and actual-weight comparisons for isolated experiments."""

from collections import Counter
from pathlib import Path

import torch


class PerfQualificationWorker:
    def begin_perf_phase(self, name: str):
        if getattr(self, "_perf_dispatch_original", None) is not None:
            raise RuntimeError("End the previous diagnostic phase first")
        manager = self.model_runner.cudagraph_manager
        original = manager.dispatch
        self._perf_counts = Counter()
        self._perf_dispatch_original = original
        self._perf_phase = name

        def dispatch(*args, **kwargs):
            desc = original(*args, **kwargs)
            self._perf_counts[(desc.cg_mode.name, desc.num_tokens, desc.num_reqs)] += 1
            return desc

        manager.dispatch = dispatch
        torch.cuda.nvtx.mark("qualification.begin." + name)
        config = self.model_runner.vllm_config
        return {
            "rank": self.rank,
            "phase": name,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
        }

    def end_perf_phase(self):
        original = getattr(self, "_perf_dispatch_original", None)
        if original is None:
            raise RuntimeError("No diagnostic phase is active")
        self.model_runner.cudagraph_manager.dispatch = original
        self._perf_dispatch_original = None
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
        }

    @torch.inference_mode()
    def check_actual_moe_weights(self, directory: str, save: bool = False):
        """Compare the loaded resident experts with a saved baseline output."""
        from vllm.model_executor.layers.quantization.rdna2_moe_resident import (
            apply_resident,
        )

        layer = next(
            module
            for module in self.model_runner.model.modules()
            if hasattr(module, "_rdna2_resident")
        )
        native = layer._rdna2_resident
        weight = native.w13_weight_scale
        generator = torch.Generator(device=weight.device).manual_seed(620)
        x = torch.randn(
            4096, 2560, dtype=weight.dtype, device=weight.device, generator=generator
        )
        ids = torch.randint(
            0,
            512,
            (4096, 10),
            device=weight.device,
            dtype=torch.int32,
            generator=generator,
        )
        routing = torch.softmax(
            torch.randn(4096, 10, device=weight.device, generator=generator), dim=-1
        )
        actual = apply_resident(layer, x, routing, ids).cpu()
        path = Path(directory) / f"resident-rank{self.rank}.pt"
        if save:
            if path.exists():
                raise RuntimeError(f"Refusing to replace {path}")
            torch.save(actual, path)
            maximum = 0.0
        else:
            expected = torch.load(path, weights_only=True)
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.01)
            maximum = (actual - expected).abs().max().item()
        return {"rank": self.rank, "saved": save, "max_abs": maximum}
