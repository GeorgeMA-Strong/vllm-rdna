# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker extension measuring prefill stages with GPU events.

Enable only in an isolated diagnostic server with --worker-extension-cls
tools.rdna2.prefill_profile_worker.PrefillProfileWorker. Arm/disarm through
collective_rpc; no synchronization is added to the measured forward calls.
"""

import os
import statistics
from collections import defaultdict
from functools import wraps

import torch


class PrefillProfileWorker:
    def set_moe_prefill_tile16(self, enabled: str):
        if enabled not in ("0", "1"):
            return {"rank": self.rank, "error": "Expected 0 or 1"}
        if getattr(self, "_prefill_profile_active", False):
            return {"rank": self.rank, "error": "Disarm profiler first"}
        os.environ["VLLM_RDNA_MOE_PREFILL_TILE16"] = enabled
        return {"rank": self.rank, "moe_prefill_tile16": enabled}

    def benchmark_moe_prefill_tile16(self):
        """Real resident expert weights/routing; unchanged MoE test tolerance."""
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
        logits = torch.randn(4096, 10, device=weight.device, generator=generator)
        routing = torch.softmax(logits, dim=-1)
        previous = os.environ.get("VLLM_RDNA_MOE_PREFILL_TILE16")
        result = {"rank": self.rank, "routes": []}
        try:
            os.environ["VLLM_RDNA_MOE_PREFILL_TILE16"] = "0"
            expected = apply_resident(layer, x, routing, ids).clone()
            for enabled in ("0", "1", "0"):
                os.environ["VLLM_RDNA_MOE_PREFILL_TILE16"] = enabled
                actual = apply_resident(layer, x, routing, ids).clone()
                errors = []
                try:
                    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.01)
                except AssertionError as error:
                    errors.append(str(error))
                for _ in range(3):
                    apply_resident(layer, x, routing, ids)
                timings = []
                for _ in range(3):
                    start = torch.Event(device=x.device, enable_timing=True)
                    end = torch.Event(device=x.device, enable_timing=True)
                    start.record()
                    for _ in range(10):
                        apply_resident(layer, x, routing, ids)
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end) / 10)
                result["routes"].append(
                    {
                        "enabled": enabled,
                        "median_ms": statistics.median(timings),
                        "samples_ms": timings,
                        "numerical_errors": errors,
                        "max_abs": (actual - expected).abs().max().item(),
                    }
                )
        finally:
            if previous is None:
                os.environ.pop("VLLM_RDNA_MOE_PREFILL_TILE16", None)
            else:
                os.environ["VLLM_RDNA_MOE_PREFILL_TILE16"] = previous
        return result

    def benchmark_hc_prefill_sp(self, rows: str = "4096"):
        """Compare actual loaded HC weights with real TP collectives."""
        if getattr(self, "_prefill_profile_active", False):
            raise RuntimeError("Disarm the profiler before benchmarking")
        count = int(rows)
        if count not in (1024, 2048, 4096):
            raise ValueError("Expected 1024, 2048 or 4096 rows")
        module = next(
            module
            for module in self.model_runner.model.modules()
            if type(module).__name__ == "GatedResidual" and module.use_combine
        )
        down = module.input_mix_weight_down_block_inject.weight
        up = module.input_mix_weight_up.weight
        generator = torch.Generator(device=down.device).manual_seed(620)
        x = torch.randn(
            count,
            down.shape[1],
            device=down.device,
            dtype=down.dtype,
            generator=generator,
        )

        def call():
            return torch.ops.vllm.rdna_hc_mix(
                x,
                down,
                None,
                None,
                up,
                None,
                None,
                module.lora_rank,
                module.hc_count,
            )

        previous = os.environ.get("VLLM_RDNA_HC_PREFILL_SP")
        result = {"rank": self.rank, "rows": count, "routes": []}
        try:
            os.environ["VLLM_RDNA_HC_PREFILL_SP"] = "0"
            expected = call()
            for enabled in ("0", "1", "0"):
                os.environ["VLLM_RDNA_HC_PREFILL_SP"] = enabled
                actual = call()
                errors = []
                for value, reference in zip(actual, expected):
                    try:
                        torch.testing.assert_close(value, reference)
                    except AssertionError as error:
                        errors.append(str(error))
                for _ in range(3):
                    call()
                timings = []
                for _ in range(3):
                    start = torch.Event(device=x.device, enable_timing=True)
                    end = torch.Event(device=x.device, enable_timing=True)
                    start.record()
                    for _ in range(10):
                        call()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end) / 10)
                result["routes"].append(
                    {
                        "enabled": enabled,
                        "median_ms": statistics.median(timings),
                        "samples_ms": timings,
                        "numerical_errors": errors,
                        "max_abs": [
                            (value - reference).abs().max().item()
                            for value, reference in zip(actual, expected)
                        ],
                    }
                )
        finally:
            if previous is None:
                os.environ.pop("VLLM_RDNA_HC_PREFILL_SP", None)
            else:
                os.environ["VLLM_RDNA_HC_PREFILL_SP"] = previous
        return result

    def set_hc_prefill_sp(self, enabled: str):
        if enabled not in ("0", "1"):
            raise ValueError("Expected 0 or 1")
        if getattr(self, "_prefill_profile_active", False):
            raise RuntimeError("Disarm the profiler before changing the route")
        os.environ["VLLM_RDNA_HC_PREFILL_SP"] = enabled
        return {"rank": self.rank, "hc_prefill_sp": enabled}

    def start_prefill_stage_profile(self, minimum_tokens: int = 512):
        if getattr(self, "_prefill_profile_active", False):
            return {"rank": self.rank, "error": "Prefill profiler already armed"}
        self._prefill_profile_events = []
        self._prefill_profile_restores = []
        self._prefill_profile_active = True

        def instrument(module, method, category):
            original = getattr(module, method)

            @wraps(original)
            def measured(*args, **kwargs):
                value = args[0] if args else kwargs.get("hidden_states")
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape[0] < minimum_tokens
                ):
                    return original(*args, **kwargs)
                start = torch.Event(device=value.device, enable_timing=True)
                end = torch.Event(device=value.device, enable_timing=True)
                start.record()
                result = original(*args, **kwargs)
                end.record()
                self._prefill_profile_events.append((category, start, end))
                return result

            self._prefill_profile_restores.append((module, method, original))
            setattr(module, method, measured)

        categories = {
            "Qwen4ExpSparseMoeBlock": "moe",
            "QwenGatedDeltaNetAttention": "gdn",
            "Qwen4ExpQSAAttention": "qsa",
            "Qwen4ExpPLELayer": "ple",
        }
        for _, module in self.model_runner.model.named_modules():
            kind = type(module).__name__
            if kind == "GatedResidual":
                instrument(module, "mix", "hc_mix")
                instrument(module, "combine_and_mix", "hc_combine_mix")
            elif kind in categories:
                instrument(module, "forward", categories[kind])
        return {"rank": self.rank, "minimum_tokens": minimum_tokens}

    def stop_prefill_stage_profile(self):
        if not getattr(self, "_prefill_profile_active", False):
            raise RuntimeError("Prefill profiler not armed")
        for module, method, original in self._prefill_profile_restores:
            setattr(module, method, original)
        self._prefill_profile_active = False
        torch.accelerator.synchronize()
        totals = defaultdict(lambda: {"calls": 0, "gpu_span_ms": 0.0})
        for category, start, end in self._prefill_profile_events:
            totals[category]["calls"] += 1
            totals[category]["gpu_span_ms"] += start.elapsed_time(end)
        self._prefill_profile_events = []
        self._prefill_profile_restores = []
        return {"rank": self.rank, "stages": dict(totals)}
