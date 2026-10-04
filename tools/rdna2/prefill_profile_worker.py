# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in worker extension measuring prefill stages with GPU events.

Enable only in an isolated diagnostic server with --worker-extension-cls
tools.rdna2.prefill_profile_worker.PrefillProfileWorker. Arm/disarm through
collective_rpc; no synchronization is added to the measured forward calls.
"""

import os
from collections import defaultdict
from functools import wraps

import torch


class PrefillProfileWorker:
    def set_hc_prefill_sp(self, enabled: str):
        if enabled not in ("0", "1"):
            raise ValueError("Expected 0 or 1")
        if getattr(self, "_prefill_profile_active", False):
            raise RuntimeError("Disarm the profiler before changing the route")
        os.environ["VLLM_RDNA_HC_PREFILL_SP"] = enabled
        return {"rank": self.rank, "hc_prefill_sp": enabled}

    def start_prefill_stage_profile(self, minimum_tokens: int = 512):
        if getattr(self, "_prefill_profile_active", False):
            raise RuntimeError("Prefill profiler already armed")
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
