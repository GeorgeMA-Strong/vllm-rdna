# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded V620 GPU trace extension, adapted from the September raw profiler."""

import ctypes
import json
import os
import sysconfig
import time
from pathlib import Path


def _sdk_library(name: str) -> Path:
    return Path(sysconfig.get_path("purelib")) / "_rocm_sdk_core/lib" / name


class V620DecodeTrace:
    """Capture selected decode steps without changing the model or its inputs."""

    def v620_start_decode_trace(self, steps: str = "4", delay: str = "2") -> dict:
        import torch

        assert os.environ.get("ROCPROF_SELECTED_REGIONS") == "1"
        assert getattr(self, "_v620_decode_trace", None) is None
        step_count, delay_count = int(steps), int(delay)
        assert 1 <= step_count <= 16 and 0 <= delay_count <= 16
        roctx = ctypes.CDLL(str(_sdk_library("librocprofiler-sdk-roctx.so.1")))
        for name in ("roctxProfilerResume", "roctxProfilerPause"):
            fn = getattr(roctx, name)
            fn.argtypes = [ctypes.c_uint32]
            fn.restype = ctypes.c_int
        state = {
            "original": self.execute_model,
            "roctx": roctx,
            "steps": step_count,
            "delay": delay_count,
            "count": 0,
            "active": False,
        }
        self._v620_decode_trace = state

        def execute(*args, **kwargs):
            if state["count"] == delay_count:
                torch.accelerator.synchronize()
                assert roctx.roctxProfilerResume(0) == 0
                state["started_ns"] = time.monotonic_ns()
                state["active"] = True
            if state["count"] == delay_count + step_count:
                self.v620_stop_decode_trace()
            state["count"] += 1
            try:
                return state["original"](*args, **kwargs)
            except BaseException:
                self.v620_stop_decode_trace()
                raise

        self.execute_model = execute
        return {"rank": self.rank, "steps": step_count, "delay": delay_count}

    def v620_stop_decode_trace(self) -> dict:
        import torch

        state = getattr(self, "_v620_decode_trace", None)
        if state is None:
            return getattr(self, "_v620_decode_result", {"rank": self.rank})
        self.execute_model = state["original"]
        if state["active"]:
            torch.accelerator.synchronize()
            assert state["roctx"].roctxProfilerPause(0) == 0
        result = {
            "rank": self.rank,
            "steps": max(0, state["count"] - state["delay"]),
            "start_ns": state.get("started_ns"),
            "end_ns": time.monotonic_ns(),
        }
        output = Path(os.environ["V620_RAW_TRACE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"steps-rank{self.rank}.json").write_text(json.dumps(result))
        self._v620_decode_trace = None
        self._v620_decode_result = result
        return result

    def v620_flush_decode_trace(self) -> dict:
        """Finalize already-paused ROCm traces before worker exit."""
        import torch

        assert getattr(self, "_v620_decode_trace", None) is None
        assert not getattr(self, "_v620_decode_flushed", False)
        torch.accelerator.synchronize()
        tool = ctypes.CDLL(
            str(_sdk_library("rocprofiler-sdk/librocprofiler-sdk-tool.so.1")),
            mode=os.RTLD_NOW | os.RTLD_NOLOAD,
        )

        class AttachConfig(ctypes.Structure):
            _fields_ = [
                ("size", ctypes.c_size_t),
                ("attach", ctypes.c_void_p),
                ("detach", ctypes.c_void_p),
                ("data", ctypes.c_void_p),
            ]

        configure = tool.rocprofiler_configure_attach
        configure.argtypes = [
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        configure.restype = ctypes.POINTER(AttachConfig)
        config = configure(10305, b"v620-rocprof-1.3.5", 0, None).contents
        assert config.size == ctypes.sizeof(AttachConfig) and config.detach
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(config.detach)(config.data)
        self._v620_decode_flushed = True
        return {"rank": self.rank, "pid": os.getpid(), "flushed": True}
