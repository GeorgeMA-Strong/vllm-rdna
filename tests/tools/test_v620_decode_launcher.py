# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling must be opt-in and leave the established decode recipe intact."""

import json
import os
import runpy
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
V620DecodeTrace = runpy.run_path(str(ROOT / "tools/rdna2/v620_decode_trace.py"))[
    "V620DecodeTrace"
]


class DecodeLauncherTests(unittest.TestCase):
    def launch(self, **settings):
        env = {k: v for k, v in os.environ.items() if not k.startswith("V620_")}
        env.update(settings)
        return subprocess.run(
            ["bash", str(ROOT / "tools/rdna2/serve_v620_baseline.sh"), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_default_stays_mtp2(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["num_speculative_tokens"], 2)

    def test_profiler_is_opt_in_and_bounded(self):
        normal = shlex.split(self.launch().stdout)
        self.assertNotIn("--profiler-config", normal)
        profiled = self.launch(V620_PROFILE_DIR="/tmp/v620 trace")
        self.assertEqual(profiled.returncode, 0, profiled.stderr)
        args = shlex.split(profiled.stdout)
        config = json.loads(args[args.index("--profiler-config") + 1])
        self.assertEqual(config["torch_profiler_dir"], "/tmp/v620 trace")
        self.assertEqual(config["max_iterations"], 16)
        self.assertTrue(config["ignore_frontend"])
        for option in ("--speculative-config", "--compilation-config"):
            self.assertEqual(
                args[args.index(option) + 1], normal[normal.index(option) + 1]
            )

    def test_profiler_path_is_json_escaped(self):
        path = '/tmp/trace "quoted" \\ newline\n tab\t'
        args = shlex.split(self.launch(V620_PROFILE_DIR=path).stdout)
        config = json.loads(args[args.index("--profiler-config") + 1])
        self.assertEqual(config["torch_profiler_dir"], path)

    def test_raw_profiler_extension_is_opt_in(self):
        normal = shlex.split(self.launch().stdout)
        self.assertNotIn("--worker-extension-cls", normal)
        profiled = self.launch(V620_WORKER_EXTENSION_CLS="tools.rdna2.trace.V620Trace")
        self.assertEqual(profiled.returncode, 0, profiled.stderr)
        args = shlex.split(profiled.stdout)
        self.assertEqual(
            args[args.index("--worker-extension-cls") + 1],
            "tools.rdna2.trace.V620Trace",
        )
        for option in ("--speculative-config", "--compilation-config"):
            self.assertEqual(
                args[args.index(option) + 1], normal[normal.index(option) + 1]
            )


class RawTraceTests(unittest.TestCase):
    def test_bounded_trace_restores_execute_model_after_two_steps(self):
        worker = V620DecodeTrace()
        worker.rank = 0
        worker.execute_model = lambda value: value + 1
        roctx = MagicMock()
        roctx.roctxProfilerResume.return_value = 0
        roctx.roctxProfilerPause.return_value = 0
        torch = SimpleNamespace(accelerator=SimpleNamespace(synchronize=MagicMock()))
        with (
            tempfile.TemporaryDirectory() as path,
            patch.dict(
                os.environ,
                {"ROCPROF_SELECTED_REGIONS": "1", "V620_RAW_TRACE_OUTPUT": path},
            ),
            patch.dict(sys.modules, {"torch": torch}),
            patch("ctypes.CDLL", return_value=roctx),
        ):
            worker.v620_start_decode_trace(steps="2", delay="1")
            self.assertEqual([worker.execute_model(n) for n in range(4)], [1, 2, 3, 4])
            result = json.loads((Path(path) / "steps-rank0.json").read_text())
            self.assertEqual(result["steps"], 2)
            self.assertEqual(roctx.roctxProfilerResume.call_count, 1)
            self.assertEqual(roctx.roctxProfilerPause.call_count, 1)
            self.assertEqual(worker.execute_model(4), 5)


if __name__ == "__main__":
    unittest.main()
