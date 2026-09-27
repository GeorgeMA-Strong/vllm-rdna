# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling must be opt-in and leave the established decode recipe intact."""

import json
import os
import shlex
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


if __name__ == "__main__":
    unittest.main()
