# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distinguish missing model answers from transport failures in quality runs."""

import copy
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

TOOL = Path(__file__).resolve().parents[2] / "tools/rdna2/check_thinking_quality.py"


class ThinkingQualityTests(unittest.TestCase):
    def run_case(self, trial):
        runner = types.ModuleType("llm_context_bench.runner")
        chat_once = Mock(return_value=trial)
        load_suite = Mock(
            return_value=(
                {
                    "cases": [{"id": "one", "max_tokens": 1, "checker": {}}],
                    "system_prompt": "",
                    "sampling": {},
                },
                "fixture-hash",
            )
        )
        artifacts = []
        runner.__dict__.update(
            chat_once=chat_once,
            load_suite=load_suite,
            render_prompt=Mock(return_value="prompt"),
            score_response=Mock(return_value=(False, "empty final answer")),
            utc_now=Mock(return_value="fixed-time"),
            write_json_atomic=lambda _, result: artifacts.append(copy.deepcopy(result)),
        )
        modules = {
            "llm_context_bench": types.ModuleType("llm_context_bench"),
            "llm_context_bench.runner": runner,
        }
        argv = [
            str(TOOL),
            "--base-url",
            "http://unused",
            "--tag",
            "test",
            "--output",
            "unused.json",
        ]
        exit_code = 0
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv):
            try:
                runpy.run_path(str(TOOL), run_name="__main__")
            except SystemExit as exc:
                if not isinstance(exc.code, int):
                    raise
                exit_code = exc.code
        return artifacts[-1], exit_code, chat_once.call_count

    def test_empty_answer_at_token_limit_is_quality_failure_not_engine_error(self):
        result, code, calls = self.run_case(
            {
                "ok": False,
                "response": "",
                "error": None,
                "http_status": 200,
                "finish_reason": "length",
            }
        )
        self.assertEqual((result["status"], code, calls), ("complete", 0, 2))
        self.assertEqual(result["engine_errors"], 0)
        self.assertEqual(result["suites"]["coding"]["passed"], 0)

    def test_transport_failure_stops_without_scoring_disconnected_cases(self):
        for status in (500, None):
            with self.subTest(http_status=status):
                result, code, calls = self.run_case(
                    {
                        "ok": False,
                        "response": "",
                        "error": "engine disconnected",
                        "http_status": status,
                    }
                )
                self.assertEqual((result["status"], code, calls), ("invalid", 2, 1))
                self.assertEqual(result["engine_errors"], 1)


if __name__ == "__main__":
    unittest.main()
