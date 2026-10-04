# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run locked quality checkers with thinking and an explicit larger token budget.

This diagnostic is NOT canonical llm-context-bench: output budgets differ.
Requires llm-context-bench on PYTHONPATH. Model weights/precision are unchanged.
"""

import argparse
from pathlib import Path

from llm_context_bench.runner import (
    chat_once,
    load_suite,
    render_prompt,
    score_response,
    utc_now,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="active")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--token-floor", type=int, default=1024)
    args = parser.parse_args()
    if args.token_floor < 1:
        parser.error("--token-floor must be positive")

    result = {
        "diagnostic": True,
        "canonical": False,
        "thinking": True,
        "token_floor": args.token_floor,
        "started_at": utc_now(),
        "status": "running",
        "suites": {},
    }
    for name in ("regular", "coding"):
        suite, suite_hash = load_suite(name)
        section = {"suite_sha256": suite_hash, "cases": [], "passed": 0, "total": 0}
        result["suites"][name] = section
        for case in suite["cases"]:
            budget = max(args.token_floor, case["max_tokens"])
            trial = chat_once(
                args.base_url,
                "",
                args.model,
                suite["system_prompt"],
                render_prompt(case),
                budget,
                suite["sampling"],
                1800,
                request_tag=f"{args.tag}-{case['id']}",
                chat_template_kwargs={"enable_thinking": True},
                engine="vllm",
            )
            trial["passed"], trial["score_detail"] = (
                score_response(trial["response"], case["checker"])
                if trial["ok"]
                else (False, trial["error"])
            )
            section["cases"].append(
                {
                    "id": case["id"],
                    "original_max_tokens": case["max_tokens"],
                    "diagnostic_max_tokens": budget,
                    "trial": trial,
                }
            )
            section["passed"] += int(trial["passed"])
            section["total"] += 1
            result["last_case"] = case["id"]
            write_json_atomic(args.output, result)
            print(
                f"{case['id']}: {trial['passed']} ({trial['score_detail']})", flush=True
            )
    result["status"] = "complete"
    result["finished_at"] = utc_now()
    write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
