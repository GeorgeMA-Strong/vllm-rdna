# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a locked coding task with/without its irrelevant source context.

Requires llm-context-bench on PYTHONPATH. This is an accuracy diagnostic, not
a speed benchmark; the manifest's sampling and output checker are reused.
"""

import argparse
import json

import regex as re
from llm_context_bench.runner import (
    chat_once,
    load_suite,
    render_prompt,
    score_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="active")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--size", choices=("8k", "16k", "32k", "64k", "128k"), default="16k"
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    suite, suite_hash = load_suite("coding")
    case = next(
        case for case in suite["cases"] if case["id"].endswith(f"context-{args.size}")
    )
    full = render_prompt(case)
    functions = re.findall(
        r"(?m)^def (?:billing_units|retry_window|shard_name)\([^\n]*\):\n"
        r"(?:[ \t]+[^\n]*\n)+",
        full,
    )
    if len(functions) != 3:
        raise ValueError("Expected exactly three authoritative fixture functions")
    task = full.rsplit("\nTASK:", 1)[1]
    short = "\n".join(functions) + "\nTASK:" + task
    output = {
        "suite_sha256": suite_hash,
        "case_id": case["id"],
        "source_sha256": case["prompt_sha256"],
        "thinking": args.thinking,
        "max_tokens": args.max_tokens or case["max_tokens"],
        "short_prompt": short,
        "trials": {},
    }
    for name, prompt in (("short", short), ("full", full)):
        trial = chat_once(
            args.base_url,
            "",
            args.model,
            suite["system_prompt"],
            prompt,
            output["max_tokens"],
            suite["sampling"],
            1800,
            request_tag=f"{args.tag}-{name}",
            chat_template_kwargs={"enable_thinking": args.thinking},
            engine="vllm",
        )
        trial["passed"], trial["score_detail"] = (
            score_response(trial["response"], case["checker"])
            if trial["ok"]
            else (False, trial["error"])
        )
        output["trials"][name] = trial
    output["passed"] = all(t["passed"] for t in output["trials"].values())
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
