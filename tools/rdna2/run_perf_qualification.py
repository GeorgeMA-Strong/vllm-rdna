# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the existing benchmark with a bounded graph census around each case."""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def rpc(base_url, method, kwargs):
    request = urllib.request.Request(
        base_url + "/collective_rpc",
        data=json.dumps({"method": method, "kwargs": kwargs}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["16k", "64k", "128k"])
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--suite", default="coding")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--request-tag-scope", default="perf-qualification-20261005")
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--save-reference", type=Path)
    parser.add_argument("--compare-reference", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False, parents=True)
    if args.save_reference or args.compare_reference:
        reference = args.save_reference or args.compare_reference
        if args.save_reference:
            reference.mkdir(exist_ok=False, parents=True)
        result = rpc(
            args.base_url,
            "check_actual_moe_weights",
            {
                "directory": str(reference),
                "save": bool(args.save_reference),
            },
        )
        (args.directory / "actual-weights.json").write_text(json.dumps(result))
        print(json.dumps({"actual_weights": result}), flush=True)
    command_description = args.launch.read_text()
    for concurrency in args.concurrencies:
        name = f"{args.profile}-c{concurrency}"
        if args.census:
            begin = rpc(args.base_url, "begin_perf_phase", {"name": name})
            print(json.dumps({"begin": begin}), flush=True)
        try:
            command = [
                sys.executable,
                "-m",
                "llm_context_bench",
                "--base-url",
                args.base_url,
                "--model",
                "active",
                "--engine",
                "vllm",
                "--profile",
                name,
                "--suite",
                args.suite,
                "--lane",
                "performance",
                "--sizes",
                *args.sizes,
                "--concurrency",
                str(concurrency),
                "--repetitions",
                str(args.repetitions),
                "--input-size-tolerance-percent",
                "12",
                "--chat-template-kwargs",
                '{"enable_thinking":false}',
                "--max-retries",
                "0",
                "--timeout",
                "1800",
                "--request-tag-scope",
                f"{args.request_tag_scope}-c{concurrency}",
                "--output",
                str(args.directory / f"c{concurrency}.json"),
                "--command",
                command_description,
                "--system",
                "4xV620 gfx1030 TP4 EP4 FP16 BF16 CPU PLE RAM KV64GiB",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = "/home/george/bench/src/llm-context-bench/src"
            env["PYTHONUNBUFFERED"] = "1"
            (args.directory / f"c{concurrency}-command.json").write_text(
                json.dumps(command, indent=2)
            )
            with (args.directory / f"c{concurrency}.log").open("x") as output:
                subprocess.run(
                    command,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report_path = args.directory / f"c{concurrency}.json"
            report = json.loads(report_path.read_text())
            rows = report.get("performance_table", [])
            if (
                report.get("status") != "complete"
                or not rows
                or any(not row.get("performance_valid") for row in rows)
            ):
                raise RuntimeError(f"Invalid benchmark report: {report_path}")
        finally:
            if args.census:
                end = rpc(args.base_url, "end_perf_phase", {})
                (args.directory / f"c{concurrency}-census.json").write_text(
                    json.dumps({"begin": begin, "end": end}, indent=2)
                )
                print(json.dumps({"end": end}), flush=True)


if __name__ == "__main__":
    main()
