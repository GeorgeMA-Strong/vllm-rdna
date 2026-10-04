# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run serialized loaded-weight and prefill-stage off/on/off comparisons."""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wait-ready", type=float, default=900)
    parser.add_argument(
        "--candidate", choices=("hc", "moe-tile16", "moe-tile32"), default="hc"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.wait_ready
    while True:
        try:
            with urllib.request.urlopen(args.base_url + "/health", timeout=2):
                break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Server did not become healthy") from None
            time.sleep(3)

    def rpc(method, values):
        request = urllib.request.Request(
            args.base_url + "/collective_rpc",
            data=json.dumps(
                {"method": method, "args": values, "timeout": 120}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=150) as response:
            return json.load(response)

    method = "set_hc_prefill_sp" if args.candidate == "hc" else "set_moe_prefill_tile"
    baseline = "0" if args.candidate == "hc" else "8"
    candidate = (
        "1" if args.candidate == "hc" else args.candidate.removeprefix("moe-tile")
    )
    micro = (
        rpc("benchmark_hc_prefill_sp", ["4096"])
        if args.candidate == "hc"
        else rpc("benchmark_moe_prefill_tiles", [candidate])
    )
    (args.output_dir / "loaded-weight-micro.json").write_text(
        json.dumps(micro, indent=2)
    )
    print(json.dumps({"micro": micro}), flush=True)
    if any(
        route["numerical_errors"]
        for rank in micro["results"]
        for route in rank["routes"]
    ):
        raise RuntimeError("Loaded-weight numerical comparison failed")

    try:
        for phase, enabled in (
            ("control-before", baseline),
            ("candidate", candidate),
            ("control-after", baseline),
        ):
            print(
                json.dumps({"phase": phase, "rpc": rpc(method, [enabled])}),
                flush=True,
            )
            with (args.output_dir / (phase + ".jsonl")).open("w") as output:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("profile_prefill_stages.py")),
                        "--base-url",
                        args.base_url,
                        "--tokens",
                        "16384",
                        "--repetitions",
                        "3",
                    ],
                    stdout=output,
                    check=True,
                )
    finally:
        rpc(method, [baseline])


if __name__ == "__main__":
    main()
