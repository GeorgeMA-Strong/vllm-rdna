# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure cold-prefix prefill stage costs on an isolated V620 server."""

import argparse
import fcntl
import hashlib
import json
import random
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    lock_path = Path(tempfile.gettempdir()) / (
        "v620-prefill-profile-" + hashlib.sha256(args.base_url.encode()).hexdigest()
    )
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, "Another prefill profile client owns this endpoint\n")

    def post(path, payload):
        request = urllib.request.Request(
            args.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    rng = random.Random(620)
    prompt = [rng.randrange(1000, 10000) for _ in range(args.tokens)]

    def inference():
        started = time.perf_counter()
        response = post(
            "/v1/completions",
            {
                "model": "active",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex,
            },
        )
        assert response["usage"]["prompt_tokens"] == args.tokens, response
        seconds = time.perf_counter() - started
        return {"seconds": seconds, "tokens_per_second": args.tokens / seconds}

    print(json.dumps({"warmup": inference()}), flush=True)
    armed = post(
        "/collective_rpc",
        {"method": "start_prefill_stage_profile", "args": [512], "timeout": 30},
    )
    if any("error" in result for result in armed["results"]):
        raise RuntimeError(armed)
    try:
        for trial in range(args.repetitions):
            print(json.dumps({"trial": trial, **inference()}), flush=True)
    finally:
        print(
            json.dumps(
                {
                    "profile": post(
                        "/collective_rpc",
                        {"method": "stop_prefill_stage_profile", "timeout": 30},
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
