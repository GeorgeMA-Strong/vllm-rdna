# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure decode responsiveness while a second chat prefills."""

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def stream_chat(
    base_url: str,
    payload: dict[str, Any],
    timestamps: list[float],
    ready: threading.Event | None = None,
) -> tuple[float, float, dict[str, Any]]:
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    began = time.monotonic()
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    timestamps.append(time.monotonic())
                    if ready is not None and len(timestamps) >= 20:
                        ready.set()
    return began, time.monotonic(), usage


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percent))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", default="active")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prefill-words", type=int, default=16_000)
    parser.add_argument("--decode-tokens", type=int, default=768)
    args = parser.parse_args()

    common = {
        "model": args.model,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    decode_payload = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Write a long passage about forests. Unique tag: {args.tag}."
                ),
            }
        ],
        "max_tokens": args.decode_tokens,
        "ignore_eos": True,
    }
    prefill_payload = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Unique tag: {args.tag}."
                    + " forest" * args.prefill_words
                    + "\nNow reply READY."
                ),
            }
        ],
        "max_tokens": 8,
    }
    decode_times: list[float] = []
    prefill_times: list[float] = []
    ready = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        decode_future = pool.submit(
            stream_chat, args.base_url, decode_payload, decode_times, ready
        )
        if not ready.wait(timeout=180):
            raise RuntimeError("decode did not emit 20 chunks before prefill")
        prefill_future = pool.submit(
            stream_chat, args.base_url, prefill_payload, prefill_times
        )
        prefill_begin, prefill_end, prefill_usage = prefill_future.result()
        decode_begin, decode_end, decode_usage = decode_future.result()

    prefill_first = prefill_times[0] if prefill_times else prefill_end
    overlap_gaps = [
        (right - left) * 1000
        for left, right in zip(decode_times, decode_times[1:])
        if left < prefill_first and right > prefill_begin
    ]
    result = {
        "tag": args.tag,
        "decode_chunks": len(decode_times),
        "decode_completion_tokens": decode_usage.get("completion_tokens"),
        "decode_wall_s": decode_end - decode_begin,
        "prefill_prompt_tokens": prefill_usage.get("prompt_tokens"),
        "prefill_ttft_s": prefill_first - prefill_begin,
        "prefill_wall_s": prefill_end - prefill_begin,
        "decode_chunks_during_prefill": sum(
            prefill_begin <= stamp < prefill_first for stamp in decode_times
        ),
        "decode_itl_p50_ms_during_prefill": (
            statistics.median(overlap_gaps) if overlap_gaps else None
        ),
        "decode_itl_p99_ms_during_prefill": percentile(overlap_gaps, 0.99),
        "decode_itl_max_ms_during_prefill": max(overlap_gaps, default=None),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
