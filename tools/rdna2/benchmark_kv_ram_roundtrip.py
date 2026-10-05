# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a long-context RAM KV-cache eviction and reload round trip."""

import argparse
import json
import time
import urllib.request
from typing import Any


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _metrics(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        body = response.read().decode()
    needles = ("offload", "prefix_cache", "kv_cache_usage")
    return [
        line
        for line in body.splitlines()
        if not line.startswith("#") and any(needle in line for needle in needles)
    ]


def _stream_chat(base_url: str, model: str, content: str) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 96,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens", 0)
    decode_seconds = max(ended - (first_token_at or ended), 1e-9)
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "total_seconds": ended - started,
        "decode_tokens_per_second": completion_tokens / decode_seconds,
        "text": "".join(text_parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="active")
    parser.add_argument("--settle-seconds", type=float, default=15)
    args = parser.parse_args()

    _get_json(f"{args.base_url}/v1/models")
    session_a = (
        "Session A sentinel is ALPHA-SESSION-200K. Preserve it."
        + " alpha" * 199_900
        + "\nReply exactly with ALPHA-SESSION-200K, then one short sentence."
    )
    session_b = (
        "Session B sentinel is BETA-SESSION-100K. Preserve it."
        + " beta" * 99_900
        + "\nReply exactly with BETA-SESSION-100K, then one short sentence."
    )

    result: dict[str, Any] = {"metrics_before": _metrics(args.base_url)}
    result["session_a_cold"] = _stream_chat(args.base_url, args.model, session_a)
    time.sleep(args.settle_seconds)
    result["metrics_after_a"] = _metrics(args.base_url)

    result["session_b_switch"] = _stream_chat(args.base_url, args.model, session_b)
    time.sleep(args.settle_seconds)
    result["metrics_after_b"] = _metrics(args.base_url)

    continuation = (
        session_a
        + result["session_a_cold"]["text"]
        + "\nContinue session A. Reply exactly with ALPHA-SESSION-RELOADED, "
        "then one short sentence."
    )
    result["session_a_reload"] = _stream_chat(args.base_url, args.model, continuation)
    time.sleep(args.settle_seconds)
    result["metrics_after_a_reload"] = _metrics(args.base_url)
    result["content_valid"] = (
        "ALPHA-SESSION-200K" in result["session_a_cold"]["text"]
        and "BETA-SESSION-100K" in result["session_b_switch"]["text"]
        and "ALPHA-SESSION-RELOADED" in result["session_a_reload"]["text"]
    )

    def counter(lines: list[str], name: str, label: str = "") -> float:
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in lines
            if line.startswith(name + "{") and label in line
        )

    for key, name, label in (
        (
            "cpu_to_gpu_bytes_reload_delta",
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="CPU_to_GPU"',
        ),
        (
            "external_hit_tokens_reload_delta",
            "vllm:external_prefix_cache_hits_total",
            "",
        ),
    ):
        result[key] = counter(result["metrics_after_a_reload"], name, label) - counter(
            result["metrics_after_b"], name, label
        )
    result["passed"] = (
        result["content_valid"]
        and result["cpu_to_gpu_bytes_reload_delta"] > 0
        and result["external_hit_tokens_reload_delta"] > 0
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
