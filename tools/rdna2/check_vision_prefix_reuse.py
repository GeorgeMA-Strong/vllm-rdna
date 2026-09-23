#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check a real image and checkpoint reuse in an appended long chat."""

import argparse
import io
import json
import time
import urllib.request
import uuid
from pathlib import Path

import pybase64 as base64
from PIL import Image, ImageDraw


def get_metrics(base_url: str) -> str:
    with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
        return response.read().decode()


def counter(snapshot: str, name: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in snapshot.splitlines()
        if line.startswith("vllm:" + name + "{")
    )


def call(
    base_url: str,
    model: str,
    messages: list[dict],
    cache_salt: str,
    name: str,
) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    chunks = []
    usage = None
    with urllib.request.urlopen(request, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                continue
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                token = choice.get("delta", {}).get("content")
                if token:
                    if first is None:
                        first = time.monotonic() - start
                    chunks.append(token)
    return {
        "case": name,
        "ttft_s": first,
        "total_s": time.monotonic() - start,
        "text": "".join(chunks),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="active")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=1260)
    parser.add_argument("--lines", type=int, default=1600)
    args = parser.parse_args()

    image = Image.new("RGB", (args.image_size, args.image_size), (25, 80, 180))
    draw = ImageDraw.Draw(image)
    midpoint = args.image_size // 2
    draw.rectangle(
        (midpoint, 0, args.image_size - 1, args.image_size - 1),
        fill=(230, 165, 25),
    )
    inset = args.image_size // 3
    draw.rectangle(
        (inset, inset, args.image_size - inset, args.image_size - inset),
        outline=(255, 255, 255),
        width=max(4, args.image_size // 52),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    background = "\n".join(
        f"Record {i:04d}: amber birch cedar delta. This line is stable background data."
        for i in range(args.lines)
    )
    messages = [
        {"role": "system", "content": "Reply with the requested word only."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": background + "\nReply exactly: blue"},
            ],
        },
    ]
    cache_salt = uuid.uuid4().hex
    before = get_metrics(args.base_url)
    initial = call(args.base_url, args.model, messages, cache_salt, "initial")
    after_initial = get_metrics(args.base_url)
    followup_messages = messages + [
        {"role": "assistant", "content": initial["text"]},
        {"role": "user", "content": "Now reply exactly: green"},
    ]
    followup = call(
        args.base_url,
        args.model,
        followup_messages,
        cache_salt,
        "followup",
    )
    after_followup = get_metrics(args.base_url)
    initial["prefix_hits_delta"] = counter(
        after_initial, "prefix_cache_hits_total"
    ) - counter(before, "prefix_cache_hits_total")
    followup["prefix_hits_delta"] = counter(
        after_followup, "prefix_cache_hits_total"
    ) - counter(after_initial, "prefix_cache_hits_total")

    result = {
        "image": {
            "width": args.image_size,
            "height": args.image_size,
            "pixels": args.image_size**2,
            "png_bytes": len(buffer.getvalue()),
        },
        "results": [initial, followup],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

    assert initial["text"].strip() == "blue", initial
    assert followup["text"].strip() == "green", followup
    prompt_tokens = followup["usage"]["prompt_tokens"]
    cached_tokens = followup["prefix_hits_delta"]
    if cached_tokens < prompt_tokens * 0.7:
        raise SystemExit(
            f"FAIL: cached {cached_tokens}/{prompt_tokens}; context was recomputed"
        )
    if followup["ttft_s"] >= initial["ttft_s"] * 0.5:
        raise SystemExit(
            f"FAIL: follow-up TTFT {followup['ttft_s']:.3f}s is not below "
            f"half of initial {initial['ttft_s']:.3f}s"
        )
    print("PASS: vision and multimodal long-chat prefix reuse")


if __name__ == "__main__":
    main()
