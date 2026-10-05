# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check three long chats through tool calls and cached continuations.

Repeated filler is deliberately a cache-pressure fixture, not a speed benchmark.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def request(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()[:4000]) from error
    return {
        "wall_seconds": time.perf_counter() - started,
        "message": result["choices"][0]["message"],
        "usage": result.get("usage", {}),
        "finish_reason": result["choices"][0].get("finish_reason"),
    }


def metrics(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    return [
        line
        for line in lines
        if not line.startswith("#")
        and any(key in line for key in ("offload", "prefix_cache", "kv_cache_usage"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8082")
    parser.add_argument("--model", default="active")
    parser.add_argument("--words", type=int, default=99_000)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stagger-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.stagger_seconds < 0:
        parser.error("Tool-return staggering must be nonnegative")
    tool = {
        "type": "function",
        "function": {
            "name": "read_session_marker",
            "description": "Read this chat's marker and checksum.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "integer"}},
                "required": ["session_id"],
                "additionalProperties": False,
            },
        },
    }
    chats = [
        [
            {
                "role": "system",
                "content": "Follow instructions and use tools when asked.",
            },
            {
                "role": "user",
                "content": (
                    f"Unique session {args.tag}-{index}. Background context follows.\n"
                    + " detail" * args.words
                    + f"\nCall read_session_marker with session_id={index} now. "
                    "Do not answer until you receive the tool result."
                ),
            },
        ]
        for index in range(3)
    ]
    common = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 192,
        "tools": [tool],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result: dict[str, Any] = {
        "tag": args.tag,
        "stagger_seconds": args.stagger_seconds,
        "metrics_before": metrics(args.base_url),
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        cold = list(
            pool.map(
                lambda messages: request(
                    args.base_url, {**common, "messages": messages}
                ),
                chats,
            )
        )
        result["cold_tool_calls"] = cold
        for index, response in enumerate(cold):
            message = response["message"]
            calls = message.get("tool_calls") or []
            if len(calls) != 1:
                raise RuntimeError(
                    f"Session {index}: expected one tool call, received {message}"
                )
            call = calls[0]
            function = call["function"]
            if function["name"] != "read_session_marker" or json.loads(
                function["arguments"]
            ) != {"session_id": index}:
                raise RuntimeError(f"Session {index}: wrong tool call: {call}")
            chats[index].append(message)
            chats[index].append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        {
                            "marker": f"{args.tag}-SESSION-{index}",
                            "checksum": 42 + index,
                        }
                    ),
                }
            )
            chats[index].append(
                {
                    "role": "user",
                    "content": (
                        "Reply only with marker:checksum from the tool result. "
                        "Do not call another tool."
                    ),
                }
            )
        time.sleep(5)
        result["metrics_after_cold"] = metrics(args.base_url)

        def continuation(item):
            index, messages = item
            delay = index * args.stagger_seconds
            time.sleep(delay)
            response = request(args.base_url, {**common, "messages": messages})
            response["tool_return_delay_seconds"] = delay
            return response

        for turn in range(2):
            responses = list(pool.map(continuation, enumerate(chats)))
            for index, response in enumerate(responses):
                expected = f"{args.tag}-SESSION-{index}:{42 + index}"
                actual = response["message"].get("content") or ""
                response["expected"] = expected
                response["valid"] = expected in actual and not response["message"].get(
                    "tool_calls"
                )
                chats[index].append(response["message"])
                chats[index].append(
                    {
                        "role": "user",
                        "content": (
                            "Repeat the same marker:checksum from the tool result. "
                            "Nothing else."
                        ),
                    }
                )
            result[f"continuation_{turn + 1}"] = responses
            time.sleep(5)
            result[f"metrics_after_continuation_{turn + 1}"] = metrics(args.base_url)
    result["passed"] = all(
        response["valid"]
        for turn in range(2)
        for response in result[f"continuation_{turn + 1}"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
