# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small text, concurrency and vision checks against an existing server.

This is a serving smoke check, not a quantization quality benchmark.
"""

import argparse
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()

    cases: list[tuple[str, str | list[dict], str]] = [
        ("addition", "What is 19 + 23? Reply with only the number.", "42"),
        ("multiply", "What is 9 * 7? Reply with only the number.", "63"),
        ("subtract", "What is 13 - 5? Reply with only the number.", "8"),
        ("divide", "What is 36 / 4? Reply with only the number.", "9"),
    ]
    if args.vision:
        import pybase64 as base64
        from PIL import Image, ImageDraw

        picture = Image.new("RGB", (384, 256), "white")
        drawing = ImageDraw.Draw(picture)
        drawing.ellipse((16, 56, 128, 168), fill="blue")
        drawing.rectangle((152, 56, 240, 168), fill="green")
        drawing.polygon([(308, 48), (260, 168), (368, 168)], fill="red")
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        cases = [
            (
                "vision-circle",
                [
                    {
                        "type": "text",
                        "text": "What color is the circle? Reply with one word.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + data},
                    },
                ],
                "blue",
            )
        ]

    def check(case: tuple[str, str | list[dict], str]) -> dict:
        name, content, expected = case
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            answer = result["choices"][0]["message"].get("content") or ""
            normalized = answer.strip().lower().strip(".\"'` \n")
            return {
                "case": name,
                "passed": normalized == expected,
                "elapsed_seconds": time.perf_counter() - started,
                "expected": expected,
                "response": result,
            }
        except Exception as error:
            return {"case": name, "passed": False, "error": str(error)}

    # Repeat requests to exercise cache reuse and request cleanup.
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(check, cases * 2))
    passed = all(result["passed"] for result in results)
    print(
        json.dumps(
            {"passed": passed, "concurrency": args.concurrency, "results": results},
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
