#!/usr/bin/env python3
import argparse
import json
import threading
import time
from pathlib import Path

import requests


def request(url, model, messages, label):
    started = time.perf_counter()
    first = None
    usage = None
    with requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False},
        },
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            if event.get("choices") and first is None:
                delta = event["choices"][0].get("delta") or {}
                if delta.get("content") is not None:
                    first = time.perf_counter()
    finished = time.perf_counter()
    return {
        "label": label,
        "ttft": (first or finished) - started,
        "elapsed": finished - started,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--messages", type=int, default=591)
    parser.add_argument("--padding-bytes", type=int, default=1750)
    parser.add_argument("--small-delay", type=float, default=0.05)
    parser.add_argument("--output")
    args = parser.parse_args()

    url = f"{args.base_url}/chat/completions"
    padding = " " * args.padding_bytes
    large_messages = []
    for index in range(args.messages - 1):
        role = "user" if index % 2 == 0 else "assistant"
        large_messages.append({"role": role, "content": f"turn-{index}{padding}"})
    large_messages.append({"role": "user", "content": "Reply only OK."})
    small_messages = [{"role": "user", "content": "Reply only OK."}]
    results = {}

    def run_large():
        results["large"] = request(url, args.model, large_messages, "large")

    thread = threading.Thread(target=run_large)
    thread.start()
    time.sleep(args.small_delay)
    results["small_during_large"] = request(
        url, args.model, small_messages, "small_during_large"
    )
    thread.join()
    results["small_idle"] = request(url, args.model, small_messages, "small_idle")
    report = {
        "large_body_bytes": len(json.dumps(large_messages, ensure_ascii=False).encode()),
        "message_count": len(large_messages),
        "results": results,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
