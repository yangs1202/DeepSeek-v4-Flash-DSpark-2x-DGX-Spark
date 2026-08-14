#!/usr/bin/env python3
import argparse
import asyncio
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request_json(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def build_prompt(base_url, model, target, nonce):
    text = f"unique request {nonce} " + "benchmark context datum " * max(1, target // 3)
    while True:
        count = request_json(
            f"{base_url.removesuffix('/v1')}/tokenize", {"model": model, "prompt": text}
        )["count"]
        if count >= target:
            return text
        text += "benchmark context datum " * max(1, (target - count) // 3)


def stream_one(base_url, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt + f"\nReturn exactly {max_tokens} numbered lowercase English words, then stop.",
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first is None and (delta.get("content") or delta.get("reasoning")):
                first = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    first = first or finished
    output_tokens = (usage or {}).get("completion_tokens", 0)
    return {
        "ttft_s": first - started,
        "elapsed_s": finished - started,
        "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / max(0.001, finished - first),
    }


async def run_case(base_url, model, target, max_tokens, concurrency, repeats):
    cases = []
    for repeat in range(repeats):
        prompts = await asyncio.gather(*[
            asyncio.to_thread(
                build_prompt,
                base_url,
                model,
                target,
                f"p{target}-c{concurrency}-r{repeat}-i{index}",
            )
            for index in range(concurrency)
        ])
        started = time.perf_counter()
        results = await asyncio.gather(*[
            asyncio.to_thread(stream_one, base_url, model, prompt, max_tokens)
            for prompt in prompts
        ])
        elapsed = time.perf_counter() - started
        total = sum(item["output_tokens"] for item in results)
        cases.append({
            "repeat": repeat + 1,
            "concurrency": concurrency,
            "elapsed_s": elapsed,
            "aggregate_tok_s": total / max(0.001, elapsed),
            "median_ttft_s": statistics.median(item["ttft_s"] for item in results),
            "median_output_tok_s": statistics.median(item["output_tok_s"] for item in results),
            "requests": results,
        })
        print(json.dumps(cases[-1], sort_keys=True), flush=True)
    return cases


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--c1-repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "conditions": {
            "prompt_tokens": args.prompt_tokens,
            "thinking": False,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": True,
        },
        "model": args.model,
        "base_url": args.base_url,
        "cases": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for concurrency in [int(value) for value in args.concurrency.split(",")]:
        repeats = args.c1_repeats if concurrency == 1 else 1
        report["cases"].extend(
            await run_case(
                args.base_url,
                args.model,
                args.prompt_tokens,
                args.max_tokens,
                concurrency,
                repeats,
            )
        )
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")


asyncio.run(main())
