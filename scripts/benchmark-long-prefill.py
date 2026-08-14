#!/usr/bin/env python3
import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def request_json(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def build_prompt(base_url, model, target_tokens, nonce):
    unit = " benchmark context datum"
    text = f"unique-{nonce}"
    while True:
        count = request_json(
            f"{base_url.removesuffix('/v1')}/tokenize",
            {"model": model, "prompt": text},
        )["count"]
        if count >= target_tokens:
            return text, count
        text += unit * max(1, (target_tokens - count) // 3)


def stream_request(url, model, messages, max_tokens, label):
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.perf_counter()
    first = None
    usage = None
    try:
        with requests.post(url, json=body, stream=True, timeout=3600) as response:
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
                choices = event.get("choices") or []
                if choices and first is None:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") is not None:
                        first = time.perf_counter()
    except Exception as error:
        return {
            "label": label,
            "error": repr(error),
            "elapsed_s": time.perf_counter() - started,
        }
    finished = time.perf_counter()
    details = (usage or {}).get("prompt_tokens_details") or {}
    return {
        "label": label,
        "ttft_s": (first or finished) - started,
        "elapsed_s": finished - started,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
    }


def metric_value(metrics, name):
    for line in metrics.splitlines():
        if line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--long-prompt-tokens", type=int, default=84000)
    parser.add_argument("--long-output-tokens", type=int, default=64)
    parser.add_argument("--long-count", type=int, default=1)
    parser.add_argument("--short-count", type=int, default=3)
    parser.add_argument("--short-delay", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts = []
    tokenized_counts = []
    for index in range(args.long_count):
        prompt, tokenized_count = build_prompt(
            args.base_url,
            args.model,
            args.long_prompt_tokens,
            f"{time.time_ns()}-{index}",
        )
        prompts.append(prompt)
        tokenized_counts.append(tokenized_count)
    chat_url = f"{args.base_url}/chat/completions"
    long_messages = [
        [{"role": "user", "content": prompt + "\nReply only OK."}]
        for prompt in prompts
    ]
    short_messages = [{"role": "user", "content": "Reply only OK."}]
    samples = []
    stop = threading.Event()

    def sample_metrics():
        metrics_url = f"{args.base_url.removesuffix('/v1')}/metrics"
        while not stop.is_set():
            try:
                metrics = requests.get(metrics_url, timeout=5).text
                samples.append(
                    {
                        "running": metric_value(metrics, "vllm:num_requests_running"),
                        "waiting": metric_value(metrics, "vllm:num_requests_waiting"),
                        "kv_usage": metric_value(metrics, "vllm:gpu_cache_usage_perc"),
                    }
                )
            except Exception:
                pass
            stop.wait(0.1)

    metrics_thread = threading.Thread(target=sample_metrics)
    metrics_thread.start()
    long_results = [None] * args.long_count

    def run_long(index):
        long_results[index] = stream_request(
            chat_url,
            args.model,
            long_messages[index],
            args.long_output_tokens,
            f"long-{index}",
        )

    long_threads = [
        threading.Thread(target=run_long, args=(index,))
        for index in range(args.long_count)
    ]
    for long_thread in long_threads:
        long_thread.start()
    time.sleep(args.short_delay)
    with ThreadPoolExecutor(max_workers=args.short_count) as executor:
        shorts = list(
            executor.map(
                lambda index: stream_request(
                    chat_url, args.model, short_messages, 8, f"short-{index}"
                ),
                range(args.short_count),
            )
        )
    for long_thread in long_threads:
        long_thread.join()
    stop.set()
    metrics_thread.join()

    report = {
        "conditions": {
            "long_prompt_target": args.long_prompt_tokens,
            "long_prompt_tokenized": tokenized_counts,
            "long_body_bytes": [
                len(json.dumps(messages).encode()) for messages in long_messages
            ],
            "long_count": args.long_count,
            "long_output_tokens": args.long_output_tokens,
            "short_count": args.short_count,
            "short_delay_s": args.short_delay,
            "thinking": False,
        },
        "longs": long_results,
        "shorts": shorts,
        "summary": {
            "short_errors": sum("error" in row for row in shorts),
            "median_short_ttft_s": statistics.median(
                row["ttft_s"] for row in shorts if "ttft_s" in row
            ),
            "max_short_ttft_s": max(
                row["ttft_s"] for row in shorts if "ttft_s" in row
            ),
            "peak_running": max((row["running"] or 0) for row in samples),
            "peak_waiting": max((row["waiting"] or 0) for row in samples),
            "peak_kv_usage": max((row["kv_usage"] or 0) for row in samples),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
