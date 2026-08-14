# Preliminary benchmarks

These measurements were taken on two NVIDIA DGX Sparks with the live
configuration documented by this recipe. They are reference numbers, not
guarantees: runtime image, model revision, cache state, fabric, prompt shape,
and vLLM version all affect results.

## Qwen Vision

Model: `qwen3.5-9b-vision` / `RedHatAI/Qwen3.5-9B-quantized.w4a16`.
Image: local 640×480 JPEG supplied as a data URI. Each case used three runs and
31 generated tokens.

| Concurrency | Median TTFT | Median request time | Median decode | Aggregate decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.53 s | 3.12 s | 12.0 tok/s | 9.9 tok/s |
| 2 | 0.65 s | 3.19 s | 12.0 tok/s | 18.8 tok/s |

The reusable benchmark is `scripts/benchmark-qwen-vision.py`. It accepts a
local image so the result does not depend on the model server fetching a public
URL.

## DeepSeek with Qwen loaded

The Mia benchmark harness was run while the Qwen sidecar was loaded:

| Prompt target | Concurrency | Median TTFT | Median prefill | Median decode | Aggregate decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 19.05 s* | 14.7K tok/s | 54.9 tok/s | 43.1 tok/s |
| 256 | 2 | 9.62 s | 29.1K tok/s | 43.8 tok/s | 79.8 tok/s |
| 256 | 4 | 0.99 s | 303.1K tok/s | 37.8 tok/s | 74.8 tok/s |
| 2,048 | 1 | 1.57 s | 1,322.2K tok/s | 65.4 tok/s | 63.3 tok/s |
| 2,048 | 2 | 4.05 s | 511.6K tok/s | 42.7 tok/s | 65.8 tok/s |

\* The first 256-token case includes cold/warm-up behavior and should not be
used as a steady-state latency figure.

## Comparison with Mia's published result

Mia's published 2K-prompt, concurrency-1 result for the same benchmark harness
reported 68.8 decode tok/s, 62.0 aggregate tok/s, 2,563K prefill tok/s, and
0.81s TTFT. The loaded-sidecar run reported 65.4 decode tok/s, 63.3 aggregate
tok/s, 1,322K prefill tok/s, and 1.57s TTFT.

That is approximately **4.9% lower single-stream decode throughput** but **2.0%
higher aggregate throughput** in this sample. The prefill and TTFT differences
are more variable and should not be attributed solely to Qwen without a clean,
same-session A/B run; the no-Qwen retry was invalidated by stale benchmark
requests that did not close cleanly.

## Clean same-session A/B

To isolate the sidecar effect, both services were stopped between conditions,
DeepSeek was started fresh for each condition, and the same fixed prompt and
generation settings were used. The first request in each condition was a
warmup; the three subsequent serial requests produced these medians:

| Condition | Median aggregate decode |
| --- | ---: |
| DeepSeek without Qwen | 38.84 tok/s |
| DeepSeek with Qwen loaded | 42.61 tok/s |

The measured difference was **+9.7% with Qwen loaded**. This small sample shows
no measurable performance penalty, but it should be treated as directional:
short serial runs have normal variance and are not a capacity or concurrency
benchmark.

Raw output: `results/deepseek-clean-ab-2026-08-08.json`.

## Resource footprint and prior A/B status

At an idle observation after the runs, Qwen used approximately 7.0 GiB of GPU
memory and 2.5% host CPU per node. DeepSeek used approximately 94.5 GiB of GPU
memory. The earlier unmatched retry had stale requests and is retained only as
historical context; the clean A/B above is the preferred comparison.

Raw outputs:

- `results/qwen-vision-2026-08-08.json`
- `results/deepseek-with-qwen-sidecar-2026-08-08.json`
- `results/deepseek-without-qwen-sidecar-2026-08-08.json`

## DeepSeek fixed-output regression suite

Use `scripts/benchmark-fixed-output.py` on the head node to compare profiles
with deterministic, non-thinking streaming requests. The default reproduces the
256-prompt/512-output C1, C2, C4, and C6 suite used for the 2026-08-11 operating
profile decision:

```bash
python3 scripts/benchmark-fixed-output.py \
  --output results/fixed-output-$(date +%F).json
```

The recovered historical 2K-prompt/128-output C1, C4, and C6 workload can be
run with:

```bash
python3 scripts/benchmark-fixed-output.py \
  --prompt-tokens 2000 \
  --max-tokens 128 \
  --concurrency 1,4,6 \
  --c1-repeats 1 \
  --output results/profile-regression-$(date +%F).json
```

Run benchmarks from the head node against loopback so client network latency is
excluded. Compare steady-state cases after model, CUDA graph, and kernel warmup;
retain the first C1 run as cold-start evidence rather than mixing it into the
warm result.

## Renderer head-of-line regression

Use `scripts/benchmark-renderer-hol.py` to reproduce the large request shape
that motivated `--renderer-num-workers 4`. It submits a 591-message request of
about 1 MiB, then sends a small request while the large request is rendering:

```bash
python3 scripts/benchmark-renderer-hol.py \
  --output results/renderer-hol-$(date +%F).json
```

The primary assertion is that `small_during_large.ttft` remains close to
`small_idle.ttft`; the large request's own TTFT still includes its legitimate
chat rendering, tokenization, and prefill cost.

## DeepSeek V4 long-prefill tuning (2026-08-14)

`renderer_num_workers=4` exposed a tokenizer concurrency bug in the DeepSeek V4
custom renderer. Concurrent requests failed with HTTP 500 and
`RuntimeError: Already borrowed` while the shared fast tokenizer changed its
truncation state. The container startup patch now wraps the tokenizer with
vLLM's existing `maybe_make_thread_pool` helper. A 40-request C4 stress run
completed with no HTTP 500 or `Already borrowed` errors after the patch.

Use `scripts/benchmark-long-prefill.py` for cache-miss long-prefill tests. It
creates an exact-size unique prompt and starts short requests while the long
prefill is active. `--long-count 2` reproduces concurrent Codex-sized requests:

```bash
python3 scripts/benchmark-long-prefill.py \
  --base-url http://127.0.0.1:8888/v1 \
  --long-count 2 \
  --long-output-tokens 8 \
  --output results/dual-long-prefill-$(date +%F).json
```

The accepted serving profile remains `max_num_batched_tokens=16384` and
`gpu_memory_utilization=0.835`. The following isolated A/B results used an
84K-token cache-miss prompt with thinking disabled:

| Profile | 84K TTFT | KV cache | C4 aggregate | C6 aggregate | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 16K / 0.835 | 44.34-51.60 s | 1,541,217 tokens* | 127.29 tok/s | 157.71 tok/s | keep |
| 18K / 0.835 | 43.08-51.43 s | 1,351,826 tokens | 117.16 tok/s | 151.71 tok/s | reject |
| 24K / 0.850 | 55.78 s | 1,204,021 tokens | not run | not run | reject |

\* KV capacity varies slightly across boots; the accepted profile has also
reported 1,524,097 tokens. The 18K result changed average 84K TTFT by only 1.5%
while reducing C4/C6 throughput. The 24K profile caused memory PSI avg10 to
reach 12.43/18.62 on the two nodes and produced swap I/O during the long
request. 24K and 32K at utilization 0.835 could not allocate enough KV cache
for one 1,048,576-token request.

Two simultaneous 84K requests on the accepted scheduler produced TTFTs of
52.44 s and 88.21 s while three short requests remained at 0.29 s median TTFT.
This image rejects `max_num_partial_prefills > 1` before model loading with
`NotImplementedError: Concurrent Partial Prefill is not supported`, so the
scheduler defaults remain `1/1/0`.

Relevant raw outputs:

- `results/tune-baseline-fixed-output-2026-08-14.json`
- `results/long-prefill-baseline-safe-renderer-2026-08-14-r1.json`
- `results/long-prefill-baseline-safe-renderer-2026-08-14-r2.json`
- `results/long-prefill-batch18k-util0835-2026-08-14-r1.json`
- `results/long-prefill-batch18k-util0835-2026-08-14-r2.json`
- `results/tune-batch18k-util0835-fixed-output-2026-08-14.json`
- `results/long-prefill-batch24k-util085-2026-08-14-r1.json`
- `results/dual-long-prefill-sched1-2026-08-14.json`
