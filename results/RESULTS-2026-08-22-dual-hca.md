# DSpark Dual-HCA A/B — 2026-08-22

Live A/B measurement on `dgx01.sg.yangs.sh` / `dgx02.sg.yangs.sh` using
fork commit `08143b82b71c50b05f793ec908829847a6d2fb45` (upstream
`ed9ef7084291e6f496b74a3c1633b6fa527ce54d`). The model image and checkpoint
were held constant:

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- endpoint: head `127.0.0.1:18888`
- prompt/output: 256 / 256 tokens, streaming, `temperature=0`, thinking off

## Result

The baseline was measured after deploying the same latest fork commit with
single HCA. The first C=2 sample after each restart was treated as cold-start
evidence when it showed the known JIT/cache penalty; the settled C=2 sample is
the separate one-case rerun.

| Profile | C=1 aggregate avg | C=1 output avg | C=2 aggregate settled | C=2 output | C=2 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single HCA | 70.41 tok/s | 76.53 tok/s | 100.13 tok/s | 55.80 tok/s | 0.428 s |
| Dual HCA | 68.99 tok/s | 74.81 tok/s | 105.38 tok/s | 58.21 tok/s | 0.417 s |
| Change | -2.0% | -2.3% | **+5.2%** | +4.3% | -2.5% |

Conclusion: Dual-HCA helps the short two-request concurrency case, but this
same-commit A/B does **not** demonstrate the requested minimum 10% improvement.
It should be kept for concurrency, not advertised as a 10% single-stream
improvement. Single-stream performance is effectively flat within normal run
variance and slightly lower in this sample.

## Deployed configuration

Both hosts are running the exact-match NCCL selector (the leading `=` is an
NCCL selector operator, so the `.env` assignment visibly contains `==`):

```text
NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1
WORKER_NCCL_IB_HCA==rocep1s0f1,roceP2p1s0f1
NCCL_IB_GID_INDEX=3
```

The start resolver confirmed both HCA members on both nodes at RoCEv2 GID
index 3. The head API, stock V2 chat smoke, and fixed-output startup warmup
passed. Both containers were checked for NCCL/CUDA/OOM/Traceback errors and
none were found in the post-start log scan.

## Raw results and rollback

Raw JSON files are retained on the head node under `/home/yangs/ds4/results/`:

- `dual-hca-ab-2026-08-22-single-latest-code.json`
- `dual-hca-ab-2026-08-22-single-latest-code-c2-repeat.json`
- `dual-hca-ab-2026-08-22-dual-exact-final.json`
- `dual-hca-ab-2026-08-22-dual-exact-c2-settled.json`

The prior single-HCA environment is preserved at
`/home/yangs/dspark-rollback/20260822T015512Z-single-hca.env`. The Dual-HCA
environment backup is at
`/home/yangs/dspark-rollback/20260822T021130Z-dual-both.env`. Code rollback is
available at commit `bf61cd3`; both nodes also retain
`local/dspark-vllm-rollback:20260819-spinfix` and the dated rollback bundle
under `/home/yangs/dspark-rollback/`.

Benchmark command:

```bash
python3 scripts/benchmark-fixed-output.py \
  --base-url http://127.0.0.1:18888/v1 \
  --model deepseek-v4-flash-0731 \
  --prompt-tokens 256 --max-tokens 256 \
  --concurrency 1,2 --c1-repeats 3 \
  --output results/<profile>.json
```
