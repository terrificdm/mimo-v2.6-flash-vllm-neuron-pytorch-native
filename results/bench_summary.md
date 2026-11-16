# Benchmark results

All numbers from `scripts/run_bench.sh` on one trn2.48xlarge (Neuron SDK 2.33.0, vllm-neuron 0.29.0.1.2.0 with the
patches of this repository, PyTorch native backend), each on a freshly started server. TTFT = time to first token,
TPOT = time per output token after the first; `vllm bench serve` rows use random-token prompts with `--ignore-eos`.
Summaries: [bench/](bench/).

## 1. Recommended configuration: `1m`, MTP k=3

| Workload | TTFT | TPOT | Output tok/s |
|---|---|---|---|
| 512 in / 128 out, concurrency 1 | 214 ms | 15.0 ms | 60.4 |
| 512 in / 128 out, concurrency 4 | 396 ms (mean) | 33.9 ms | 101.2 |
| 8,192 in / 512 out, concurrency 1 | 890 ms | 10.7 ms | 80.9 |
| 8,192 in / 512 out, concurrency 4 | 1,733 ms (mean) | 29.6 ms | 115.5 |
| 32,768 in / 256 out, concurrency 1 | 3,687 ms | 16.2 ms | 32.8 |
| 131,072 in / 128 out, concurrency 1 | 15,762 ms | 17.1 ms | 7.1 |
| 30K needle prompt: first / repeat / follow-up | 3.24 / 0.30 / 0.31 s | — | — |
| 940K needle prompt: first / repeat / follow-up | 148.8 / 3.38 / 3.20 s | — | — |
| image / video / audio (TTFT, median of 3) | 0.22 / 0.33 / 0.22 s | — | — |
| image / video / audio (total time, median of 3) | 0.25 / 0.41 / 0.41 s | — | — |

GSM8K 98.0 %, MMLU 79.5 % (200 questions each). HBM per logical core: 20.96 GiB at ready, 21.67 GiB on the busiest
core after the 940K prompt (limit 24 GiB). First start from an empty compile cache: 1,877 s.

GSM8K / MMLU wall time with 4 workers: 196 s / 46 s. Summary: [bench/1m-mtp3.summary.json](bench/1m-mtp3.summary.json).

## 2. Native backend against the earlier PyTorch/XLA deployment

The earlier deployment of this model used the vllm-neuron PyTorch/XLA backend (Neuron SDK 2.32, vllm-neuron 0.24,
[mimo-v2.6-flash-vllm-neuron-pytorch-xla](https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron-pytorch-xla)) with
a 1M context and 4 sequences and no speculative decoding. The same suite ran on the native `1m_c4` profile (1M
context, 4 sequences) without speculation and with MTP k=3.

Configuration differences:

| | PyTorch/XLA deployment | native `1m_c4` |
|---|---|---|
| software | Neuron SDK 2.32, vllm-neuron 0.24, vLLM 0.24 | Neuron SDK 2.33, vllm-neuron 0.29, vLLM 0.29 |
| expert weights | FP8 codes × BF16 activations (converted checkpoint) | FP8 codes × BF16 activations (encoded at load time) |
| `--max-num-batched-tokens` / prefill buckets | 4096 / [4096] | 8192 / [512, 1024, 8192] |
| decode | graphs per context-length bucket (4K-1M) | one graph per batch bucket, NKI kernel reads only the used KV |
| KV pool | 36,864 × 32 tokens; admission capped at 1 running request at 1M | 38,587 × 32 tokens; admission by actual length, 4 running requests |
| expert placement | linear | placement3_s5 |
| multimodal limits | 2 images, 1 video, 1 audio per request; images capped at about 1024×1024 | 64 / 4 / 8 per request |
| speculative decoding | none | off and MTP k=3 |

Throughput and latency (`vllm bench serve`):

| Workload | Metric | PyTorch/XLA | native, spec off | native, MTP k=3 |
|---|---|---|---|---|
| 512 in / 128 out, c1 | TTFT | 198.0 ms | 201.2 ms | 200.1 ms |
| | TPOT | 47.75 ms | 15.91 ms | 16.01 ms |
| | output tok/s | 20.44 | 57.59 | 57.29 |
| 512 in / 128 out, c4 | mean TTFT | 16,587 ms | 494 ms | 365 ms |
| | TPOT | 47.69 ms | 24.76 ms | 29.93 ms |
| | output tok/s | 20.49 | 140.57 | 114.17 |
| 8,192 in / 512 out, c1 | TTFT | 1,069 ms | 876 ms | 878 ms |
| | TPOT | 48.28 ms | 16.08 ms | 7.99 ms |
| | output tok/s | 19.89 | 56.30 | 103.23 |
| 8,192 in / 512 out, c4 | mean TTFT | 58,694 ms | 2,092 ms | 1,611 ms |
| | output tok/s | 19.98 | 138.46 | 135.93 |
| 32,768 in / 256 out, c1 | TTFT | 4,261 ms | 3,367 ms | 3,682 ms |
| | TPOT | 50.78 ms | 16.46 ms | 12.72 ms |
| 131,072 in / 128 out, c1 | TTFT | 18,491 ms | 14,204 ms | 15,722 ms |
| | TPOT | 221.42 ms | 17.68 ms | 21.04 ms |

Long prompts and prefix cache (needle prompts, 64 output tokens):

| Prompt | Request | PyTorch/XLA | native, spec off | native, MTP k=3 |
|---|---|---|---|---|
| 30,005 tokens | first / identical repeat / follow-up turn | 3.864 / 0.274 / 0.280 s | 2.909 / 0.275 / 0.269 s | 3.224 / 0.271 / 0.284 s |
| 940,004 tokens | first / identical repeat / follow-up turn | 204.98 / 3.302 / 3.311 s | 136.61 / 2.856 / 2.989 s | 148.47 / 3.095 / 2.936 s |

Image / video / audio (median of 3, thinking off, greedy; answers identical on all three):

| Input | PyTorch/XLA TTFT / total | native, spec off | native, MTP k=3 |
|---|---|---|---|
| image `number_739.png` | 0.22 / 0.37 s | 0.22 / 0.27 s | 0.22 / 0.24 s |
| video `digits_1234.mp4` | 0.27 / 0.75 s | 0.27 / 0.43 s | 0.32 / 0.40 s |
| audio `jfk.wav` | 0.22 / 1.46 s | 0.21 / 0.62 s | 0.21 / 0.38 s |

Accuracy (200 questions each; wall time with 4 workers):

| Task | PyTorch/XLA | native, spec off | native, MTP k=3 |
|---|---|---|---|
| GSM8K | 98.0 % (1,981 s) | 97.0 % (277 s) | 98.0 % (180 s) |
| MMLU | 80.5 % (49 s) | 79.5 % (41 s) | 79.5 % (43 s) |

Reading the comparison:

- **Decode** is about 3× faster at up to 32K context and 12.5× at 128K: the native decode kernel's cost follows the
  actual context, while the XLA deployment's 128K request ran in its 512K decode bucket.
- **Concurrency 4**: the XLA deployment admitted one request at a time at 1M (its KV pool held one 1M sequence), so
  its c4 rows are mostly queueing. The native scheduler admits by actual length and runs all 4.
- **Long prompts** prefill 18-33 % faster.
- **Short prompts** reach the first token as fast as before (201 ms against 198 ms); without the 512-token prefill
  bucket the native server took 253 ms, because a 512-token prompt was padded to 1024 tokens.
- **MTP k=3** helps where drafts are accepted (8K row: TPOT 16.1 → 8.0 ms; GSM8K 35 % less wall time) and costs
  8-11 % prefill time on long prompts; on the random-token rows it gains little, and at concurrency 4 with short
  prompts it lowers throughput (140.6 → 114.2 tok/s).
- Accuracy differences are within the noise of 200 questions (paired McNemar p ≈ 0.6).
- First start from an empty compile cache: 105 minutes on the XLA deployment, 31 minutes native (`1m`, MTP k=3).
