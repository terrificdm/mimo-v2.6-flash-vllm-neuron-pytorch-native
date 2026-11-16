# Speculative decoding

The checkpoint ships two draft models: 3 MTP layers (multi-token prediction, built into the main checkpoint) and a
5-layer DFlash draft in `dflash/` that proposes 7 tokens at once. Both run on Trainium in the same server as the
target model. Choose with `scripts/serve.sh --spec off|mtp1|mtp2|mtp3|dflash`; the default is `mtp3`.

## 1. What to choose

| Situation | Option |
|---|---|
| general serving, natural-language or code traffic | `mtp3` (default) |
| evaluations, regression tests, data generation that must be bit-exact or use `seed` | `off` |
| speech recognition (ASR) evaluation, reproducible transcripts | `off` |
| synthetic or random-token traffic (load tests) | `off`, or measure both on your traffic |
| one request at a time, latency first, mostly greedy | `dflash` can be faster; see the limits below |

## 2. How much it helps

The gain depends on how predictable the output is. Draft tokens are accepted most often on natural text and code
and least often on random-token prompts.

- **Natural text and code** (20 original English / Chinese / code prompts and a 7.4K-token document), MTP k=3 ÷ off,
  output throughput, greedy / GPU-aligned sampling (temperature 1.0, top-p 0.95):

  | Profile | concurrency 1 | concurrency 4 | concurrency 16 |
  |---|---|---|---|
  | 8K | 1.73 / 1.43 | 2.00 / 1.75 | 1.72 / 1.59 |
  | 1M | 1.87 / 1.54 | — | — |

  GSM8K through `scripts/run_bench.sh` (200 questions, 4 workers, greedy) finishes in 180 s with MTP k=3 against
  277 s without speculation on `1m_c4` (scores 98.0 % and 97.0 %, within the noise of 200 questions).
- **Random-token prompts** (the worst case): with GPU-aligned sampling the 8K server at concurrency 4-16 is 0.97-1.04×
  of `off`, the 1M server at concurrency 4 0.89-0.96× and at concurrency 8 1.01-1.11×; greedy stays at 1.20-1.64×.
  At concurrency 1: 8K 1.43× / 2.14×, 1M 1.39× / 2.17× (GPU-aligned / greedy). The random-prompt rows of
  `scripts/run_bench.sh` show this case: decode time per token barely changes at 512 input tokens, while the
  8192-token row drops from 16.1 to 8.0 ms per output token (`1m_c4`).
- **Long prompts**: the draft layer adds work to prefill, so time to first token at 32K-940K tokens is 8-11 % longer
  with MTP k=3 than with `off`.
- **DFlash** is fastest at concurrency 1 (8K natural text: 2.0-2.3× of `off`; random prompts greedy 8K 2.68×, 1M
  2.40×, but only 1.06× with GPU-aligned sampling). At concurrency 4 and above it is usually behind MTP k=3.

## 3. Behavior and limits

- **Not bit-exact with `off`.** The verify graph processes 1+k rows per request and the decode graph 1 row; the
  different matrix shapes round differently in BF16, so greedy output usually diverges from `off` after some dozens
  of tokens, at positions where the top two candidates are nearly tied. Use `off` when outputs must match exactly.
- **ASR**: word error rate on 1,551 LibriSpeech test-clean words is 2.19-2.26 % with `off` and 2.32-2.39 % with MTP
  k=3; the extra errors come from such near-ties on the text side.
- **`seed`**: with any speculative option, a request with `seed` gets HTTP 400 with an explanation (the server stays
  healthy). Without speculation, `seed` works.
- **Penalties, `logit_bias`, structured outputs**: requests that use them produce the same output as on GPU, but each
  verify step emits only 1 token for that request, so it loses the speedup. `min_p` and `logit_bias` with
  speculative decoding are rejected by vLLM's front end, as on GPU.
- **logprobs and prompt_logprobs** work with speculative decoding.
- **High concurrency**: decode steps with 17 or more scheduled requests run without drafts
  (`neuron_config.spec_disable_by_batch_size`, default 17; 0 = never). On the 8K profile requests 17-32 fall into the
  32-row decode bucket, where speculation is slower than `off`. A greedy-only deployment loses 3-6 % throughput at
  17-32 requests from this pause and may set it to 0. The 1M profiles (at most 8 requests) never reach it.
- **Leave headroom below `--max-model-len`**: a request that gets within 3+2k tokens of `--max-model-len` (9 for
  k=3, 17 for DFlash) stops using drafts.
- k is 1-3 for MTP and exactly 7 for DFlash. `num_speculative_tokens_per_batch_size` is not supported.
- **DFlash**: the 8K profile serves at most 16 concurrent requests with DFlash (the server lowers `--max-num-seqs`
  from 32 and logs `lowering max_num_seqs 32 -> 16`) and has 18 % fewer KV blocks; DFlash does not use the expert
  placement table.

## 4. Memory

HBM per logical core at ready / after the full test set (limit 24 GiB):

| Option | 8K | 1M |
|---|---|---|
| `off` | 17.79 / 17.81 GiB | 20.28 / 20.31 GiB |
| `mtp3` | 18.57 / 19.16 GiB | 20.86 / 21.78 GiB (peak 21.94 with all four modalities in use) |
| `dflash` | 17.84 / 19.19 GiB | 20.21 / 21.10 GiB |

MTP keeps the same number of KV blocks as `off` (the draft layer's KV joins the sliding-window group).
