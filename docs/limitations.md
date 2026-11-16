# Known limitations

## Platform

- Validated only on one trn2.48xlarge with the Neuron multi-framework DLAMI for Neuron SDK 2.33.0, vllm-neuron
  0.29.0.1.2.0 (plus the patches in this repository), vLLM 0.29.0, torch 2.13.0, neuronx-cc 2.28.7631.0, and the
  vllm-neuron PyTorch native backend. Other versions are not tested.
- The deployment uses the whole instance: tensor parallel 64 with expert parallel 64 over the 64 logical
  NeuronCores. One server per instance.
- The first start compiles for about 25-35 minutes and uses a large part of the 2 TB host memory (64 ranks trace at
  the same time; `scripts/check_host.sh` requires at least 1,900 GiB). Weights and compile cache live on the instance-store NVMe and are lost when the instance stops.

## Requests

- **`seed` with speculative decoding** returns HTTP 400 (on GPU, vLLM's rejection sampler keeps a generator per
  request). Start with `--spec off` when you need `seed`.
- **Speculative decoding is not bit-exact** with `--spec off` (different matrix shapes, BF16 rounding); greedy outputs
  diverge at near-tied positions. ASR evaluation should use `--spec off`.
- **Penalties, `logit_bias`, structured outputs** give GPU-identical output but no speculative speedup for that
  request; requests with penalties, `logit_bias`, `min_p`, `min_tokens`, `bad_words` or `allowed_token_ids` are
  processed on the host and slow down the other requests of the same batch.
- `thinking_token_budget` and `logprob_token_ids` return HTTP 400. `logprobs_mode` in a logits mode returns 400.
- On-device sampling draws from the 256 most likely tokens (`max_top_k` 256); against full-vocabulary top-p on GPU the
  mean total-variation distance is 0.0094.
- Thinking is on by default (the chat template treats a missing `enable_thinking` as true); pass
  `"chat_template_kwargs": {"enable_thinking": false}` to turn it off.
- With `--reasoning-parser mimo`, a `/v1/completions` prompt without `</think>` gets no grammar from structured
  outputs (the same on GPU): use the chat API, or end the prompt with `<think></think>`.

## Context length and concurrency

- `1m`: at most 8 running requests (`1m_c4`: 4). The KV pool is 38,587 blocks × 32 tokens = 1,234,784 tokens, so
  only one request can use the full 1M context at a time; others wait.
- A 1M-token prompt from an empty cache takes about 2.5-3 minutes to the first token; a repeated or extended prompt
  hits the prefix cache and takes seconds.
- Speculative decoding: keep a few hundred tokens between prompt + `max_tokens` and `--max-model-len`
  ([speculative-decoding.md](speculative-decoding.md#3-behavior-and-limits)).

## Multimodal

- One image, video or audio item can have at most 114,688 embedding tokens on the 1M profiles and 8,192 on the 8K
  profiles; larger items get HTTP 400 ([configuration.md](configuration.md#4-prefix-caching-and-the-size-of-one-multimodal-item)).
  On GPU this limit can be raised with a larger `--max-num-batched-tokens`; on trn2 the extra 131,072-row prefill
  graph does not fit in HBM, so **videos of about 128K tokens and more are not supported**. Sample fewer frames,
  lower the resolution or split the video.
- The checkpoint's `audio_tokenizer/` directory is required: without it the server does not start (GPU vLLM
  disables audio instead).
- Offline `video_audio` input (audio track of a video) is not supported, as on GPU.

## Numerics

- FP8 linear layers are numerically FP8 × FP8 (the GPU algorithm), but the matrix multiplication runs at BF16 rate:
  trn2 has no kernel for 128 × 128 block-scaled FP8.
- The MoE cross-expert accumulation is BF16 (FP32 on GPU). Measured against an FP32 reference, the logit error is close to
  that of a CPU emulation of the GPU algorithm (p95 ratio 0.84-0.94, σ ratio 1.00-1.04) ([design-notes.md](design-notes.md#accuracy)).

## Performance

- Prefix-cache hits on 1M-token prompts spend most of their time in host-side preprocessing of the 1M-token text.
- On the 1M profiles the expert placement table is fitted on natural text; prefix-hit and short-request traffic is
  0-4 % slower with it ([configuration.md](configuration.md#3-expert-placement-configplacement3_s5json)).
