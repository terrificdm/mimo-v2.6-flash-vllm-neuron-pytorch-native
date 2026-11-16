# Design notes

How the port works and why some choices were made. Configuration values: [configuration.md](configuration.md).

## Model

XiaomiMiMo/MiMo-V2.6-Flash-MOPD (revision `b12f5573350bb1704513fcf119fca48e498b0423`):

| Part | Shape |
|---|---|
| text backbone | 48 layers: 9 global-attention layers (0, 5, 11, 17, 23, 29, 35, 41, 47) and 39 sliding-window layers (window 128, with an attention-sink logit per head); layer 0 dense SwiGLU MLP, layers 1-47 MoE with 256 experts, top-8, sigmoid routing with a correction bias, no shared expert |
| attention | 64 query heads; 8 KV heads on sliding-window layers, 4 on global layers; QK head dim 192 (RoPE on the first 64 dims), V head dim 128 |
| context | 1,048,576 tokens |
| draft models | 3 MTP layers; a 5-layer sliding-window DFlash draft that proposes 7 tokens |
| encoders | vision transformer (window and full-attention blocks); two audio towers (audio tokenizer encoder with residual VQ, and the audio encoder) |
| checkpoint precision | linear layers block-scaled FP8 (E4M3FN), experts MXFP4, the rest BF16 |

## Where the code lives

All model code is in vllm-neuron, the vLLM plugin for Neuron. The patches add one model package and change some
existing modules ([development.md](development.md#patches)):

- `vllm_neuron/model/mimo_v2/`: configuration, weight loading, the text backbone, MoE, FP8 layers, MTP and DFlash
  draft models, vision and audio encoders, expert placement, on-device prompt logprobs.
- `vllm_neuron/_kernels/mimo_v2/`: NKI attention kernels (decode and prefill).
- `vllm_neuron/model/mimo_v2/nki_moe_tkg/`: a copy of nkilib's MoE token-generation kernels.
- existing modules: model registry, NeuronConfig, KV-cache layout, scheduler (KV admission, encoder-cache
  accounting), sampling, speculative decoding, model runner / worker.

The model runs on vllm-neuron's **PyTorch native backend** (`VLLM_NEURON_BACKEND=neuron_native`): the model is
written as ordinary PyTorch modules, captured with torch.compile and compiled by neuronx-cc per graph shape (one graph
per prefill bucket and per decode bucket). The startup log shows `Using neuron_native backend` and no
`on the XLA route` line.

## Precision

Same scheme as the official H100 deployment: FP8 linear layers, FP8 experts × BF16 activations, BF16 KV cache.

- **Linear layers (qkv, layer-0 MLP)**: the GPU algorithm, block FP8 × FP8 with 128 × 128 weight blocks and
  per-token, per-128-column dynamic activation scales. trn2 has no block-scaled FP8 matmul kernel, so the FP8 codes
  are multiplied at BF16 rate with FP32 accumulation and the block scales applied in FP32: numerically FP8 × FP8,
  speed of BF16. trn2's FP8 E4M3 has a smaller range than E4M3FN; the loader halves the codes and doubles the scales,
  which is exact. `o_proj`, embedding, LM head and router stay BF16, as on GPU.
- **Experts**: the checkpoint's MXFP4 values are re-encoded as FP8 E4M3 with one scale per output row. For this
  checkpoint every one of the 36,096 expert tensors is reproduced exactly, so no layer falls back to BF16. The
  kernels convert FP8 tiles to BF16 in SBUF and multiply with BF16 activations.
- **No conversion step**: the original Hugging Face checkpoint is read as is; all re-encoding happens at load time.

## Attention

- The shared vllm-neuron attention kernels assume equal K and V head dims or a head dim of at most 128, and MiMo has
  K 192 / V 128, so the model has its own NKI kernels:
  - **prefill / segmented prefill**: QK 192 split into 128 + 64 partition tiles accumulated in PSUM; the global layers
    stream the cached prefix from the paged KV cache in 128-token folds, with the number of folds decided at run
    time; calls of more than 8192 rows are split.
  - **decode**: one position-major kernel for every profile, batch size and speculative width; it reads only the KV
    blocks a request has, so decode time depends on the actual context, not on `--max-model-len`.
- The 8K profiles also offer the torch implementation (`8k_torch`); on the 1M profiles it would compute over the full
  1M positions.
- K and V of different widths are stored as one buffer per layer (all K blocks, then all V blocks). The sliding-window
  layers use vLLM's sliding-window KV spec, so blocks outside the window are freed (hybrid memory allocator).

## MoE

- Prefill: vllm-neuron's `moe_cte` kernel; with FP8 experts, a loop over only the non-empty token blocks (a long
  prompt reserves blocks for the worst case, most stay empty).
- Decode: nkilib's `moe_tkg` all-expert kernel (copy with a fix of the FP8 row-scale PSUM offset); every local
  expert's weights are read once per step.
- Routing follows the checkpoint: sigmoid scores, top-8 chosen by score + correction bias, weights from the
  unbiased scores, renormalized. Router logits in BF16 as configured by the checkpoint; exact ties go to the lower
  expert index, as in vLLM on GPU.
- **Expert placement** (1M profiles): a static table that adds one prefill replica of a hot expert per rank
  ([configuration.md](configuration.md#3-expert-placement-configplacement3_s5json)). It is fixed at start-up; there
  is no dynamic re-balancing.

## Speculative decoding

- **MTP**: draft step i uses MTP layer i (k ≤ 3), each taking the target's final-norm hidden state, the same scheme
  as the official SGLang command (`--enable-multi-layer-eagle`); embedding and LM head are shared with the target.
  The proposer reuses vLLM's EAGLE pipeline.
- **DFlash**: the target returns auxiliary hidden states of selected layers; the 5-layer draft proposes 7 tokens.
- The verify step runs 1+k rows per request in the decode graph. At 17 or more scheduled requests the scheduler
  pauses drafting for the whole step ([speculative-decoding.md](speculative-decoding.md)).

## Multimodal

- The vision encoder and both audio towers run on Trainium in the same server; images, videos and audio go through
  vLLM's MiMo processor unchanged.
- Vision buckets: 1024, 4096, 16384 and 32768 patches per encoder call; larger videos are split by frame group
  (attention is per frame group, as in the reference code).
- Encoder outputs are stored in an encoder cache of 512-token blocks; one item may use at most 224 blocks on the 1M
  profiles (114,688 tokens). Longer items are rejected with HTTP 400 before they reach the engine.

## Sampling and API behavior aligned with GPU vLLM

- On-device sampling for the common case (temperature, top-k, top-p over the 256 most likely tokens). `seed`,
  penalties, `logit_bias`, `min_p`, `min_tokens`, `bad_words` and `allowed_token_ids` run on the host with vLLM's own
  code, so outputs follow GPU semantics.
- Greedy ties take the lowest token id, like `torch.argmax`.
- prompt_logprobs are computed on the device; structured outputs apply the grammar bitmask on the host.

## Scheduler and KV cache

- 1M profiles: `--num-gpu-blocks-override 38587` fixes the KV pool (38,587 × 32 tokens) so the graphs fit in HBM.
- Requests are admitted by the KV blocks they actually need (`kv_admission: actual_blocks_reserve`), not by
  `--max-model-len`, so short requests run next to a long one.

## Accuracy

There is no GPU on the instance. "Same as GPU" means one of two things: read from vLLM 0.29's source, or reproduced
on the CPU with the GPU's dtype path.

- **Logits** (16 prompts × 64 steps against a true FP32 reference): the error of the four default configurations is
  0.84-0.94× (p95) of a CPU emulation of the GPU algorithm and their σ is 1.00-1.04× of it.
- **Downstream** (FP8 vs BF16 weights on the same server; GSM8K 1,319 questions, MMLU-Pro 1,000): GSM8K greedy
  0.9712 / 0.9712, MMLU-Pro greedy 0.778 / 0.779; all paired differences within one paired standard error.
- **Multimodal**: audio word error rate 2.19-2.26 % on 1,551 LibriSpeech words (GPU dtype path on CPU: 1.93 %; FP32:
  2.19 %); the vision encoder matches the reference within the tested tolerance.
- **Long context**: needle-in-a-haystack at 32K and 128K 20/20; 1M (999,996 tokens) correct.
- Results of `scripts/run_bench.sh`: [results/bench_summary.md](../results/bench_summary.md).

## Notes on the source tree

`vllm_neuron/model/mimo_v2/README.md` inside the patched source is a development-time note (in Chinese); some status
rows there predate the NKI decode kernel and the audio towers. The descriptions in this file are the current state.
