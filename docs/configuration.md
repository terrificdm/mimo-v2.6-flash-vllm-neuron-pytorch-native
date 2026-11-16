# Configuration reference

All serving parameters live in one file, [config/serve_args.json](../config/serve_args.json) (another file can be
used with `SERVE_ARGS=<file>`).
`scripts/serve.sh` renders a profile and a speculative option from it into the environment and the `vllm serve`
command line (`scripts/render_serve_args.py`); `bash scripts/serve.sh --dry-run` prints the result without starting.

## 1. Profiles

One trn2.48xlarge runs one server at a time. Pick the profile when you deploy; a running server does not switch.

| | `1m` (default) | `1m_c4` | `8k` | `8k_torch` |
|---|---|---|---|---|
| Use for | long documents, up to 1M context, image / video / audio in the same server, repeated or extended long prompts | as `1m`, when concurrency rarely exceeds 4 and single-request latency matters more | text chat with contexts up to 8K, many concurrent short requests | as `8k` with the torch attention (fallback / comparison) |
| `--max-model-len` / `--max-num-batched-tokens` | 1,048,576 / 8192 | same as `1m` | 8192 / 4096 | same as `8k` |
| prefill buckets (`num_batched_tokens_buckets`) | [512, 1024, 8192] | [512, 1024, 8192] | [1024, 4096] | [1024, 4096] |
| decode buckets (`num_seqs_buckets`) / `--max-num-seqs` | [1, 4, 8] / 8 | [1, 4] / 4 | [1, 4, 8, 16, 32] / 32 | same as `8k` |
| KV cache | `--num-gpu-blocks-override 38587` (blocks of 32 tokens) | same as `1m` | sized automatically from the HBM budget | same as `8k` |
| prefix caching | on, `--prefix-cache-retention-interval 1024` | on | off | off |
| multimodal | `--limit-mm-per-prompt {"image": 64, "video": 4, "audio": 8}`, `--allowed-local-media-path $MEDIA_DIR` | same as `1m` | text only | text only |
| text attention | NKI kernels (`mimo_attn_kernel: nki`) | NKI | NKI | torch |
| expert placement | `config/placement3_s5.json` (not with DFlash) | same as `1m` | linear | linear |
| default speculative decoding | MTP k=3 | MTP k=3 | MTP k=3 | MTP k=3 |

Common to all profiles: FP8 weights (block-scaled FP8 linear layers, FP8 experts expanded from the checkpoint's
MXFP4, BF16 activations), BF16 KV cache, tensor parallel 64 with expert parallel 64 over the 64 logical NeuronCores
(LNC=2), `--reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice --generation-config vllm`,
`--served-model-name mimo-v2.6-flash`, `--host 127.0.0.1`.

```bash
bash scripts/serve.sh                              # 1m, MTP k=3
bash scripts/serve.sh --profile 1m_c4              # 1M, max-num-seqs 4
bash scripts/serve.sh --profile 8k                 # 8K, MTP k=3
bash scripts/serve.sh --profile 8k --spec off      # 8K without speculative decoding
```

Notes:

- **Prefill buckets.** A prefill step is padded to the smallest bucket that holds it, and the padded rows are
  computed. A 500-token prompt on a profile whose smallest bucket is 1024 costs the same as a 1000-token prompt.
  The 512 bucket of the 1M profiles brings a 512-token request's time to first token from about 253 ms to about
  201 ms. Prompts longer than `--max-num-batched-tokens` are prefilled in several steps of at most that many tokens.
- **Decode buckets.** Decode graphs are compiled once per decode bucket for the full `--max-model-len`; the NKI
  decode kernel reads only the KV blocks a request actually has, so a short request on the 1M server decodes about
  as fast as on the 8K server.
- **`1m` vs `1m_c4`.** With 8 requests in flight, system throughput on `1m` is about 15-30 % higher than with
  4 (21 % without speculation, 15-29 % with MTP k=3 depending on sampling); at concurrency 1 the two are within 1 %. The cost is per-request latency when 8 requests run together.
  `1m_c4` has different decode buckets and compiles its own graphs on its first start.
- **torch attention** computes over `--max-model-len`, so it is offered for 8K only. On the 8K server the NKI kernels
  are 10-17 % faster at concurrency 1-32 with speculative decoding off.

## 2. Speculative decoding options

`--spec off | mtp1 | mtp2 | mtp3 | dflash`, default `mtp3`. Details, measurements and how to choose:
[speculative-decoding.md](speculative-decoding.md).

| Option | `--speculative-config` | Notes |
|---|---|---|
| `off` | not passed | needed for bit-exact greedy output, `seed`, and ASR evaluation |
| `mtp1` / `mtp2` / `mtp3` | `{"method": "mtp", "num_speculative_tokens": k}` | the checkpoint's 3 MTP layers are used in turn (step i uses layer i); KV blocks as without speculation |
| `dflash` | `{"method": "dflash", "model": "<checkpoint>/dflash", "num_speculative_tokens": 7}` | k must be 7; 1M: `--num-gpu-blocks-override 71772` and KV budget fraction 0.31, no expert placement; 8K: budget fraction 0.25, at most 16 concurrent requests (the server lowers `--max-num-seqs` and logs a warning) |

## 3. Expert placement (`config/placement3_s5.json`)

With expert parallel 64 every rank holds 4 of the 256 experts, and the rank with the busiest experts sets the time
of each MoE layer. `placement3_s5.json` gives every rank 5 slots: 4 primary copies (each expert once; decode uses
only these) and 1 replica of a hot expert used during prefill, with the rows of a replicated expert split between
its copies. Routing is unchanged and replicas are bit-identical to the original expert; the only numerical
difference from linear placement is the order of the cross-rank sum.

- Set by `neuron_config.mimo_expert_placement` (a file path). Unset = linear placement.
  `scripts/render_serve_args.py` refuses to start if the file's sha256 differs from `placement_sha256` in
  serve_args.json.
- Used by `1m` / `1m_c4` with `off` and MTP. Not used with DFlash or on the 8K profiles.
- Effect (placement ÷ linear, time to first token): code 0.90-0.92, English documents 0.94-0.95, Chinese documents
  0.97-0.98, legal text 1.01-1.02, synthetic 128K / 1M prompts 1.00-1.01. Prefix-cache hits and short requests are
  0-4 % slower. Decode is within ±1 % (3 % faster at concurrency 8 without speculation). About 1.1 GiB more HBM
  per core.
- The table was fitted on prefill expert loads of natural text (English and Chinese documents, Python source).
  If your traffic is very different, the gain can shrink; for prefix-hit or short-request traffic you can drop
  the placement: copy serve_args.json, remove `mimo_expert_placement` from the profile and start with
  `SERVE_ARGS=<your copy> bash scripts/serve.sh` (see [development.md](development.md#changing-the-configuration)).
- To use another table, give it a **new file name** and update `placement_sha256`. The table content is a graph
  input: a table with the same number of slots per rank reuses the compiled graphs.

## 4. Prefix caching and the size of one multimodal item

- `1m` / `1m_c4` enable prefix caching with `--prefix-cache-retention-interval 1024`. With vLLM's default (0) the
  first "append to the end" request on a model with sliding-window layers does not hit the cache (the same on GPU);
  the flag only decides which sliding-window blocks keep their hash and does not change outputs.
- **One image, video or audio item can have at most 114,688 embedding tokens on the 1M profiles** (224 encoder-cache
  blocks of 512; the same limit as vLLM 0.29 with the model's official GPU command) and 8,192 on the 8K profiles,
  where the whole prompt plus output must also fit in 8192. An item longer than `--max-num-batched-tokens` is encoded
  once and consumed over several prefill steps. A larger item gets HTTP 400 with the reason; the server keeps
  running.
- Estimated tokens per item (vLLM samples 32 frames per video by default, so video size depends on resolution, not
  duration):

  | Input (default 32 frames) | Tokens (estimate) | `1m` (114,688) | `8k` (8,192) |
  |---|---|---|---|
  | 4K 3840×2160 video | about 130,560 | too large (HTTP 400) | too large |
  | 1080p video | about 32,640 | fits | too large |
  | 720p video | about 14,080 | fits | too large |
  | 480p video | about 6,480 | fits | fits (plus text and output) |
  | one image | 1 per 32×32 pixels, at most 8,192 | fits | fits within 8192 together with text and output |
  | audio | about 6.25 per second | up to about 5 hours | up to about 21 minutes |

  Lower the resolution with the request's `mm_processor_kwargs` (`max_pixels`), sample fewer frames with
  `--media-io-kwargs`, or split long media into several requests.
- `--limit-mm-per-prompt {"image": 64, "video": 4, "audio": 8}` is a deployment policy (requests with more items
  get HTTP 400), not a hardware limit. vLLM's default is 999 per modality.

## 5. Compile cache

- `scripts/serve.sh` sets `VLLM_CACHE_ROOT=$CACHE_BASE/<first 12 characters of the vllm-neuron tree hash>/<profile>-<spec>`.
  - `neuron/torch_compile` is separate for every (profile, option). The torch.compile fingerprint does not include
    the speculative k, so sharing one directory between options makes later cold starts accumulate guarded code and
    use more and more host memory.
  - `neuron/native_neff_cache`, `neuron/compile_cache` and `modelinfos` are links to `$CACHE_BASE/<tree>/shared/`:
    NEFFs are keyed by graph content, so a graph two options have in common compiles once.
  - New code (other patches) has another tree hash and therefore a fresh set of directories. If you set the cache
    directory by hand, use a new one after changing the code: changing only module-level constants can make the
    precompiled cache load stale graphs.
- First start from an empty cache: about 30 minutes for `1m` with MTP k=3, 35 minutes for `8k` with MTP k=3,
  24 minutes for `8k` without speculation; later starts of the same option load from the cache (about 13-15
  minutes, mostly loading 178 GB of weights onto 64 cores).
- Size: about 8-25 GB per (profile, option); every option of `1m` and `8k` together is about 150-200 GB.
- `CACHE_BASE`, `NEURON_COMPILE_CACHE_URL` and the NEFF lock directory must be on a local disk: the compile lock is
  a flock, which is not reliable on NFS.
- Do not clear `/var/tmp/neuron-compile-cache` unless an error mentions `[NLA001]`.

## 6. Environment

Set by serve_args.json `env` (rendered by `scripts/serve.sh`):

| Variable | Value | Why |
|---|---|---|
| `VLLM_NEURON_BACKEND` | `neuron_native` | the PyTorch native backend of vllm-neuron |
| `NEURON_LOGICAL_NC_CONFIG` | 2 | 64 logical NeuronCores of 2 physical cores each |
| `NEURON_SKIP_EFA_AFFINITY` | 1 | skip binding workers to EFA devices (a CPU-side optimization; this deployment uses no EFA) |
| `VLLM_NEURON_SWITCH_CC` | 1 | contiguous 8×8 groups for the collectives instead of the custom trn2 mesh; used for every validated run |
| `NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS` | 12 | trace threads per rank during the first compile; much higher values ran the 2 TB host out of memory |
| `TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT` | 43200 | the default 1200 s lock timeout makes all 64 ranks recompile a long-running graph at once; the lock is a kernel flock and is released when its holder exits |
| `VLLM_RPC_TIMEOUT` | 1800000 | engine RPC timeout (ms) for long first compiles |

Set by `scripts/env.sh`: `MIMO_ROOT` (default `/mnt/nvme/mimo`) and the paths under it (`VENV`, `SRC_DIR`,
`HF_HOME`, `CACHE_BASE`, `NEURON_COMPILE_CACHE_URL`, `NKI_COMPILE_CACHE_URL`, `TMPDIR`, `TORCH_NEURONX_DEBUG_DIR`,
`LOG_DIR`, `RUN_DIR`), `MEDIA_DIR` (default `assets/`), `HF_HUB_OFFLINE=1` while serving. It unsets
`XLA_IR_DEBUG`, `XLA_HLO_DEBUG` and `NEURON_FRAMEWORK_DEBUG`, which change the NEFF cache key.

## 7. Defaults that need no flag

Listed under `defaults_in_code` in serve_args.json for reference:

| Setting (`neuron_config`) | Default | Meaning |
|---|---|---|
| `spec_disable_by_batch_size` | 17 | decode steps with 17 or more scheduled requests run without draft tokens; drafts resume when the count drops (0 = never pause) |
| `mimo_prompt_logprobs` | on | prompt_logprobs computed on the device (false: such requests get HTTP 400) |
| `mimo_router_dtype` | the checkpoint's `moe_router_dtype` (bfloat16) | as vLLM on GPU |
| `mimo_fp8_linear` | `block` | block-scaled FP8 linear layers, the GPU algorithm |
| `kv_admission` | `actual_blocks_reserve` | requests are admitted by the KV blocks they actually need |
| `max_logprobs` (vLLM flag) | 20 | do not set 0: structured outputs and host-side sampling need gathered logits |

## 8. Scripts

| Script | What it does |
|---|---|
| `scripts/check_host.sh` | read-only host checks (instance type, DLAMI venv, devices, memory, disk) |
| `scripts/setup_venv.sh` | creates `$VENV` from `requirements/serve-freeze.txt` and `requirements/audio.txt` |
| `scripts/setup_source.sh` | clones vllm-neuron, applies the patches, checks the tree hashes, editable install |
| `scripts/download_weights.sh` | downloads the pinned checkpoint and checks it against `weights.lock.json` |
| `scripts/serve.sh` | starts `vllm serve` for a profile / option in the foreground |
| `scripts/wait_ready.sh` | waits for `/v1/models` |
| `scripts/stop.sh` | stops the server started by `serve.sh` and checks the NeuronCores are free |
| `scripts/smoke_test.sh` | functional smoke test (text, streaming, thinking, tools, JSON schema, concurrency, media) |
| `scripts/run_bench.sh` | benchmark suite: throughput / latency, long prompts, prefix cache, media, GSM8K / MMLU |
| `scripts/api_tests/ask.py` | small streaming client for text / image / video / audio questions |
