# Tutorial: Deploy MiMo-V2.6-Flash-MOPD with vLLM Neuron (PyTorch native backend)

[中文版](README.zh-CN.md)

This tutorial deploys [XiaomiMiMo/MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) on one
AWS **trn2.48xlarge** with vLLM and the vllm-neuron plugin on its **PyTorch native backend**, serving a 1M-token
context together with image, video and audio input, prefix caching and MTP speculative decoding from one server.
The model support is a set of patches on vllm-neuron 0.29 in [patches/](patches/); the original checkpoint is used
as is, with no conversion step.

Flow: launch the instance → set up the venv and the patched vllm-neuron → download the weights → start the server →
send requests → benchmark.

## Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1. Set up the software](#step-1-set-up-the-software)
- [Step 2. Download the weights](#step-2-download-the-weights)
- [Step 3. Start the server](#step-3-start-the-server)
- [Step 4. Send requests](#step-4-send-requests)
- [Step 5. Smoke test](#step-5-smoke-test)
- [Step 6. Benchmark](#step-6-benchmark)
- [Restarting and recovery](#restarting-and-recovery)
- [Other configurations](#other-configurations)
- [Accuracy](#accuracy)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Related resources](#related-resources)
- [License](#license)

## Overview

What the recommended configuration (`1m` profile, MTP k=3) provides, validated with the code in this repository:

- **1M context** (1,048,576 tokens) with FP8 weights (block-scaled FP8 linear layers; experts FP8-encoded from the
  checkpoint's MXFP4, exactly) and a BF16 KV cache: the precision scheme of the official H100 deployment.
- **Image, video and audio** in the same server; the vision encoder and both audio towers run on Trainium.
- **Prefix caching**: a repeated or extended long prompt reaches its first token in seconds.
- **MTP speculative decoding** with the checkpoint's 3 MTP layers; DFlash and no speculation are options.
- Up to **8 concurrent requests**; reasoning and tool calling with vLLM's `mimo` parsers; structured outputs;
  sampling parameters with GPU vLLM semantics.

Key numbers (one trn2.48xlarge, `1m` profile with MTP k=3, `scripts/run_bench.sh`; details in [Step 6](#step-6-benchmark)):

| Measurement | Result |
|---|---|
| Short request (512 in / 128 out), concurrency 1 | TTFT 214 ms, TPOT 15.0 ms |
| 8,192 in / 512 out, concurrency 1 | TTFT 890 ms, TPOT 10.7 ms |
| First token at 940K / 30K tokens (first request) | 148.8 s / 3.24 s |
| Same prompt again (prefix-cache hit), 940K / 30K | 3.38 s / 0.30 s |
| GSM8K / MMLU (200 questions each, 0-shot, greedy) | 98.0 % / 79.5 % |
| First start from an empty compile cache | 31 minutes |

The GSM8K / MMLU row is a quick regression check with thinking off; MMLU there asks for the answer letter only, which
keeps the score below what the model reaches when it can reason (MMLU-Pro with thinking on: 0.80, see
[Accuracy](#accuracy)).

Time from a fresh instance to the first request, as measured:

| Step | Time |
|---|---|
| Step 1: check, venv, patched vllm-neuron | about 8 minutes |
| Step 2: download 178 GB and check sha256 | depends on the bandwidth to Hugging Face (not measured) |
| Step 3: first start, compiling every graph | about 31 minutes |
| Step 5 / 6: smoke test / benchmark | under 1 minute / about 12 minutes |

## Prerequisites

**Instance**

- One **trn2.48xlarge** (16 Trainium2 devices, 64 logical NeuronCores at LNC=2, 24 GiB HBM per logical core, 2 TB
  host memory), from an EC2 Capacity Block for ML or on-demand capacity.
- Inbound **SSH only** in the security group. The server listens on `127.0.0.1:8000`; reach it with
  `ssh -L 8000:127.0.0.1:8000 ...`.
- A root volume of at least 512 GB (the DLAMI's minimum for Trn instances); the weights, venv and compile cache go on
  the local NVMe drives (below).

**AMI and software**: the Neuron multi-framework DLAMI for **Neuron SDK 2.33.0** on Ubuntu 24.04. The validated
image is named `Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`; find its ID in your Region with:

```bash
aws ec2 describe-images --region <region> --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId, Name]' --output text
```

Do not use the `.../latest/image_id` SSM parameter of the
[DLAMI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/setup/multiframework-dlami.html): it points to
the newest Neuron SDK, on which `scripts/check_host.sh` fails. The DLAMI ships the venv
`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`, whose package list this tutorial copies into a separate
venv; the DLAMI venv itself is not modified.

> [!IMPORTANT]
> Validated only on these versions:
>
> | Component | Version |
> |---|---|
> | Neuron SDK | 2.33.0 |
> | vllm-neuron | 0.29.0.1.2.0 + the patches in this repository |
> | vLLM | 0.29.0 |
> | torch | 2.13.0 |
> | neuronx-cc | 2.28.7631.0 |
> | nki | 0.7.0 |
> | Backend | vllm-neuron PyTorch native (`VLLM_NEURON_BACKEND=neuron_native`) |
>
> Do not upgrade or add packages in the serving venv.

**Disk**: weights 178 GB + serving venv about 13 GB + compile cache for each configuration you start (15 GB for `1m`
with MTP k=3, 8-25 GB for the others; about 150-200 GB if you start every option of `1m` and `8k`). Use the four instance-store NVMe drives in RAID 0 (about
6.9 TiB; check the device names with `lsblk`):

```bash
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1
sudo mkfs.xfs /dev/md0
sudo mkdir -p /mnt/nvme && sudo mount -o noatime /dev/md0 /mnt/nvme && sudo chown "$USER:$USER" /mnt/nvme
mkdir -p /mnt/nvme/mimo
```

> [!WARNING]
> Instance-store data is lost when the instance stops. After a reboot:
> `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme`.

All scripts put their files under `MIMO_ROOT` (default `/mnt/nvme/mimo`); export it to use another directory on a
local disk. Paths and defaults: [scripts/env.sh](scripts/env.sh).

## Step 1. Set up the software

About 8 minutes.

```bash
cd /mnt/nvme/mimo
git clone https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron-pytorch-native.git
cd mimo-v2.6-flash-vllm-neuron-pytorch-native
bash scripts/check_host.sh      # read-only: instance type, DLAMI venv, 16 idle devices, memory, disk
bash scripts/setup_venv.sh      # new venv /mnt/nvme/mimo/venv with the DLAMI vLLM venv's packages + av, soundfile
bash scripts/setup_source.sh    # clone vllm-neuron, apply the patches, check the hashes, editable install
```

- `check_host.sh` ends with `all checks passed`.
- `setup_venv.sh` installs the DLAMI venv's exact package list ([requirements/serve-freeze.txt](requirements/serve-freeze.txt))
  with `--no-deps` from PyPI and the Neuron pip index, then checks the result line by line (about 7 minutes).
- `setup_source.sh` clones [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron) at
  `release-0.29.0.1.2.0`, applies the three patches with `git am`, checks the tree hashes against
  [sources.lock.json](sources.lock.json) and installs it in editable mode (needed: vLLM finds the plugin through its
  entry points). It ends with `vllm-neuron installed (editable)`.

## Step 2. Download the weights

```bash
tmux new -d -s download "bash scripts/download_weights.sh > /mnt/nvme/mimo/download.log 2>&1"
tail -f /mnt/nvme/mimo/download.log          # ends with "weights ready: ..."; exit code 3 on failure
```

178 GB, 154 files, revision `b12f5573350bb1704513fcf119fca48e498b0423`, into `/mnt/nvme/mimo/hf`. The weights are
public; no token is needed. The script then checks every file's size and sha256 against
[weights.lock.json](weights.lock.json). Re-run the same command after an interruption: the download resumes.
The checkpoint must include `audio_tokenizer/` (audio towers) and `dflash/` (DFlash draft); the script checks both.

## Step 3. Start the server

Open a tmux session for the server (detach with Ctrl-b d, reattach with `tmux attach -t serve`):

```bash
tmux new -s serve
```

Inside it, from the repository directory:

```bash
bash scripts/serve.sh            # profile 1m, MTP k=3
```

In another shell, wait until it is ready:

```bash
bash scripts/wait_ready.sh       # prints the last log line every 5 minutes; ends with "ready ... ['mimo-v2.6-flash']"
```

**Time**: the first start from an empty compile cache takes about **31 minutes**: 7-10 minutes to
load the weights onto 64 cores, then the graphs compile in parallel (one per prefill bucket and decode bucket, plus
the encoders and the draft layer). Later starts of the same configuration load from the cache and take about 13
minutes. The log is `/mnt/nvme/mimo/logs/serve-latest.log`; it shows `Using neuron_native backend`.

`scripts/serve.sh` renders the command from [config/serve_args.json](config/serve_args.json); `--dry-run` prints it.
For the `1m` profile with MTP k=3 it is:

```bash
env VLLM_NEURON_BACKEND=neuron_native NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWITCH_CC=1 \
  TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=43200 NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=12 VLLM_RPC_TIMEOUT=1800000 \
  vllm serve <checkpoint> \
  --trust-remote-code --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --generation-config vllm --tensor-parallel-size 64 --enable-expert-parallel \
  --max-model-len 1048576 --max-num-batched-tokens 8192 --max-num-seqs 8 --num-gpu-blocks-override 38587 \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --limit-mm-per-prompt '{"image": 64, "video": 4, "audio": 8}' --allowed-local-media-path <repo>/assets \
  --served-model-name mimo-v2.6-flash --host 127.0.0.1 --port 8000 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --additional-config '{"neuron_config": {"quantization": "fp8", "num_batched_tokens_buckets": [512, 1024, 8192],
      "num_seqs_buckets": [1, 4, 8], "on_device_sampling_config": {"all_greedy": false},
      "mimo_attn_kernel": "nki", "mimo_expert_placement": "<repo>/config/placement3_s5.json"}}'
```

Use `scripts/serve.sh` rather than typing this: it also sets a separate compile-cache directory per configuration
([why](docs/configuration.md#5-compile-cache)) and the cache, temporary and log paths under `MIMO_ROOT`.

| Parameter | Why |
|---|---|
| `VLLM_NEURON_BACKEND=neuron_native` | vllm-neuron's PyTorch native backend |
| `--tensor-parallel-size 64 --enable-expert-parallel` | all 64 logical cores; the 256 experts split 64 ways |
| `--max-model-len 1048576`, `--max-num-batched-tokens 8192` | 1M context; longer prompts are prefilled in 8192-token segments |
| `num_batched_tokens_buckets [512, 1024, 8192]` | prefill graphs; a step is padded to the smallest bucket that holds it |
| `--max-num-seqs 8`, `num_seqs_buckets [1, 4, 8]` | up to 8 concurrent requests; decode graphs for 1, 4 and 8 rows |
| `--num-gpu-blocks-override 38587` | KV pool of 38,587 × 32 tokens; required so the graphs fit in HBM |
| `--enable-prefix-caching --prefix-cache-retention-interval 1024` | prefix caching, including "append to the end" of a prompt on the sliding-window layers |
| `quantization fp8` | FP8 linear layers and FP8 experts read from the original checkpoint |
| `mimo_attn_kernel nki` | the model's NKI attention kernels (prefill and decode) |
| `mimo_expert_placement` | static expert placement with prefill replicas of hot experts ([details](docs/configuration.md#3-expert-placement-configplacement3_s5json)) |
| `--speculative-config` mtp, 3 tokens | MTP speculative decoding with the 3 MTP layers ([options](docs/speculative-decoding.md)) |
| `--limit-mm-per-prompt` | up to 64 images, 4 videos and 8 audio clips per request |
| `--generation-config vllm` | the model's generation_config.json does not override request defaults |
| `NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=12`, `TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=43200` | keep the first compile within host memory |

Stop the server with Ctrl-C in its tmux window or `bash scripts/stop.sh`.

> [!CAUTION]
> The server has **no authentication**. Keep `--host 127.0.0.1`; put an authenticating proxy in front of it if other
> machines must reach it.

## Step 4. Send requests

Model name `mimo-v2.6-flash`; thinking is on by default (the reply first streams `reasoning`, then `content`).

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "max_tokens": 2048, "temperature": 1.0, "top_p": 0.95,
  "messages": [{"role": "user", "content": "Introduce the Great Wall in three sentences."}]}'
```

Thinking off: add `"chat_template_kwargs": {"enable_thinking": false}`. Streaming: add `"stream": true`.

Image / video / audio by `file://` URL (the file must be under `--allowed-local-media-path`, by default the
repository's `assets/`; set `MEDIA_DIR` before `scripts/serve.sh` to use another directory):

```bash
A=$PWD/assets
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mimo-v2.6-flash", "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "file://'"$A"'/number_739.png"}},
    {"type": "text", "text": "What number is written in the image? Answer with the number only."}]}]}'
# video: {"type": "video_url", "video_url": {"url": "file://'"$A"'/digits_1234.mp4"}}   -> "1, 2, 3, 4"
# audio: {"type": "audio_url", "audio_url": {"url": "file://'"$A"'/audio/jfk.wav"}}     -> JFK quote
```

Tool calling (OpenAI format, `"tools": [...]`, `"tool_choice": "auto"`) and structured outputs (`response_format`
with a JSON schema) work as with any vLLM server.

From your own computer, forward the port over SSH and use any OpenAI-compatible client:

```bash
ssh -i <your SSH key file> -N -L 8000:127.0.0.1:8000 ubuntu@<instance public DNS or IP>     # keep it open
```

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")   # the server checks no key
r = client.chat.completions.create(
    model="mimo-v2.6-flash", max_tokens=2048, temperature=1.0, top_p=0.95,
    messages=[{"role": "user", "content": "Introduce the Great Wall in three sentences."}])
print(getattr(r.choices[0].message, "reasoning", None))   # thinking (vLLM's extra field)
print(r.choices[0].message.content)                      # answer
```

[scripts/api_tests/ask.py](scripts/api_tests/ask.py) is a small streaming client that sends local files as base64
(standard library only):

```bash
python3 scripts/api_tests/ask.py "What is 17×23?"
python3 scripts/api_tests/ask.py --no-think --image assets/number_739.png "What number is in the image?"
python3 scripts/api_tests/ask.py --no-think --video assets/digits_1234.mp4 "Which digits appear, in order?"
python3 scripts/api_tests/ask.py --no-think --audio assets/audio/jfk.wav "Transcribe this audio exactly."
```

It prints `思考` (reasoning), `回答` (answer), `[结束原因]` (finish reason), `[用量]` (usage) and `[首字时间]`
(time to first token).

## Step 5. Smoke test

```bash
bash scripts/smoke_test.sh       # on the 8k profiles: bash scripts/smoke_test.sh --no-mm
```

13 checks: model list, English / Chinese / arithmetic chat, completions, streaming, thinking, tool calling, JSON
schema, 4 concurrent requests, image, video and audio. Expected: `smoke: 13 pass, 0 fail`. Results:
`/mnt/nvme/mimo/results/smoke.{json,md}`.

## Step 6. Benchmark

```bash
OUT=/mnt/nvme/mimo/bench/1m-mtp3 bash scripts/run_bench.sh       # about 12 minutes; SMOKE=1 for a tiny version
```

[scripts/run_bench.sh](scripts/run_bench.sh) runs, over HTTP only: `vllm bench serve` with random prompts (512 / 8K /
32K / 128K tokens, concurrency 1 and 4), long needle prompts (30K and 940K tokens: first request, identical repeat,
follow-up turn), image / video / audio requests, and GSM8K / MMLU (200 questions each). Summary:
`$OUT/summary/summary.md`.

Results of the recommended configuration (`1m`, MTP k=3):

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

GSM8K 98.0 %, MMLU 79.5 % (200 questions each). HBM per logical core: 20.96 GiB at ready, 21.67 GiB on the busiest
core after the 940K prompt (limit 24 GiB). First start from an empty compile cache: 1,877 s.

The same suite was run on the earlier deployment of this model with the PyTorch/XLA backend (Neuron SDK 2.32,
vllm-neuron 0.24); the comparison and the 1M configuration with 4 sequences are in
[results/bench_summary.md](results/bench_summary.md).

Notes:

- The random-token prompts of `vllm bench serve` are the worst case for speculative decoding (drafts are rarely
  accepted); on natural text MTP k=3 gives 1.4-2.0× the output throughput of no speculation
  ([speculative-decoding.md](docs/speculative-decoding.md#2-how-much-it-helps)).
- One request can use the whole 1M KV pool; with several long requests, later ones wait.

## Restarting and recovery

| Situation | What is left | What to do |
|---|---|---|
| server stopped (Ctrl-C, `scripts/stop.sh`) or crashed | everything | `bash scripts/serve.sh` again; ready in about 13 minutes from the cache |
| instance rebooted | data on the NVMe RAID, but the RAID is not mounted | `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme`, then `bash scripts/serve.sh` (about 13 minutes) |
| instance stopped and started, or replaced (for example at the end of a Capacity Block) | nothing: instance storage is erased | repeat the RAID setup and Steps 1-3, including the download and the 31-minute first compile |
| another profile or speculative option | the other configurations' caches | `bash scripts/stop.sh`, then `bash scripts/serve.sh --profile ... --spec ...`; the first start of a new combination compiles its graphs |

To skip the download on a new instance, copy `$MIMO_ROOT/hf` to durable storage (EBS or S3) beforehand and put it
back before Step 2; `scripts/download_weights.sh` then only checks it. Reusing a compile cache on another instance
has not been tested.

Disk: `$MIMO_ROOT/cache/<tree>/<profile>-<spec>` belongs to one configuration (deleting it makes that configuration
rebuild its torch.compile cache on its next start); `$MIMO_ROOT/cache/<tree>/shared` holds the compiled graphs of all configurations;
directories of an older `<tree>` are unused after the code changed. `$MIMO_ROOT/logs`, `results` and `bench` can be
deleted at any time; `$MIMO_ROOT/tmp` while no server runs.

## Other configurations

[config/serve_args.json](config/serve_args.json) has four profiles and five speculative options:

| Profile | Context | Concurrency | Multimodal | Notes |
|---|---|---|---|---|
| `1m` (default) | 1M | 8 | yes | recommended |
| `1m_c4` | 1M | 4 | yes | lower latency at concurrency ≤ 4 |
| `8k` | 8K | 32 | no | many short text requests |
| `8k_torch` | 8K | 32 | no | torch attention instead of the NKI kernels |

```bash
bash scripts/serve.sh --profile 8k                  # --spec off | mtp1 | mtp2 | mtp3 | dflash
```

Use `--spec off` for evaluations, regression tests or data generation that need bit-exact greedy output or `seed`,
and for ASR evaluation. Each new (profile, option) compiles its own graphs on its first start.
Details: [docs/configuration.md](docs/configuration.md), [docs/speculative-decoding.md](docs/speculative-decoding.md).

## Accuracy

- **Logits** against a true FP32 reference (16 prompts × 64 steps), relative to a CPU emulation of the GPU algorithm
  with the same weights: p95 error ratio 0.84-0.94 and σ ratio 1.00-1.04 for all four default configurations (below 1
  = closer to FP32 than the emulated GPU).
- **GSM8K / MMLU-Pro** (1,319 / 1,000 questions, thinking on): FP8 equals BF16 weights within one paired standard
  error (GSM8K greedy 0.9712 both). MMLU-Pro with GPU-aligned sampling and `max_tokens` 8192: 0.797 and 0.804 in two
  runs.
- **Audio**: word error rate 2.19-2.26 % on 1,551 LibriSpeech test-clean words without speculation.
- `scripts/run_bench.sh`: GSM8K 98.0 %, MMLU 79.5 % (200 questions each).

Method and numbers: [results/accuracy_summary.md](results/accuracy_summary.md),
[docs/design-notes.md](docs/design-notes.md#accuracy).

## Limitations

Details: [docs/limitations.md](docs/limitations.md).

- Validated only on Neuron SDK 2.33.0 / vllm-neuron 0.29.0.1.2.0 with the PyTorch native backend.
- `seed` with speculative decoding returns HTTP 400; speculative decoding is not bit-exact with `--spec off`.
- One image, video or audio item can have at most 114,688 tokens on the 1M profiles (videos of about 128K tokens and
  more are not supported); a 4K video at the default 32 frames is too large.
- At most 8 running requests on `1m`; one 1M-token request fills the KV pool.
- A 1M-token first prompt takes about 2.5-3 minutes to the first token (prefix-cache hits take seconds).
- First compile about 31 minutes; weights and compile cache live on instance storage.

## Troubleshooting

More cases: [docs/troubleshooting.md](docs/troubleshooting.md).

- **Not ready after 20 minutes**: the first start compiles; follow `scripts/wait_ready.sh`.
- **`fp8 quantization is currently not supported in neuron`**: run `scripts/setup_source.sh` (editable install).
- **HTTP 400 with `seed`**: speculative decoding is on; use `--spec off`.
- **HTTP 400 `... encoder-cache blocks ...`**: one media item is too large; sample fewer frames or lower the resolution.
- **Only reasoning, no answer**: raise `max_tokens` or turn thinking off.
- **`/mnt/nvme` empty after reboot**: re-assemble the RAID (after stop / start the data is gone).

## Repository layout

| Path | Content |
|---|---|
| [patches/vllm-neuron/](patches/vllm-neuron/) | the model support: 3 patches on vllm-neuron `release-0.29.0.1.2.0` |
| [sources.lock.json](sources.lock.json) | pinned base commit, patch sha256 values, expected tree hashes |
| [requirements/](requirements/) | package list of the serving venv (copied from the DLAMI vLLM venv) and the audio decoders |
| [weights.lock.json](weights.lock.json) | the pinned checkpoint's 154 files with size and sha256 |
| [config/](config/) | `serve_args.json` (all profiles and speculative options), `placement3_s5.json` (expert placement) |
| [scripts/](scripts/) | setup, serve / stop, smoke test and benchmark; `scripts/api_tests/` HTTP clients; `scripts/dev/` release checks |
| [assets/](assets/), [data/task_eval/](data/task_eval/) | test media with expected answers; GSM8K / MMLU questions for the benchmark |
| [docs/](docs/) | configuration, speculative decoding, design notes, limitations, troubleshooting, development, release checklist |
| [results/](results/) | measured benchmark and accuracy results |

Every file is described in [docs/development.md](docs/development.md#repository-layout).

## Related resources

- [XiaomiMiMo/MiMo-V2.6-Flash-MOPD on Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)
- [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
- [Neuron multi-framework DLAMI (2.33.0)](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.33.0/setup/multiframework-dlami.html)
- [mimo-v2.6-flash-vllm-neuron-pytorch-xla](https://github.com/terrificdm/mimo-v2.6-flash-vllm-neuron-pytorch-xla):
  the same model on the vllm-neuron PyTorch/XLA backend (Neuron SDK 2.32); differences:
  [design notes](docs/design-notes.md#compared-with-the-pytorchxla-deployment)
- In this repository: [configuration](docs/configuration.md), [speculative decoding](docs/speculative-decoding.md),
  [design notes](docs/design-notes.md), [limitations](docs/limitations.md),
  [troubleshooting](docs/troubleshooting.md), [development](docs/development.md), [results](results/README.md),
  [release checklist](docs/release-checklist.md).

## License

Apache-2.0, see [LICENSE](LICENSE). The model weights are MIT-licensed by Xiaomi and are not included. Third-party
sources are listed in [NOTICE](NOTICE).
