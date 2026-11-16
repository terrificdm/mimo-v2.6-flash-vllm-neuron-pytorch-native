# Troubleshooting

The server log is `$MIMO_ROOT/logs/serve-latest.log` (a link to the log of the latest `scripts/serve.sh` start).

## Setup

| Symptom | Cause and fix |
|---|---|
| `check_host.sh`: `missing /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` | not the Neuron SDK 2.33.0 DLAMI; launch the instance from that AMI |
| `setup_venv.sh`: `venv differs from requirements/serve-freeze.txt` | something was installed into `$VENV` by hand; remove `$VENV` and run the script again |
| `setup_source.sh`: `patch files do not match sources.lock.json` | the patch files changed (for example by a Git setting that rewrites line endings); clone the repository again |
| `setup_source.sh`: `exists but its tree is ...` | `$SRC_DIR/vllm-neuron` is from another version; move it away and run again |
| `setup_source.sh`: `git am failed` | the clone is not at the pinned base commit; move `$SRC_DIR/vllm-neuron` away and run again |
| `download_weights.sh` exits 3 during the download | network interruption; run it again (it resumes) |
| `download_weights.sh`: `sha256 ... !=` | a damaged file; delete that file's blob under `$HF_HOME/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD/blobs/` and run again |

## Start-up

| Symptom | Cause and fix |
|---|---|
| not ready after 20 minutes | the first start compiles for 25-35 minutes; `bash scripts/wait_ready.sh` prints the last log line every 5 minutes |
| `fp8 quantization is currently not supported in neuron` | the vllm-neuron plugin is not registered: run `scripts/setup_source.sh` (editable install), and do not set `VLLM_PLUGINS` without `neuron` |
| the log shows `on the XLA route`, or no `Using neuron_native backend` | `VLLM_NEURON_BACKEND` is not `neuron_native`; start through `scripts/serve.sh`, which sets it |
| host out of memory during the first compile | keep `NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=12`, use `scripts/serve.sh` (separate compile directory per option), do not run other large jobs at the same time |
| `max_num_batched_tokens ... not one of the supported KV segment sizes` or `Automatic Prefix Caching (APC) requires segmented prefill` | you changed `--max-num-batched-tokens` with prefix caching on: also set `neuron_config.kv_segment_size_buckets`, for example `[8192]` |
| DFlash on 8K logs `lowering max_num_seqs 32 -> 16` | expected; DFlash serves at most 16 concurrent requests |
| `a server already answers on port 8000` | stop the running server first (`bash scripts/stop.sh`, or Ctrl-C in its tmux window) |
| `[NLA001]` in the log | the only case in which `/var/tmp/neuron-compile-cache` should be cleared |

## Requests

| Symptom | Cause and fix |
|---|---|
| HTTP 400 with `seed` | speculative decoding is on; start with `--spec off` to use `seed` |
| HTTP 400 `... needs N encoder-cache blocks of 512 tokens; this server allows 224 per item` | one video / audio item has more than 114,688 tokens: sample fewer frames, lower the resolution or split it |
| HTTP 400 for too many images / videos / audio clips | more items than `--limit-mm-per-prompt` allows (64 / 4 / 8) |
| `file://` media refused | the file is not under `--allowed-local-media-path` (`$MEDIA_DIR`, default `assets/`); send it as base64 or set `MEDIA_DIR` before `scripts/serve.sh` |
| only reasoning, no answer, `finish_reason` `length` | thinking used up `max_tokens`; raise `max_tokens` or pass `"chat_template_kwargs": {"enable_thinking": false}` |
| requests queue on the 1M server | at most 8 run at once (`1m_c4`: 4), and one request can take the whole 1M KV pool |
| completions with a JSON schema are not constrained | with `--reasoning-parser mimo` the grammar starts after `</think>`; use the chat API or end the prompt with `<think></think>` |

## Stopping

| Symptom | Cause and fix |
|---|---|
| `stop.sh`: `NeuronCores still in use` | another process holds the devices; look it up with `neuron-ls` and stop it only if it is yours |
| `/mnt/nvme` is empty after a reboot | re-assemble the RAID: `sudo mdadm --assemble --scan && sudo mount -o noatime /dev/md0 /mnt/nvme` (after a stop / start of the instance the data is gone and the steps have to be repeated) |
