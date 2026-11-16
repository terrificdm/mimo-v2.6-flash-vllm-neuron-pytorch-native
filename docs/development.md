# Development

## Repository layout

Top level:

| File | Content |
|---|---|
| `README.md`, `README.zh-CN.md` | the tutorial (English / Chinese) |
| `LICENSE`, `NOTICE` | Apache-2.0; third-party sources and what was copied or adapted |
| `sources.lock.json` | vllm-neuron repository, base commit, sha256 of each patch, expected tree and `vllm_neuron/` subtree hashes |
| `weights.lock.json` | Hugging Face repository, revision, and path / size / sha256 of each of the 154 checkpoint files |
| `.gitignore`, `.gitattributes` | keeps weights, caches and logs out of Git; keeps the patches byte-exact |

| Directory | File | Content |
|---|---|---|
| `patches/vllm-neuron/` | `0001-...`, `0002-...`, `0003-...` | the model support as patches on vllm-neuron `release-0.29.0.1.2.0` ([Patches](#patches)) |
| `requirements/` | `serve-freeze.txt` | the DLAMI vLLM venv's 370 packages with exact versions (installed with `--no-deps`) |
| | `audio.txt` | `av` and `soundfile` for audio decoding |
| `config/` | `serve_args.json` | environment, profiles (`1m`, `1m_c4`, `8k`, `8k_torch`), speculative options, `placement_sha256` ([configuration.md](configuration.md)) |
| | `placement3_s5.json` | expert placement of the 1M profiles: 64 ranks × 5 slots for each of the 47 MoE layers |
| `scripts/` | `env.sh` | paths and environment shared by all scripts (`MIMO_ROOT` and the directories under it) |
| | `check_host.sh` | read-only host checks |
| | `setup_venv.sh` | creates the serving venv |
| | `setup_source.sh` | clones vllm-neuron, applies and checks the patches, editable install |
| | `download_weights.sh`, `verify_weights.py` | downloads the checkpoint; checks it against `weights.lock.json` |
| | `render_serve_args.py` | turns a profile and option of `serve_args.json` into the environment and the `vllm serve` command |
| | `serve.sh`, `wait_ready.sh`, `stop.sh` | starts the server in the foreground, waits for it, stops it |
| | `smoke_test.sh` | runs `api_tests/smoke.py` |
| | `run_bench.sh` | the benchmark suite (`vllm bench serve`, needle / prefix cache, media, GSM8K / MMLU) |
| `scripts/api_tests/` | `common.py` | HTTP helpers and the result writer shared by the clients (standard library only) |
| | `smoke.py` | the 13 smoke checks |
| | `ask.py` | streaming client for text / image / video / audio questions |
| | `needle.py`, `prefix_cache_test.py` | builds needle prompts; sends a prompt cold, repeated and with a follow-up turn |
| | `task_eval.py` | GSM8K / MMLU on the questions in `data/task_eval/` |
| `scripts/dev/` | `check_independence.sh`, `check_doc_paths.py`, `scan_sensitive.sh` | release checks ([release-checklist.md](release-checklist.md)) |
| `assets/` | images, `digits_1234.mp4`, `audio/*.wav`, `answers.json`, `SHA256SUMS` | test media and their expected answers |
| `data/task_eval/` | `gsm8k_test_200_seed0.jsonl`, `mmlu_test_200_seed0.jsonl`, `task_eval_item_ids.json` | 200 questions each and their ids in the official test splits |
| `docs/` | `configuration.md` | profiles, speculative options, expert placement, multimodal limits, compile cache, environment, scripts |
| | `speculative-decoding.md` | which option to choose, measured gains, limits, memory |
| | `design-notes.md` | how the port works, accuracy, comparison with the PyTorch/XLA deployment |
| | `limitations.md`, `troubleshooting.md` | known limits; symptoms and fixes |
| | `development.md` | this file |
| | `release-checklist.md` | checks before a release |
| `results/` | `bench_summary.md`, `accuracy_summary.md`, `README.md` | measured results and how they were obtained |
| `results/bench/` | `<profile>-<option>.summary.json` | summaries of `scripts/run_bench.sh` runs |

Created on the instance, not in the repository (`MIMO_ROOT` defaults to `/mnt/nvme/mimo`): `$MIMO_ROOT/venv`,
`$MIMO_ROOT/src/vllm-neuron`, `$MIMO_ROOT/hf` (checkpoint), `$MIMO_ROOT/cache` (compile caches),
`$MIMO_ROOT/neuron-compile-cache`, `$MIMO_ROOT/logs`, `$MIMO_ROOT/run`, `$MIMO_ROOT/tmp`, `$MIMO_ROOT/results`,
`$MIMO_ROOT/bench`.

## Patches

`patches/vllm-neuron/` holds three patches on vllm-neuron `release-0.29.0.1.2.0` (commit `eb6cb36`), applied in
order by `scripts/setup_source.sh`:

| Patch | Content |
|---|---|
| `0001-vllm-neuron-framework-changes-for-MiMo-V2.6-Flash.patch` | changes 21 existing vllm-neuron files and adds 4 (MTP proposer, host logits processing, encoder-cache manager, an accuracy helper): registry, NeuronConfig, KV cache, scheduler, sampling, speculative decoding, structured outputs, platform checks, model runner / worker, accuracy tooling |
| `0002-mimo_v2-vendored-copy-of-the-nkilib-moe_tkg-kernels.patch` | `vllm_neuron/model/mimo_v2/nki_moe_tkg/`: copy of nkilib's MoE token-generation kernels |
| `0003-mimo_v2-MiMo-V2.6-Flash-model-for-the-native-backend.patch` | `vllm_neuron/model/mimo_v2/` and `vllm_neuron/_kernels/mimo_v2/`: the model package and its NKI attention kernels |

The patches change only `vllm_neuron/`. After `git am`, `setup_source.sh` checks that the tree and the
`vllm_neuron/` subtree have the hashes recorded in `sources.lock.json`; the subtree hash is the one all published
measurements were taken with.

### Working on the code

`$SRC_DIR/vllm-neuron` is an editable install, so changes there take effect at the next server start.

```bash
cd $MIMO_ROOT/src/vllm-neuron            # branch mimo-v2.6-flash, three commits on the base
# edit, then commit
git commit -am "mimo_v2: ..."
```

Commit before you restart the server: `scripts/serve.sh` puts the hash of the committed tree (`HEAD^{tree}`) into
`VLLM_CACHE_ROOT`, so a new commit gets a new compile-cache directory and the next start compiles again. Uncommitted
edits keep the old directory, which can load stale graphs. A checkout with your commits no longer matches
`sources.lock.json`, so `scripts/setup_source.sh` refuses it until the lock file is updated. To publish the change,
regenerate the patches and the lock file:

```bash
cd $MIMO_ROOT/src/vllm-neuron
rm <repo>/patches/vllm-neuron/*.patch
git format-patch --zero-commit -o <repo>/patches/vllm-neuron eb6cb360f7cff6082e22a9437d8ce0ba3891304a..HEAD
# then update "patches" (sha256 per file), "expected_head_tree" (git rev-parse 'HEAD^{tree}') and
# "expected_vllm_neuron_subtree" (git rev-parse HEAD:vllm_neuron) in sources.lock.json
```

## Changing the configuration

Copy `config/serve_args.json`, change your copy, and start with it:

```bash
cp config/serve_args.json ~/my_serve_args.json
# edit ~/my_serve_args.json
SERVE_ARGS=~/my_serve_args.json bash scripts/serve.sh --profile 1m --dry-run   # check the rendered command
SERVE_ARGS=~/my_serve_args.json bash scripts/serve.sh --profile 1m
```

- Anything that changes graph shapes (buckets, `--max-model-len`, `--max-num-seqs`, `--num-gpu-blocks-override`,
  speculative k) compiles new graphs on the next start; graphs that stay the same are reused from the shared NEFF
  cache.
- When prefix caching is on and you change `--max-num-batched-tokens`, set `neuron_config.kv_segment_size_buckets`
  as well (for example `[8192]`).
- Every new prefill bucket changes the numerics of the requests that fall into it (a different graph rounds
  differently in BF16); check greedy outputs and an accuracy task after adding one.

## Checks before a release

[release-checklist.md](release-checklist.md).
