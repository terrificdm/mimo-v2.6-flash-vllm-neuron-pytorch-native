# Development

## Repository layout

```
README.md, README.zh-CN.md   tutorial (English / Chinese)
LICENSE, NOTICE              Apache-2.0; third-party sources
sources.lock.json            vllm-neuron base commit, patch sha256 values, expected tree hashes
patches/vllm-neuron/         the model code, as patches on vllm-neuron release-0.29.0.1.2.0
requirements/                serve-freeze.txt (the DLAMI vLLM venv's package set), audio.txt (av, soundfile)
weights.lock.json            the pinned checkpoint's files: size and sha256
config/serve_args.json       all serving profiles and speculative options
config/placement3_s5.json    expert placement table of the 1M profiles
scripts/                     setup, serve / stop, smoke test, benchmark (see docs/configuration.md, section 8)
scripts/api_tests/           HTTP clients used by the smoke test and the benchmark (standard library only)
scripts/dev/                 repository checks run before a release
assets/                      test media (image, video, audio) with expected answers and SHA256SUMS
data/task_eval/              GSM8K and MMLU subsets (200 questions each) for scripts/run_bench.sh
docs/                        configuration, speculative decoding, design notes, limitations, troubleshooting
results/                     measured results
```

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
