# Release checklist

Run from the repository root before a release or a pull request. Everything must pass.

| # | Check | Command | Pass |
|---|---|---|---|
| 1 | self-contained copy | `bash scripts/dev/check_independence.sh` | `== result: PASS` |
| 2 | documentation paths | `python3 scripts/dev/check_doc_paths.py` | `missing=0` |
| 3 | sensitive patterns | `bash scripts/dev/scan_sensitive.sh` (`SHOW=1` prints the matches) | `total_findings=0` |
| 4 | patches match the lock file | `shasum -a 256 patches/vllm-neuron/*.patch`, compare with `patches` in `sources.lock.json` (`scripts/setup_source.sh` does the same check before `git am`) | all equal |
| 5 | test media | `(cd assets && sha256sum -c SHA256SUMS)` | all `OK` |
| 6 | no weights, NEFFs or caches | `find . -path ./.git -prune -o -type f \( -name '*.safetensors' -o -name '*.neff' -o -name '*.bin' -o -name '*.pt' \) -print` | no output |
| 7 | LICENSE and NOTICE | present; NOTICE lists every copied or adapted source | — |
| 8 | end-to-end on a trn2.48xlarge | follow README Steps 1-6 in a fresh `MIMO_ROOT` | `smoke_test.sh` all pass; `run_bench.sh` no failed steps |

`check_independence.sh` copies the repository without `.git` into a temporary directory and runs checks 2 and 3 there,
plus `py_compile` for every `.py`, `bash -n` for every `.sh`, and searches for development references (internal
decision or task numbers, development paths) and commit hashes. `patches/` is excluded from the last two searches
(the vllm-neuron source carries its own code comments and the patch headers contain blob hashes), and
`requirements/` from the hash search (pinned versions such as `neuronx-cc==2.28.7631.0+<build hash>`). The pinned upstream
references a reader needs (Hugging Face revision, vllm-neuron base commit, expected tree hashes) are allowed.
