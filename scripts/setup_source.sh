#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Clone vllm-project/vllm-neuron, check out the pinned base commit, apply patches/vllm-neuron/*.patch with git am,
# check the resulting tree hashes against sources.lock.json, then install it into $VENV in editable mode.
# Editable install is required: vLLM finds the neuron plugin through its entry points (vllm.general_plugins /
# vllm.platform_plugins); putting the source on PYTHONPATH does not register them.
# Usage: bash scripts/setup_source.sh [--no-install]
# Exit code: 0 done; 2 bad argument; 4 clone / patch / hash / install failed.
set -uo pipefail
source "$(dirname "$0")/env.sh"
install=true
case "${1:-}" in
  "") ;;
  --no-install) install=false ;;
  *) die 2 "usage: $0 [--no-install]" ;;
esac
LOCK=$REPO_DIR/sources.lock.json
q() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$LOCK" "$1"; }
url=$(q 'd["url"]'); base=$(q 'd["base_commit"]'); pdir=$REPO_DIR/$(q 'd["patches_dir"]')
want_tree=$(q 'd["expected_head_tree"]'); want_sub=$(q 'd["expected_vllm_neuron_subtree"]')

# Patch files must be byte-identical to the ones the lock file was made from.
python3 - "$LOCK" "$pdir" <<'PY' || die 4 "patch files do not match sources.lock.json (re-clone this repository)"
import hashlib, json, os, sys
lock, pdir = json.load(open(sys.argv[1])), sys.argv[2]
want = {p["file"]: p["sha256"] for p in lock["patches"]}
have = sorted(f for f in os.listdir(pdir) if f.endswith(".patch"))
assert have == sorted(want), f"patch files {have} != {sorted(want)}"
for f in have:
    got = hashlib.sha256(open(os.path.join(pdir, f), "rb").read()).hexdigest()
    assert got == want[f], f"{f}: sha256 {got} != {want[f]}"
print(f"{len(have)} patch files match sources.lock.json")
PY

dir=$VLLM_NEURON_DIR
g=(git -C "$dir")
if [ -d "$dir/.git" ]; then
  cur=$("${g[@]}" rev-parse 'HEAD^{tree}' 2>/dev/null || true)
  [ "$cur" = "$want_tree" ] || die 4 "$dir exists but its tree is ${cur:-unknown}, not $want_tree; move it away and run again"
  log "$dir is already at the expected tree; skipping clone and patches"
else
  mkdir -p "$(dirname "$dir")"
  log "cloning $url"
  git clone -q "$url" "$dir" || die 4 "git clone failed"
  "${g[@]}" checkout -q -b mimo-v2.6-flash "$base" || die 4 "checkout $base failed"
  for p in "$pdir"/*.patch; do
    # A fixed committer identity, so git am does not need user.name / user.email on the instance.
    if ! GIT_COMMITTER_NAME=mimo GIT_COMMITTER_EMAIL=mimo@localhost "${g[@]}" am -q --3way "$p"; then
      "${g[@]}" am --abort 2>/dev/null
      die 4 "git am failed: $(basename "$p")"
    fi
  done
  log "applied $(ls "$pdir"/*.patch | wc -l) patches"
fi
tree=$("${g[@]}" rev-parse 'HEAD^{tree}'); sub=$("${g[@]}" rev-parse HEAD:vllm_neuron)
[ "$tree" = "$want_tree" ] || die 4 "tree $tree != expected $want_tree"
[ "$sub" = "$want_sub" ] || die 4 "vllm_neuron/ subtree $sub != expected $want_sub"
log "tree $tree and vllm_neuron/ subtree $sub match sources.lock.json"

if $install; then
  [ -x "$VENV/bin/pip" ] || die 4 "no venv at $VENV; run scripts/setup_venv.sh first"
  "$VENV/bin/pip" install -q --no-deps -e "$dir" || die 4 "pip install -e $dir failed"
  [ -z "${VLLM_PLUGINS:-}" ] || [[ ",$VLLM_PLUGINS," == *,neuron,* ]] \
    || die 4 "VLLM_PLUGINS='$VLLM_PLUGINS' excludes neuron; unset VLLM_PLUGINS"
  "$VENV/bin/python" - <<'PY' || die 4 "vLLM plugin entry-point check failed"
import importlib.metadata as m
import vllm_neuron
eps = {(e.group, e.name) for e in m.entry_points() if e.group.startswith("vllm.")}
missing = {("vllm.general_plugins", "neuron"), ("vllm.platform_plugins", "neuron")} - eps
print("vllm_neuron imported from", vllm_neuron.__file__)
raise SystemExit(f"missing entry points {sorted(missing)}" if missing else 0)
PY
  log "vllm-neuron installed (editable) into $VENV"
fi
