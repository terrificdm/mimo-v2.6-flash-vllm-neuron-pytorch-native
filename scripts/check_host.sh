#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Read-only checks before the deployment: instance type, the pinned DLAMI venv, python3.12, 16 idle Neuron devices,
# host memory, local (non-network) disk for MIMO_ROOT and enough free space.
# Usage: bash scripts/check_host.sh
# Exit code: 0 all checks pass; 1 at least one check failed (each failure is printed).
set -uo pipefail
source "$(dirname "$0")/env.sh"

WEIGHTS_GB=178        # 177,765,031,044 bytes in weights.lock.json
CACHE_FLOOR_GB=150    # one (profile, speculative option) cache is about 8-25 GB; every option of 1m and 8k is about 150-200 GB
ROOT_FLOOR_GB=10
fail=0
bad() { log "FAIL: $*"; fail=1; }
ok() { log "ok: $*"; }

tok=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
itype=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/meta-data/instance-type || true)
[ "$itype" = trn2.48xlarge ] && ok "instance type $itype" || bad "instance type '${itype:-unknown}', expected trn2.48xlarge"

if [ -x "$DLAMI_VENV/bin/python" ]; then
  vers=$("$DLAMI_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm-neuron"), m.version("vllm"))' 2>/dev/null || true)
  [ "$vers" = "0.29.0.1.2.0 0.29.0" ] && ok "DLAMI venv vllm-neuron / vllm = $vers" \
    || bad "DLAMI venv has vllm-neuron / vllm '$vers', expected '0.29.0.1.2.0 0.29.0' (Neuron SDK 2.33.0 DLAMI)"
else
  bad "missing $DLAMI_VENV: use the Neuron multi-framework DLAMI for Neuron SDK 2.33.0"
fi
[ -x /usr/bin/python3.12 ] && ok "/usr/bin/python3.12" || bad "missing /usr/bin/python3.12 (setup_venv.sh needs it)"
for c in git curl neuron-ls; do command -v "$c" >/dev/null && ok "command $c" || bad "command $c not found"; done

if command -v neuron-ls >/dev/null; then
  js=$(neuron-ls --json-output 2>/dev/null || echo '[]')
  ndev=$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)))' <<<"$js" 2>/dev/null || echo 0)
  [ "$ndev" = 16 ] && ok "$ndev Neuron devices" || bad "$ndev Neuron devices, expected 16"
  busy=$(python3 -c 'import json,sys; print(sum(len(x.get("neuron_processes") or []) for x in json.load(sys.stdin)))' <<<"$js" 2>/dev/null || echo 0)
  [ "${busy:-0}" = 0 ] && ok "NeuronCores idle" || bad "$busy process(es) hold NeuronCores (see neuron-ls); stop them first"
fi

mem=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
[ "$mem" -ge 1900 ] && ok "host memory ${mem} GiB" || bad "host memory ${mem} GiB < 1900 GiB (the first compile traces on 64 ranks at once)"

fs_of() { local p=$1; while [ ! -e "$p" ]; do p=$(dirname "$p"); done; echo "$p"; }
avail_gb() { df -BG --output=avail "$(fs_of "$1")" | tail -1 | tr -dc 0-9; }
fstype() { df --output=fstype "$(fs_of "$1")" | tail -1 | tr -d ' '; }
for d in "$MIMO_ROOT" "$CACHE_BASE"; do
  case $(fstype "$d") in
    nfs*|cifs|smb*|fuse.*) bad "$d is on a network file system ($(fstype "$d")); the compile-cache lock (flock) is not reliable there";;
  esac
done
if [ -d "$HF_SNAPSHOT" ]; then ok "checkpoint already present at $HF_SNAPSHOT (download_weights.sh will verify it)"
else
  av=$(avail_gb "$HF_HOME")
  [ "${av:-0}" -ge $WEIGHTS_GB ] && ok "HF_HOME free ${av} GB >= ${WEIGHTS_GB} GB" || bad "HF_HOME ($(fs_of "$HF_HOME")) free ${av} GB < ${WEIGHTS_GB} GB"
fi
av=$(avail_gb "$CACHE_BASE")
[ "${av:-0}" -ge $CACHE_FLOOR_GB ] && ok "CACHE_BASE free ${av} GB" || bad "CACHE_BASE ($(fs_of "$CACHE_BASE")) free ${av} GB < ${CACHE_FLOOR_GB} GB"
av=$(avail_gb /)
[ "${av:-0}" -ge $ROOT_FLOOR_GB ] && ok "root disk free ${av} GB" || bad "root disk free ${av} GB < ${ROOT_FLOOR_GB} GB"

[ $fail = 0 ] && log "all checks passed" || log "some checks failed (FAIL lines above)"
exit $fail
