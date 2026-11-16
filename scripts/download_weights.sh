#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Download the pinned revision of XiaomiMiMo/MiMo-V2.6-Flash-MOPD (178 GB, 154 files, public, no token needed)
# into $HF_HOME and check every file against weights.lock.json. The checkpoint is used as is: no conversion.
# It must include audio_tokenizer/ (audio towers; the server does not start without it) and dflash/ (DFlash draft).
# Usage: bash scripts/download_weights.sh [--verify-only] [--size-only]
#   --verify-only  do not download, only check an existing snapshot
#   --size-only    check file count and sizes only (quick re-check; do not use for the first deployment)
# Re-running resumes an interrupted download.
# Exit code: 0 ready; 2 bad argument; 3 download or check failed.
set -uo pipefail
source "$(dirname "$0")/env.sh"
verify_only=false vargs=()
while [ $# -gt 0 ]; do
  case $1 in
    --verify-only) verify_only=true; shift ;;
    --size-only) vargs+=(--size-only); shift ;;
    *) die 2 "usage: $0 [--verify-only] [--size-only]" ;;
  esac
done
PY=$VENV/bin/python
[ -x "$PY" ] || die 2 "no venv at $VENV; run scripts/setup_venv.sh first (it provides huggingface_hub)"
if ! $verify_only; then
  log "downloading $HF_REPO@$HF_REVISION into $HF_HOME"
  HF_HUB_OFFLINE=0 "$PY" - <<'PY' || die 3 "snapshot_download failed (run the script again to resume)"
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=os.environ["HF_REPO"], revision=os.environ["HF_REVISION"], max_workers=8)
print("snapshot:", p)
assert os.path.realpath(p) == os.path.realpath(os.environ["HF_SNAPSHOT"]), (p, os.environ["HF_SNAPSHOT"])
PY
fi
[ -d "$HF_SNAPSHOT/audio_tokenizer" ] || die 3 "the checkpoint has no audio_tokenizer/ (required)"
[ -d "$HF_SNAPSHOT/dflash" ] || die 3 "the checkpoint has no dflash/ (DFlash draft)"
nice -n 10 python3 "$REPO_DIR/scripts/verify_weights.py" --snapshot "$HF_SNAPSHOT" "${vargs[@]}" || die 3 "weights check failed"
log "weights ready: $HF_SNAPSHOT"
