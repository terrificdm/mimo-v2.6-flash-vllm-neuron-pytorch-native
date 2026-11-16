#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create the serving venv $VENV as an exact copy of the DLAMI vLLM venv's package set, plus the audio decoders.
# The DLAMI venv itself is not modified.
#   /usr/bin/python3.12 -m venv $VENV
#   pip install --no-deps -r requirements/serve-freeze.txt   (the DLAMI venv's pip freeze, Neuron pip index)
#   pip install --no-deps -r requirements/audio.txt          (av, soundfile: WAV / FLAC / MP3 / Ogg decoding)
# --no-deps keeps pip from re-resolving torch / vllm versions. The patched vllm-neuron is installed by setup_source.sh.
# Usage: bash scripts/setup_venv.sh        (re-running only installs what is missing, then checks again)
# Exit code: 0 ready; 1 install or check failed.
set -uo pipefail
source "$(dirname "$0")/env.sh"
INDEX=https://pip.repos.neuron.amazonaws.com
FREEZE=$REPO_DIR/requirements/serve-freeze.txt
AUDIO=$REPO_DIR/requirements/audio.txt

if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV"
  /usr/bin/python3.12 -m venv "$VENV" || die 1 "python3.12 -m venv failed"
else
  log "$VENV exists; installing anything missing"
fi
pipv=$("$DLAMI_VENV/bin/pip" --version 2>/dev/null | awk '{print $2}')
[ -n "$pipv" ] && { "$VENV/bin/pip" install -q "pip==$pipv" || die 1 "pip==$pipv failed"; }
log "installing requirements/serve-freeze.txt (about 370 packages, several minutes)"
"$VENV/bin/pip" install -q --no-deps --extra-index-url "$INDEX" -r "$FREEZE" || die 1 "serve-freeze.txt failed"
"$VENV/bin/pip" install -q --no-deps --extra-index-url "$INDEX" -r "$AUDIO" || die 1 "audio.txt failed"

# The venv must match the freeze line by line (editable packages and the audio extras excluded).
diff <("$VENV/bin/pip" freeze 2>/dev/null | grep -v '^-e ' | grep -v '^#' | grep -vxF -f "$AUDIO" | sort) \
     <(sed '/^\s*$/d' "$FREEZE" | sort) > "$TMPDIR/venv-freeze.diff"
if [ -s "$TMPDIR/venv-freeze.diff" ]; then
  log "venv differs from requirements/serve-freeze.txt:"; head -20 "$TMPDIR/venv-freeze.diff"; exit 1
fi
"$VENV/bin/python" -c 'import torch, vllm; print("torch", torch.__version__, "vllm", vllm.__version__)' || die 1 "import torch / vllm failed"
log "venv ready: $VENV"
