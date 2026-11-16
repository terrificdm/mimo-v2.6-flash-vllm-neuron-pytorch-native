#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start vllm serve in the foreground (run it inside tmux) with one profile and speculative option of
# config/serve_args.json. The output also goes to $LOG_DIR/serve-<profile>-<spec>-<time>.log.
#   bash scripts/serve.sh                          # profile 1m, MTP k=3 (recommended)
#   bash scripts/serve.sh --profile 8k --spec off  # other profiles / options: docs/configuration.md
#   bash scripts/serve.sh --dry-run                # print the environment and command, do not start
#   SERVE_ARGS=my_serve_args.json bash scripts/serve.sh   # use a modified copy of config/serve_args.json
# Compile cache: VLLM_CACHE_ROOT=$CACHE_BASE/<first 12 chars of the vllm-neuron tree hash>/<profile>-<spec>.
#   neuron/torch_compile is separate for each (profile, option); the NEFF directories are links to a shared/
#   directory, so graphs that two options have in common are compiled once. A new code tree (new patches) gets a
#   new set of directories automatically.
# Stop: Ctrl-C in the tmux window, or bash scripts/stop.sh from another shell.
# Exit code: 2 bad argument or missing setup; otherwise the exit code of vllm serve.
set -uo pipefail
source "$(dirname "$0")/env.sh"
profile="" spec="" port=8000 dry=false
while [ $# -gt 0 ]; do
  case $1 in
    --profile) profile=${2:-}; shift 2 ;;
    --spec) spec=${2:-}; shift 2 ;;
    --port) port=${2:-}; shift 2 ;;
    --dry-run) dry=true; shift ;;
    *) die 2 "unknown argument $1 (usage: $0 [--profile P] [--spec S] [--port N] [--dry-run])" ;;
  esac
done
[[ "$port" =~ ^[0-9]+$ ]] || die 2 "--port must be a number"
$dry || [ -x "$VENV/bin/vllm" ] || die 2 "no vllm in $VENV; run scripts/setup_venv.sh and scripts/setup_source.sh"
$dry || [ -f "$HF_SNAPSHOT/config.json" ] || die 2 "no checkpoint at $HF_SNAPSHOT; run scripts/download_weights.sh"

SERVE_ARGS=${SERVE_ARGS:-$REPO_DIR/config/serve_args.json}
[ -f "$SERVE_ARGS" ] || die 2 "no $SERVE_ARGS"
render=$(python3 "$REPO_DIR/scripts/render_serve_args.py" --file "$SERVE_ARGS" --model "$HF_SNAPSHOT" --repo-dir "$REPO_DIR" \
  --media-dir "$MEDIA_DIR" --port "$port" ${profile:+--profile "$profile"} ${spec:+--spec "$spec"} --format json) || exit 2
profile=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["profile"])' <<<"$render")
spec=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["spec"])' <<<"$render")
ARGV=() ENVV=()
while IFS= read -r l; do ARGV+=("$l"); done < <(python3 -c 'import json,sys; [print(a) for a in json.load(sys.stdin)["argv"]]' <<<"$render")
while IFS= read -r l; do ENVV+=("$l"); done < <(python3 -c 'import json,sys; [print(f"{k}={v}") for k, v in json.load(sys.stdin)["env"].items()]' <<<"$render")

tree=$(git -C "$VLLM_NEURON_DIR" rev-parse 'HEAD^{tree}' 2>/dev/null || echo no-source-yet)
export VLLM_CACHE_ROOT=$CACHE_BASE/${tree:0:12}/$profile-$spec
shared=$CACHE_BASE/${tree:0:12}/shared
log "profile=$profile spec=$spec port=$port"
log "VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"
if $dry; then
  python3 "$REPO_DIR/scripts/render_serve_args.py" --file "$SERVE_ARGS" --model "$HF_SNAPSHOT" --repo-dir "$REPO_DIR" --media-dir "$MEDIA_DIR" \
    --port "$port" --profile "$profile" --spec "$spec" --format shell
  exit 0
fi
if curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
  die 2 "a server already answers on port $port (scripts/stop.sh stops the one started by this script)"
fi
mkdir -p "$VLLM_CACHE_ROOT/neuron/torch_compile"
for d in neuron/native_neff_cache neuron/compile_cache modelinfos; do
  mkdir -p "$shared/$d"
  [ -e "$VLLM_CACHE_ROOT/$d" ] || ln -s "$shared/$d" "$VLLM_CACHE_ROOT/$d"
  [ -L "$VLLM_CACHE_ROOT/$d" ] || die 2 "$VLLM_CACHE_ROOT/$d must be a link to $shared/$d; use an empty CACHE_BASE"
done
logf=$LOG_DIR/serve-$profile-$spec-$(date -u +%Y%m%dT%H%M%SZ).log
ln -sfn "$logf" "$LOG_DIR/serve-latest.log"
echo "$$" > "$RUN_DIR/serve.pid"
log "log: $logf (also $LOG_DIR/serve-latest.log)"
log "ready when: curl -sf http://127.0.0.1:$port/v1/models (first start from an empty cache: about 30-40 min)"
export HF_HUB_OFFLINE=1
for e in "${ENVV[@]}"; do export "${e?}"; done
# vllm-neuron logs its backend only when get_backend() is called; print it once so the log shows the route.
"$VENV/bin/python" -c "from vllm_neuron.backend import get_backend; print('vllm-neuron backend:', get_backend())" 2>&1 | tee -a "$logf"
exec > >(tee -a "$logf") 2>&1
exec "$VENV/bin/vllm" "${ARGV[@]:1}"
