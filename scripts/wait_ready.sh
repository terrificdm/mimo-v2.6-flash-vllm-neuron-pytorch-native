#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wait until the server answers /v1/models, printing progress every 5 minutes.
# Usage: bash scripts/wait_ready.sh [--port 8000] [--timeout 7200]
# Exit code: 0 ready; 1 timeout or the server process exited; 2 bad argument.
set -uo pipefail
source "$(dirname "$0")/env.sh"
port=8000 timeout=7200
while [ $# -gt 0 ]; do
  case $1 in
    --port) port=${2:-}; shift 2 ;;
    --timeout) timeout=${2:-}; shift 2 ;;
    *) die 2 "usage: $0 [--port N] [--timeout SECONDS]" ;;
  esac
done
t0=$(date +%s) last=0
while ! curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null; do
  now=$(( $(date +%s) - t0 ))
  if [ -f "$RUN_DIR/serve.pid" ] && ! kill -0 "$(cat "$RUN_DIR/serve.pid")" 2>/dev/null; then
    log "the server process exited; see $LOG_DIR/serve-latest.log"; tail -n 20 "$LOG_DIR/serve-latest.log" 2>/dev/null; exit 1
  fi
  [ "$now" -ge "$timeout" ] && { log "not ready after $timeout s"; exit 1; }
  if [ $((now - last)) -ge 300 ]; then
    last=$now
    log "waiting ($now s); last log line: $(tail -n 1 "$LOG_DIR/serve-latest.log" 2>/dev/null | cut -c1-160)"
  fi
  sleep 10
done
log "ready after $(( $(date +%s) - t0 )) s of waiting: $(curl -s "http://127.0.0.1:$port/v1/models" | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])')"
