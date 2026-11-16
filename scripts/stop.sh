#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stop the server started by scripts/serve.sh (pid in $RUN_DIR/serve.pid), then check that the NeuronCores are free.
# Sends SIGTERM to vllm serve and waits; if it is still running after 120 s, sends SIGTERM to its process group.
# Processes that serve.sh did not start are never signalled.
# Usage: bash scripts/stop.sh
# Exit code: 0 stopped and NeuronCores idle; 1 something still holds NeuronCores; 2 no pid file.
set -uo pipefail
source "$(dirname "$0")/env.sh"
pidf=$RUN_DIR/serve.pid
[ -f "$pidf" ] || die 2 "no $pidf (was the server started with scripts/serve.sh?)"
pid=$(cat "$pidf")
alive() { kill -0 "$pid" 2>/dev/null && grep -q vllm "/proc/$pid/cmdline" 2>/dev/null; }
if alive; then
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  log "SIGTERM to vllm serve pid $pid"
  kill -TERM "$pid"
  for _ in $(seq 120); do alive || break; sleep 1; done
  if alive && [ -n "$pgid" ]; then
    log "still running after 120 s; SIGTERM to process group $pgid"
    kill -TERM -- "-$pgid" 2>/dev/null
    for _ in $(seq 60); do alive || break; sleep 1; done
  fi
  alive && die 1 "pid $pid is still running"
else
  log "pid $pid is not running"
fi
rm -f "$pidf"
for _ in $(seq 60); do
  busy=$(neuron-ls --json-output 2>/dev/null | python3 -c 'import json,sys; print(sum(len(x.get("neuron_processes") or []) for x in json.load(sys.stdin)))' 2>/dev/null || echo 0)
  [ "${busy:-0}" = 0 ] && { log "stopped; NeuronCores idle"; exit 0; }
  sleep 2
done
log "NeuronCores still in use; check neuron-ls (do not kill processes you did not start)"
exit 1
