#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Functional smoke test of the running server: chat (English / Chinese / math), completions, streaming, thinking,
# tool calling, structured output, 4 concurrent requests, and image / video / audio on the 1M profiles.
# Usage: bash scripts/smoke_test.sh [--no-mm]        (--no-mm for the text-only 8k profiles)
# Results: $MIMO_ROOT/results/smoke.{json,md}. Exit code: 0 all pass; 1 a case failed; 2 server unreachable.
set -uo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$MIMO_ROOT/results"
exec python3 "$REPO_DIR/scripts/api_tests/smoke.py" --out "$MIMO_ROOT/results/smoke" "$@"
