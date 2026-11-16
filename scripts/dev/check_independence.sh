#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Self-containment check: copy this repository alone (without .git) into a temporary directory and verify that it
# does not depend on any other local repository or on the authors' development notes.
#   1. no references to development documents, internal decision / task numbers or development paths
#      (patterns below). patches/ is excluded: the vllm-neuron source code carries its own code comments.
#   2. no git commit hashes (7-40 hex digits) outside patches/ and requirements/ (pinned package versions such as
#      neuronx-cc==2.28.7631.0+8f7b0cf5 carry a build hash), except the pinned upstream references a reader
#      needs (ALLOWED_HEX: Hugging Face model revision, vllm-neuron base commit and the expected tree hashes);
#      lines that carry a SHA-256 value and the SHA256SUMS file are not checked
#   3. documentation paths resolve (scripts/dev/check_doc_paths.py)
#   4. every .py passes python3 -m py_compile; 5. every .sh passes bash -n
#   6. generic sensitive-pattern scan is clean (scripts/dev/scan_sensitive.sh)
# Usage: bash scripts/dev/check_independence.sh        Exit code: 0 all pass, 1 otherwise.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SELF=scripts/dev/$(basename "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir "$TMP/repo"
(cd "$ROOT" && tar --exclude=./.git --exclude='*/__pycache__' -cf - .) | (cd "$TMP/repo" && tar -xf -)
cd "$TMP/repo" || exit 1
echo "isolated copy: $(find . -type f | wc -l | tr -d ' ') files"

# Internal decision numbers (D-123), task labels (T12, T54a), rule labels, development-only file names and paths.
PATTERN='(^|[^A-Za-z0-9_-])D-[0-9]{2,3}([^0-9]|$)|(^|[^A-Za-z0-9_])T[0-9]{2}[a-z]?([^A-Za-z0-9_]|$)'
PATTERN+='|规则 ?[A-Z]{1,2}([^A-Za-z]|$)|decisions\.md|tasks\.json|agent_artifacts|apl-development|trn-study|mimo2\.6-adaption'
PATTERN+='|mimo-work/|/home/ubuntu/work|rehearsal/2026|dev-reference|编排者|orchestrator|wf_[0-9a-f]{6}'
fail=0
echo "== 1. development references"
hits=$(grep -rnIE --exclude-dir=.git --exclude-dir=patches -- "$PATTERN" . | grep -v "^\./$SELF:" || true)
n=$(printf '%s' "$hits" | grep -c . || true)
[ "$n" -gt 0 ] && { printf '%s\n' "$hits"; fail=1; }
echo "hits=$n"

ALLOWED_HEX='b12f5573350bb1704513fcf119fca48e498b0423 eb6cb360f7cff6082e22a9437d8ce0ba3891304a 2d69ea6bfdd7285d330daa2cfe70ab1aa7f1eb40 3910a640fb5a87c0604eebc22555dc5946222639'
echo "== 2. commit hashes"
hex=$(grep -rnIoE --exclude-dir=.git --exclude-dir=patches --exclude-dir=requirements --exclude=SHA256SUMS -- '\b[0-9a-f]{7,40}\b' . | grep -v "^\./$SELF:" \
  | awk -F: -v allow="$ALLOWED_HEX" '{t=$NF; ok=0; n=split(allow, a, " "); for (i = 1; i <= n; i++) if (index(a[i], t) == 1) ok=1}
       t ~ /[a-f]/ && t !~ /^[0-9]+e$/ && !ok {print $1 ":" $2 ":" t}' \
  | while IFS=: read -r f l t; do sed -n "${l}p" "$f" | grep -qiE 'sha-?256' || echo "$f:$l: $t"; done)
nh=$(printf '%s' "$hex" | grep -c . || true)
[ "$nh" -gt 0 ] && { printf '%s\n' "$hex"; fail=1; }
echo "hash_hits=$nh"

echo "== 3. documentation paths"
python3 scripts/dev/check_doc_paths.py || fail=1

echo "== 4. py_compile"
np=0; fp=0
while IFS= read -r -d '' f; do
  np=$((np + 1)); python3 -m py_compile "$f" 2>/dev/null || { fp=$((fp + 1)); echo "FAIL $f"; }
done < <(find . -name '*.py' -type f -print0)
echo "py files=$np fail=$fp"; [ "$fp" -eq 0 ] || fail=1

echo "== 5. bash -n"
ns=0; fs=0
while IFS= read -r -d '' f; do
  ns=$((ns + 1)); bash -n "$f" 2>/dev/null || { fs=$((fs + 1)); echo "FAIL $f"; }
done < <(find . -name '*.sh' -type f -print0)
echo "sh files=$ns fail=$fs"; [ "$fs" -eq 0 ] || fail=1

echo "== 6. sensitive patterns"
find . -name __pycache__ -type d -prune -exec rm -rf {} +
bash scripts/dev/scan_sensitive.sh . || fail=1

echo "== result: $([ "$fail" -eq 0 ] && echo PASS || echo FAIL)"
exit "$fail"
