#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark suite (items 1-6) for a RUNNING MiMo-V2.6-Flash vLLM server on trn2.48xlarge.
# The same suite was run on the earlier PyTorch/XLA-backend deployment (Neuron SDK 2.32 / vllm-neuron 0.24), so the
# numbers in results/bench_summary.md are directly comparable.
# It talks to the server only over HTTP (OpenAI API + /metrics) and uses `vllm bench serve`, the tutorial's
# client scripts (scripts/api_tests/*.py) and generic tools. No SDK-internal APIs.
#
# Assumed server config (see serve_command.txt for the full command):
#   a 1M profile of config/serve_args.json (scripts/serve.sh --profile 1m or 1m_c4): --served-model-name
#   mimo-v2.6-flash, --max-model-len 1048576, max-num-seqs >= 4 (max concurrency used: 4), prefix caching,
#   image / video / audio enabled, reasoning parser mimo.
#   Start the suite on a freshly started server if possible (empty prefix cache), after it answers /v1/models.
#
# Items (ITEMS selects a subset; the same OUT can be reused, files are overwritten per run):
#   1 short     vllm bench serve random 512/128, concurrency 1 (8 prompts) and 4 (16 prompts)
#   2 medium    random 8192/512, concurrency 1 (4 prompts) and 4 (8 prompts)
#   3 long      random 32768/256 c1 (4 prompts); random 131072/128 c1 (3 prompts)
#   4 ultra     needle.py prompt (seed 0) + prefix_cache_test.py: 30K and ~940K prompt, cold then identical
#               repeat (prefix-cache hit) then one follow-up turn
#   5 mm        ask.py (streamed, thinking off, greedy): number_739.png, digits_1234.mp4, audio/jfk.wav,
#               3 runs each, TTFT + total time
#   6 accuracy  task_eval.py GSM8K + MMLU, 200 questions each, 0-shot, greedy, 4 workers
#
# Seeds: every measured `vllm bench serve` run has its own fixed seed (0..5) and the warm-up uses seed 100.
# With one shared seed, RandomDataset regenerates the same token sequences (e.g. the 32K prompt #0 is a prefix
# of the 128K prompt #0), which the prefix cache would hit. Distinct seeds were checked: no shared prefix.
#
# Usage (on the instance):
#   OUT=$MIMO_ROOT/bench/<name> bash scripts/run_bench.sh                     # all items, about 15-25 min
#   ITEMS="1 2" OUT=... bash scripts/run_bench.sh                             # subset
#   SMOKE=1 OUT=/tmp/smoke bash scripts/run_bench.sh                          # tiny version of every item
# Env (defaults from scripts/env.sh): BASE_URL (http://127.0.0.1:8000), SERVED_MODEL (mimo-v2.6-flash), MODEL_DIR
#   (tokenizer dir, default $HF_SNAPSHOT), VENV (venv to activate), REPO (this repository), MEDIA_DIR (test media,
#   default assets/), EVAL_DATA (dir with gsm8k_test_200_seed0.jsonl / mmlu_test_200_seed0.jsonl, default
#   data/task_eval), PYDEPS (prepended to PYTHONPATH if set and it exists), ITEMS ("1 2 3 4 5 6"), SMOKE (0).
# Output: OUT/master.log, OUT/summary/{status.tsv,summary.md,summary.json}, OUT/item<N>/..., OUT/metrics/*.prom
#   (/metrics before/after every perf run), OUT/snapshots/ (neuron-ls, per-core HBM, free -g).
set -u
set -o pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/env.sh"

BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
SERVED_MODEL=${SERVED_MODEL:-mimo-v2.6-flash}
MODEL_DIR=${MODEL_DIR:-$HF_SNAPSHOT}
VENV=${VENV:?}
REPO=${REPO:-$REPO_DIR}
MEDIA_DIR=${MEDIA_DIR:-$REPO_DIR/assets}
EVAL_DATA=${EVAL_DATA:-$REPO_DIR/data/task_eval}
PYDEPS=${PYDEPS:-}
ITEMS=${ITEMS:-1 2 3 4 5 6}
SMOKE=${SMOKE:-0}
OUT=${OUT:?set OUT to the output directory}

mkdir -p "$OUT"/{summary,metrics,snapshots} || exit 2
OUT=$(cd "$OUT" && pwd)
LOG=$OUT/master.log
STATUS=$OUT/summary/status.tsv
[ -f "$STATUS" ] || printf 'item\tstep\trc\tstart_utc\tend_utc\tseconds\n' > "$STATUS"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "cannot activate $VENV" >&2; exit 2; }
[ -n "$PYDEPS" ] && [ -d "$PYDEPS" ] && export PYTHONPATH="$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
export MIMO_API_BASE=$BASE_URL MIMO_API_MODEL=$SERVED_MODEL MODEL_DIR
PY=python3
API=$REPO/scripts/api_tests

# run_step <item> <step> <logfile> <cmd...>: run, log rc + duration to status.tsv; never aborts the suite.
run_step() {
  local item=$1 step=$2 lf=$3; shift 3
  local t0 s0 rc
  t0=$(date -u +%FT%TZ); s0=$(date +%s)
  log "item $item: $step start (log ${lf#"$OUT"/})"
  "$@" > "$lf" 2>&1; rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$item" "$step" "$rc" "$t0" "$(date -u +%FT%TZ)" "$(( $(date +%s) - s0 ))" >> "$STATUS"
  log "item $item: $step rc=$rc ($(( $(date +%s) - s0 )) s)"
  return $rc
}

metrics_snap() {  # metrics_snap <name>: prefix-cache / mm-cache / request counters from /metrics
  curl -s -m 20 "$BASE_URL/metrics" \
    | grep -E '^vllm:(prefix_cache_(queries|hits)|gpu_prefix_cache_(queries|hits)|prompt_tokens_cached|mm_cache_(queries|hits)|num_preemptions|request_success|prompt_tokens|generation_tokens)' \
    > "$OUT/metrics/$1.prom" || log "metrics_snap $1: /metrics not readable"
}

snapshot() {  # snapshot <tag>: neuron-ls (instance id removed), per-core HBM from one neuron-monitor sample, free -g
  local d=$OUT/snapshots
  date -u +%FT%TZ > "$d/$1_time.txt"
  free -g > "$d/$1_free.txt" 2>&1
  if command -v neuron-ls >/dev/null; then
    neuron-ls 2>&1 | grep -viE 'instance-id|instance_id' > "$d/$1_neuron-ls.txt"
  else
    echo "neuron-ls not found" > "$d/$1_neuron-ls.txt"
  fi
  if command -v neuron-monitor >/dev/null; then
    timeout 60 neuron-monitor 2>/dev/null | head -1 | "$PY" -c '
import json, sys
line = sys.stdin.readline()
if not line.strip():
    print("neuron-monitor: no sample"); sys.exit(0)
d = json.loads(line)
G = 2 ** 30
rows = []
for r in d.get("neuron_runtime_data") or []:
    mu = ((r.get("report") or {}).get("memory_used") or {}).get("neuron_runtime_used_bytes") or {}
    rows.append((r.get("neuron_runtime_tag"), r.get("pid"), mu.get("neuron_device", 0) / G, mu.get("host", 0) / G))
print("per neuron runtime (one per vLLM worker / logical NeuronCore): device HBM GiB, host GiB")
for tag, pid, dev, host in rows:
    print(f"  tag={tag} pid={pid} device={dev:.2f} host={host:.2f}")
if rows:
    dv = [x[2] for x in rows]
    print(f"runtimes={len(rows)} device_GiB min={min(dv):.2f} max={max(dv):.2f} mean={sum(dv)/len(dv):.2f} total={sum(dv):.1f}")
mi = (d.get("system_data") or {}).get("memory_info") or {}
if mi:
    used, total = mi.get("memory_used_bytes", 0) / G, mi.get("memory_total_bytes", 0) / G
    print(f"host memory used {used:.0f} / {total:.0f} GiB (neuron-monitor; includes page cache)")
' > "$d/$1_hbm.txt" 2>&1
  else
    echo "neuron-monitor not found" > "$d/$1_hbm.txt"
  fi
  command -v neuron-top >/dev/null || echo "(neuron-top not installed)" >> "$d/$1_hbm.txt"
  log "snapshot $1: $(tail -2 "$d/$1_hbm.txt" | head -1)"
}

# Optional `vllm bench serve` flags: pin "no extra requests before the measured run" when the flag exists
# (an implicit test / warm-up request reuses prompt #0 and would make it a prefix-cache hit).
BENCH_HELP=$(vllm bench serve --help=all 2>/dev/null)
BENCH_EXTRA=()
grep -q -- '--num-warmups' <<<"$BENCH_HELP" && BENCH_EXTRA+=(--num-warmups 0)
grep -q -- '--ready-check-timeout-sec' <<<"$BENCH_HELP" && BENCH_EXTRA+=(--ready-check-timeout-sec 0)

# bench <item> <name> <input_len> <output_len> <concurrency> <num_prompts> <seed>
bench() {
  local item=$1 name=$2 inp=$3 outl=$4 conc=$5 n=$6 seed=$7 d=$OUT/item$1
  mkdir -p "$d"
  metrics_snap "${name}_before"
  run_step "$item" "$name" "$d/$name.log" \
    vllm bench serve --backend vllm --base-url "$BASE_URL" --endpoint /v1/completions \
      --model "$MODEL_DIR" --served-model-name "$SERVED_MODEL" --tokenizer "$MODEL_DIR" \
      --dataset-name random --random-input-len "$inp" --random-output-len "$outl" --ignore-eos \
      --max-concurrency "$conc" --num-prompts "$n" --seed "$seed" "${BENCH_EXTRA[@]}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --save-detailed --result-dir "$d" --result-filename "$name.json"
  metrics_snap "${name}_after"
}

warmup() {  # 2 short requests through the same path as the perf runs; seed 100 / 256 tokens, never measured
  bench 0 warmup 256 16 1 2 100
}

item1() {
  if [ "$SMOKE" = 1 ]; then bench 1 smoke_256x16_c1 256 16 1 2 900; return; fi
  bench 1 short_512x128_c1 512 128 1 8 0
  bench 1 short_512x128_c4 512 128 4 16 1
}
item2() {
  [ "$SMOKE" = 1 ] && { log "item 2: skipped in SMOKE mode (same code path as item 1)"; return; }
  bench 2 medium_8192x512_c1 8192 512 1 4 2
  bench 2 medium_8192x512_c4 8192 512 4 8 3
}
item3() {
  [ "$SMOKE" = 1 ] && { log "item 3: skipped in SMOKE mode (same code path as item 1)"; return; }
  bench 3 long_32768x256_c1 32768 256 1 4 4
  bench 3 long_131072x128_c1 131072 128 1 3 5
}

# Item 4: build a needle prompt with needle.py (deterministic from tokenizer + seed 0; same prompt ids as the
# tutorial's prefix-cache runs), then prefix_cache_test.py sends it cold, identical repeat, follow-up turn.
pc_case() {  # pc_case <name> <length> <depths> <id>
  local name=$1 len=$2 depths=$3 id=$4 d=$OUT/item4
  mkdir -p "$d/prompts"
  run_step 4 "${name}_build" "$d/${name}_build.log" \
    "$PY" "$API/needle.py" --dry-run --ctx 1048576 --lengths "$len" --depths "$depths" --seed 0 \
      --tokenizer "$MODEL_DIR" --out "$d/prompts/$name" || return 1
  metrics_snap "${name}_before"
  run_step 4 "$name" "$d/$name.log" \
    "$PY" "$API/prefix_cache_test.py" --prompts "$d/prompts/${name}_prompts.json" --ids "$id" \
      --max-tokens 64 --timeout 3600 --out "$d/$name.json"
  metrics_snap "${name}_after"
}
item4() {
  if [ "$SMOKE" = 1 ]; then pc_case smoke_pc_2k 2000 0.5 N1@2000@50; return; fi
  pc_case pc_30k 30000 0.1,0.5 N2@30000@50
  pc_case pc_940k 940000 0.1 N1@940000@10
  snapshot item4_after
}

# Item 5: ask.py (base64 data URL, streamed, thinking off, temperature 0); prints first-token and total time.
mm_run() {  # mm_run <kind> <file> <question> <runs>
  local kind=$1 f=$2 q=$3 runs=$4 d=$OUT/item5 i name
  mkdir -p "$d"
  for i in $(seq 1 "$runs"); do
    name=${kind}_$(basename "${f%.*}")_run$i
    metrics_snap "${name}_before"
    run_step 5 "$name" "$d/$name.txt" \
      "$PY" "$API/ask.py" --no-think --temperature 0 --max-tokens 64 --"$kind" "$MEDIA_DIR/$f" \
        --url "$BASE_URL/v1/chat/completions" --model "$SERVED_MODEL" "$q"
    metrics_snap "${name}_after"
  done
}
item5() {
  local runs=3 img=number_739.png aud=audio/jfk.wav
  [ "$SMOKE" = 1 ] && { runs=1; img=red_square.png; aud=audio/beeps_3.wav; }
  mm_run image "$img" "What number is written in the image? Answer with the number only." "$runs"
  mm_run video digits_1234.mp4 "Which digits appear in the video, in order? Answer with the digits separated by commas." "$runs"
  mm_run audio "$aud" "Transcribe this audio exactly." "$runs"
}

# Item 6: task_eval.py reads <its dir>/../../data/task_eval; run it from a symlink tree that points that path at
# EVAL_DATA, so a different data directory can be used without changing the repository.
item6() {
  local d=$OUT/item6 shim n=200 t
  [ "$SMOKE" = 1 ] && n=2
  shim=$d/_shim
  mkdir -p "$shim/scripts/api_tests" "$shim/data"
  ln -sf "$API/task_eval.py" "$shim/scripts/api_tests/task_eval.py"
  ln -sf "$API/common.py" "$shim/scripts/api_tests/common.py"
  ln -sfn "$EVAL_DATA" "$shim/data/task_eval"
  for t in gsm8k mmlu; do
    [ -f "$EVAL_DATA/${t}_test_200_seed0.jsonl" ] || { log "item 6: missing $EVAL_DATA/${t}_test_200_seed0.jsonl"; continue; }
    metrics_snap "${t}_before"
    run_step 6 "$t" "$d/$t.log" "$PY" "$shim/scripts/api_tests/task_eval.py" --task "$t" --n "$n" --workers 4 \
      --out "$d/$t"
    metrics_snap "${t}_after"
  done
}

summarize() {
  "$PY" - "$OUT" <<'EOF' > "$OUT/summary/summary.md"
import glob, json, os, re, statistics, sys
out = sys.argv[1]
S = {"bench": {}, "prefix_cache": {}, "mm": {}, "accuracy": {}, "metrics_delta": {}}

def prom(path):
    tot = {}
    if not os.path.exists(path):
        return None
    for ln in open(path):
        m = re.match(r"^([a-zA-Z_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", ln.strip())
        if m:
            tot[m.group(1)] = tot.get(m.group(1), 0.0) + float(m.group(2))
    return tot

for b in sorted(glob.glob(os.path.join(out, "metrics", "*_before.prom"))):
    name = os.path.basename(b)[:-len("_before.prom")]
    p0, p1 = prom(b), prom(os.path.join(out, "metrics", name + "_after.prom"))
    if p0 is None or p1 is None:
        continue
    dd = {k: p1[k] - p0.get(k, 0.0) for k in p1}
    q = dd.get("vllm:prefix_cache_queries_total", dd.get("vllm:gpu_prefix_cache_queries_total"))
    h = dd.get("vllm:prefix_cache_hits_total", dd.get("vllm:gpu_prefix_cache_hits_total"))
    S["metrics_delta"][name] = {"prefix_queries": q, "prefix_hits": h,
                                "prefix_hit_rate": round(h / q, 4) if q else None,
                                "mm_cache_queries": dd.get("vllm:mm_cache_queries_total"),
                                "mm_cache_hits": dd.get("vllm:mm_cache_hits_total"),
                                "preemptions": dd.get("vllm:num_preemptions_total")}

KEYS = ["completed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
        "p90_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms", "total_input_tokens", "total_output_tokens"]
for f in sorted(glob.glob(os.path.join(out, "item[0-3]", "*.json"))):
    d = json.load(open(f))
    S["bench"][os.path.basename(f)[:-5]] = {k: d.get(k) for k in KEYS}

for f in sorted(glob.glob(os.path.join(out, "item4", "*.json"))):
    d = json.load(open(f))
    for c in d.get("cases", []):
        S["prefix_cache"][os.path.basename(f)[:-5]] = {"id": c["id"], "prompt_tokens": c["prompt_tokens"],
            "ttft_s": c["ttft_s"], "cached_tokens": c["cached_tokens"], "repeat_identical": c["repeat_identical"],
            "elapsed_s": {k: r.get("elapsed_s") for k, r in c["records"].items()},
            "answer_cold": c["records"]["cold"]["text"][:80]}

pat = re.compile(r"\[首字时间\]\s*([0-9.]+).*\[总耗时\]\s*([0-9.]+)")
use = re.compile(r"\[用量\] 输入 (\d+) token，输出 (\d+) token")
groups = {}
for f in sorted(glob.glob(os.path.join(out, "item5", "*_run*.txt"))):
    txt = open(f, errors="replace").read()
    m, u = pat.search(txt), use.search(txt)
    key = re.sub(r"_run\d+$", "", os.path.basename(f)[:-4])
    ans = txt.split("========== 回答 ==========")[-1].split("[结束原因]")[0].split("[用量]")[0].strip()[:120] if "回答" in txt else None
    groups.setdefault(key, []).append({"run": os.path.basename(f)[:-4], "ttft_s": float(m.group(1)) if m else None,
                                       "total_s": float(m.group(2)) if m else None,
                                       "prompt_tokens": int(u.group(1)) if u else None,
                                       "completion_tokens": int(u.group(2)) if u else None, "answer": ans})
for k, runs in groups.items():
    tt = [r["ttft_s"] for r in runs if r["ttft_s"] is not None]
    tl = [r["total_s"] for r in runs if r["total_s"] is not None]
    S["mm"][k] = {"runs": runs, "median_ttft_s": statistics.median(tt) if tt else None,
                  "median_total_s": statistics.median(tl) if tl else None}

for t in ("gsm8k", "mmlu"):
    f = os.path.join(out, "item6", t + ".json")
    if os.path.exists(f):
        S["accuracy"][t] = json.load(open(f))["summary"]

json.dump(S, open(os.path.join(out, "summary", "summary.json"), "w"), indent=1, ensure_ascii=False)
f2 = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
print("# benchmark summary\n")
print("| run | completed | mean TTFT ms | p50 | p90 | p99 | mean TPOT ms | p50 | p99 | mean ITL ms | mean E2E ms | out tok/s | prefix hit rate |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, v in S["bench"].items():
    md = S["metrics_delta"].get(k, {})
    print(f"| {k} | {f2(v['completed'])} | {f2(v['mean_ttft_ms'])} | {f2(v['median_ttft_ms'])} | {f2(v['p90_ttft_ms'])} | "
          f"{f2(v['p99_ttft_ms'])} | {f2(v['mean_tpot_ms'])} | {f2(v['median_tpot_ms'])} | {f2(v['p99_tpot_ms'])} | "
          f"{f2(v['mean_itl_ms'])} | {f2(v['mean_e2el_ms'])} | {f2(v['output_throughput'])} | {f2(md.get('prefix_hit_rate'))} |")
print("\n| prefix-cache case | prompt tokens | TTFT cold s | TTFT repeat s | TTFT follow-up s | cached tokens (cold/repeat/follow-up) |")
print("|---|---|---|---|---|---|")
for k, v in S["prefix_cache"].items():
    t, c = v["ttft_s"], v["cached_tokens"]
    print(f"| {k} {v['id']} | {v['prompt_tokens']} | {t['cold']} | {t['repeat']} | {t['follow_up']} | "
          f"{c['cold']}/{c['repeat']}/{c['follow_up']} |")
print("\n| mm | runs TTFT s | runs total s | median TTFT s | median total s | answer (run 1) |")
print("|---|---|---|---|---|---|")
for k, v in S["mm"].items():
    print(f"| {k} | {[r['ttft_s'] for r in v['runs']]} | {[r['total_s'] for r in v['runs']]} | "
          f"{f2(v['median_ttft_s'])} | {f2(v['median_total_s'])} | {(v['runs'][0]['answer'] or '').replace('|', '/')[:60]} |")
print("\n| task | n | correct | accuracy | unparsed | wall s |")
print("|---|---|---|---|---|---|")
for k, v in S["accuracy"].items():
    print(f"| {k} | {v['n']} | {v['correct']} | {v['accuracy']} | {v['unparsed']} | {v['wall_s']} |")
EOF
}

log "run_bench.sh start: ITEMS='$ITEMS' SMOKE=$SMOKE OUT=$OUT BASE_URL=$BASE_URL MODEL_DIR=$MODEL_DIR REPO=$REPO"
log "vllm $("$PY" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null) bench extra flags: ${BENCH_EXTRA[*]:-<none>}"
if ! curl -sf -m 20 "$BASE_URL/v1/models" > "$OUT/summary/models.json"; then
  log "server not reachable at $BASE_URL/v1/models"; exit 2
fi
snapshot start
warmup
for it in $ITEMS; do
  case $it in
    1) item1 ;; 2) item2 ;; 3) item3 ;; 4) item4 ;; 5) item5 ;; 6) item6 ;;
    *) log "unknown item '$it' (valid: 1-6)" ;;
  esac
done
summarize || log "summarize failed"
curl -sf -m 20 "$BASE_URL/v1/models" > /dev/null && log "server still answering /v1/models" || log "server NOT answering /v1/models"
log "run_bench.sh done; failed steps: $(awk -F'\t' 'NR>1 && $3!=0 {printf "%s/%s(rc=%s) ", $1, $2, $3}' "$STATUS")"
