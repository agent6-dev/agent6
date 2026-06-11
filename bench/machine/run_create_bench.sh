#!/usr/bin/env bash
# Bench `agent6 machine create`: attempts, spend, wall time, and whether the
# produced bundle passes `machine check` + `machine test`, over N runs of a
# fixed representative task (a poll -> classify -> record paper-trader).
#
# Each run gets a fresh git repo under $BENCH_ROOT. The model comes from the
# operator's normal agent6 config (set it with `agent6 model worker ...`),
# so this measures exactly what a user gets.
#
# Usage:
#   bash bench/machine/run_create_bench.sh            # 3 runs
#   RUNS=5 BENCH_ROOT=/tmp/agent6-create-bench bash bench/machine/run_create_bench.sh
#
# Results land in $BENCH_ROOT/results.jsonl (one line per run) and a summary
# is printed at the end.

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-create-bench}"
RUNS="${RUNS:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
TASK="${AGENT6_CREATE_TASK:-poll a public price feed every 10 minutes; when an \
LLM judges the latest headline for that asset as clearly positive, record a \
paper-trade buy with the current price so performance can be simulated later}"

mkdir -p "$BENCH_ROOT"
RESULTS="$BENCH_ROOT/results.jsonl"
: > "$RESULTS"

for i in $(seq 1 "$RUNS"); do
  dir="$BENCH_ROOT/run_$i"
  rm -rf "$dir"; mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" -c user.email=bench@bench -c user.name=bench commit -q --allow-empty -m init

  log="$dir/create.log"
  start=$(date +%s)
  set +e
  (cd "$dir" && "$REPO/.venv/bin/agent6" machine create "$TASK" \
      --max-attempts "$MAX_ATTEMPTS") >"$log" 2>&1
  code=$?
  set -e
  secs=$(( $(date +%s) - start ))

  attempts=$(grep -cE "machine create: attempt" "$log" || true)
  spent=$(grep -oE "spent ~\\\$[0-9.]+" "$log" | grep -oE "[0-9.]+" | tail -1)
  machine_file=$(ls "$dir"/*.asm.toml 2>/dev/null | head -1)
  check=fail; test_ok=fail
  if [ -n "$machine_file" ]; then
    (cd "$dir" && "$REPO/.venv/bin/agent6" machine check "$(basename "$machine_file")" >/dev/null 2>&1) && check=ok
    (cd "$dir" && "$REPO/.venv/bin/agent6" machine test "$(basename "$machine_file")" >/dev/null 2>&1) && test_ok=ok
  fi
  scripts=$(ls "$dir/scripts" 2>/dev/null | wc -l)
  tests=$(ls "$dir/scripts" 2>/dev/null | grep -c "_test.py" || true)

  printf '{"run":%d,"exit":%d,"attempts":%s,"spent_usd":%s,"wall_secs":%d,"scripts":%d,"mock_tests":%d,"check":"%s","test":"%s"}\n' \
    "$i" "$code" "${attempts:-0}" "${spent:-0}" "$secs" "$scripts" "$tests" "$check" "$test_ok" \
    | tee -a "$RESULTS"
done

echo
echo "== summary ($RUNS runs) =="
python3 - "$RESULTS" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
ok = [r for r in rows if r["exit"] == 0 and r["check"] == "ok" and r["test"] == "ok"]
print(f"success: {len(ok)}/{len(rows)}")
if rows:
    print(f"attempts: {[r['attempts'] for r in rows]}")
    print(f"spent_usd: {[r['spent_usd'] for r in rows]}  total=${sum(r['spent_usd'] for r in rows):.2f}")
    print(f"wall_secs: {[r['wall_secs'] for r in rows]}")
PY
