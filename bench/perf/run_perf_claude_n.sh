#!/usr/bin/env bash
# Run bench/perf/run_perf_claude.sh N times, saving each result_claude-code.json
# under $BENCH_ROOT/_samples/<label>_s<i>.json.
#
# Mirrors run_perf_v2_n.sh so the claude-code side of the head-to-head can be
# re-baselined at n>1 (the single-sample claude-code baseline has too much
# variance for fair comparison).
#
# Honours the same env knobs as run_perf_claude.sh (CLAUDE_BUDGET_USD,
# CLAUDE_MODEL, CLAUDE_BIN).
#
# Usage:
#   bash bench/perf/run_perf_claude_n.sh <N> <label>
#
# Example:
#   # claude-code sonnet milestone n=3 (~$15)
#   bash bench/perf/run_perf_claude_n.sh 3 cc_sonnet_milestone

set -euo pipefail
N="${1:?usage: run_perf_claude_n.sh <N> <label>}"
LABEL="${2:?usage: run_perf_claude_n.sh <N> <label>}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-perf}"
OUT="$BENCH_ROOT/_samples"
mkdir -p "$OUT"

for i in $(seq 1 "$N"); do
  echo "==== $LABEL sample $i / $N ===="
  bash "$(dirname "$0")/run_perf_claude.sh" 2>&1 | tail -25
  # run_perf_claude.sh writes result_claude.json (not result_claude-code.json)
  cp "$BENCH_ROOT/result_claude.json" "$OUT/${LABEL}_s${i}.json"
  echo "Saved $OUT/${LABEL}_s${i}.json"
done

echo "==== summary $LABEL ===="
for f in "$OUT/${LABEL}_s"*.json; do
  python3 -c "import json,sys; d=json.load(open('$f')); print('${f##*/}', 'speedup=%.2fx cost=\$%.2f wall=%ss in=%s out=%s exit=%s tests_clean=%s' % (d['speedup_over_baseline'], d['cost_usd'], d['wall_seconds'], d['input_tokens'], d['output_tokens'], d['agent_exit'], d['tests_clean']))"
done
