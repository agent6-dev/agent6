#!/usr/bin/env bash
# Run bench/perf/run_perf.sh N times, saving each result_agent6.json under
# a labelled name. Usage: run_perf_n.sh <N> <label>
set -euo pipefail
N="${1:-3}"
LABEL="${2:-baseline}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-perf}"
OUT="$BENCH_ROOT/_samples"
mkdir -p "$OUT"
for i in $(seq 1 "$N"); do
  echo "==== $LABEL sample $i / $N ===="
  bash "$(dirname "$0")/run_perf.sh" 2>&1 | tail -25
  cp "$BENCH_ROOT/result_agent6.json" "$OUT/${LABEL}_s${i}.json"
  echo "Saved $OUT/${LABEL}_s${i}.json"
done
echo "==== summary $LABEL ===="
for f in "$OUT/${LABEL}_s"*.json; do
  python3 -c "import json,sys; d=json.load(open('$f')); print('${f##*/}', 'speedup=%.2fx cost=\$%.2f wall=%ss in=%s out=%s exit=%s tests_clean=%s' % (d['speedup_over_baseline'], d['cost_usd'], d['wall_seconds'], d['input_tokens'], d['output_tokens'], d['agent_exit'], d['tests_clean']))"
done
