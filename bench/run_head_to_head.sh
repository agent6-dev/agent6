#!/usr/bin/env bash
# Head-to-head benchmark: agent6 vs claude-code on the same 8 synthetic tasks.
#
# Requires baseline fixtures from a prior `bash bench/run_bench.sh` run (we
# read the root commit of each task repo and reset to it). Spawns claude-code
# in non-interactive --print mode with --dangerously-skip-permissions so it
# can edit files inside the fixture, then runs the same verify command.
#
# Each claude-code run is capped via --max-budget-usd so a runaway agent
# cannot blow the budget. The harness sums per-task costs and aborts if the
# cumulative spend exceeds CLAUDE_TOTAL_BUDGET_USD (default 5.00).
#
# Usage:
#   BENCH_SRC=/tmp/agent6-bench-baseline bash bench/run_head_to_head.sh
#
# Output: $BENCH_ROOT/head_to_head.md plus per-task result.json.

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BENCH_SRC="${BENCH_SRC:-/tmp/agent6-bench-baseline}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-bench-claude}"
PER_TASK_BUDGET_USD="${PER_TASK_BUDGET_USD:-0.50}"
CLAUDE_TOTAL_BUDGET_USD="${CLAUDE_TOTAL_BUDGET_USD:-5.00}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-5}"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.npm-global/bin/claude}"

cd "$REPO"
[ -x "$CLAUDE_BIN" ] || { echo "claude-code not found at $CLAUDE_BIN" >&2; exit 1; }
[ -d "$BENCH_SRC" ] || { echo "baseline fixtures missing at $BENCH_SRC; run bench/run_bench.sh first" >&2; exit 1; }

mkdir -p "$BENCH_ROOT/logs"

total_cost=0
total_wall=0
pass=0
total=0

reset_fixture() {
  local src="$1" dst="$2"
  rm -rf "$dst"
  cp -a "$src" "$dst"
  ( cd "$dst" \
    && git reset -q --hard "$(git rev-list --max-parents=0 HEAD | tail -1)" \
    && git clean -qfdx -e final_pytest.txt -e result.json )
  # remove agent6-specific artifacts before claude runs
  rm -rf "$dst/.agent6" "$dst/.agent6-self-review"
}

run_one() {
  local name="$1"
  local src="$BENCH_SRC/$name"
  local dir="$BENCH_ROOT/$name"
  [ -d "$src" ] || { echo "skip $name (source missing)"; return; }
  echo
  echo "================================================================"
  echo "TASK: $name (claude-code)"
  echo "================================================================"
  reset_fixture "$src" "$dir"
  local task_text; task_text=$(cat "$dir/TASK.md")
  local log="$BENCH_ROOT/logs/${name}.json"
  local stderr_log="$BENCH_ROOT/logs/${name}.stderr"

  local start_ns end_ns wall_s
  start_ns=$(date +%s%N)
  set +e
  ( cd "$dir" && "$CLAUDE_BIN" \
      --print \
      --dangerously-skip-permissions \
      --model "$CLAUDE_MODEL" \
      --output-format json \
      --max-budget-usd "$PER_TASK_BUDGET_USD" \
      --bare \
      "$task_text" \
  ) > "$log" 2> "$stderr_log"
  local exit_code=$?
  set -e
  end_ns=$(date +%s%N)
  wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.1f", (e-s)/1e9}')

  # verify
  set +e
  ( cd "$dir" && python3 -m unittest -v ) > "$dir/final_pytest.txt" 2>&1
  local verify=$?
  set -e

  # parse claude json
  local cost duration_api turns
  cost=$(python3 -c "import json,sys; d=json.load(open('$log')); print(d.get('total_cost_usd', d.get('cost_usd', 0)))" 2>/dev/null || echo 0)
  duration_api=$(python3 -c "import json,sys; d=json.load(open('$log')); print(d.get('duration_ms', 0)/1000)" 2>/dev/null || echo 0)
  turns=$(python3 -c "import json,sys; d=json.load(open('$log')); print(d.get('num_turns', d.get('turns', 0)))" 2>/dev/null || echo 0)

  cat > "$dir/result.json" <<EOF
{
  "task": "$name",
  "runner": "claude-code",
  "exit_code": $exit_code,
  "wall_seconds": $wall_s,
  "verify_pass": $([ $verify -eq 0 ] && echo true || echo false),
  "cost_usd": $cost,
  "duration_api_s": $duration_api,
  "turns": $turns
}
EOF
  echo "  exit=$exit_code  verify=$([ $verify -eq 0 ] && echo PASS || echo FAIL)  wall=${wall_s}s  cost=\$${cost}  turns=${turns}"

  total_cost=$(awk -v a="$total_cost" -v b="$cost" 'BEGIN{printf "%.4f", a+b}')
  total_wall=$(awk -v a="$total_wall" -v b="$wall_s" 'BEGIN{printf "%.1f", a+b}')
  total=$((total+1))
  [ $verify -eq 0 ] && pass=$((pass+1))

  # safety cap
  local over; over=$(awk -v a="$total_cost" -v b="$CLAUDE_TOTAL_BUDGET_USD" 'BEGIN{print (a>b)?1:0}')
  if [ "$over" = "1" ]; then
    echo "TOTAL COST $total_cost EXCEEDED CAP $CLAUDE_TOTAL_BUDGET_USD — aborting" >&2
    exit 2
  fi
}

for name in $(ls "$BENCH_SRC" | grep -E '^[0-9]+-' | sort); do
  run_one "$name"
done

# emit summary markdown
{
  echo "# Head-to-head: agent6 vs claude-code"
  echo
  echo "Model: $CLAUDE_MODEL. Per-task budget cap: \$$PER_TASK_BUDGET_USD. Baseline fixtures: $BENCH_SRC."
  echo
  printf "| # | Task | claude verify | claude wall | claude cost | claude turns |\n"
  printf "|---|------|---------------|-------------|-------------|--------------|\n"
  for d in "$BENCH_ROOT"/[0-9]*/; do
    n=$(basename "$d")
    python3 -c "
import json
r=json.load(open('$d/result.json'))
num=r['task'].split('-')[0]
short='-'.join(r['task'].split('-')[1:])
print(f'| {num} | {short} | {\"PASS\" if r[\"verify_pass\"] else \"FAIL\"} | {r[\"wall_seconds\"]:.1f}s | \${r[\"cost_usd\"]:.4f} | {r[\"turns\"]} |')
"
  done
  echo
  echo "**claude-code total: $pass/$total verify, \$$total_cost, ${total_wall}s wall.**"
} > "$BENCH_ROOT/head_to_head.md"

echo
echo "Summary written to $BENCH_ROOT/head_to_head.md"
echo "claude-code: $pass/$total PASS, total \$$total_cost, ${total_wall}s wall"
