#!/usr/bin/env bash
# One runner for the synthetic tiers: creates each tier's task repos under
# $BENCH_ROOT, runs `agent6 run` on each with a hard budget cap, captures wall
# time + cost + verify-pass, and writes one result.json per task.
#
# Usage: bash bench/run_tier.sh <core|hard|extreme|megaextreme|tier5>
#
# A tier file (bench/tiers/<tier>.sh) supplies the fixtures + tier config:
#   DEFAULT_BENCH_ROOT   where task repos land when $BENCH_ROOT is unset
#   tier_toml            the agent6.toml every task repo gets
#   tier_gitignore       the .gitignore body
#   tier_agents_md       the AGENTS.md body
#   TASKS                "setup_fn task-name" per entry, run order
#   setup_task*          one function per task; echoes the task dir
#   tier_final_verify    (optional) the post-run check; defaults to unittest
#
# Constraints:
#   * The jail only exposes the Python stdlib (no pytest, no extra modules),
#     so tasks use plain `unittest`.

set -euo pipefail

TIER="${1:?usage: bash bench/run_tier.sh <core|hard|extreme|megaextreme|tier5>}"
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${AGENT6_REPO:-$(cd "$BENCH_DIR/.." && pwd)}"
TIER_FILE="$BENCH_DIR/tiers/$TIER.sh"
[ -f "$TIER_FILE" ] || {
  echo "unknown tier '$TIER'; have: $(ls "$BENCH_DIR/tiers" | sed 's/\.sh$//' | paste -sd' ' -)" >&2
  exit 2
}

cd "$REPO"
export AGENT6_JAIL_BIN="${AGENT6_JAIL_BIN:-$REPO/src/agent6/jail/target/release/agent6-jail}"
AGENT6_BIN="$REPO/.venv/bin/agent6"
[ -x "$AGENT6_BIN" ] || { echo "agent6 not found at $AGENT6_BIN — run 'uv sync' in $REPO first" >&2; exit 1; }

# The post-run pass/fail check; a tier file may override (tier5 excludes its
# hidden suite here: the agent must never see it).
tier_final_verify() { python3 -m unittest -v; }

# shellcheck source=/dev/null
source "$TIER_FILE"

BENCH_ROOT=${BENCH_ROOT:-$DEFAULT_BENCH_ROOT}
mkdir -p "$BENCH_ROOT/logs"

init_repo() {
  local dir="$1"
  rm -rf "$dir"
  mkdir -p "$dir"
  ( cd "$dir" && git init -q && git config user.email bench@agent6 && git config user.name bench )
  tier_toml > "$dir/agent6.toml"
  # Fail on a dead config key here, in 200ms, not after a task's spend:
  # `config show` loads the same merged layers the run will.
  ( cd "$dir" && "$AGENT6_BIN" --config agent6.toml config show >/dev/null ) || exit 1
  tier_gitignore > "$dir/.gitignore"
  tier_agents_md > "$dir/AGENTS.md"
}

run_task() {
  local dir="$1" name="$2"
  echo
  echo "================================================================"
  echo "TASK: $name"
  echo "DIR : $dir"
  echo "================================================================"
  local task_text; task_text=$(cat "$dir/TASK.md")
  local start_ns end_ns wall_s log
  log="$BENCH_ROOT/logs/${name}.log"
  start_ns=$(date +%s%N)
  set +e
  ( cd "$dir" && "$AGENT6_BIN" --config "$dir/agent6.toml" run "$task_text" ) \
    > "$log" 2>&1
  local exit_code=$?
  set -e
  end_ns=$(date +%s%N)
  wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.1f", (e-s)/1e9}')

  set +e
  ( cd "$dir" && tier_final_verify ) > "$dir/final_pytest.txt" 2>&1
  local verify=$?
  set -e

  local cost_line; cost_line=$(grep -E '^\s*TOTAL: in=' "$log" | tail -1 || echo "")
  local commits; commits=$( cd "$dir" && git rev-list --count HEAD)
  local diff_lines; diff_lines=$( cd "$dir" && git diff --shortstat HEAD~$((commits-1)) HEAD 2>/dev/null | tail -1 || echo "")
  local test_modified="false"
  ( cd "$dir" && git diff HEAD~$((commits-1)) HEAD --name-only | grep -E '^test_|/test_' >/dev/null ) && test_modified="true" || true

  cat > "$dir/result.json" <<EOF
{
  "task": "$name",
  "exit_code": $exit_code,
  "wall_seconds": $wall_s,
  "verify_pass": $([ $verify -eq 0 ] && echo true || echo false),
  "cost_summary": $(printf '%s' "$cost_line" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))"),
  "commits": $commits,
  "diff_shortstat": $(printf '%s' "$diff_lines" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))"),
  "test_file_modified": $test_modified
}
EOF
  echo "  exit=$exit_code  verify=$([ $verify -eq 0 ] && echo PASS || echo FAIL)  wall=${wall_s}s"
  echo "  $cost_line"
}

for entry in "${TASKS[@]}"; do
  fn="${entry%% *}"
  name="${entry#* }"
  dir="$($fn)"
  run_task "$dir" "$name"
done

echo
echo "Per-task JSON results saved under $BENCH_ROOT/*/result.json"
