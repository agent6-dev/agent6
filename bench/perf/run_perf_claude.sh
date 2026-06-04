#!/usr/bin/env bash
# claude-code side of the perf-takehome head-to-head. Mirrors
# bench/perf/run_perf.sh but uses the `claude` CLI in --print mode with
# --max-budget-usd 5.00. Scoring is done by bench/perf/score.sh, identical
# to the agent6 side.

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-perf}"
PERF_REPO_URL="${PERF_REPO_URL:-https://github.com/anthropics/original_performance_takehome.git}"
PERF_REPO_COMMIT="${PERF_REPO_COMMIT:-5452f74bd977807ac2e74f3d29432b9df6f25197}"
CLAUDE_BUDGET_USD="${CLAUDE_BUDGET_USD:-5.00}"
CLAUDE_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-5}"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.npm-global/bin/claude}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "$REPO"
[ -x "$CLAUDE_BIN" ] || { echo "claude-code not found at $CLAUDE_BIN" >&2; exit 1; }

mkdir -p "$BENCH_ROOT/_logs/claude"
WORKDIR="$BENCH_ROOT/perf_claude"
LOGDIR="$BENCH_ROOT/_logs/claude"

rm -rf "$WORKDIR"
# init + fetch-by-SHA pattern (supports both SHA and branch).
git init --quiet "$WORKDIR"
git -C "$WORKDIR" remote add origin "$PERF_REPO_URL"
git -C "$WORKDIR" fetch --quiet --depth 1 origin "$PERF_REPO_COMMIT"
git -C "$WORKDIR" checkout --quiet -B main FETCH_HEAD
git -C "$WORKDIR" update-ref refs/remotes/origin/main FETCH_HEAD
git -C "$WORKDIR" config user.email "bench@agent6"
git -C "$WORKDIR" config user.name "bench"

# TASK.md mirrors agent6's task + HARD RULES, but omits agent6-specific
# bits (verify_command, [workflow.metric], jailed-python path) that don't
# apply to claude-code. Keep the goal description + HARD RULES sections
# in lock-step when editing; the agent-tool-specific framing is allowed
# to differ per runner.
cat > "$WORKDIR/TASK.md" <<'MD'
# Optimize KernelBuilder.build_kernel for the simulated machine

This repo is Anthropic's original performance take-home. The simulated
machine in `problem.py` runs the kernel emitted by
`KernelBuilder.build_kernel` (in `perf_takehome.py`) and reports a cycle
count. The goal is to minimize that cycle count.

## How to measure cycles

Run:

```
python tests/submission_tests.py
```

It prints lines like `CYCLES:  18532` and tells you which speed-tier
tests pass. Lower is better. The baseline starting point is 147734
cycles; Claude Opus 4.5's best at launch was 1487 cycles.

## HARD RULES — read carefully, violating these voids the result

1. **Do not modify anything in `tests/`.** Not `tests/submission_tests.py`,
   not `tests/frozen_problem.py`, nothing under that directory. The
   scoring harness will run `git diff origin/main -- tests/` after you
   finish; any non-empty diff is treated as cheating and the score is
   discarded. The `tests/frozen_problem.py` file is intentionally a
   frozen copy of the original `problem.py` — changing it to make the
   simulator "faster" is the exact failure mode the upstream README
   warns about.
2. **Do not change `N_CORES` in `problem.py`.** Multicore is disabled
   intentionally. Setting `N_CORES > 1` is documented as the canonical
   cheat and will be detected (any change to that constant is flagged).
3. **Correctness first.** `tests/submission_tests.py`'s
   `CorrectnessTests` must pass. If it does not, the cycle count is
   doubled (the upstream test does this automatically).
4. **All optimization happens in `perf_takehome.py`.** You may also
   edit `problem.py` for changes that are pure performance improvements
   to the simulator's *implementation* without changing its semantics
   (e.g. replacing `list` with `array.array`, hoisting a dict lookup),
   but do NOT change anything that affects the cycle count of a given
   kernel: the cycle accounting in `Machine.run`, `SLOT_LIMITS`,
   `VLEN`, `N_CORES`, `SCRATCH_SIZE`, `HASH_STAGES`, or the engine
   model. When in doubt, only edit `perf_takehome.py`.

## How to work

- Read `perf_takehome.py` top-to-bottom first — the docstring explains
  the engine model (alu / load / store / flow), the VLIW packing, and
  what `build_kernel` is allowed to emit.
- Read `problem.py` for the instruction semantics and cycle accounting.
- After each change, run `python tests/submission_tests.py` and look at
  the `CYCLES:` line. Keep changes that reduce cycles AND keep
  correctness; revert anything else.
- Commit working improvements as you go so we have a history. If you
  break correctness, revert.

You have a fixed compute budget. Iterate as many times as the budget
allows. Stopping early is wasted budget — keep optimizing right up
until the cap.
MD

# Capture starting cycles.
( cd "$WORKDIR" && "$PYTHON_BIN" tests/submission_tests.py 2>&1 ) > "$LOGDIR/cycles_before.txt" || true
start_cycles=$(grep -oE 'CYCLES:[[:space:]]+[0-9]+' "$LOGDIR/cycles_before.txt" | head -1 | awk '{print $2}')
echo "Starting cycles: ${start_cycles:-unknown}"

task_text=$(cat "$WORKDIR/TASK.md")

start_ns=$(date +%s%N)
set +e
( cd "$WORKDIR" && "$CLAUDE_BIN" \
    --print \
    --dangerously-skip-permissions \
    --model "$CLAUDE_MODEL" \
    --output-format json \
    --max-budget-usd "$CLAUDE_BUDGET_USD" \
    "$task_text" \
) > "$LOGDIR/claude.json" 2> "$LOGDIR/claude.stderr"
agent_exit=$?
set -e
end_ns=$(date +%s%N)
wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.1f", (e-s)/1e9}')

cost=$("$PYTHON_BIN" -c "import json; d=json.load(open('$LOGDIR/claude.json')); print(d.get('total_cost_usd', d.get('cost_usd', 0)))" 2>/dev/null || echo 0)
# claude-code reports per-model token totals under modelUsage (top-level
# `usage` is zero across the board for --print mode). Sum across models so
# the head-to-head agent6 vs claude-code token comparison is meaningful.
in_tok=$("$PYTHON_BIN" -c "import json; d=json.load(open('$LOGDIR/claude.json')); m=d.get('modelUsage', {}); print(sum(int(v.get('inputTokens', 0)) + int(v.get('cacheReadInputTokens', 0)) + int(v.get('cacheCreationInputTokens', 0)) for v in m.values()))" 2>/dev/null || echo 0)
out_tok=$("$PYTHON_BIN" -c "import json; d=json.load(open('$LOGDIR/claude.json')); m=d.get('modelUsage', {}); print(sum(int(v.get('outputTokens', 0)) for v in m.values()))" 2>/dev/null || echo 0)

bash "$REPO/bench/perf/score.sh" "$WORKDIR" "claude-code" \
  "$wall_s" "$cost" "$in_tok" "$out_tok" "$agent_exit" "$start_cycles" \
  > "$BENCH_ROOT/result_claude.json"

cat "$BENCH_ROOT/result_claude.json"
