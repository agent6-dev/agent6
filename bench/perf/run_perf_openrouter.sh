#!/usr/bin/env bash
# Model-agnostic OpenRouter variant of run_perf.sh - bench agent6 with a
# single model on EVERY role (no opus escalation, no haiku summarizer)
# against the Anthropic perf-takehome. Used to capture current-state
# data for any OpenRouter model (Kimi K2.6, Qwen3-Max, ...).
#
# Pick the model with AGENT6_OR_MODEL (default: moonshotai/kimi-k2.6).
# Results are labelled by AGENT6_OR_LABEL (default: derived from the
# model slug) and land in $BENCH_ROOT/result_agent6.json.
#
# At Kimi/Qwen OpenRouter pricing (~$0.7/M in, ~$3.5-3.9/M out) one
# milestone run (1.5M in, 120k out cap) tops out at ~$1.50 -
# substantially cheaper than the equivalent sonnet bench (~$5).
#
# Set OPENROUTER_API_KEY before running.
#
# Usage:
#   OPENROUTER_API_KEY=... bash bench/perf/run_perf_openrouter.sh
#   AGENT6_OR_MODEL=qwen/qwen3-max OPENROUTER_API_KEY=... \
#     BENCH_ROOT=/tmp/agent6-qwen-perf bash bench/perf/run_perf_openrouter.sh

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-perf}"
PERF_REPO_URL="${PERF_REPO_URL:-https://github.com/anthropics/original_performance_takehome.git}"
PERF_REPO_COMMIT="${PERF_REPO_COMMIT:-5452f74bd977807ac2e74f3d29432b9df6f25197}"
MODEL="${AGENT6_OR_MODEL:-moonshotai/kimi-k2.6}"
# Result label, used in result_agent6.json. Derive from the model slug
# (e.g. qwen/qwen3-max -> agent6-qwen3-max) unless overridden.
# Budget envelope. Two modes:
#   - Token mode (default): same token caps as the sonnet milestone.
#   - USD mode: set AGENT6_PERF_MAX_USD to give every model the SAME dollar
#     budget. The config loader sizes per-model token ceilings from each
#     model's pricing (budget.usd_budget_to_tokens), so we raise the raw
#     token caps high enough that the USD cap is the binding constraint
#     (config.resolve takes min(token_cap, usd_derived_cap)).
LABEL="${AGENT6_OR_LABEL:-agent6-$(printf '%s' "$MODEL" | sed 's#.*/##; s/[^A-Za-z0-9._-]/-/g')}"
MAX_USD="${AGENT6_PERF_MAX_USD:-0}"
if [ "$(printf '%s' "$MAX_USD" | awk '{print ($1>0)}')" = "1" ]; then
  MAX_INPUT_TOKENS="${AGENT6_PERF_MAX_IN:-1000000000}"
  MAX_OUTPUT_TOKENS="${AGENT6_PERF_MAX_OUT:-1000000000}"
else
  MAX_INPUT_TOKENS="${AGENT6_PERF_MAX_IN:-1500000}"
  MAX_OUTPUT_TOKENS="${AGENT6_PERF_MAX_OUT:-120000}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set" >&2; exit 1; }

cd "$REPO"
export AGENT6_JAIL_BIN="${AGENT6_JAIL_BIN:-$REPO/src/agent6/jail/target/release/agent6-jail}"
AGENT6_BIN="$REPO/.venv/bin/agent6"
[ -x "$AGENT6_BIN" ] || { echo "agent6 not found at $AGENT6_BIN" >&2; exit 1; }
[ -x "$AGENT6_JAIL_BIN" ] || { echo "jail launcher missing at $AGENT6_JAIL_BIN" >&2; exit 1; }

# v1-shape benchmark: disable architect/editor split so this run uses
# the same model on a single edit loop. (No-op on the current single
# loop workflow; kept for parity with older result JSONs.)
export AGENT6_DISABLE_ARCHITECT_EDITOR=1

# Force the OpenAI provider onto its SSE streaming
# code path even though bench stderr is not a TTY. OpenRouter emits
# `: OPENROUTER PROCESSING` SSE comment heartbeats for long requests;
# on the non-streaming code path those land in the response body and
# break resp.json() with a JSONDecodeError mid-stream. Streaming
# consumes the heartbeats as protocol-level keep-alives, so the
# connection stays healthy and we get clean delta accumulation.
export AGENT6_FORCE_STREAM=1

mkdir -p "$BENCH_ROOT/_logs/agent6"
WORKDIR="$BENCH_ROOT/perf_agent6"
LOGDIR="$BENCH_ROOT/_logs/agent6"

rm -rf "$WORKDIR"
git init --quiet "$WORKDIR"
git -C "$WORKDIR" remote add origin "$PERF_REPO_URL"
git -C "$WORKDIR" fetch --quiet --depth 1 origin "$PERF_REPO_COMMIT"
git -C "$WORKDIR" checkout --quiet -B main FETCH_HEAD
git -C "$WORKDIR" update-ref refs/remotes/origin/main FETCH_HEAD
git -C "$WORKDIR" config user.email "bench@agent6"
git -C "$WORKDIR" config user.name "bench"

# TASK.md - same content as run_perf.sh except the python3 path
# is the same (agent6 jail's allowed Python).
cat > "$WORKDIR/TASK.md" <<'MD'
# Optimize KernelBuilder.build_kernel for the simulated machine

This repo is Anthropic's original performance take-home. The simulated
machine in `problem.py` runs the kernel emitted by
`KernelBuilder.build_kernel` (in `perf_takehome.py`) and reports a cycle
count. The goal is to minimize that cycle count.

## How to measure cycles

Run:

```
/usr/bin/python3 tests/submission_tests.py
```

It prints lines like `CYCLES:  18532`. Lower is better. The baseline
starting point is 147734 cycles; Claude Opus 4.5's best at launch was
1487 cycles.

The configured `verify_command` is
`pytest tests/submission_tests.py::CorrectnessTests` - only correctness
is checked, not speed tiers.

This run has `[workflow.metric]` configured: after every passing step,
the harness runs the cycles command, parses `CYCLES:` and feeds the
trajectory back to you as a `[harness metric]` block on the next step.
Keep verified edits only when that block shows a new best; if it is
flat or worse, revert or pivot to a different bottleneck.

## HARD RULES - violating these voids the result

1. Do NOT modify anything in `tests/`.
2. Do NOT change `N_CORES` in `problem.py`. Multicore is disabled
   intentionally; setting `N_CORES > 1` is the canonical cheat.
3. Correctness first - CorrectnessTests must pass.
4. All optimization happens in `perf_takehome.py`. You may also edit
   `problem.py` for pure performance improvements to the simulator's
   implementation that don't change its semantics.

## How to work

- Read `perf_takehome.py` and `problem.py` top to bottom first.
- After each change, run `/usr/bin/python3 tests/submission_tests.py`
  and look at the `CYCLES:` line. Keep changes that reduce cycles AND
  preserve correctness; revert anything else.
- Commit working improvements as you go. The scoring harness rescues
  the best (lowest-cycles) commit on this branch after the run ends.

You have a fixed compute budget. Iterate as many times as the budget
allows.
MD

# OpenRouter wants HTTP-Referer + X-Title. agent6 forwards extra_headers
# verbatim (per config.py).
cat > "$WORKDIR/agent6.toml" <<EOF
[agent6]
config_version = 1

[providers.openrouter]
api_format = "openai"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
extra_headers = { "HTTP-Referer" = "https://github.com/elesiuta/agent6", "X-Title" = "agent6-bench" }

[models.worker]
provider = "openrouter"
model = "$MODEL"

[models.reviewer]
provider = "openrouter"
model = "$MODEL"

[sandbox]
profile = "auto"
agent_network = "providers"
tool_network = "block"
run_commands = "yes"
protect_git = true

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
verify_command = [
  "/usr/bin/pytest",
  "-q",
  "tests/submission_tests.py::CorrectnessTests",
]
verify_timeout_s = 30.0

[prompt]
revise_prompt = "${AGENT6_PERF_REVISE_PROMPT:-off}"

[workflow.metric]
command = ["/usr/bin/python3", "tests/submission_tests.py"]
pattern = 'CYCLES:\s*(\d+)'
goal = "minimize"

[budget]
max_input_tokens = $MAX_INPUT_TOKENS
max_output_tokens = $MAX_OUTPUT_TOKENS
best_effort_usd_limit = $MAX_USD
EOF

{
  echo "agent6.toml"
  echo "TASK.md"
  echo ".agent6/"
} >> "$WORKDIR/.gitignore"
git -C "$WORKDIR" add .gitignore
git -C "$WORKDIR" commit -q -m "bench: gitignore harness files"

( cd "$WORKDIR" && "$PYTHON_BIN" tests/submission_tests.py 2>&1 ) > "$LOGDIR/cycles_before.txt" || true
start_cycles=$(grep -oE 'CYCLES:[[:space:]]+[0-9]+' "$LOGDIR/cycles_before.txt" | head -1 | awk '{print $2}')
echo "Starting cycles: ${start_cycles:-unknown}"
echo "Model: $MODEL (via OpenRouter)"

task_text=$(cat "$WORKDIR/TASK.md")

start_ns=$(date +%s%N)
set +e
# Current CLI takes only a positional `task` arg; legacy --yes /
# --no-tui flags have been removed.
( cd "$WORKDIR" && "$AGENT6_BIN" --config agent6.toml run "$task_text" ) \
  > "$LOGDIR/agent6.stdout" 2> "$LOGDIR/agent6.stderr"
ag_exit=$?
set -e
end_ns=$(date +%s%N)
wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.1f", (e-s)/1e9}')

combined="$LOGDIR/agent6.stdout $LOGDIR/agent6.stderr"
cost=$(cat $combined 2>/dev/null | grep -oE 'cost~\$[0-9.]+' | tail -1 | tr -d '$' | sed 's/cost~//')
[ -z "$cost" ] && cost=0
in_tok=$(cat $combined 2>/dev/null | grep -oE 'TOTAL: in=[0-9]+' | tail -1 | sed 's/TOTAL: in=//')
[ -z "$in_tok" ] && in_tok=0
out_tok=$(cat $combined 2>/dev/null | grep -oE 'out=[0-9]+/' | tail -1 | tr -d '/' | sed 's/out=//')
[ -z "$out_tok" ] && out_tok=0

bash "$REPO/bench/perf/score.sh" "$WORKDIR" "$LABEL" \
  "$wall_s" "$cost" "$in_tok" "$out_tok" "$ag_exit" "$start_cycles" \
  > "$BENCH_ROOT/result_agent6.json"

cat "$BENCH_ROOT/result_agent6.json"
