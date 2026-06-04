#!/usr/bin/env bash
# Smoke variant of run_perf_claude.sh — $1 budget (vs $5 for the full run).
# Paired with run_perf_smoke.sh so we always have a same-budget reference
# when measuring iter-over-iter changes on the agent6 side.
#
# Usage:
#   ANTHROPIC_API_KEY=... bash bench/perf/run_perf_smoke_claude.sh
set -euo pipefail
export CLAUDE_BUDGET_USD="${CLAUDE_BUDGET_USD:-1.00}"
exec bash "$(dirname "$0")/run_perf_claude.sh" "$@"
