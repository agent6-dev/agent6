#!/usr/bin/env bash
# Smoke variant of run_perf.sh — ~$1 budget (vs ~$5 for the full run).
#
# Purpose: fast per-iter regression / improvement signal during prompt or
# routing changes. Same task, same scoring harness, same metric (cycles),
# just a smaller compute envelope. Use the full run_perf.sh / run_perf_n.sh
# for milestone scoring; use this for go/no-go signal on each iter.
#
# ~$1 at sonnet-4.5 pricing ($3/M in, $15/M out) ≈ 300k in + 24k out.
#
# Usage:
#   ANTHROPIC_API_KEY=... bash bench/perf/run_perf_smoke.sh
set -euo pipefail
export AGENT6_PERF_MAX_IN="${AGENT6_PERF_MAX_IN:-300000}"
export AGENT6_PERF_MAX_OUT="${AGENT6_PERF_MAX_OUT:-24000}"
exec bash "$(dirname "$0")/run_perf.sh" "$@"
