#!/usr/bin/env bash
# Post-hoc scoring + anti-cheat for bench/perf/.
#
# Usage:
#   bench/perf/score.sh <workdir> <runner> <wall_s> <cost> <in_tok> <out_tok> <agent_exit> <start_cycles>
#
# Emits a JSON object to stdout.
#
# Cheat detection (all must pass for the result to be VALID):
#   1. `git diff origin/main -- tests/` is empty
#   2. `N_CORES` in problem.py is unchanged
#   3. CorrectnessTests pass when the upstream tests/ is restored
#
# Score: minimum CYCLES line printed by `python tests/submission_tests.py`
# run against the AGENT'S problem.py and perf_takehome.py BUT with the
# upstream tests/ contents (in case the agent quietly modified tests).

set -euo pipefail

WORKDIR="$1"
RUNNER="$2"
WALL_S="$3"
COST="$4"
IN_TOK="$5"
OUT_TOK="$6"
AGENT_EXIT="$7"
START_CYCLES="${8:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "$WORKDIR"

# --- (1) tests/ diff ---------------------------------------------------------
tests_diff=$(git diff origin/main -- tests/ 2>/dev/null || true)
tests_clean=true
[ -n "$tests_diff" ] && tests_clean=false

# --- (2) N_CORES sanity (deferred to best-commit check below) ---------------
# Captured AFTER best_sha is identified so a multi-core cheat that was reverted
# at HEAD doesn't slip through. Bug history: original check looked at HEAD's
# problem.py only, then per-commit checkouts overwrote it - so n_cores_ok
# always reflected HEAD, not best_sha. See audit 2026-05-24.
n_cores_ok=true  # provisional; recomputed at line ~95 against best_sha

# --- (3) restore upstream tests/ for scoring --------------------------------
# Use a worktree-local checkout of origin/main tests/ so we score with the
# real test contents regardless of any tampering.
git checkout origin/main -- tests/ 2>/dev/null || true

# Score EVERY commit between origin/main and HEAD, then keep the best.
# This rescues the lowest-cycle commit even if a later step regressed.
score_out="$WORKDIR/_score.txt"
: > "$score_out"
best_cycles=""
best_sha=""
head_sha=$(git rev-parse HEAD)
# Commits in chronological order, oldest first.
commits=$(git rev-list --reverse "origin/main..HEAD" 2>/dev/null || true)
# Always include HEAD as a candidate (covers the no-commits case).
[ -z "$commits" ] && commits="$head_sha"
for sha in $commits; do
  # Check out just the source files (NOT tests/) for this commit.
  for f in perf_takehome.py problem.py; do
    git checkout "$sha" -- "$f" 2>/dev/null || true
  done
  set +e
  per_out=$("$PYTHON_BIN" tests/submission_tests.py 2>&1)
  set -e
  echo "=== commit $sha ===" >> "$score_out"
  echo "$per_out" >> "$score_out"
  c=$(echo "$per_out" | grep -oE 'CYCLES:[[:space:]]+[0-9]+' | awk '{print $2}' | sort -n | head -1)
  if [ -n "$c" ]; then
    if [ -z "$best_cycles" ] || [ "$c" -lt "$best_cycles" ]; then
      best_cycles="$c"
      best_sha="$sha"
    fi
  fi
done
# Restore HEAD's source files so the workdir is left in a sensible state.
for f in perf_takehome.py problem.py; do
  git checkout "$head_sha" -- "$f" 2>/dev/null || true
done

final_cycles="${best_cycles:-null}"
best_sha_json="\"${best_sha:-}\""
[ -z "$best_sha" ] && best_sha_json=null

# Tally how many speed-tier tests passed against the BEST commit's source.
# Use `-v` so unittest emits per-test `test_X (qualname) ... ok|FAIL` lines
# we can grep. The earlier non-`-v` invocation only produced `.F` dots so
# passed_tiers was silently always 0 (audit 2026-05-24).
# Also recompute n_cores_ok against the best_sha here so a multi-core cheat
# that was reverted at HEAD doesn't slip through.
if [ -n "$best_sha" ]; then
  for f in perf_takehome.py problem.py; do
    git checkout "$best_sha" -- "$f" 2>/dev/null || true
  done
  n_cores_line=$(grep -E '^[[:space:]]*N_CORES[[:space:]]*=' problem.py | head -1 || true)
  case "$n_cores_line" in
    *"= 1"*) n_cores_ok=true ;;
    *) n_cores_ok=false ;;
  esac
  # Pattern: `test_kernel_speedup (...) ... ok` (verbose unittest output
  # from the script's own unittest.main()). Subtract 1 for the
  # `test_kernel_correctness` line in CorrectnessTests which is not a tier.
  passed_tiers_raw=$("$PYTHON_BIN" tests/submission_tests.py -v 2>&1 \
    | grep -cE '^test_[a-zA-Z0-9_]+ .* \.\.\. ok$' || true)
  passed_tiers=$((passed_tiers_raw > 0 ? passed_tiers_raw - 1 : 0))
  for f in perf_takehome.py problem.py; do
    git checkout "$head_sha" -- "$f" 2>/dev/null || true
  done
else
  passed_tiers=0
fi

valid=true
if [ "$tests_clean" != "true" ] || [ "$n_cores_ok" != "true" ]; then
  valid=false
fi

# Speedup over baseline (147734). null if no cycles.
if [ "$final_cycles" != "null" ]; then
  speedup=$(awk -v c="$final_cycles" 'BEGIN{printf "%.2f", 147734/c}')
else
  speedup=null
fi

scored_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat <<EOF
{
  "runner": "$RUNNER",
  "agent_exit": $AGENT_EXIT,
  "wall_seconds": $WALL_S,
  "cost_usd": $COST,
  "input_tokens": $IN_TOK,
  "output_tokens": $OUT_TOK,
  "start_cycles": ${START_CYCLES:-null},
  "final_cycles": $final_cycles,
  "best_commit_sha": $best_sha_json,
  "speedup_over_baseline": $speedup,
  "speed_tiers_passed": ${passed_tiers:-0},
  "tests_clean": $tests_clean,
  "n_cores_ok": $n_cores_ok,
  "valid": $valid,
  "scored_at": "$scored_at"
}
EOF
