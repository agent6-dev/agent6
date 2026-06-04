#!/usr/bin/env bash
# Summarize bench/perf/ results into a markdown table after both runners
# have written their result_*.json into $BENCH_ROOT. Safe to re-run.
set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-/tmp/agent6-perf}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

"$PYTHON_BIN" - <<PY
import json, os
from pathlib import Path
root = Path("${BENCH_ROOT}")
rows = []
for runner in ("agent6", "claude-code"):
    fname = "result_agent6.json" if runner == "agent6" else "result_claude.json"
    p = root / fname
    if not p.exists():
        continue
    rows.append(json.loads(p.read_text()))

if not rows:
    print(f"No results in {root}")
    raise SystemExit(1)

print("# bench/perf/ — Anthropic perf-takehome head-to-head")
print()
print("| runner | final cycles | speedup over baseline | speed tiers passed | wall (s) | cost (\$) | in tok | out tok | valid | tests clean | N_CORES ok |")
print("|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|")
for r in rows:
    fc = r.get("final_cycles")
    sp = r.get("speedup_over_baseline")
    print(
        f"| {r['runner']} | {fc if fc is not None else '—'} "
        f"| {sp if sp is not None else '—'}x "
        f"| {r.get('speed_tiers_passed', 0)}/8 "
        f"| {r['wall_seconds']} | {r['cost_usd']} | {r['input_tokens']} | {r['output_tokens']} "
        f"| {'YES' if r['valid'] else 'NO'} "
        f"| {'yes' if r['tests_clean'] else 'NO'} "
        f"| {'yes' if r['n_cores_ok'] else 'NO'} |"
    )
print()
print("Baseline: 147734 cycles. Reference scores from upstream README:")
print("- 1487 cycles: Claude Opus 4.5 at launch (best human ~= 1790)")
print("- 1363 cycles: Claude Opus 4.5 in improved harness")
PY
