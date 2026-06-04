#!/usr/bin/env bash
# overnight FULL bench. For each of the three winning open-weights
# candidates from the smoke gate (GLM-4.6, Minimax-m2.7, Kimi-k2.6 with the
# loop-guard feature active), runs the full 11-task realworld suite
# and the perf takehome. Same env-var driver as run_overnight_smoke.sh.
#
# Budget envelope (per-model estimates from smoke):
#   GLM-4.6:        11 tasks * ~$0.008 = ~$0.09 realworld + ~$0.5  perf  = ~$0.6
#   Minimax-m2.7:   11 tasks * ~$0.009 = ~$0.10 realworld + ~$0.5  perf  = ~$0.6
#   Kimi-k2.6:      11 tasks * ~$0.031 = ~$0.35 realworld + ~$0.5  perf  = ~$0.85
# Total ~$2-2.5. Well inside the $20 overnight budget.

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVR_ROOT=${OVR_ROOT:-/tmp/agent6-overnight-full}

MODELS=(
  "z-ai/glm-4.6"
  "minimax/minimax-m2.7"
  "moonshotai/kimi-k2.6"
)

mkdir -p "$OVR_ROOT"

pids=()
for model in "${MODELS[@]}"; do
  slug=$(echo "$model" | tr '/.' '__')
  rwdir="$OVR_ROOT/$slug/realworld"
  perfdir="$OVR_ROOT/$slug/perf"
  mkdir -p "$rwdir" "$perfdir"
  echo "[overnight-full] launching $model"

  # Realworld (all 11 tasks).
  (
    AGENT6_REALWORLD_MODEL="$model" \
    AGENT6_REALWORLD_OPENROUTER=1 \
    AGENT6_REALWORLD_CRITIC=off \
    AGENT6_FORCE_STREAM=1 \
    BENCH_ROOT="$rwdir" \
      bash "$REPO/bench/realworld/run_realworld.sh" \
      > "$rwdir/driver.stdout" 2> "$rwdir/driver.stderr"
    echo "[overnight-full] $model realworld done (exit $?)"
  ) &
  pids+=($!)

  # Perf takehome. run_perf_openrouter.sh is the OpenRouter-wired perf
  # driver; it reads AGENT6_OR_MODEL as MODEL.
  (
    AGENT6_OR_MODEL="$model" \
    BENCH_ROOT="$perfdir" \
      bash "$REPO/bench/perf/run_perf_openrouter.sh" \
      > "$perfdir/driver.stdout" 2> "$perfdir/driver.stderr"
    echo "[overnight-full] $model perf done (exit $?)"
  ) &
  pids+=($!)
done

echo "[overnight-full] launched ${#pids[@]} workers; waiting..."
wait
echo "[overnight-full] all finished"

# Aggregate.
python3 - <<PY > "$OVR_ROOT/summary.md"
import json, os, glob
root = "$OVR_ROOT"
print("# overnight full bench")
print()
print("## Realworld")
print()
print("| model | task | pass | wall (s) | tok_in | tok_out |")
print("|---|---|---|---|---|---|")
for slug in sorted(os.listdir(root)):
    d = os.path.join(root, slug, "realworld")
    if not os.path.isdir(d): continue
    for r in sorted(glob.glob(os.path.join(d, "_logs", "*", "result.json"))):
        try:
            j = json.load(open(r))
        except Exception:
            continue
        task = os.path.basename(os.path.dirname(r)).rsplit("_", 1)[0]
        print(f"| {slug} | {task} | {j['verify_pass']} | {j['wall_seconds']:.1f} | {j['input_tokens']} | {j['output_tokens']} |")
print()
print("## Perf")
print()
for slug in sorted(os.listdir(root)):
    d = os.path.join(root, slug, "perf")
    if not os.path.isdir(d): continue
    print(f"### {slug}")
    print()
    print("```")
    for f in sorted(glob.glob(os.path.join(d, "_logs", "*"))):
        print(f"  {f}")
    print("```")
    print()
PY
echo "[overnight-full] summary -> $OVR_ROOT/summary.md"
