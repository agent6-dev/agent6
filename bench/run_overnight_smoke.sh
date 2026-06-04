#!/usr/bin/env bash
# Overnight smoke: try 4 top open-weights OpenRouter models against 2 cheap
# realworld tasks each, in parallel. Drives bench/realworld/run_realworld.sh
# via AGENT6_REALWORLD_* env vars. Captures per-model results under
# /tmp/agent6-overnight-smoke/<model-slug>/.
#
# Cheap by design: ~$0.05 per (model x task) at Kimi-class pricing, ~$0.5 total.
# Goal is "does this model even drive apply_edit on a bounded task" — a
# go/no-go gate before spending real bench dollars.

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVR_ROOT=${OVR_ROOT:-/tmp/agent6-overnight-smoke}
TASK_FILTER=${TASK_FILTER:-roman-tampering,click-format-filename}

# Models to smoke. Order matters only for logging.
MODELS=(
  "moonshotai/kimi-k2.6"            # baseline for cross-check
  "deepseek/deepseek-v3.2-exp"      # widely-deployed, fewer surprises than v4-pro
  "qwen/qwen3-coder"                # code-specialized, MoE 480B
  "z-ai/glm-4.6"                    # strong recent OS coder
  "minimax/minimax-m2.7"            # newer minimax line
)

mkdir -p "$OVR_ROOT"

pids=()
for model in "${MODELS[@]}"; do
  slug=$(echo "$model" | tr '/.' '__')
  rundir="$OVR_ROOT/$slug"
  mkdir -p "$rundir"
  echo "[overnight-smoke] launching $model -> $rundir"
  (
    AGENT6_REALWORLD_MODEL="$model" \
    AGENT6_REALWORLD_OPENROUTER=1 \
    AGENT6_REALWORLD_CRITIC=off \
    AGENT6_REALWORLD_TASK_FILTER="$TASK_FILTER" \
    AGENT6_FORCE_STREAM=1 \
    BENCH_ROOT="$rundir" \
      bash "$REPO/bench/realworld/run_realworld.sh" \
      > "$rundir/driver.stdout" 2> "$rundir/driver.stderr"
    echo "[overnight-smoke] $model finished (exit $?)"
  ) &
  pids+=($!)
done

echo "[overnight-smoke] launched ${#pids[@]} workers; waiting..."
wait
echo "[overnight-smoke] all models finished"

# Aggregate.
python3 - <<PY > "$OVR_ROOT/summary.md"
import json, os, glob
root = "$OVR_ROOT"
rows = []
for slug in sorted(os.listdir(root)):
    d = os.path.join(root, slug)
    if not os.path.isdir(d):
        continue
    for rj in sorted(glob.glob(os.path.join(d, "_logs", "*", "result.json"))):
        try:
            r = json.load(open(rj))
        except Exception as e:
            rows.append((slug, "(parse error)", "?", 0, 0, str(e)))
            continue
        rows.append((
            slug,
            r.get("task", "?"),
            "PASS" if r.get("verify_pass") else "FAIL",
            r.get("wall_seconds", 0),
            r.get("cost", 0),
            r.get("metric_score", "—"),
        ))
print("# overnight smoke summary")
print()
print("| model_slug | task | verify | wall_s | cost_usd | metric |")
print("|---|---|---|---|---|---|")
for row in rows:
    print("| " + " | ".join(str(x) for x in row) + " |")
PY
cat "$OVR_ROOT/summary.md"
