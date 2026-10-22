#!/usr/bin/env bash
# Real-world benchmark harness for agent6.
#
# Loads each task JSON under bench/realworld/tasks/, clones the named OSS
# repo shallowly at the pinned commit, sets up a per-task venv, applies the
# breakage, then runs `agent6 run` and records cost / wall / verify-pass.
#
# Toolset comparison: set AGENT6_REALWORLD_TOOLSET to "index" (default: the
# symbol-tool surface) or "baseline" (no symbol tools at all — the
# rg-via-run_command floor) and run once per arm; the results filename
# includes the toolset. The arms map onto the AGENT6_SYMBOL_TOOLS switch the
# dispatcher reads.
#
# Usage:
#   bash bench/realworld/run_realworld.sh
#   AGENT6_REALWORLD_MODEL=provider/model bash bench/realworld/run_realworld.sh
#
# The worker is the operator's configured one unless AGENT6_REALWORLD_MODEL names another.
#
# Outputs: $BENCH_ROOT/results_<toolset>.md
#          $BENCH_ROOT/_logs/<task>_<toolset>/result.json

set -euo pipefail

REPO="${AGENT6_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_ROOT=${BENCH_ROOT:-/tmp/agent6-realworld}
TOOLSET=${AGENT6_REALWORLD_TOOLSET:-index}
PER_TASK_BUDGET_USD=${PER_TASK_BUDGET_USD:-1.00}
# Plan points one task may use; unset keeps the operator's [budget].max_plan_points.
PER_TASK_MAX_PLAN_POINTS=${PER_TASK_MAX_PLAN_POINTS:-}
# [PROVIDER/]MODEL for `run --model`; unset runs the operator's configured worker.
MODEL=${AGENT6_REALWORLD_MODEL:-}
# Wall-clock bound on one task's `agent6 run`.
TASK_TIMEOUT_S=${AGENT6_REALWORLD_TASK_TIMEOUT_S:-3600}
# Default: python3 on PATH. Use PYTHON_BIN to match the task's pinned pytest.
# The interpreter must be visible in the sandbox, such as /usr/bin/python3.
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
export PYTHON_BIN

cd "$REPO"
AGENT6_BIN="$REPO/.venv/bin/agent6"
[ -x "$AGENT6_BIN" ] || { echo "agent6 not found at $AGENT6_BIN, run 'uv sync' in $REPO first" >&2; exit 1; }
# Nobody answers an approval here: deny it rather than park the run.
export AGENT6_DETACHED_AWAY="${AGENT6_DETACHED_AWAY:-deny}"

mkdir -p "$BENCH_ROOT"

# --- shared agent6.toml --------------------------------------------------------
emit_toml() {
  local verify_cmd_json="$1"
  local metric_block="${2:-}"
  cat <<EOF
[agent6]
config_version = 1

[sandbox]
isolation = "auto"
network = "session"
run_commands = "yes"
protect_git = "on"

[git]
dirty_tree = "ask"
branch_per_run = true

[harness]
verify_command = $verify_cmd_json

[review]
# Optional critic-in-loop. Set AGENT6_REALWORLD_CRITIC=before_finish (or
# on_verify_fail / periodic) to evaluate critic modes against the
# premature-finish task variants. Default: "off".
trigger = "${AGENT6_REALWORLD_CRITIC:-off}"

[prompt]
revise_prompt = "${AGENT6_REALWORLD_REVISE_PROMPT:-off}"
${metric_block}

[budget]
max_usd = $PER_TASK_BUDGET_USD
max_tokens_fallback = 1500000
$(if [ -n "$PER_TASK_MAX_PLAN_POINTS" ]; then echo "max_plan_points = $PER_TASK_MAX_PLAN_POINTS"; fi)
EOF
}

setup_task() {
  local task_json="$1" dir="$2"
  python3 - "$task_json" "$dir" <<'PY'
import json, os, subprocess, sys
from pathlib import Path

task = json.loads(Path(sys.argv[1]).read_text())
dir = Path(sys.argv[2]).absolute()

if dir.exists():
    import shutil
    shutil.rmtree(dir)
dir.mkdir(parents=True)

# self-contained tasks may declare a "files" mapping {relpath: content}
# instead of cloning an upstream repo. Useful for premature-finish task
# variants where we author the entire scaffold ourselves.
if "repo_url" in task:
    # Shallow clone at the pinned tag/commit. --branch accepts both tags and branch names.
    subprocess.run(
        # Depth 200 so the run's recent-commit context has history to show
        # (a depth-1 clone shows none); it adds <2MB for click-class repos.
        ["git", "clone", "--depth", "200", "--branch", task["commit"], "--quiet",
         task["repo_url"], str(dir)],
        check=True,
    )
else:
    subprocess.run(["git", "init", "--quiet", str(dir)], check=True)
    for relpath, content in task.get("files", {}).items():
        p = dir / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

# Configure git identity (clone leaves the directory bare of user config).
subprocess.run(["git", "-C", str(dir), "config", "user.email", "bench@agent6"], check=True)
subprocess.run(["git", "-C", str(dir), "config", "user.name", "bench"], check=True)

# Initial commit for self-contained tasks (so subsequent `break` apply on a
# committed worktree, same shape as the clone path).
if "repo_url" not in task:
    subprocess.run(["git", "-C", str(dir), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(dir), "commit", "-q", "-m", "bench: seed"], check=True
    )

# Install via the requested argv list. Each entry runs in the task dir.
# Any "python3" token in the argv is replaced by $PYTHON_BIN so tasks can be
# version-pinned externally (older fixtures vs new system Python).
python_bin = os.environ.get("PYTHON_BIN", "python3")
for argv in task.get("install", []):
    argv = [python_bin if a == "python3" else a for a in argv]
    subprocess.run(argv, cwd=dir, check=True)

# Apply breakages.
for b in task.get("break", []):
    p = dir / b["path"]
    src = p.read_text()
    if b["find"] not in src:
        raise SystemExit(f"break.find not present in {b['path']}; check the task definition")
    if src.count(b["find"]) != 1:
        raise SystemExit(f"break.find not unique in {b['path']}; refine the task definition")
    p.write_text(src.replace(b["find"], b["replace"]))

# Commit the broken state so agent6's dirty-tree question does not fire.
# Also pick up any pre-staged files (agent6.toml, TASK.md, .gitignore tweaks).
# Skip when there's nothing staged (self-contained tasks with no
# `break` entries — the seed commit already captured everything).
subprocess.run(["git", "-C", str(dir), "add", "-A"], check=True)
_dirty = subprocess.run(
    ["git", "-C", str(dir), "diff", "--cached", "--quiet"]
).returncode
if _dirty != 0:
    subprocess.run(
        ["git", "-C", str(dir), "commit", "-q", "-m", "bench: introduce gap"],
        check=True,
    )

# Write TASK.md (after the commit so the agent doesn't blame TASK.md changes
# on itself; .agent6/ is auto-added to .gitignore by agent6 on first run).
(dir / "TASK.md").write_text(task["task_md"])
# The fixture repos ship no AGENTS.md; this one states the task's ground rules.
if not (dir / "AGENTS.md").exists():
    (dir / "AGENTS.md").write_text(
        "# AGENTS.md (bench harness)\n\n"
        "This is a real OSS repository checked out at a pinned tag for the\n"
        "agent6 realworld benchmark. A small piece of working code has been\n"
        "replaced with `raise NotImplementedError(\"agent6-realworld: restore "
        "this implementation\")`.\n\n"
        "Your job is to restore the original implementation so the configured\n"
        "verify_command passes. The verify_command runs only the test(s) that\n"
        "exercise the broken function — keep your change minimal and do not\n"
        "modify test files.\n"
    )
# And re-commit TASK.md so the worktree is clean when agent6 starts.
subprocess.run(["git", "-C", str(dir), "add", "-A"], check=True)
subprocess.run(["git", "-C", str(dir), "commit", "-q", "-m", "bench: add TASK.md"], check=True)
PY
}

# Print one field of the `sessions show --json` capture, or nothing when it is absent.
session_field() {
  python3 -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" \
    "$1" "$2" 2>/dev/null || true
}

# ----- main loop --------------------------------------------------------------

case "$TOOLSET" in
  index)
    unset AGENT6_SYMBOL_TOOLS
    echo "Toolset: INDEX (symbol tools on)"
    ;;
  baseline)
    export AGENT6_SYMBOL_TOOLS=none
    echo "Toolset: BASELINE (no symbol tools)"
    ;;
  *)
    echo "unknown AGENT6_REALWORLD_TOOLSET: $TOOLSET (index|baseline)" >&2
    exit 2
    ;;
esac

total_cost=0
total_wall=0
total_in=0
total_out=0
pass=0
total=0
unscored_n=0
run_model=""
results_file="$BENCH_ROOT/results_${TOOLSET}.md"

for task_json in "$REPO"/bench/realworld/tasks/*.json; do
  name=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$task_json")
  # Optional filter: only run tasks whose name matches the substring in
  # AGENT6_REALWORLD_TASK_FILTER (or list, comma-separated). Useful for
  # iterating on a single regression without paying for the whole matrix.
  if [ -n "${AGENT6_REALWORLD_TASK_FILTER:-}" ]; then
    match=0
    IFS=',' read -ra _filters <<< "$AGENT6_REALWORLD_TASK_FILTER"
    for f in "${_filters[@]}"; do
      case "$name" in *"$f"*) match=1; break ;; esac
    done
    [ "$match" = 1 ] || { echo "[skip] $name (not in TASK_FILTER)"; continue; }
  fi
  desc=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['description'])" "$task_json")
  verify_cmd_json=$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['verify_command']))" "$task_json")
  # Optional metric block. Task JSON may declare a `metric` object
  # with {command: argv, pattern: regex, goal: 'minimize' | 'maximize'}.
  # When present we emit a [harness.metric] block so the agent's
  # `run_metric_command` tool is wired up, AND we run the metric ourselves
  # post-run as an independent score (so the harness number isn't subject
  # to whatever the agent did or didn't measure).
  metric_block=$(python3 - "$task_json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))
m = t.get("metric")
if not m:
    print("")
    raise SystemExit(0)
cmd_toml = "[" + ", ".join(json.dumps(a) for a in m["command"]) + "]"
print(
    "\n[harness.metric]\n"
    f"command = {cmd_toml}\n"
    f"pattern = {json.dumps(m['pattern'])}\n"
    f"goal = {json.dumps(m['goal'])}\n"
)
PY
)
  metric_cmd_json=$(python3 -c "import json,sys; t=json.load(open(sys.argv[1])); m=t.get('metric'); print(json.dumps(m['command']) if m else '')" "$task_json")
  metric_pattern=$(python3 -c "import json,sys; t=json.load(open(sys.argv[1])); m=t.get('metric'); print(m['pattern'] if m else '')" "$task_json")
  metric_goal=$(python3 -c "import json,sys; t=json.load(open(sys.argv[1])); m=t.get('metric'); print(m['goal'] if m else '')" "$task_json")
  dir="$BENCH_ROOT/${name}_${TOOLSET}"

  echo
  echo "================================================================"
  echo "TASK: $name [$TOOLSET]"
  echo "  $desc"
  echo "================================================================"

  # A setup failure is unscored; keep the other tasks' results and continue.
  set +e
  setup_task "$task_json" "$dir"
  setup_exit=$?
  set -e
  if [ "$setup_exit" -ne 0 ]; then
    echo "  SETUP FAILED (exit $setup_exit), skipping $name" >&2
    logdir="$BENCH_ROOT/_logs/${name}_${TOOLSET}"
    mkdir -p "$logdir"
    cat > "$logdir/result.json" <<EOF
{
  "task": "$name",
  "toolset": "$TOOLSET",
  "unscored": "setup failed (exit $setup_exit)",
  "agent_exit": null,
  "wall_seconds": 0,
  "verify_pass": false,
  "metric_score": null,
  "metric_goal": "$metric_goal",
  "cost_usd": 0,
  "input_tokens": 0,
  "output_tokens": 0,
  "plan_points": 0
}
EOF
    unscored_n=$((unscored_n + 1))
    echo "  agent_exit=-  verify=UNSCORED (setup failed, exit $setup_exit)  metric=-  wall=0s  cost=\$0"
    continue
  fi
  emit_toml "$verify_cmd_json" "$metric_block" > "$dir/agent6.toml"
  # Fail on an invalid config before any provider call:
  # `config show` loads the same merged layers the run will.
  ( cd "$dir" && "$AGENT6_BIN" --config agent6.toml config show >/dev/null ) || exit 1
  # Ignore the untracked agent6.toml so it doesn't dirty the tree.
  # Ensure a trailing newline before appending (some upstream .gitignores end
  # without one, which would join our pattern to the previous line).
  [ -f "$dir/.gitignore" ] && [ -s "$dir/.gitignore" ] && \
    [ "$(tail -c1 "$dir/.gitignore" | wc -l)" = "0" ] && echo "" >> "$dir/.gitignore"
  echo "agent6.toml" >> "$dir/.gitignore"
  ( cd "$dir" && git add .gitignore && git commit -q -m "bench: ignore agent6.toml" )
  # Harness side-channel files (stdout/stderr/result.json) live OUTSIDE the
  # task dir so they don't pollute the agent's git worktree.
  logdir="$BENCH_ROOT/_logs/${name}_${TOOLSET}"
  mkdir -p "$logdir"

  task_text=$(cat "$dir/TASK.md")

  model_args=()
  if [ -n "$MODEL" ]; then model_args=(--model "$MODEL"); fi
  start_ns=$(date +%s%N)
  set +e
  ( cd "$dir" && timeout --kill-after=30 "$TASK_TIMEOUT_S" \
      "$AGENT6_BIN" --config agent6.toml run ${model_args[@]+"${model_args[@]}"} "$task_text" ) \
    > "$logdir/agent6.stdout" 2> "$logdir/agent6.stderr"
  ag_exit=$?
  set -e
  end_ns=$(date +%s%N)
  wall_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.1f", (e-s)/1e9}')

  # Final-state verify (independent of whatever agent6's last verify said).
  set +e
  ( cd "$dir" && eval "$(python3 -c "import json,sys; print(' '.join(json.dumps(a) for a in json.loads(sys.argv[1])))" "$verify_cmd_json")" ) \
    > "$logdir/final_verify.txt" 2>&1
  ver_exit=$?
  set -e

  # Independent metric score, if the task declared one. The agent
  # may or may not have driven `run_metric_command`; this re-runs the same
  # argv post-finish and parses the score out via the task's pattern.
  metric_score="null"
  if [ -n "$metric_cmd_json" ]; then
    set +e
    ( cd "$dir" && eval "$(python3 -c "import json,sys; print(' '.join(json.dumps(a) for a in json.loads(sys.argv[1])))" "$metric_cmd_json")" ) \
      > "$logdir/final_metric.txt" 2>&1
    set -e
    metric_score=$(python3 - "$logdir/final_metric.txt" "$metric_pattern" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
m = re.search(sys.argv[2], txt)
print(m.group(1) if m else "null")
PY
)
    [ -z "$metric_score" ] && metric_score="null"
  fi

  # Pull cost/tokens out of the budget summary that agent6 prints.
  # The end-of-run summary may go to stdout or stderr depending on path; check both.
  combined="$logdir/agent6.stdout $logdir/agent6.stderr"
  # `|| true` on each: under `set -e -o pipefail` a grep that matches nothing
  # exits 1 and kills the whole (often multi-hour) run AFTER the task already
  # passed, and the regex-validate fallbacks below never get to run. The
  # summary format is allowed to drift; the numbers just fall back to 0.
  cost=$( { cat $combined 2>/dev/null | grep -oE 'cost[~=]\$[0-9.]+' | tail -1 | sed 's/cost[~=]\$//'; } || true)
  [[ "$cost" =~ ^[0-9]+(\.[0-9]+)?$ ]] || cost=0
  in_tok=$( { cat $combined 2>/dev/null | grep -oE 'TOTAL: in=[0-9]+' | tail -1 | sed 's/TOTAL: in=//'; } || true)
  [[ "$in_tok" =~ ^[0-9]+$ ]] || in_tok=0
  out_tok=$( { cat $combined 2>/dev/null | grep -oE 'TOTAL: in=[0-9]+ out=[0-9]+' | tail -1 | sed 's/.* out=//'; } || true)
  [[ "$out_tok" =~ ^[0-9]+$ ]] || out_tok=0

  # The session's end reason separates provider failures from agent failures.
  # Plan points are what a subscription run costs.
  sid=$( { cat $combined 2>/dev/null | grep -oE 'session id: [^ ]+' | head -1 | sed 's/session id: //'; } || true)
  : > "$logdir/session.json"
  if [ -n "$sid" ]; then
    ( cd "$dir" && "$AGENT6_BIN" --config agent6.toml sessions show "$sid" --json ) \
      > "$logdir/session.json" 2>/dev/null || true
  fi
  end_reason=$(session_field "$logdir/session.json" reason)
  plan_points=$(session_field "$logdir/session.json" plan_consumed)
  [[ "$plan_points" =~ ^[0-9]+(\.[0-9]+)?$ ]] || plan_points=0
  model_ran=$(session_field "$logdir/session.json" model)
  # The header names the last model that ran.
  if [ -n "$model_ran" ]; then run_model="$model_ran"; fi

  # A verdict needs a run that ended on agent6's own terms.
  unscored=""
  case "$ag_exit" in
    0|3|4|5) ;;
    1) case "$end_reason" in
         provider_error) unscored="provider error" ;;
         verify_command_unexecutable) unscored="verify command cannot run in the jail" ;;
         "") unscored="agent6 exit 1, end reason unread" ;;
       esac ;;
    2) unscored="agent6 refused to start (exit 2)" ;;
    124|137) unscored="timeout after ${TASK_TIMEOUT_S}s" ;;
    *) unscored="agent6 exit $ag_exit" ;;
  esac
  if [ -n "$unscored" ]; then verdict="UNSCORED ($unscored)"
  elif [ $ver_exit -eq 0 ]; then verdict=PASS
  else verdict=FAIL; fi
  unscored_json=null
  if [ -n "$unscored" ]; then unscored_json="\"$unscored\""; fi

  cat > "$logdir/result.json" <<EOF
{
  "task": "$name",
  "toolset": "$TOOLSET",
  "agent_exit": $ag_exit,
  "session_id": "$sid",
  "model": "$model_ran",
  "end_reason": "$end_reason",
  "unscored": $unscored_json,
  "wall_seconds": $wall_s,
  "verify_pass": $([ $ver_exit -eq 0 ] && echo true || echo false),
  "metric_score": $metric_score,
  "metric_goal": "$metric_goal",
  "cost_usd": $cost,
  "plan_points": $plan_points,
  "input_tokens": $in_tok,
  "output_tokens": $out_tok
}
EOF
  metric_disp="-"
  [ "$metric_score" != "null" ] && metric_disp="$metric_score"
  echo "  agent_exit=$ag_exit  verify=$verdict  metric=$metric_disp  wall=${wall_s}s  cost=\$${cost}  plan=${plan_points}pt  in=${in_tok}  out=${out_tok}"

  total_cost=$(awk -v a="$total_cost" -v b="$cost" 'BEGIN{printf "%.4f", a+b}')
  total_wall=$(awk -v a="$total_wall" -v b="$wall_s" 'BEGIN{printf "%.1f", a+b}')
  total_in=$((total_in + in_tok))
  total_out=$((total_out + out_tok))
  if [ -n "$unscored" ]; then
    unscored_n=$((unscored_n + 1))
  else
    total=$((total + 1))
    if [ $ver_exit -eq 0 ]; then pass=$((pass + 1)); fi
  fi
done

{
  echo "# agent6 real-world benchmark — toolset: $TOOLSET"
  echo
  echo "Model: ${run_model:-${MODEL:-operator config}}. Per-task caps: \$$PER_TASK_BUDGET_USD metered, ${PER_TASK_MAX_PLAN_POINTS:-operator config} plan points, ${TASK_TIMEOUT_S}s wall."
  echo
  printf "| Task | verify | wall | cost | plan pts | in toks | out toks |\n"
  printf "|------|--------|------|------|----------|---------|----------|\n"
  for d in "$BENCH_ROOT"/_logs/*_"${TOOLSET}"/; do
    [ -f "$d/result.json" ] || continue
    python3 -c "
import json
r = json.load(open('$d/result.json'))
verify = f'UNSCORED: {r[\"unscored\"]}' if r.get('unscored') else ('PASS' if r['verify_pass'] else 'FAIL')
print(f'| {r[\"task\"]} | {verify} | {r[\"wall_seconds\"]:.1f}s | \${r[\"cost_usd\"]:.4f} | {r.get(\"plan_points\", 0):g} | {r[\"input_tokens\"]} | {r[\"output_tokens\"]} |')
"
  done
  echo
  echo "**Total: $pass/$total verify pass, $unscored_n unscored, \$$total_cost, ${total_wall}s wall, in=${total_in} out=${total_out} tokens.**"
} > "$results_file"

echo
echo "Summary written to $results_file"
echo "Total: $pass/$total PASS, $unscored_n unscored, \$$total_cost, ${total_wall}s wall"
