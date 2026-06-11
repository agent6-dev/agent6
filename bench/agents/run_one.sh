#!/usr/bin/env bash
# Run one agent on one task. Usage: run_one.sh <task> <agent>
# Tasks: go-logwindow | rust-ratelimit | go-kvstore-debug
# Agents: agent6 | aider | opencode | claude
set -u
TASK="$1"; AGENT="$2"
ROOT="${AGENTBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RUN="${AGENTBENCH_RUNS:-$HOME/agentbench-runs}/${TASK}__${AGENT}"
TPL="$ROOT/tasks/$TASK"
RESULTS="${AGENTBENCH_RUNS:-$HOME/agentbench-runs}/results.jsonl"
PROMPT="Read README.md and implement what it specifies so that ./verify.sh passes (it runs the test suite). Do not modify the test files or verify.sh. Iterate until the tests pass."
TIMEOUT=1500

export PATH="$HOME/.local/bin:$HOME/.opencode/bin:/usr/bin:$PATH"
OPENROUTER_API_KEY=$(python3 -c 'import tomllib,pathlib;print(tomllib.loads((pathlib.Path.home()/".config/agent6/secrets.toml").read_text())["providers"]["openrouter"]["api_key"])')
ANTHROPIC_API_KEY=$(python3 -c 'import tomllib,pathlib;print(tomllib.loads((pathlib.Path.home()/".config/agent6/secrets.toml").read_text())["providers"]["anthropic"]["api_key"])')

or_usage() {
  curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["usage"])' 2>/dev/null || echo 0
}

rm -rf "$RUN"; mkdir -p "$(dirname "$RUN")"
cp -r "$TPL" "$RUN"
cd "$RUN"
git config user.email bench@bench; git config user.name bench

case "$TASK" in
  go-logwindow) IMPL=logwindow.go ;;
  rust-ratelimit) IMPL=src/lib.rs ;;
  go-kvstore-debug) IMPL=kvstore.go ;;
esac

BEFORE=$(or_usage)
START=$(date +%s)
STATUS=0
case "$AGENT" in
  agent6)
    mkdir -p .agent6
    cat > .agent6/config.toml <<CFG
[workflow]
verify_command = ["./verify.sh"]
[sandbox]
run_commands = "yes"
CFG
    git add -A && git commit -qm "agent6 repo config"
    AGENT6_FORCE_STREAM=1 timeout $TIMEOUT "${AGENT6_BIN:-agent6}" run "$PROMPT" --max-usd 0.60 > agent.log 2>&1
    STATUS=$?
    ;;
  aider)
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" timeout $TIMEOUT aider \
      --model openrouter/moonshotai/kimi-k2.6 --yes-always --no-check-update \
      --auto-test --test-cmd ./verify.sh --message "$PROMPT" "$IMPL" > agent.log 2>&1
    STATUS=$?
    ;;
  opencode)
    OPENROUTER_API_KEY="$OPENROUTER_API_KEY" timeout $TIMEOUT opencode run \
      -m openrouter/moonshotai/kimi-k2.6 "$PROMPT" > agent.log 2>&1
    STATUS=$?
    ;;
  claude)
    export CLAUDE_CONFIG_DIR="$ROOT/claude-cfg"
    mkdir -p "$CLAUDE_CONFIG_DIR"
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" timeout $TIMEOUT claude -p "$PROMPT" \
      --model claude-haiku-4-5 --dangerously-skip-permissions \
      --output-format json > agent.log 2> agent.err
    STATUS=$?
    ;;
esac
WALL=$(( $(date +%s) - START ))
sleep 12
AFTER=$(or_usage)

# independent verdict
./verify.sh > verdict.log 2>&1
PASS=$([ $? -eq 0 ] && echo true || echo false)
# test-file tamper check vs the template's committed copy
TAMPER=false
case "$TASK" in
  go-logwindow) TESTF=logwindow_test.go ;;
  rust-ratelimit) TESTF=tests/ratelimit_test.rs ;;
  go-kvstore-debug) TESTF=kvstore_test.go ;;
esac
if ! cmp -s "$TPL/$TESTF" "$RUN/$TESTF"; then TAMPER=true; fi
if ! cmp -s "$TPL/verify.sh" "$RUN/verify.sh"; then TAMPER=true; fi

COST=$(python3 -c "print(round($AFTER - $BEFORE, 4))" 2>/dev/null || echo "?")
CLAUDE_COST=""
if [ "$AGENT" = "claude" ]; then
  CLAUDE_COST=$(python3 -c 'import json;print(round(json.load(open("agent.log")).get("total_cost_usd",0),4))' 2>/dev/null || echo "?")
  COST=$CLAUDE_COST
fi
printf '{"task":"%s","agent":"%s","pass":%s,"tampered":%s,"wall_secs":%d,"cost_usd":%s,"exit":%d}\n' \
  "$TASK" "$AGENT" "$PASS" "$TAMPER" "$WALL" "${COST:-0}" "$STATUS" | tee -a "$RESULTS"
