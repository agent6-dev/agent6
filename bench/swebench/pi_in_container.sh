#!/usr/bin/env bash
# Runs INSIDE a SWE-bench instance container: drives the pi coding agent
# (badlogic/pi-mono) on the issue text in /testbed and exports the git diff,
# mirroring in_container.sh so predictions are extracted identically. Node and
# the pi install are mounted read-only by the orchestrator (--container-script
# + --extra-mount); the OpenRouter key is read from the same mounted
# secrets.toml the agent6 leg uses, never printed.
set -uo pipefail
export HOME=/root
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export PYTHONUTF8=1
export PATH="/mnt/node/bin:/mnt/pi/bin:$PATH"

MODEL="${AGENT6_SB_MODEL:?set AGENT6_SB_MODEL}"
TIMEOUT_S="${AGENT6_SB_TIMEOUT:-1500}"
OPENROUTER_API_KEY=$(sed -n 's/.*api_key *= *"\(sk-or-[^"]*\)".*/\1/p' \
  /root/.config/agent6/secrets.toml | head -1)
export OPENROUTER_API_KEY

cd /testbed
git config user.email "swebench@agent6" 2>/dev/null
git config user.name "pi" 2>/dev/null
BASE=$(git rev-parse HEAD)

# Same instance scaffolding the agent6 leg plants: symmetric task materials.
if [ ! -f AGENTS.md ]; then
  cat > AGENTS.md <<'AEOF'
If the task matches a known public issue, still derive the fix from
this checkout: never spend turns recalling or fetching the canonical
upstream commit. Anything remembered about the upstream fix is an
unverified hint, not a source.
AEOF
fi

timeout "$TIMEOUT_S" pi -p --no-session --provider openrouter --model "$MODEL" \
  "$(cat /mnt/problem.txt)" > /out/pi.log 2>&1 \
  || echo "[pi_in_container] pi exited nonzero or timed out" >> /out/pi.log

git -C /testbed add -u
git -C /testbed diff --cached "$BASE" -- . ':(exclude).agent6' > /out/patch.diff
echo "[pi_in_container] patch lines: $(wc -l < /out/patch.diff)"
