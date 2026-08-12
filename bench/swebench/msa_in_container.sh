#!/usr/bin/env bash
# Runs INSIDE a SWE-bench instance container: drives mini-swe-agent on the
# issue text in /testbed and exports the git diff, mirroring in_container.sh.
# Installed per-container from PyPI via the mounted uv (same pattern as the
# agent6 wheel); the OpenRouter key is read from the mounted secrets.toml,
# never printed. Cost tracking is litellm-based and lacks prices for new
# OpenRouter models, so it is relaxed and the bounds are wall clock plus a
# call limit.
set -uo pipefail
export HOME=/root
export PATH="/root/.local/bin:$PATH"
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export PYTHONUTF8=1

MODEL="${AGENT6_SB_MODEL:?set AGENT6_SB_MODEL}"
TIMEOUT_S="${AGENT6_SB_TIMEOUT:-1500}"
OPENROUTER_API_KEY=$(sed -n 's/.*api_key *= *"\(sk-or-[^"]*\)".*/\1/p' \
  /root/.config/agent6/secrets.toml | head -1)
export OPENROUTER_API_KEY
export MSWEA_CONFIGURED=true
export MSWEA_COST_TRACKING=ignore_errors
export MSWEA_GLOBAL_CALL_LIMIT="${AGENT6_SB_MSA_CALLS:-60}"

uv python install 3.14 >/dev/null 2>&1
uv tool install --python 3.14 mini-swe-agent >/dev/null 2>&1

cd /testbed
git config user.email "swebench@agent6" 2>/dev/null
git config user.name "msa" 2>/dev/null

# Same instance scaffolding the other legs plant: symmetric task materials.
if [ ! -f AGENTS.md ]; then
  cat > AGENTS.md <<'AEOF'
If the task matches a known public issue, still derive the fix from
this checkout: never spend turns recalling or fetching the canonical
upstream commit. Anything remembered about the upstream fix is an
unverified hint, not a source.
AEOF
  git add AGENTS.md && git commit -q -m "bench scaffolding" 2>/dev/null
fi
BASE=$(git rev-parse HEAD)

timeout "$TIMEOUT_S" mini -m "openrouter/${MODEL}" -y -l 0 \
  -t "$(cat /mnt/problem.txt)" > /out/msa.log 2>&1 \
  || echo "[msa_in_container] mini exited nonzero or timed out" >> /out/msa.log

git -C /testbed add -u
git -C /testbed diff --cached "$BASE" -- . ':(exclude).agent6' > /out/patch.diff
echo "[msa_in_container] patch lines: $(wc -l < /out/patch.diff)"
