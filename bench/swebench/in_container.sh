#!/usr/bin/env bash
# Runs INSIDE a pulled SWE-bench instance container. Installs agent6 (uv-managed
# Python 3.14 + the mounted wheel — no Rust toolchain needed), points it at the
# worker model, runs it on the issue text in /testbed, and writes the resulting
# git diff to /out/patch.diff for SWE-bench's evaluator. The container is the
# isolation boundary; agent6 runs hardened inside it (no privileged, no userns),
# with the repo's conda env granted read+exec via sandbox.extra_read_paths.
set -uo pipefail
export HOME=/root
export PATH="/root/.local/bin:$PATH"
# Force a UTF-8 locale: the SWE-bench images default to ASCII (C/POSIX), so the
# conda python 3.6 launcher below would UnicodeEncodeError when the issue text
# contains a non-ASCII char (e.g. a zero-width space) passed as a subprocess
# argv -- crashing BEFORE agent6 starts and silently yielding an empty patch.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export PYTHONUTF8=1
export AGENT6_STATE_HOME=/root/a6state   # keep agent6's run state OUT of /testbed
export AGENT6_FORCE_STREAM=1             # OpenRouter SSE heartbeat-safe path
export AGENT6_ALLOW_ROOT=1               # SWE-bench images run as root; the container IS the boundary

MODEL="${AGENT6_SB_MODEL:?set AGENT6_SB_MODEL}"
MAX_USD="${AGENT6_SB_MAX_USD:-3.0}"
TIMEOUT_S="${AGENT6_SB_TIMEOUT:-1500}"
WHL=$(ls /mnt/wheel/*.whl | head -1)

# agent6's best_effort_usd_limit is a no-op for providers it can't price
# (Anthropic returns no pricing), so we ALSO bound spend by token caps derived
# from MAX_USD x list price (85% input / 15% output, SWE-bench is input-heavy).
# This is the real enforcer for unpriced providers and a backstop for the rest.
case "$MODEL" in
  claude-opus-*)    IN_PRICE=5 ; OUT_PRICE=25 ;;
  claude-sonnet-*)  IN_PRICE=3 ; OUT_PRICE=15 ;;
  moonshotai/kimi*) IN_PRICE=0.66 ; OUT_PRICE=3.41 ;;
  z-ai/glm*)        IN_PRICE=0.98 ; OUT_PRICE=3.08 ;;
  qwen/*)           IN_PRICE=0.29 ; OUT_PRICE=3.17 ;;
  deepseek/*)       IN_PRICE=0.09 ; OUT_PRICE=0.18 ;;
  *)                IN_PRICE=1 ; OUT_PRICE=5 ;;
esac
MAX_IN=$(/opt/miniconda3/envs/testbed/bin/python -c "print(max(50000,int($MAX_USD*0.85/$IN_PRICE*1e6)))")
MAX_OUT=$(/opt/miniconda3/envs/testbed/bin/python -c "print(max(8000,int($MAX_USD*0.15/$OUT_PRICE*1e6)))")

uv python install 3.14 >/dev/null 2>&1
uv tool install --python 3.14 "$WHL" >/dev/null 2>&1

# Provider is chosen from the model slug: claude-* -> Anthropic (key from
# secrets.toml [providers.anthropic]); everything else -> OpenRouter.
if [[ "$MODEL" == claude-* ]]; then
  PROVIDER=anthropic
  PROVIDER_BLOCK='[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true'
else
  PROVIDER=openrouter
  PROVIDER_BLOCK='[providers.openrouter]
api_format = "openai"
api_key_env = "OPENROUTER_API_KEY"
base_url = "https://openrouter.ai/api/v1"
extra_headers = { "HTTP-Referer" = "https://github.com/elesiuta/agent6", "X-Title" = "agent6-swebench" }'
fi

# Optional review panel (Fugu dimension). AGENT6_SB_REVIEW_SEATS is a
# semicolon-separated list of "persona@provider/model" seats; when set the panel
# reviews before finish_run and gates per AGENT6_SB_REVIEW_DECISION (default
# quorum). Same-model vs distinct-model panels are just different seat lists.
REVIEW_LINES=""
if [ -n "${AGENT6_SB_REVIEW_SEATS:-}" ]; then
  ARR=""
  IFS=';' read -ra _SEATS <<< "$AGENT6_SB_REVIEW_SEATS"
  for s in "${_SEATS[@]}"; do ARR="${ARR}\"${s}\", "; done
  REVIEW_LINES="[review]
trigger = \"before_finish\"
decision = \"${AGENT6_SB_REVIEW_DECISION:-quorum}\"
quorum = ${AGENT6_SB_REVIEW_QUORUM:-2}
tier = \"diff\"
seats = [${ARR%, }]"
fi

# Verify command. The jail forces child PATH=/usr/bin:/bin, so a bare `python3`
# won't resolve the container's conda interpreter; use its ABSOLUTE path (exec is
# granted via sandbox.extra_read_paths). Auto-detect django's runner; default to
# pytest. Override with AGENT6_SB_VERIFY (space-separated argv) for odd repos.
CONDA_PY=$(ls /opt/miniconda3/envs/*/bin/python 2>/dev/null | head -1)
CONDA_PY="${CONDA_PY:-python3}"
if [ -z "${AGENT6_SB_VERIFY:-}" ]; then
  if [ -f /testbed/tests/runtests.py ]; then
    AGENT6_SB_VERIFY="$CONDA_PY tests/runtests.py --verbosity 1"
  else
    AGENT6_SB_VERIFY="$CONDA_PY -m pytest -q"
  fi
fi
VARR=""
for w in $AGENT6_SB_VERIFY; do VARR="${VARR}\"${w}\", "; done
VERIFY_TOML="verify_command = [${VARR%, }]"

cat > /root/agent6.toml <<EOF
[agent6]
config_version = 1

$PROVIDER_BLOCK

[models.worker]
provider = "$PROVIDER"
model = "$MODEL"

[models.reviewer]
provider = "$PROVIDER"
model = "$MODEL"

[sandbox]
# UNSANDBOXED: the container is the isolation. agent6's jail fights the container
# here (couldn't exec the conda interpreter under hardened/strict), so we opt out
# of the kernel sandbox entirely -- the standard SWE-bench setup where Docker is
# the boundary. profile="none" is allowed because in_container is detected; on a
# bare host it would refuse without AGENT6_ALLOW_NO_SANDBOX=1.
profile = "none"
agent_network = "providers"
tool_network = "block"
run_commands = "yes"
protect_git = false
extra_read_paths = ["/opt/miniconda3"]

[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = false
commit_strategy = "none"
allow_push = false
allow_force = false
allow_history_rewrite = false

[workflow]
$VERIFY_TOML

[prompt]
revise_prompt = "off"
structural_priors = ${AGENT6_SB_STRUCTURAL_PRIORS:-true}

$REVIEW_LINES

[budget]
max_input_tokens = $MAX_IN
max_output_tokens = $MAX_OUT
best_effort_usd_limit = $MAX_USD
EOF

cd /testbed
git config user.email "swebench@agent6" 2>/dev/null
git config user.name "agent6" 2>/dev/null
BASE=$(git rev-parse HEAD)

# Pass the (long, special-char-laden) issue text as a single argv via Python so
# no shell quoting can corrupt it.
AGENT6_SB_TIMEOUT="$TIMEOUT_S" /opt/miniconda3/envs/testbed/bin/python - <<'PYEOF'
import os, subprocess
problem = open("/mnt/problem.txt", encoding="utf-8").read()
try:
    subprocess.run(
        ["agent6", "--config", "/root/agent6.toml", "run", problem],
        cwd="/testbed", timeout=float(os.environ.get("AGENT6_SB_TIMEOUT", "1500")),
    )
except subprocess.TimeoutExpired:
    print("[in_container] agent6 run timed out")
PYEOF

# The model's patch = everything changed under /testbed vs the base commit,
# excluding agent6's own artifacts (which live outside /testbed anyway).
mkdir -p /out
git -C /testbed add -A
git -C /testbed diff --cached "$BASE" -- . ':(exclude).agent6' ':(exclude)agent6.toml' \
    > /out/patch.diff
echo "[in_container] patch lines: $(wc -l < /out/patch.diff)"
