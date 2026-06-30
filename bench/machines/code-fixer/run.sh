#!/usr/bin/env bash
# Run the code-fixer machine against a fresh copy of the seeded buggy repo.
#
# The machine bundle (code-fixer.asm.toml + scripts/) and the buggy source
# (seed/stats.py) are copied into a throwaway git repo so each run starts from
# the same failing state and the agent's edits never touch this checkout.
#
# Runs under the default sandbox config: tool states get their own network
# namespace (strict, from the jail launcher) and the agent confines its egress
# to the provider API. On a host that supports only the hardened profile, a
# tool state is refused under the default `sandbox.tool_network = "block"`, and
# agent6 prints the one-line config opt-in to apply (it never relaxes the
# sandbox unattended). Nothing here touches your global config.
#
# Usage:  bash bench/machines/code-fixer/run.sh [workdir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-code-fixer}"

rm -rf "$WORK"; mkdir -p "$WORK/scripts"
cp "$HERE/code-fixer.asm.toml" "$WORK/"
cp "$HERE/scripts/"*.py "$WORK/scripts/"
cp "$HERE/seed/stats.py" "$WORK/"
git -C "$WORK" init -q
git -C "$WORK" -c user.email=bench@bench -c user.name=bench add -A
git -C "$WORK" -c user.email=bench@bench -c user.name=bench commit -q -m "seed: buggy median"

# Keep all agent6 state (per-repo config + machine journal) inside the workspace
# so each run is hermetic and re-runnable.
export AGENT6_STATE_HOME="$WORK/.agent6-state"
# The mode="run" agent commits its fix. This host has no git identity at all, so
# give agent6 one to commit under (resolved on the host, exported into the
# confined agent which can't read ~/.gitconfig). A real repo with local or
# global git identity needs none of this.
(cd "$WORK" && "$AGENT6" config set git.commit.name "agent6 code-fixer" --repo >/dev/null)
(cd "$WORK" && "$AGENT6" config set git.commit.email "code-fixer@agent6.local" --repo >/dev/null)

echo "== before: verify reports failing =="
(cd "$WORK" && python3 scripts/verify.py)
echo
echo "== running code-fixer machine =="
(cd "$WORK" && "$AGENT6" machine run code-fixer.asm.toml)
echo
echo "== after: verify =="
(cd "$WORK" && python3 scripts/verify.py)
echo
echo "== agent's fix (git diff of stats.py) =="
git -C "$WORK" --no-pager diff -- stats.py 2>/dev/null
git -C "$WORK" --no-pager log --oneline -5
echo
echo "== machine status =="
(cd "$WORK" && "$AGENT6" machine status code-fixer)
