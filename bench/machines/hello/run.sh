#!/usr/bin/env bash
# The smallest useful machine: one `agent` state that returns a structured
# greeting, then a terminal. No tools, ~1 cent on anthropic.
#
# Usage:  bash bench/machines/hello/run.sh [workdir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-hello}"
rm -rf "$WORK" "${WORK}-state"; mkdir -p "$WORK"
cp "$HERE/hello.asm.toml" "$WORK/"
git -C "$WORK" init -q
# `machine run` only accepts a committed bundle.
git -C "$WORK" add -A
git -C "$WORK" -c user.email=b@b -c user.name=b commit -q -m init
# agent6 refuses a state dir inside the workspace.
export XDG_STATE_HOME="${WORK}-state"
(cd "$WORK" && "$AGENT6" machine run hello.asm.toml)
(cd "$WORK" && "$AGENT6" machine status hello)
