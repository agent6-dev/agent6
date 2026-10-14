#!/usr/bin/env bash
# Exercise the two timing behaviors: `until` (deadline) and the `--exit-on-wait`
# persisted-wake driver (pulse).
# Pure wait/terminal machines, so this needs no provider keys.
#
# Usage:  bash bench/machines/wait-clock/run.sh [workdir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT6="$(cd "$HERE/../../.." && pwd)/.venv/bin/agent6"
WORK="${1:-/tmp/agent6-machine-wait}"

rm -rf "$WORK" "${WORK}-state"; mkdir -p "$WORK"
cp "$HERE"/*.asm.toml "$WORK/"
git -C "$WORK" init -q
# `machine run` only accepts a committed bundle.
git -C "$WORK" add -A
git -C "$WORK" -c user.email=b@b -c user.name=b commit -q -m init
# agent6 refuses a state dir inside the workspace.
export XDG_STATE_HOME="${WORK}-state"
cd "$WORK"

echo "== deadline: until a past instant fires immediately =="
"$AGENT6" machine run deadline.asm.toml

echo
echo "== pulse: --exit-on-wait persists the wake and exits, no blocking =="
"$AGENT6" machine run pulse.asm.toml --exit-on-wait
echo "-- status (note next wake) --"
"$AGENT6" machine status pulse
echo "-- resume immediately: still waiting (wake not reached) --"
"$AGENT6" machine run pulse.asm.toml --exit-on-wait
echo "-- wait out the interval, then resume: advances to done --"
sleep 5
"$AGENT6" machine run pulse.asm.toml --exit-on-wait
