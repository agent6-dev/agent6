#!/usr/bin/env bash
# Web workflow tour video: a run starts three children from the browser, one is
# steered and one stopped, the tour switches between sessions, reads a reviewed
# run's findings and merges the result. The children are real runs: each one
# replays the committed temperature-fix cassette through llm_proxy.py. The other
# sessions are the seeded ones. Recorded with Playwright (web_demo.py --mode
# workflow) as out/web-workflow.webm, with its chapter track out/web-workflow.vtt
# and its poster out/web-workflow.png; no API key.
#
#   WEB_DEMO_PY=/tmp/pw/bin/python bash docs/screenshots/web_workflow_demo.sh
#
# Needs `agent6`, `python3` (agent6 importable), `ffmpeg`, and a
# Playwright-capable Python in $WEB_DEMO_PY (see web_demo.sh's header).
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
SS="$ROOT/docs/screenshots"
OUT="$SS/out"
PROXY_PORT=8907
WEB_PORT="${WEB_WORKFLOW_DEMO_PORT:-8991}"
CASSETTE="$SS/seed/temps-cassette.jsonl"
PW_PY="${WEB_DEMO_PY:-python3}"

[ -x "$ROOT/.venv/bin/agent6" ] && export PATH="$ROOT/.venv/bin:$PATH"
export AGENT6_FORCE_STREAM=1

for bin in agent6 python3 ffmpeg ffprobe; do
  command -v "$bin" >/dev/null 2>&1 || { echo "web_workflow_demo.sh: missing tool: $bin" >&2; exit 1; }
done
"$PW_PY" -c "from playwright.sync_api import sync_playwright" 2>/dev/null || {
  echo "web_workflow_demo.sh: \$WEB_DEMO_PY ($PW_PY) has no playwright; see web_demo.sh." >&2; exit 1; }

# The children's commands run jailed; the jail needs unprivileged userns
# (Ubuntu 24.04 AppArmor blocks it by default).
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 || true

TMP="$(mktemp -d)"
# A tour that fails midway leaves its run and children live: stop them first.
trap '( cd "$AGENT6_DEMO_REPO" && agent6 stop --all ) >/dev/null 2>&1 || true
  kill "${PROXY_PID:-0}" "${SERVER:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT
export XDG_CONFIG_HOME="$TMP/config"
export XDG_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="$TMP/acme-stats"
python3 -m docs.screenshots.workflow_repo "$PROXY_PORT"

echo "web workflow demo: REPLAY proxy on :$PROXY_PORT <- $CASSETTE"
# One replay per conversation, so each child gets the whole cassette. 250ms
# chunks keep a child working about 45s: long enough to be messaged before its
# last turn and stopped on camera.
AGENT6_PROXY_MODE=replay AGENT6_PROXY_CASSETTE="$CASSETTE" AGENT6_PROXY_PORT="$PROXY_PORT" \
  AGENT6_PROXY_PER_CONVERSATION=1 AGENT6_PROXY_CHUNK_MS=250 python3 "$SS/llm_proxy.py" & PROXY_PID=$!
sleep 1

echo "web workflow demo: starting agent6 web on :$WEB_PORT"
# The server requires a secret; web_demo.py signs in with this one, fresh for every tour.
AGENT6_WEB_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export AGENT6_WEB_SECRET
# Tree-sitter grammars stay in the caller's cache home: an empty one downloads each at first use.
export TREE_SITTER_LANGUAGE_PACK_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
# Child clones live under the cache home. Only the server gets a throwaway one: Playwright
# looks for its browser under the real one.
( cd "$AGENT6_DEMO_REPO" && XDG_CACHE_HOME="$TMP/cache" exec agent6 web --host 127.0.0.1 --port "$WEB_PORT" ) \
  > "$TMP/web.log" 2>&1 &
SERVER=$!
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1 && break
  sleep 0.25
done
curl -sf "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1 || {
  echo "web_workflow_demo.sh: agent6 web did not answer on :$WEB_PORT within 10s; its output:" >&2
  cat "$TMP/web.log" >&2
  exit 1
}

mkdir -p "$OUT"
rm -f "$OUT"/web-workflow.webm "$OUT"/web-workflow.vtt "$OUT"/web-workflow.png
"$PW_PY" "$SS/web_demo.py" --url "http://127.0.0.1:$WEB_PORT" --out "$OUT/web-workflow.webm" --mode workflow
rm -rf "$OUT"/_web_workflow_raw
for f in web-workflow.webm web-workflow.vtt web-workflow.png; do
  [ -s "$OUT/$f" ] || { echo "web_workflow_demo.sh: failed to produce $f" >&2; exit 1; }
done
echo "web workflow demo: done -> $OUT/web-workflow.webm"
