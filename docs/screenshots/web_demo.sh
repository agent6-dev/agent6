#!/usr/bin/env bash
# Record the web-UI tour videos (desktop + phone) for the docs site.
#
#   bash docs/screenshots/web_demo.sh
#
# Seeds the committed fixtures (docs/screenshots/seed/) into an isolated,
# temporary agent6 home, starts `agent6 web` against them, and drives it with
# Playwright (web_demo.py) to capture web-desktop.webm + web-phone.webm under
# docs/screenshots/out/ (gitignored). No live LLM calls, no network, no API key.
#
# Needs `agent6`, `python3` (agent6 importable), `ffmpeg` (trims the loading-screen
# head so each video opens on the hub), and a Playwright-capable Python with
# Chromium installed. Point $WEB_DEMO_PY at it, e.g.:
#   python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright \
#     && /tmp/pw/bin/playwright install chromium
#   WEB_DEMO_PY=/tmp/pw/bin/python bash docs/screenshots/web_demo.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/docs/screenshots/out"
PORT="${WEB_DEMO_PORT:-8987}"
PW_PY="${WEB_DEMO_PY:-python3}"
mkdir -p "$OUT"

for bin in agent6 python3 ffmpeg ffprobe; do
  command -v "$bin" >/dev/null 2>&1 || { echo "web_demo.sh: missing tool: $bin" >&2; exit 1; }
done
"$PW_PY" -c "from playwright.sync_api import sync_playwright" 2>/dev/null || {
  echo "web_demo.sh: \$WEB_DEMO_PY ($PW_PY) has no playwright; see the header." >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'kill "${SERVER:-0}" 2>/dev/null || true; rm -rf "$TMP"' EXIT
# The server requires a secret; web_demo.py signs in with this one, fresh for every run.
AGENT6_WEB_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export AGENT6_WEB_SECRET

# Record one tour: tour MODE OUT [web_demo.py flags]. A tour saves a file and commits it, so
# each one gets its own seeded home, clone and server.
tour() {
  local mode="$1" out="$2"; shift 2
  local home="$TMP/$mode"
  export XDG_CONFIG_HOME="$home/config"
  export XDG_STATE_HOME="$home/state"
  # The tours run in a clone of this checkout: the Git and Files pages show its history, and
  # the seeded runs' reviewed trees come from the stats demo repository's bundle.
  export AGENT6_DEMO_REPO="$home/demo"
  git clone -q --no-hardlinks "$ROOT" "$AGENT6_DEMO_REPO"  # /tmp is often another filesystem
  git -C "$AGENT6_DEMO_REPO" fetch -q "$ROOT/docs/screenshots/seed/stats-demo.bundle" \
    '+refs/heads/*:refs/screenshots/heads/*' '+refs/agent6/*:refs/screenshots/agent6/*'
  # The Git page commits under the repository's own identity.
  git -C "$AGENT6_DEMO_REPO" config user.name "agent6 demo"
  git -C "$AGENT6_DEMO_REPO" config user.email "demo@agent6.dev"
  # The pages the Files stop opens, untracked in the working tree.
  cp "$ROOT"/docs/screenshots/seed/pages/* "$AGENT6_DEMO_REPO/"

  echo "web_demo: seeding fixtures into $home"
  python3 docs/screenshots/seed.py >/dev/null

  echo "web_demo: starting agent6 web on :$PORT"
  ( cd "$AGENT6_DEMO_REPO" && exec agent6 web --host 127.0.0.1 --port "$PORT" ) > "$home/web.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
    sleep 0.25
  done
  curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 || {
    echo "web_demo.sh: agent6 web did not answer on :$PORT within 10s; its output:" >&2
    cat "$home/web.log" >&2
    exit 1
  }

  rm -f "$out"
  "$PW_PY" docs/screenshots/web_demo.py --url "http://127.0.0.1:$PORT" --out "$out" --mode "$mode" "$@"
  kill "$SERVER"
  wait "$SERVER" 2>/dev/null || true
  rm -rf "$OUT/_web_${mode}_raw"
  [ -s "$out" ] || { echo "web_demo.sh: failed to produce $out" >&2; exit 1; }
}

# The hero records the desktop tour at 1600x900 without the narration banner.
if [ "${1:-}" = "hero" ]; then
  tour hero "$OUT/hero-src-web.webm" --no-narration
  echo "web_demo: done -> $OUT/hero-src-web.webm"
  exit 0
fi

tour desktop "$OUT/web-desktop.webm"
tour phone "$OUT/web-phone.webm"
echo "web_demo: done -> $OUT/web-desktop.webm, $OUT/web-phone.webm"
