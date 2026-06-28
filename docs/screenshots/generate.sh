#!/usr/bin/env bash
# Generate the TUI screenshots + tour video for the docs site.
#
#   bash docs/screenshots/generate.sh
#
# Seeds committed fixtures (docs/screenshots/seed/) into an isolated, temporary
# agent6 home, then drives the TUI with vhs (docs/screenshots/tour.tape) to
# capture PNGs + tour.webm under docs/screenshots/out/ (gitignored).
# No live LLM calls, no network, no API key; everything renders from the seeded
# run logs. The pages workflow runs this before `mkdocs build`.
#
# AGENT6_SCREENSHOTS=placeholder writes flat placeholder images instead (a fast,
# tool-free docs-only build path). Needs `vhs`, `ttyd`, `ffmpeg`, `agent6`, and
# `python3` (with agent6 importable) on PATH in the default (full) mode.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/docs/screenshots/out"
mkdir -p "$OUT"

# The screenshots tour.tape writes, kept in sync with the docs pages that embed
# them. generate.sh fails if `full` mode does not produce every one.
SHOTS=(01-hub 02-run-dashboard 03-config 04-config-search 05-transcript 08-help 09-logs)

write_placeholders() {
  python3 - "$OUT" "${SHOTS[@]}" <<'PY'
import struct, zlib, sys
from pathlib import Path
out = Path(sys.argv[1]); names = sys.argv[2:]
def png(path, w=1600, h=900, rgb=(0x16, 0x16, 0x18)):
    raw = bytearray()
    row = bytes(rgb) * w
    for _ in range(h):
        raw.append(0); raw += row
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
for n in names:
    png(out / f"{n}.png")
(out / "tour.webm").write_bytes(b"")
print(f"wrote {len(names)} placeholder PNGs + empty tour.webm")
PY
}

if [ "${AGENT6_SCREENSHOTS:-full}" = "placeholder" ]; then
  echo "screenshots: placeholder mode"
  write_placeholders
  exit 0
fi

for bin in vhs ttyd ffmpeg agent6 python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "generate.sh: missing required tool: $bin" >&2; exit 1; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export AGENT6_CONFIG_HOME="$TMP/config"
export AGENT6_STATE_HOME="$TMP/state"
export AGENT6_DEMO_REPO="${AGENT6_DEMO_REPO:-$ROOT}"

echo "screenshots: seeding fixtures into $TMP"
python3 docs/screenshots/seed.py

# Fresh media each run; out/ is gitignored (committed: the .tape + seed only).
rm -f "$OUT"/*.png "$OUT"/tour.webm "$OUT"/_capture.webm

echo "screenshots: capturing PNGs with vhs (tour.tape)"
vhs docs/screenshots/tour.tape

missing=0
for s in "${SHOTS[@]}"; do
  [ -s "$OUT/$s.png" ] || { echo "generate.sh: tour.tape did not produce $s.png" >&2; missing=1; }
done
[ "$missing" = 0 ] || exit 1

rm -f "$OUT"/_capture.webm  # the throwaway video from the screenshot pass

# The tour.webm reel is one vhs recording of a single TUI session (reel.tape)
# with animated keystroke toasts overlaid (keystroke_overlay.py). The overlay
# scales the keypress timeline to the recording's ACTUAL duration, so the toasts
# stay aligned even though the live dashboard records a little fast (it redraws
# faster than vhs captures at this resolution).
echo "screenshots: recording reel.tape with vhs"
rm -f "$OUT/_reel-raw.webm"
vhs docs/screenshots/reel.tape
python3 docs/screenshots/keystroke_overlay.py \
  docs/screenshots/reel.tape "$OUT/_reel-raw.webm" "$OUT/tour.webm"
rm -f "$OUT/_reel-raw.webm"
[ -s "$OUT/tour.webm" ] || { echo "generate.sh: failed to build tour.webm" >&2; exit 1; }

echo "screenshots: done -> $OUT"
