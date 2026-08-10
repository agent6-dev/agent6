#!/usr/bin/env bash
# Build the three README/site heroes, one per surface: hero-tui, hero-cli,
# hero-web (each .webm + .gif).
#
#   bash docs/screenshots/hero.sh
#
# House style for all three: open mid-action (no launch preamble), stay on one
# surface the whole video (no cross-surface cuts), keep something moving in
# every frame, rest on the payoff frames, end on a beat that states what
# agent6 is, loop clean. Time compression is honest: the UI's own durations
# and costs stay visible.
#
#   tui  conversation streaming -> approval modal over the dashboard (ask) ->
#        verify ✓ auto-commit + diff pane -> hub receipt -> config: sandbox.
#   cli  the failing suite -> one command, the run streams -> /exit ->
#        sessions diff (the fix) -> sessions show (the receipt).
#   web  hub -> session view -> expanded tool detail -> config: sandbox.
#
# Sources (record first):
#   bash docs/screenshots/hero_run.sh                       # -> hero-src-tui.webm
#   bash docs/screenshots/hero_cli.sh                       # -> hero-src-cli.webm
#   WEB_DEMO_PY=… bash docs/screenshots/web_demo.sh hero    # -> hero-src-web.webm
#
# Each segment cuts a window from its source and retimes it onto a target
# length, so the pace comes from how much happens in the window; a per-segment
# hold freezes the last frame so payoffs rest (the run TUI exits the moment
# the run ends, so a rest cannot come from the source). The 8fps TUI/CLI
# recordings keep ~20 distinct frames/s after a ~2.5x speed-up.
#
# Segment windows are tuned to the committed cassette's replay pacing and the
# committed web tour script; after re-recording a source, re-check the cuts by
# extracting frames (scene changes = YAVG discontinuities:
#   ffmpeg -i src.webm -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG -f null -).
#
# One ffmpeg pass per video: cutting to intermediate files and concatenating
# them with `-c copy` produced a file that decoded as six seconds of the LAST
# clip.
#
# Writes .webm (the site) and .gif (GitHub, which animates a GIF through its
# image proxy but will not play a webm from another host; the proxy refuses
# anything over 5 MB -- holds compress to almost nothing, so full width fits).
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/docs/screenshots/out"

command -v ffmpeg >/dev/null 2>&1 || { echo "hero.sh: missing required tool: ffmpeg" >&2; exit 1; }

BG="0x161618"   # agent6-dark's background; see src/agent6/ui/tui/theme.py
FPS=24
GIF_FPS=14      # every frame of a scrolling transcript is a full redraw, so the
                # gif's size scales with fps; 14 keeps the 10s cuts under the 5 MB cap

# Segment spec: src : cut-in : window length : target length : hold.
TUI_SEGMENTS=(
  "hero-src-tui.webm:5.4:6.8:2.0:0"
  "hero-src-tui.webm:13.5:4.0:1.2:0"
  "hero-src-tui.webm:20.3:2.3:1.5:0"
  "hero-src-tui.webm:22.80:0.28:0.30:1.3"
  "hero-src-tui.webm:34.75:2.4:1.3:0"
  "hero-src-tui.webm:38.2:4.9:2.0:0.9"
)
CLI_SEGMENTS=(
  "hero-src-cli.webm:0.3:3.3:1.4:0"
  "hero-src-cli.webm:3.9:22.2:3.8:0"
  "hero-src-cli.webm:26.9:6.9:2.2:0.5"
  "hero-src-cli.webm:35.4:4.6:1.4:0.9"
)
WEB_SEGMENTS=(
  "hero-src-web.webm:0.6:3.3:1.9:0"
  "hero-src-web.webm:3.8:5.5:2.6:0"
  "hero-src-web.webm:21.2:2.1:1.4:0"
  "hero-src-web.webm:25.4:2.5:1.6:1.0"
)

build() {
  local name="$1" w="$2" h="$3" gif_w="$4"; shift 4
  local segments=("$@")

  local missing=0
  for seg in "${segments[@]}"; do
    [ -s "$OUT/${seg%%:*}" ] || { echo "hero.sh: missing source $OUT/${seg%%:*}" >&2; missing=1; }
  done
  [ "$missing" = 0 ] || return 1

  local inputs=() chain="" labels="" i=0
  local src start window target hold
  for seg in "${segments[@]}"; do
    IFS=: read -r src start window target hold <<<"$seg"
    inputs+=(-ss "$start" -t "$window" -i "$OUT/$src")
    # setpts retimes the cut window onto its target; scale+pad normalises onto
    # one canvas without cropping; tpad rests on the last frame for the hold.
    chain+="[${i}:v]setpts=PTS*${target}/${window},"
    chain+="scale=${w}:${h}:force_original_aspect_ratio=decrease,"
    chain+="pad=${w}:${h}:(ow-iw)/2:(oh-ih)/2:color=${BG},fps=${FPS},"
    chain+="tpad=stop_mode=clone:stop_duration=${hold},setsar=1[v${i}];"
    labels+="[v${i}]"
    i=$((i + 1))
  done

  ffmpeg -v error -y "${inputs[@]}" \
    -filter_complex "${chain}${labels}concat=n=${i}:v=1:a=0[out]" \
    -map "[out]" -an -c:v libvpx-vp9 -crf 32 -b:v 0 "$OUT/hero-$name.webm"

  ffmpeg -v error -y -i "$OUT/hero-$name.webm" \
    -vf "fps=${GIF_FPS},scale=${gif_w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=5" \
    "$OUT/hero-$name.gif"

  printf 'hero-%s: %s (%ss), gif %s\n' "$name" \
    "$(du -h "$OUT/hero-$name.webm" | cut -f1)" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/hero-$name.webm")" \
    "$(du -h "$OUT/hero-$name.gif" | cut -f1)"
}

# The TUI/CLI sources are 1600x900 recordings; their gifs downscale to 1440
# (the last ~0.5 MB under the proxy cap). The web tour records 1280x800 and
# stays at its native size.
fail=0
build tui 1600 900 1440 "${TUI_SEGMENTS[@]}" || fail=1
build cli 1600 900 1440 "${CLI_SEGMENTS[@]}" || fail=1
build web 1280 800 1280 "${WEB_SEGMENTS[@]}" || fail=1
if [ "$fail" != 0 ]; then
  cat >&2 <<'EOF'
hero.sh: record the missing sources first:
  bash docs/screenshots/hero_run.sh                       # -> hero-src-tui.webm
  bash docs/screenshots/hero_cli.sh                       # -> hero-src-cli.webm
  WEB_DEMO_PY=… bash docs/screenshots/web_demo.sh hero    # -> hero-src-web.webm
EOF
  exit 1
fi
