#!/usr/bin/env bash
# Build the README hero: one fast tour across the three surfaces.
#
#   bash docs/screenshots/hero.sh
#
# Three separate recordings, cut and sped up, then stitched: the CLI run
# (cli_demo.sh), the TUI reel (generate.sh), and the web tour (web_demo.sh
# hero). No keystroke toasts and no narration banner -- the hero carries no
# overlay at all; the narrated videos stay for the docs pages.
#
# Each source is cut to one window and sped to SEG_S seconds, so the pace comes
# from how much happens in that window rather than from squeezing a whole demo.
# Everything lands on one 1280x720 canvas over agent6-dark's background, so a
# clip of another shape (the web tour records 1280x800) is padded in the theme's
# own colour rather than black.
#
# One ffmpeg pass: cutting to intermediate files and concatenating them with
# `-c copy` produced a file that decoded as six seconds of the LAST clip.
#
# Writes hero.webm (the site) and hero.gif (GitHub, which animates a GIF
# through its image proxy but will not play a webm from another host).
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
OUT="$ROOT/docs/screenshots/out"

command -v ffmpeg >/dev/null 2>&1 || { echo "hero.sh: missing required tool: ffmpeg" >&2; exit 1; }

# One segment per surface: source, where to cut in, how long a window to take.
# The window is what gets compressed into SEG_S, so a longer one moves faster.
SEG_S=6.5
SEGMENTS=(
  "cli-demo.webm:6:20"
  "hero-tui.webm:2:18"
  "hero-web.webm:1:17"
)
BG="0x161618"   # agent6-dark's background; see src/agent6/ui/tui/theme.py
W=1280
H=720
FPS=24

missing=0
for seg in "${SEGMENTS[@]}"; do
  [ -s "$OUT/${seg%%:*}" ] || { echo "hero.sh: missing source $OUT/${seg%%:*}" >&2; missing=1; }
done
if [ "$missing" != 0 ]; then
  cat >&2 <<'EOF'
hero.sh: record the sources first:
  bash docs/screenshots/cli_demo.sh                       # -> cli-demo.webm
  bash docs/screenshots/generate.sh                       # -> hero-tui.webm (the bare reel)
  WEB_DEMO_PY=… bash docs/screenshots/web_demo.sh hero    # -> hero-web.webm
EOF
  exit 1
fi

inputs=()
chain=""
labels=""
i=0
for seg in "${SEGMENTS[@]}"; do
  IFS=: read -r src start window <<<"$seg"
  inputs+=(-ss "$start" -t "$window" -i "$OUT/$src")
  # setpts scales the cut window onto SEG_S; scale+pad normalises every source
  # onto one canvas without cropping content.
  chain+="[${i}:v]setpts=PTS*${SEG_S}/${window},"
  chain+="scale=${W}:${H}:force_original_aspect_ratio=decrease,"
  chain+="pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:color=${BG},fps=${FPS},setsar=1[v${i}];"
  labels+="[v${i}]"
  i=$((i + 1))
done

ffmpeg -v error -y "${inputs[@]}" \
  -filter_complex "${chain}${labels}concat=n=${i}:v=1:a=0[out]" \
  -map "[out]" -an -c:v libvpx-vp9 -crf 32 -b:v 0 "$OUT/hero.webm"

# GitHub renders an external webm as a broken image but proxies an animated GIF,
# so the README needs one. GitHub's image proxy refuses anything over 5 MB, and
# a 24fps full-scale GIF of this is 4 MB before it has been looked at, so drop
# to 10fps at 800px over a shared 96-colour palette: ~2.4 MB, still legible.
ffmpeg -v error -y -i "$OUT/hero.webm" \
  -vf "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
  "$OUT/hero.gif"

printf 'hero: %s (%s, %ss), %s (%s)\n' \
  "$OUT/hero.webm" "$(du -h "$OUT/hero.webm" | cut -f1)" \
  "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/hero.webm")" \
  "$OUT/hero.gif" "$(du -h "$OUT/hero.gif" | cut -f1)"
