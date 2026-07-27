#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the README cover image from the three UI screenshots.

The three screenshots are stacked full-frame with the same positioning, and two
vertical cuts split the canvas into equal thirds, revealing the layers: the CLI
on the left, the TUI dashboard in the middle, the web run view on the right.
Each panel therefore shows THAT region of its own UI, as if one interface were
peeled between its three skins. The dressing reuses the app's own look (the web
UI's design tokens): accent hairlines on the cuts, a status-pill tag per panel,
a hairline border and rounded corners on the canvas.

Inputs come from the docs media pipeline (docs/screenshots/out/): the TUI PNG
from tour.tape, web-shot.png from web_demo.py's desktop tour, and a frame
pulled out of cli-demo.webm. Output: out/cover.png (1600x900, RGBA). Runs in
the pages workflow after the media steps; needs Pillow and ffmpeg.

  python3 docs/screenshots/make_cover.py [--out PATH]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - docs tool
    sys.exit("make_cover.py needs Pillow: pip install pillow")

OUT_DIR = Path(__file__).parent / "out"

W, H = 1600, 900
# The web UI's dark-theme tokens (src/agent6/ui/web/styles.css :root).
BG = "#0e1116"  # --bg
BORDER = "#2a3140"  # --border
ACCENT = "#6ea8fe"  # --accent
PILL_FILL = (14, 17, 22, 235)  # --bg, near-opaque, so tags read over content

CUTS = (W // 3, 2 * W // 3)  # even thirds: CLI | TUI | WEB
GAP = 10  # bg-coloured breathing room each side of a cut
LINE = 2  # the accent hairline itself
RADIUS = 18  # canvas corner radius
SS = 2  # supersample factor for the overlay (pills, border) and corner mask


def _font(size: int) -> ImageFont.FreeTypeFont:
    for pattern in ("JetBrains Mono:bold", "DejaVu Sans Mono:bold", "DejaVu Sans:bold"):
        try:
            path = subprocess.check_output(
                ["fc-match", "-f", "%{file}", pattern], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if path:
            return ImageFont.truetype(path, size)
    raise SystemExit("make_cover.py: no usable bold font found (need fontconfig)")


def _cli_frame(webm: Path) -> Image.Image:
    """A mid-demo frame of the CLI video (content is replay-deterministic)."""
    dur = float(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(webm)],
            text=True,
        ).strip()
    )  # fmt: skip
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        frame = Path(tf.name)
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{dur * 0.45:.2f}",
         "-i", str(webm), "-frames:v", "1", str(frame)],
    )  # fmt: skip
    img = Image.open(frame).convert("RGB")
    frame.unlink(missing_ok=True)
    return img


def _cover_fit(img: Image.Image, top_crop: int = 0, x_shift: int = 0) -> Image.Image:
    """Scale to cover the canvas and crop. *top_crop* drops rows from the top
    (past the CLI's scrollback tail, down to the web page's side-card stack);
    *x_shift* slides the layer sideways (positive = rightward) so its panel's
    slice lands on the layer's densest columns. Background pads any overhang."""
    scale = max(W / img.width, (H + top_crop) / img.height)
    scaled = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    out = Image.new("RGB", (W, H), BG)
    x0 = (W - scaled.width) // 2 + x_shift
    out.paste(scaled, (x0, -top_crop))
    return out


def _tracked_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    track: float,
    fill: str,
) -> None:
    """Draw *text* with *track* extra px between characters (PIL has none)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR / "cover.png")
    args = ap.parse_args()

    tui_png = OUT_DIR / "02-run-dashboard.png"
    web_png = OUT_DIR / "web-shot.png"
    cli_webm = OUT_DIR / "cli-demo.webm"
    for p in (tui_png, web_png, cli_webm):
        if not p.exists():
            sys.exit(f"make_cover.py: missing input {p} (run the media pipeline first)")

    cli = _cover_fit(_cli_frame(cli_webm), top_crop=48, x_shift=36)
    # 422 puts the middle cut in the gap between the transcript's line-number
    # gutter and its event column (measured: no x is clean for both that and
    # the menu bar's "View", and a ragged gutter is the worse clip).
    tui = _cover_fit(Image.open(tui_png).convert("RGB"), top_crop=0, x_shift=422)
    # x_shift lands the WEB slice on the run view's details drawer (task graph /
    # budget / tool calls, logical x ~216-513 of the 1280-wide shot): the
    # conversation-primary layout keeps the widgets there, and the slice right
    # of the cut would otherwise show the conversation's quiet right margin.
    web = _cover_fit(Image.open(web_png).convert("RGB"), top_crop=285, x_shift=942)

    canvas = Image.new("RGB", (W, H), BG)
    for img, x0, x1 in ((cli, 0, CUTS[0]), (tui, CUTS[0], CUTS[1]), (web, CUTS[1], W)):
        canvas.paste(img.crop((x0, 0, x1, H)), (x0, 0))

    # The cuts: a bg-coloured gap with a centred accent hairline.
    draw = ImageDraw.Draw(canvas)
    for x in CUTS:
        draw.rectangle((x - GAP, 0, x + GAP, H), fill=BG)
        draw.line([(x, 0), (x, H)], fill=ACCENT, width=LINE)

    # The dressing, supersampled for crisp curves: a CLI / TUI / WEB tag per
    # panel styled like the app's status pills (accent text and border on a
    # near-bg fill), plus the canvas hairline border.
    font = _font(23 * SS)
    track = 4 * SS
    overlay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    centres = (CUTS[0] // 2, (CUTS[0] + CUTS[1]) // 2, (CUTS[1] + W) // 2)
    for label, cx in zip(("CLI", "TUI", "WEB"), centres, strict=True):
        tw = sum(odraw.textlength(ch, font=font) for ch in label) + track * (len(label) - 1)
        top, bottom = odraw.textbbox((0, 0), label, font=font)[1::2]
        pad_x, pad_y = 17 * SS, 10 * SS
        tx, ty = cx * SS - tw / 2, (H - 62) * SS
        box = (tx - pad_x, ty + top - pad_y, tx + tw + pad_x, ty + bottom + pad_y)
        odraw.rounded_rectangle(
            box, radius=(box[3] - box[1]) / 2, fill=PILL_FILL, outline=ACCENT, width=SS
        )
        _tracked_text(odraw, (tx, ty), label, font, track, ACCENT)
    odraw.rounded_rectangle(
        (0, 0, W * SS - 1, H * SS - 1), radius=RADIUS * SS, outline=BORDER, width=2 * SS
    )
    overlay = overlay.resize((W, H), Image.LANCZOS)
    out = Image.alpha_composite(canvas.convert("RGBA"), overlay)

    # Rounded corners: transparent outside the border, so the cover sits as a
    # card on any README background.
    mask = Image.new("L", (W * SS, H * SS), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, W * SS - 1, H * SS - 1), radius=RADIUS * SS, fill=255
    )
    out.putalpha(mask.resize((W, H), Image.LANCZOS))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"make_cover: wrote {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
