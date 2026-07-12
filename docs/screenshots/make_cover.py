#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the README cover image from the three UI screenshots.

The three screenshots are stacked full-frame with the same positioning, and two
slightly-leaning vertical cuts reveal the layers: the CLI on the left, the TUI
dashboard in the middle, the web run view on the right. Each panel therefore
shows THAT region of its own UI, as if one interface were peeled between its
three skins. Thin accent hairlines mark the cuts; a small CLI / TUI / WEB tag
sits at the bottom of each panel.

Inputs come from the docs media pipeline (docs/screenshots/out/): the TUI PNG
from tour.tape, web-shot.png from web_demo.py's desktop tour, and a frame
pulled out of cli-demo.webm. Output: out/cover.png (1600x900). Runs in the
pages workflow after the media steps; needs Pillow and ffmpeg.

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
BG = "#0e1116"  # the app background
ACCENT = "#6ea8fe"

# The two cuts, as (x at y=0, x at y=H): near-vertical, leaning the same way
# (~5 degrees). Left of CUTA: the CLI. Between: the TUI (the widest panel).
# Right of CUTB: the web run view.
CUTA = (590, 500)
CUTB = (1160, 1070)
GAP = 9  # bg-coloured breathing room each side of a cut
LINE = 2  # the accent hairline itself


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


def _panel_mask(left: tuple[int, int] | None, right: tuple[int, int] | None) -> Image.Image:
    """A full-canvas mask for the panel between two cuts (None = canvas edge)."""
    l0, l1 = left if left else (0, 0)
    r0, r1 = right if right else (W, W)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon([(l0, 0), (r0, 0), (r1, H), (l1, H)], fill=255)
    return mask


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
    tui = _cover_fit(Image.open(tui_png).convert("RGB"), top_crop=0, x_shift=430)
    web = _cover_fit(Image.open(web_png).convert("RGB"), top_crop=285)

    canvas = Image.new("RGB", (W, H), BG)
    for img, left, right in ((cli, None, CUTA), (tui, CUTA, CUTB), (web, CUTB, None)):
        canvas = Image.composite(img, canvas, _panel_mask(left, right))

    # The cuts: a bg-coloured gap with a centred accent hairline.
    draw = ImageDraw.Draw(canvas)
    for x0, x1 in (CUTA, CUTB):
        draw.line([(x0, 0), (x1, H)], fill=BG, width=GAP * 2)
        draw.line([(x0, 0), (x1, H)], fill=ACCENT, width=LINE)

    # A small tag at the bottom of each panel, on a quiet pill so it reads over
    # any content: CLI / TUI / WEB, centred between that panel's bottom cuts.
    font = _font(22)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    centres = ((0 + CUTA[1]) // 2, (CUTA[1] + CUTB[1]) // 2, (CUTB[1] + W) // 2)
    for label, cx in zip(("CLI", "TUI", "WEB"), centres, strict=True):
        tw = odraw.textlength(label, font=font)
        tx, ty = cx - tw / 2, H - 52
        odraw.rounded_rectangle(
            (tx - 14, ty - 8, tx + tw + 14, ty + 30), radius=8, fill=(14, 17, 22, 220)
        )
        odraw.text((tx, ty), label, font=font, fill="#8b95a5")
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"make_cover: wrote {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
