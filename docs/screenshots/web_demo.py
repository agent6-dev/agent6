#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Record a web-UI tour with Playwright (dev tool, not run in CI by default).

Drives `agent6 web` in a headless Chromium, recording a .webm at a given
viewport (desktop or phone) while touring the hub, a run dashboard, the
transcript, and the config page. A virtual circle cursor glides between targets
and a bottom-center toast narrates each step, the browser analogue of the TUI
reel's keystroke overlay (keystroke_overlay.py).

The cursor moves via a CSS transition on left/top so the compositor interpolates
it; a per-frame JS animation janks in a headless screencast. Everything renders
from committed seed fixtures (docs/screenshots/seed/), so the tour is
deterministic and needs no API key or network. Driven by web_demo.sh.

  python3 web_demo.py --url http://127.0.0.1:PORT --out out/web-desktop.webm --mode desktop
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import Page, sync_playwright
except ImportError:  # pragma: no cover - dev tool
    sys.exit("web_demo.py needs playwright: pip install playwright && playwright install chromium")

VIEWPORTS = {
    "desktop": {"width": 1280, "height": 800},
    "phone": {"width": 390, "height": 844},
}

# A virtual circle cursor (headless recordings have no OS pointer), a click
# ripple, and a bottom-center toast. Self-installing once per document via
# add_init_script so it survives navigations. Motion is a CSS transition on
# left/top so the compositor drives it (a per-frame JS animation janks here).
OVERLAY_INIT_SCRIPT = r"""
(() => {
  if (window.__a6Overlay) return;
  window.__a6Overlay = true;
  const install = () => {
    if (!document.body || document.getElementById('__a6_cursor')) return;
    const cursor = document.createElement('div');
    cursor.id = '__a6_cursor';
    cursor.style.cssText = (
      'position:fixed;left:50%;top:45%;width:22px;height:22px;'
      + 'border-radius:50%;background:rgba(255,255,255,0.94);'
      + 'border:2px solid #0b0e14;pointer-events:none;z-index:2147483646;'
      + 'transform:translate(-50%,-50%);box-shadow:0 0 10px rgba(0,0,0,0.55);'
      + 'transition:left 650ms cubic-bezier(.25,.46,.45,.94),'
      + 'top 650ms cubic-bezier(.25,.46,.45,.94);'
    );
    document.documentElement.appendChild(cursor);
    const toast = document.createElement('div');
    toast.id = '__a6_toast';
    toast.style.cssText = (
      'position:fixed;bottom:32px;left:50%;'
      + 'transform:translateX(-50%) translateY(24px);'
      + 'background:rgba(18,20,38,0.94);color:#f7f9ff;'
      + 'padding:11px 20px;border-radius:10px;'
      + 'font:600 15px ui-monospace,SFMono-Regular,Menlo,monospace;'
      + 'letter-spacing:0.03em;z-index:2147483647;opacity:0;'
      + 'pointer-events:none;transition:opacity 220ms ease,transform 220ms ease;'
      + 'border:1px solid rgba(110,168,254,0.55);box-shadow:0 10px 28px rgba(0,0,0,0.5);'
    );
    document.documentElement.appendChild(toast);
    const sty = document.createElement('style');
    sty.textContent = (
      '@keyframes __a6Ripple { '
      + '0% { width:22px; height:22px; opacity:0.85; border-width:2px; } '
      + '100% { width:90px; height:90px; opacity:0; border-width:1px; } }'
    );
    document.head.appendChild(sty);
    window.__a6MoveCursor = (x, y) => { cursor.style.left = x + 'px'; cursor.style.top = y + 'px'; };
    window.__a6Ripple = () => {
      const r = document.createElement('div');
      r.style.cssText = (
        'position:fixed;left:' + (cursor.style.left || '50%')
        + ';top:' + (cursor.style.top || '50%')
        + ';border-radius:50%;border:2px solid #6ea8fe;pointer-events:none;'
        + 'z-index:2147483645;transform:translate(-50%,-50%);'
        + 'animation:__a6Ripple 0.7s ease-out forwards;'
      );
      document.documentElement.appendChild(r);
      setTimeout(() => r.remove(), 750);
    };
    let toastTimer;
    window.__a6Toast = (msg, ms) => {
      toast.textContent = msg;
      toast.style.opacity = '1';
      toast.style.transform = 'translateX(-50%) translateY(0)';
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(24px)';
      }, ms || 2000);
    };
  };
  if (document.body) install();
  else document.addEventListener('DOMContentLoaded', install);
})();
"""


def toast(page: Page, msg: str, ms: int = 2400, after: int = 450) -> None:
    page.evaluate(f"window.__a6Toast({json.dumps(msg)}, {ms})")
    if after:
        page.wait_for_timeout(after)


def move_to(page: Page, selector: str, *, settle: int = 760) -> tuple[float, float] | None:
    loc = page.locator(selector).first
    loc.scroll_into_view_if_needed()
    box = loc.bounding_box()
    if box is None:
        return None
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.evaluate(f"window.__a6MoveCursor({cx}, {cy})")
    page.wait_for_timeout(settle)  # let the 650ms CSS transition finish
    return cx, cy


def click(page: Page, selector: str, *, label: str | None = None, settle: int = 800) -> None:
    center = move_to(page, selector)
    if label:
        toast(page, label, after=320)
    page.evaluate("window.__a6Ripple()")
    page.wait_for_timeout(180)
    if center is not None:
        page.mouse.click(*center)
    else:
        page.locator(selector).first.click()
    page.wait_for_timeout(settle)


def scroll_to(page: Page, y: int, *, wait: int = 1500) -> None:
    """Native compositor smooth-scroll (steady in the recording), then settle."""
    page.evaluate("(y) => window.scrollTo({ top: y, behavior: 'smooth' })", y)
    page.wait_for_timeout(wait)


def drive(page: Page, base: str, mode: str, t0: float) -> float:
    page.goto(base, wait_until="networkidle")
    page.wait_for_selector(".item")
    page.wait_for_timeout(500)
    # The hub is painted now; everything recorded before this point is the SPA
    # loading screen. Return its offset so the recording can be trimmed to open
    # on the hub -- the still users see as the poster frame before pressing play.
    hub_ready = time.monotonic() - t0
    toast(page, "agent6 web — the hub: runs, machines, new work")
    move_to(page, ".item")
    page.wait_for_timeout(900)

    # Open the featured run (the first, newest).
    click(page, ".item", label="Open a run", settle=1100)
    page.wait_for_selector(".tree")
    toast(page, "Run dashboard: task, reasoning, tools, log, budget")
    page.wait_for_timeout(1400)
    scroll_to(page, 520, wait=1900)
    toast(page, "Steer, merge, approve, and answer — from the browser")
    page.wait_for_timeout(1500)
    scroll_to(page, 0, wait=1200)

    # Full transcript.
    click(page, "button:has-text('Transcript')", label="Read the full transcript")
    page.wait_for_timeout(700)
    scroll_to(page, 620, wait=2100)
    scroll_to(page, 0, wait=1000)

    # Config page: the left nav rail on desktop, the bottom tab bar on phone
    # (data-tab is the stable hook; the other nav is display:none per viewport).
    click(
        page,
        "nav.tabs a[data-tab='config']" if mode == "phone" else "aside.rail a[data-tab='config']",
        label="Every setting, with its source",
    )
    click(page, "input.filter", settle=300)
    page.keyboard.type("sandbox", delay=120)
    page.wait_for_timeout(1600)

    # Back to the hub to close the loop.
    page.evaluate("location.hash = '#/'")
    page.wait_for_timeout(700)
    toast(page, "Drivable from a desktop or a phone", ms=1800)
    page.wait_for_timeout(1500)
    return hub_ready


def write_trimmed(raw: Path, out: Path, trim_s: float) -> None:
    """Re-encode `raw` dropping the leading `trim_s` seconds so the first frame is
    the loaded hub, not the SPA loading screen (the poster the browser shows
    before play). Falls back to a plain move if the trim is negligible or ffmpeg
    is unavailable (a standalone run without the pipeline's ffmpeg)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if trim_s < 0.1 or shutil.which("ffmpeg") is None:
        raw.replace(out)
        return
    # -ss after -i re-encodes from the seek point, so the first output frame is
    # exactly at trim_s (frame-accurate); matches the overlay step's VP9 settings.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ss", f"{trim_s:.3f}",
         "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-an", str(out)],
        check=True,
    )  # fmt: skip
    raw.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=list(VIEWPORTS), default="desktop")
    args = ap.parse_args()

    vp = VIEWPORTS[args.mode]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out.parent / f"_web_{args.mode}_raw"

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport=vp,
            record_video_dir=str(tmp_dir),
            record_video_size=vp,
            device_scale_factor=2,
        )
        page = context.new_page()
        page.add_init_script(OVERLAY_INIT_SCRIPT)
        t0 = time.monotonic()
        trim_s = drive(page, args.url, args.mode, t0)
        video = page.video
        context.close()  # flushes the recording
        browser.close()
        if video is not None:
            write_trimmed(Path(video.path()), args.out, trim_s)
    print(f"web_demo: wrote {args.out} (trimmed {trim_s:.2f}s loading head)")


if __name__ == "__main__":
    main()
