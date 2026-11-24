#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Record a web-UI tour with Playwright (dev tool, not run in CI by default).

Drives `agent6 web` in a headless Chromium, recording a .webm at a given
viewport (desktop, phone or hero) while touring the hub, a run view, the
conversation, the config page, and the Files and Git pages of the repository
`agent6 web` runs in. A virtual circle cursor glides between targets
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
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import webm

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page, ViewportSize

VIEWPORTS: dict[str, ViewportSize] = {
    "desktop": {"width": 1280, "height": 800},
    "hero": {"width": 1600, "height": 900},
    "phone": {"width": 390, "height": 844},
    # The state-machine tour records at the desktop viewport; it is a separate
    # mode because it needs the replay-proxy environment (web_machine_demo.sh),
    # not the static seeds web_demo.sh serves.
    "machine": {"width": 1280, "height": 800},
    # The workflow tour also needs a replay proxy, with one replay per child
    # (web_workflow_demo.sh).
    "workflow": {"width": 1280, "height": 800},
}

# A virtual circle cursor (headless recordings have no OS pointer), a click
# ripple, and a bottom-center toast. Self-installing once per document via
# add_init_script so it survives navigations; the top document only, so a file
# preview's frame gets no cursor of its own. Motion is a CSS transition on
# left/top so the compositor drives it (a per-frame JS animation janks here).
OVERLAY_INIT_SCRIPT = r"""
(() => {
  if (window !== top || window.__a6Overlay) return;
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
    window.__a6MoveCursor = (x, y) => { cursor.style.left = x + 'px'; cursor.style.top = y + 'px'; };
    window.__a6Ripple = () => {
      const r = document.createElement('div');
      r.style.cssText = (
        'position:fixed;left:' + (cursor.style.left || '50%')
        + ';top:' + (cursor.style.top || '50%')
        + ';border-radius:50%;border:2px solid #6ea8fe;pointer-events:none;'
        + 'z-index:2147483645;transform:translate(-50%,-50%);'
      );
      document.documentElement.appendChild(r);
      r.animate([
        {width:'22px', height:'22px', opacity:0.85},
        {width:'90px', height:'90px', opacity:0, borderWidth:'1px'},
      ], {duration:700, easing:'ease-out', fill:'forwards'}).onfinish = () => r.remove();
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


# Whether the narration banner is drawn. Mutated, never rebound, so every
# call site sees main()'s choice without a `global` statement.
_narration: dict[str, bool] = {"show": True}


def toast(page: Page, msg: str, ms: int = 2400, after: int = 450) -> None:
    """Narrate one step, unless `--no-narration` asked for a bare recording.

    The wait happens either way, so a bare take is paced exactly like a
    narrated one and a clip cut from either lines up frame for frame."""
    if _narration["show"]:
        page.evaluate(f"window.__a6Toast({json.dumps(msg)}, {ms})")
    if after:
        page.wait_for_timeout(after)


def _centre(loc: Locator) -> tuple[float, float] | None:
    box = loc.bounding_box()
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2) if box else None


def move_to(page: Page, target: str | Locator, *, settle: int = 760) -> tuple[float, float] | None:
    """Glide the cursor to a target: a selector, or a locator for an element inside a frame."""
    loc = page.locator(target).first if isinstance(target, str) else target
    # Centre the target: a row scrolled to the nearest edge sits under the phone's docked
    # composer and tab bar, where the click would land on them.
    loc.evaluate("el => el.scrollIntoView({block: 'center', inline: 'nearest'})")
    page.wait_for_timeout(250)
    centre = _centre(loc)
    if centre is not None:
        page.evaluate(f"window.__a6MoveCursor({centre[0]}, {centre[1]})")
        page.wait_for_timeout(settle)  # let the 650ms CSS transition finish
    return centre


def click(
    page: Page, target: str | Locator, *, label: str | None = None, settle: int = 800
) -> None:
    loc = page.locator(target).first if isinstance(target, str) else target
    move_to(page, loc)
    if label:
        toast(page, label, after=320)
    page.evaluate("window.__a6Ripple()")
    page.wait_for_timeout(180)
    # A live view moves its controls while the cursor glides: click where the target is now.
    centre = _centre(loc)
    if centre is not None:
        page.evaluate(f"window.__a6MoveCursor({centre[0]}, {centre[1]})")
        page.mouse.click(*centre)
    else:
        loc.click()
    page.wait_for_timeout(settle)


def scroll_to(page: Page, y: int, *, wait: int = 1500) -> None:
    """Native compositor smooth-scroll (steady in the recording), then settle."""
    page.evaluate("(y) => window.scrollTo({ top: y, behavior: 'smooth' })", y)
    page.wait_for_timeout(wait)


def scroll_card(page: Page, selector: str, y: int, *, wait: int = 1500) -> None:
    """Smooth-scroll INSIDE a scroll-capped card (the desktop conversation
    scrolls internally). On a phone the boxes flow at natural height and the
    PAGE is the one scroller, so fall back to scrolling the window."""
    page.evaluate(
        "([sel, y]) => { const c = document.querySelector(sel);"
        " if (c && c.scrollHeight > c.clientHeight + 4) c.scrollTo({ top: y, behavior: 'smooth' });"
        " else window.scrollTo({ top: y, behavior: 'smooth' }); }",
        [selector, y],
    )
    page.wait_for_timeout(wait)


def drive_repo(page: Page, mode: str) -> None:
    """The Files page (a task box, an edit, an HTML preview), then the Git page (stage, commit, diff)."""
    rail = "nav.tabs a[data-tab='%s']" if mode == "phone" else "aside.rail a[data-tab='%s']"
    click(page, rail % "files", label="Files: the working tree, or any revision", settle=1200)
    page.wait_for_selector(".gtree")
    click(page, ".gtree [data-path='release.md']", label="A Markdown page, rendered", settle=1400)
    # Rendered Markdown lives in the preview frame.
    preview = page.frame_locator(".files-preview iframe")
    task = preview.get_by_role("checkbox", name="Update the changelog")
    task.wait_for()
    click(page, task, label="Tick a task: Save writes it to the file", settle=900)
    click(page, "dialog[open] button.primary", settle=1300)
    preview.get_by_role("checkbox", name="Update the changelog", checked=True).wait_for()
    edit = page.get_by_role("button", name="Edit", exact=True)
    click(page, edit, label="Edit the page's text", settle=900)
    # The editor enables its text once the file has loaded.
    page.locator("dialog[open] textarea.file-text:enabled").press("Control+End")
    page.keyboard.type("- [ ] Announce the release\n", delay=45)
    page.wait_for_timeout(700)
    click(page, "dialog[open] footer button.primary", settle=1400)
    preview.get_by_role("checkbox", name="Announce the release").wait_for()
    if mode == "phone":  # the open file covers the tree
        click(page, "button.files-back", settle=700)
    click(page, ".gtree [data-path='report.html']", label="An HTML page", settle=1200)
    show = page.get_by_role("button", name="Preview", exact=True)
    click(page, show, label="Preview it: its scripts stay off", settle=1600)
    page.get_by_text("Scripts are off for this page.", exact=True).wait_for()
    preview.get_by_role("heading", name="Test report").wait_for()
    page.wait_for_timeout(1200)

    # The Git page: stage the edited page, commit it, then read the commit.
    click(page, rail % "git", label="Git: the commit graph and your changes", settle=1200)
    click(page, ".gc-row.gc-wt", settle=1200)
    staged = ".git-side .gf[data-path='release.md']"
    click(page, f"{staged} button[data-act=stage]", label="Stage a file", settle=1200)
    # The message box opens once the panel shows the staged file.
    click(page, ".git-commit textarea:enabled", settle=300)
    page.keyboard.type("Add the release checklist", delay=45)
    click(page, ".git-commit button.primary", label="Commit what is staged", settle=1600)
    commit = page.locator(".gc-row:not(.gc-wt)", has_text="Add the release checklist")
    commit.wait_for(state="attached")
    if mode == "phone":  # the working-tree panel covers the graph
        click(page, page.get_by_role("button", name="Graph", exact=True), settle=700)
    click(page, commit, label="The commit: its message and files", settle=1400)
    click(page, staged, label="A file's diff, inline or split", settle=2000)


def drive_findings(page: Page, mode: str) -> None:
    """A reviewed run: its Findings card with two open warnings, then the receipt."""
    page.evaluate("location.hash = '#/'")
    page.wait_for_selector(".item")
    page.wait_for_timeout(700)
    click(
        page,
        ".item:has-text('merry-fountain')",
        label="A reviewed run: findings stay open until you resolve them",
        settle=1300,
    )
    page.wait_for_selector(".card-findings .finding-counts", state="attached")
    if mode == "phone":  # the phone shows one widget at a time, picked from the menu
        click(page, ".wmenu-btn", settle=500)
        click(page, ".wmenu button[data-w='findings']", settle=900)
    page.wait_for_selector(".card-findings .finding-counts")
    page.wait_for_timeout(900)
    click(
        page,
        ".card-findings button.mini:has-text('Receipt')",
        label="The receipt: tests, review, risk, how to revert",
        settle=1800,
    )
    page.wait_for_selector("dialog .receipt pre")
    page.wait_for_timeout(1500)


def drive(page: Page, base: str, mode: str, t0: float, shot: Path | None = None) -> float:
    page.goto(base, wait_until="networkidle")
    page.wait_for_selector(".item")
    page.wait_for_timeout(500)
    # The hub is painted now; everything recorded before this point is the SPA
    # loading screen. Return its offset so the recording can be trimmed to open
    # on the hub -- the still users see as the poster frame before pressing play.
    hub_ready = time.monotonic() - t0
    toast(page, "agent6 web — the hub: your runs, new work one click away")
    move_to(page, ".item")
    page.wait_for_timeout(900)

    # Open the featured run (the first, newest).
    click(page, ".item", label="Open a run", settle=1100)
    page.wait_for_selector(".conv .ci")
    if shot is not None:  # the curated web still for the README cover (make_cover.py):
        # taken BEFORE the narration toast so neither it nor the cursor is baked in.
        page.evaluate("document.getElementById('__a6_cursor').style.visibility = 'hidden'")
        page.wait_for_timeout(400)
        page.screenshot(path=str(shot))
        page.evaluate("document.getElementById('__a6_cursor').style.visibility = ''")
    toast(page, "A run: the conversation, task graph, tools, budget")
    page.wait_for_timeout(1400)
    scroll_card(page, ".card-conv .conv-box", 400, wait=1600)
    toast(page, "Steer, merge, approve, and answer — from the browser")
    page.wait_for_timeout(1500)
    scroll_to(page, 0, wait=1200)

    # The conversation IS the page: cycle the detail level, then (desktop) fold
    # the details drawer away to show the focus mode.
    click(page, "button.mini:has-text('detail:')", label="Cycle the detail level", settle=900)
    scroll_card(page, ".card-conv .conv-box", 0, wait=1400)
    scroll_card(page, ".card-conv .conv-box", 4000, wait=1600)
    if mode == "desktop":
        click(
            page,
            "button:has-text('Details')",
            label="Fold the details away — pure conversation",
            settle=1400,
        )
        click(page, "button:has-text('Details')", settle=900)

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

    drive_repo(page, mode)
    drive_findings(page, mode)

    # Back to the hub to close the loop.
    page.evaluate("location.hash = '#/'")
    page.wait_for_timeout(700)
    toast(page, "Drivable from a desktop or a phone", ms=1800)
    page.wait_for_timeout(1500)
    return hub_ready


def drive_machine(page: Page, base: str, t0: float) -> float:
    """The state-machine tour: start the seeded code-fixer machine from the
    Machines page and watch it run to green, all from the browser. The machine
    is a REAL `agent6 machine run` whose agent calls the replay proxy serves
    (web_machine_demo.sh), so the stream is deterministic and key-free."""
    page.goto(base + "/#/machines", wait_until="networkidle")
    run = ".machine-files button[aria-label^='Run code-fixer']"
    page.wait_for_selector(run)
    page.wait_for_timeout(600)
    ready = time.monotonic() - t0
    toast(page, "State machines: author, run, and watch — from the browser")
    page.wait_for_timeout(1400)

    # Start it: POST /api/machine/run spawns a detached `agent6 machine run`.
    # The Run button asks first (a paid run must not start from one click);
    # the question's Run starts it.
    click(page, run, label="Run the code-fixer machine", settle=900)
    click(page, "dialog[open] button.primary", settle=1300)
    # The instance registers its journal + worker pid almost immediately; the
    # page re-renders itself once, then the instance row is clickable.
    instance = ".list:not(.machine-files) .item"
    page.wait_for_selector(instance, timeout=30000)
    toast(page, "The instance appears in the list — open it to watch")
    page.wait_for_timeout(600)
    click(page, instance, label="Watch it live", settle=1300)
    page.wait_for_selector(".tree", timeout=15000)
    toast(page, "A fix-loop: an agent edits, a tool re-checks, a branch routes on the result")
    page.wait_for_timeout(1800)

    # The agent state streams its reasoning into the current-state pane.
    page.wait_for_selector(".conv .ci", timeout=90000)
    page.wait_for_timeout(2200)
    toast(page, "The current agent state streams its reasoning, like any run")
    page.wait_for_timeout(2000)
    scroll_card(page, ".conv-box", 4000, wait=2200)

    # The status turns ok as the machine ends; linger on the finished machine.
    page.wait_for_selector(".machine-grid .pill.ok", timeout=180000)
    toast(page, "Green: the check passes and the machine ends ok", ms=2400)
    page.wait_for_timeout(2400)
    scroll_to(page, 0, wait=1400)
    page.wait_for_timeout(1000)
    return ready


WORKFLOW_TASK = "A test in tests/test_temps.py is failing. Find and fix the bug in temps.py."
WORKFLOW_STEER = "Keep the change to one line."
CHILD_ROWS = ".drawer details.card-children .child-row"


def start_children(page: Page) -> None:
    """Type a /parallel task into the hub's composer and start it."""
    toast(page, "One task, three attempts: /parallel starts a child for each")
    click(page, ".dock textarea", settle=300)
    page.keyboard.type(f"/parallel 3 -- {WORKFLOW_TASK}", delay=24)
    page.wait_for_timeout(1500)
    click(page, ".dock button.primary", label="Start the run", settle=1200)
    page.locator(".drawer details.card-children").wait_for(timeout=30000)


def watch_children(page: Page) -> None:
    """Open the run's Children card, then one child's prompt and conversation."""
    click(page, ".drawer details.card-children > summary", label="Its children, each in a clone")
    rows = page.locator(CHILD_ROWS)
    rows.nth(2).wait_for(timeout=30000)
    page.wait_for_timeout(1200)
    click(page, rows.nth(0).locator(":scope > summary"), label="Open one: its prompt, its work")
    rows.nth(0).get_by_role("textbox", name="Message").wait_for(timeout=30000)
    page.wait_for_timeout(1500)


def steer_and_stop(page: Page) -> None:
    """Message the open child, then stop the third."""
    rows = page.locator(CHILD_ROWS)
    message = rows.nth(0).get_by_role("textbox", name="Message")
    click(page, message, label="Message this child: it reads it at its next turn", settle=300)
    page.keyboard.type(WORKFLOW_STEER, delay=40)
    click(page, rows.nth(0).get_by_role("button", name="Send", exact=True), settle=1500)
    stopped = rows.nth(2)
    click(page, stopped.locator(":scope > summary"), label="Stop another", settle=700)
    click(page, stopped.get_by_role("button", name="Stop", exact=True), settle=900)
    click(page, "dialog[open] button.danger", settle=1200)
    stopped.locator(":scope > summary", has_text="stopped").wait_for(timeout=30000)
    page.wait_for_timeout(1200)


def switch_sessions(page: Page) -> None:
    """Ctrl+K to the child that got the message: its conversation shows it read."""
    child = page.locator(CHILD_ROWS).nth(0).get_by_role("link").first.inner_text()
    page.keyboard.press("Control+k")
    toast(page, "Ctrl+K switches session: here, to the child that got the message")
    page.wait_for_selector("dialog[open] input.field")
    page.keyboard.type(child[-5:], delay=90)
    page.wait_for_timeout(900)
    page.keyboard.press("Enter")
    page.wait_for_url(f"**/#/session/{child}")
    # The child reads a queued message at its next turn.
    page.locator(".conv .ci", has_text=WORKFLOW_STEER).first.wait_for(timeout=60000)
    toast(page, "The child read the message and kept working")
    page.wait_for_timeout(2200)
    scroll_card(page, ".card-conv .conv-box", 4000, wait=1800)


def read_review(page: Page) -> None:
    """Ctrl+K to the seeded reviewed run: its two open findings."""
    page.keyboard.press("Control+k")
    page.wait_for_selector("dialog[open] input.field")
    page.keyboard.type("percentile", delay=90)
    page.wait_for_timeout(900)
    page.keyboard.press("Enter")
    page.wait_for_selector(".card-findings .finding-counts")
    toast(page, "A reviewed run: its findings stay open until you resolve them")
    move_to(page, ".card-findings .finding-title")
    page.wait_for_timeout(2600)


def merge_run(page: Page) -> None:
    """Back on the run its children have ended; merge it into its base branch and see the commit."""
    page.evaluate("location.hash = '#/'")
    run = page.locator(".item:not(.child)", has_text=WORKFLOW_TASK)
    click(page, run.locator("a.title"), label="Back to the run that started them", settle=1200)
    children = page.locator(".drawer details.card-children > summary")
    click(page, children, label="Each row follows its child to its end")
    children.get_by_text("children: 1 merged, 1 ranked, 1 stopped").wait_for(timeout=120000)
    toast(page, "One merged into the run, one ranked second, one stopped")
    page.wait_for_timeout(2400)
    page.evaluate("location.hash = '#/'")
    merge = run.get_by_role("button", name="Merge", exact=True)
    click(page, merge, label="Merge the run into its base branch", settle=900)
    click(page, "dialog[open] button.primary", settle=1500)
    # The row drops "unmerged" from its status once the merge is recorded.
    run.locator(".pill", has_text=re.compile("^finished · children ranked$")).wait_for(
        timeout=30000
    )
    click(page, "aside.rail a[data-tab='git']", label="Its commit is on master", settle=1200)
    page.locator(".gc-row:not(.gc-wt)", has_text="tests/test_temps.py is failing").first.wait_for()
    page.wait_for_timeout(2400)


# The workflow tour's parts in order: each title names a chapter of the recording.
WORKFLOW_PARTS = (
    ("Start children", start_children),
    ("Watch them work", watch_children),
    ("Steer one, stop one", steer_and_stop),
    ("Switch sessions", switch_sessions),
    ("Read a review", read_review),
    ("Merge", merge_run),
)


def drive_workflow(
    page: Page, base: str, t0: float, shot: Path | None = None
) -> tuple[float, list[tuple[str, float]]]:
    """The workflow tour: a run starts three children; steer one, stop one, switch, review, merge.

    The children are REAL runs replaying the temperature cassette (web_workflow_demo.sh), so
    their states are waited for, never assumed. Returns the loading head to trim and each
    part's title with its start, both in seconds since `t0`; a last untitled entry is the end.
    The first part starts where the trimmed recording does.
    """
    page.goto(base, wait_until="networkidle")
    page.wait_for_selector(".item")
    page.wait_for_timeout(500)
    if shot is not None:  # the poster, without the cursor
        page.evaluate("document.getElementById('__a6_cursor').style.visibility = 'hidden'")
        page.screenshot(path=str(shot))
        page.evaluate("document.getElementById('__a6_cursor').style.visibility = ''")
    ready = time.monotonic() - t0
    starts = [ready]
    for _title, stop in WORKFLOW_PARTS:
        stop(page)
        starts.append(time.monotonic() - t0)
    titles = [*(title for title, _stop in WORKFLOW_PARTS), ""]
    return ready, list(zip(titles, starts, strict=True))


def write_chapters(out: Path, parts: list[tuple[str, float]], trim_s: float) -> None:
    """Write the parts as a WebVTT chapter track; the last entry only ends the part before it."""

    def stamp(seconds: float) -> str:
        whole, millis = divmod(round(max(seconds - trim_s, 0) * 1000), 1000)
        return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}.{millis:03d}"

    cues = [
        f"{stamp(start)} --> {stamp(end)}\n{title}\n"
        for (title, start), (_, end) in itertools.pairwise(parts)
    ]
    out.write_text("WEBVTT\n\n" + "\n".join(cues), encoding="utf-8")


def write_trimmed(raw: Path, out: Path, trim_s: float) -> None:
    """Trim the loading screen and encode BT.709, including recordings with no trim."""
    out.parent.mkdir(parents=True, exist_ok=True)
    webm.encode(raw, out, trim_s=trim_s)
    raw.unlink(missing_ok=True)


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415 - only recording needs it
    except ImportError:  # pragma: no cover - dev tool
        sys.exit(
            "web_demo.py needs playwright: pip install playwright && playwright install chromium"
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=list(VIEWPORTS), default="desktop")
    ap.add_argument(
        "--no-narration",
        action="store_true",
        help="Record without the narration banner (for the README hero, which carries no overlay).",
    )
    args = ap.parse_args()

    _narration["show"] = not args.no_narration
    vp = VIEWPORTS[args.mode]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out.parent / f"_web_{args.mode}_raw"

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport=vp,
            color_scheme="dark",
            record_video_dir=str(tmp_dir),
            record_video_size=vp,
            device_scale_factor=2,
        )
        # Sign in off camera: the token the server mints rides a header the page adds from its
        # origin's storage, so every page of the tour starts with it planted.
        secret = os.environ["AGENT6_WEB_SECRET"]
        login = context.request.post(args.url + "/api/login", data={"secret": secret})
        if not login.ok:
            sys.exit(f"web_demo.py: sign-in refused ({login.status}): {login.text()}")
        token = json.dumps(login.json()["token"])
        context.add_init_script(f"localStorage.setItem('a6-token', {token})")
        page = context.new_page()
        page.add_init_script(OVERLAY_INIT_SCRIPT)
        t0 = time.monotonic()
        if args.mode == "machine":
            trim_s = drive_machine(page, args.url, t0)
        elif args.mode == "workflow":
            trim_s, parts = drive_workflow(page, args.url, t0, args.out.with_suffix(".png"))
            write_chapters(args.out.with_suffix(".vtt"), parts, trim_s)
        else:
            shot = args.out.parent / "web-shot.png" if args.mode == "desktop" else None
            mode = "desktop" if args.mode == "hero" else args.mode
            trim_s = drive(page, args.url, mode, t0, shot=shot)
        video = page.video
        context.close()  # flushes the recording
        browser.close()
        if video is not None:
            write_trimmed(Path(video.path()), args.out, trim_s)
    print(f"web_demo: wrote {args.out} (trimmed {trim_s:.2f}s loading head)")


if __name__ == "__main__":
    main()
