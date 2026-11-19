"""The session page's conversation driven in a browser: its items, keys, heading and notes."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.web import session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


def test_a_drag_inside_a_conversation_item_selects_its_text(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A drag across an expandable item leaves its text selected and the item as it was.

    The drag ended in a click that toggled the item, and its repaint dropped the selection.
    """
    from playwright.sync_api import expect

    argv = ["sh", "-c", "echo " + "word " * 400]
    rows = [
        {"type": "tool.call", "name": "run_command", "args_preview": {"argv": argv}},
        {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    session_logs.write_run(tmp_path, "drag-AAA111", rows)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/drag-AAA111")
    item = page.locator(".card-conv .ci.exp").filter(has=page.locator(".s-call"))
    expect(item).to_have_attribute("title", "click to expand")
    box = item.bounding_box()
    assert box
    page.mouse.move(box["x"] + 4, box["y"] + 8)
    page.mouse.down()
    page.mouse.move(box["x"] + 240, box["y"] + 8, steps=8)
    page.mouse.up()
    assert page.evaluate("getSelection().toString()").startswith("→ run_command")
    expect(item).to_have_attribute("title", "click to expand")
    assert not errors


def test_a_live_runs_conversation_keeps_a_selection_across_frames(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A frame that adds no conversation item leaves the conversation's nodes and selection.

    Every frame refolded the conversation and repainted it, dropping a selection within a second.
    """
    from playwright.sync_api import expect

    call = {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}}
    session = session_logs.write_run(tmp_path, "live-AAA111", [call], live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/live-AAA111")
    item = page.locator(".card-conv .ci").filter(has=page.locator(".s-call"))
    expect(item).to_be_visible()
    item.evaluate("e => getSelection().selectAllChildren(e)")
    refolded = page.expect_response(lambda response: response.url.endswith("/conversation"))
    with refolded, (session / "logs.jsonl").open("a") as log:
        log.write(json.dumps({"type": "budget.update", "usd_total": 0.5}) + "\n")
    page.wait_for_timeout(200)  # the response's paint
    assert page.evaluate("getSelection().toString()").startswith("→ read_file")
    assert not errors


def test_a_wrapped_conversation_line_hangs_under_its_text(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A wrapped line's later lines start under its first word, past its indent and marker.

    They restarted at the item's left edge, under the "└" of a tool's result; then a line
    indented past the column hung its words outside it.
    """
    from playwright.sync_api import expect

    rows = [
        {"type": "tool.call", "name": "run_command", "args_preview": {"argv": ["make"]}},
        {"type": "tool.result", "name": "run_command", "ok": True, "summary": "word " * 60},
    ]
    session_logs.write_run(tmp_path, "wrap-AAA111", rows)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/wrap-AAA111")
    line = page.locator(".card-conv .ci > div").filter(has_text="└ word")
    expect(line).to_be_visible()
    # The text starts at the line's fifth character, after two spaces, the marker and a space.
    lines, last, text = line.evaluate(
        """e => { const r = document.createRange(); r.selectNodeContents(e);
        const lines = [...r.getClientRects()].map(x => Math.round(x.top));
        const tops = [...new Set(lines)], all = [...r.getClientRects()];
        const walker = document.createTreeWalker(e, NodeFilter.SHOW_TEXT);
        let node, seen = 0, first = null;
        while ((node = walker.nextNode())) {
            if (seen + node.length > 4) { const c = document.createRange();
                c.setStart(node, 4 - seen); c.setEnd(node, 5 - seen);
                first = c.getBoundingClientRect().left; break; }
            seen += node.length;
        }
        return [tops.length, Math.min(...all.filter(x => Math.round(x.top) === tops.at(-1))
            .map(x => x.left)), first]; }"""
    )
    assert lines > 1
    assert last == pytest.approx(text, abs=0.5)  # layout rounds to 1/64 px
    # The hang stops at 40% of the box, so a line indented 200 spaces keeps its words in it.
    deep = {"type": "model.result", "role": "worker", "text": "Deep:\n" + " " * 200 + "word " * 40}
    session_logs.write_run(tmp_path, "deep-AAA111", [deep])
    page.goto(f"http://127.0.0.1:{port}/#/session/deep-AAA111")
    deep_line = page.locator(".card-conv .ci > div").filter(has_text="word word")
    expect(deep_line).to_be_visible()
    inside = deep_line.evaluate(
        """e => { const t = e.querySelector('span').firstChild, r = document.createRange();
        r.setStart(t, t.data.search(/\\S/)); r.setEnd(t, t.data.length);
        const box = e.closest('.conv-box').getBoundingClientRect();
        return [...r.getClientRects()].every(x => x.left >= box.left && x.right <= box.right); }"""
    )
    assert inside
    assert not errors


@pytest.mark.parametrize("width", [1280, 820])
def test_the_conversation_heading_lines_up_with_the_drawers_first_card(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """The Conversation heading and its box start where the run card's heading and body start.

    The detail button set the heading row's height, so the heading sat 3px and the box 6px
    lower than the run card's beside them. At 820px with the drawer open the button wrapped.
    """
    from playwright.sync_api import expect

    session_logs.write_run(
        tmp_path, "heads-AAA111", [{"type": "model.call", "role": "worker", "model": "m1"}]
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/heads-AAA111")
    details = page.get_by_role("button", name="Details", exact=True)
    if details.get_attribute("aria-expanded") == "false":
        details.click()
    expect(page.locator(".card-head .kv")).to_be_visible()
    tops = page.evaluate(
        """() => ['.card-head > h2', '.card-conv h2', '.card-head > .card-body',
        '.card-conv .conv-box'].map(s => document.querySelector(s).getBoundingClientRect().top)"""
    )
    assert tops[0] == tops[1]
    assert tops[2] == tops[3]
    assert not errors


def test_a_stale_run_says_why_under_its_conversation(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A run whose worker is gone says why under its conversation, in the read model's words.

    Only an empty conversation's note gave the reason, so a stale run with items read "stale"
    and nothing more; "the composer below resumes it" then showed on a draft, which has none.
    With no items above it, the note sat under a rule that parted it from nothing.
    """
    from playwright.sync_api import expect

    rows = [
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
    ]
    session_logs.write_run(tmp_path, "stale-AAA111", rows)
    created = paths.state_dir(tmp_path) / "sessions" / "runs" / "created-AAA111"
    created.mkdir(parents=True)
    (created / "logs.jsonl").write_text("")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/stale-AAA111")
    expect(page.locator(".card-conv .ci").first).to_be_visible()
    note = page.locator(".card-conv .conv-live")
    expect(note).to_have_text("worker exited without finishing (crashed or killed)")
    expect(page.locator(".card-conv .conv-empty")).to_have_count(0)
    assert note.evaluate("e => getComputedStyle(e).borderTopWidth") == "1px"
    page.goto(f"http://127.0.0.1:{port}/#/session/created-AAA111")
    expect(note).to_have_text("the run has not started")
    assert note.evaluate("e => getComputedStyle(e).borderTopWidth") == "0px"
    assert not errors


def test_a_conversation_item_opens_and_closes_from_the_keyboard(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The conversation is one Tab stop; arrows move between items, Enter opens, Space closes.

    An item opened on a click alone: it took no focus and said nothing of its state. Then each
    item was a Tab stop, so a run with 60 calls took 73 presses from Details to the composer.
    """
    from playwright.sync_api import expect

    argv = ["sh", "-c", "echo " + "word " * 400]
    call = {"type": "tool.call", "name": "run_command", "args_preview": {"argv": argv}}
    result = {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"}
    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    session_logs.write_run(
        tmp_path, "keys-AAA111", [call, result, call, result, call, result, stop]
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/keys-AAA111")
    heads = page.locator(".card-conv .ci > [role=button]")
    expect(heads).to_have_count(4)  # the task and three calls
    page.get_by_role("button", name="detail: collapsed").focus()
    page.keyboard.press("Tab")
    expect(heads.last).to_be_focused()  # the newest item holds the stop
    page.keyboard.press("Tab")
    assert page.evaluate("() => !document.activeElement.closest('.card-conv')")
    page.keyboard.press("Shift+Tab")
    expect(heads.last).to_be_focused()
    for key, at in (("ArrowUp", 2), ("Home", 0), ("ArrowDown", 1), ("End", 3)):
        page.keyboard.press(key)
        expect(heads.nth(at)).to_be_focused()
    item = page.locator(".card-conv .ci.exp").last
    head = item.get_by_role("button")
    expect(head).to_have_attribute("aria-expanded", "false")
    lines = item.locator("> div").count()
    page.keyboard.press("Enter")
    expect(head).to_have_attribute("aria-expanded", "true")
    expect(head).to_be_focused()
    assert item.locator("> div").count() > lines
    ring = item.evaluate(
        """e => [getComputedStyle(e).outlineStyle,
        e.getBoundingClientRect().left - e.closest('.conv-box').getBoundingClientRect().left]"""
    )
    assert ring[0] == "solid"
    assert ring[1] >= 0
    page.keyboard.press(" ")
    expect(head).to_have_attribute("aria-expanded", "false")
    expect(head).to_be_focused()
    assert not errors


def test_the_detail_toggle_holds_still_as_it_cycles(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The detail button keeps one place and width through collapsed, expanded and hidden.

    Its width followed its label, so its left edge moved 18px between clicks.
    """
    from playwright.sync_api import expect

    rows = [
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
    ]
    session_logs.write_run(tmp_path, "detail-AAA111", rows)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/detail-AAA111")
    button = page.locator(".card-conv .card-head-row > button")
    places = set()
    for level in ("collapsed", "expanded", "hidden"):
        expect(button).to_have_accessible_name(f"detail: {level}")
        box = button.bounding_box()
        assert box
        places.add((box["x"], box["width"]))
        button.click()
    expect(button).to_have_accessible_name("detail: collapsed")
    assert len(places) == 1
    assert not errors
