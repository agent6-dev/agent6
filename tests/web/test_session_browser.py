"""The session page driven in a browser: its prompts, run card, drawer cards and conversation."""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths
from agent6.sessions import event_log, journal, layout, process
from tests.web import test_review_evidence, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
evidence = test_review_evidence.evidence

LONG_COMMAND = (
    "sh -c 'rm -rf build dist && python -m pip install --no-deps -e . && .venv/bin/python -m "
    'pytest tests/test_csv_parse.py -k "quoted and not slow" --maxfail=1 2>&1 | tee /tmp/LAST.log\''
)


def _session(
    tmp_path: pathlib.Path, name: str, rows: list[dict[str, Any]], *, live: bool = False
) -> pathlib.Path:
    """Write a run's log under the served state dir; a live run's worker is this process."""
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / name
    session.mkdir(parents=True)
    start = {"type": "session.start", "mode": "run", "user_task": "Parse CSV rows"}
    (session / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [start, *rows]))
    if live:
        process.write_worker_pid(session, os.getpid())
    return session


def test_an_approval_shows_its_whole_command(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The command under judgment wraps inside its box, so its tail shows beside Allow.

    The box kept the command on one line and cut it at the box's edge: at 1024px a pipe into
    `tee` was out of sight.
    """
    from playwright.sync_api import expect

    ask = {
        "type": "approval.prompt",
        "id": "approval-1",
        "prompt": f"Allow run_command: {LONG_COMMAND}",
    }
    _session(tmp_path, "asking-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto(f"http://127.0.0.1:{port}/#/session/asking-AAA111")
    command = page.locator(".prompt-box .cmd")
    expect(command).to_contain_text("/tmp/LAST.log")
    assert command.evaluate("e => e.scrollWidth <= e.clientWidth")
    assert not errors


@pytest.mark.parametrize("width", [1280, 390])
def test_prompts_stand_under_the_top_bar_and_the_action_bar_clips_nothing(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """A prompt box stands 10px under the top bar, over an action bar in the page's gutter.

    The box touched the top bar. The action bar was a scroll container, which cut its buttons'
    focus rings, and on a phone it sat 6px right of the boxes and cards.
    """
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow run_command: ls"}
    _session(tmp_path, "asking-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/asking-AAA111")
    box = page.locator(".prompt-box")
    expect(box).to_be_visible()
    actions = page.locator(".run-app .actions")
    header, top = page.locator("header").bounding_box(), box.bounding_box()
    first = actions.locator("button:visible").first.bounding_box()
    assert header is not None and top is not None and first is not None
    gap = top["y"] - (header["y"] + header["height"])
    overflow = actions.evaluate("e => getComputedStyle(e).overflow")
    assert (gap, overflow, first["x"]) == (10, "visible", top["x"])
    assert not errors


@pytest.mark.parametrize("width", [1280, 390])
def test_a_question_box_spaces_its_questions_and_wraps_a_narrow_field(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """Each question and Submit all stand 14px under a field; a narrow field takes its own row.

    Question 2 and Submit all touched the field above them, and on a phone the field shrank
    beside its options until its placeholder was cut.
    """
    from playwright.sync_api import expect

    ask = {
        "type": "question.prompt",
        "id": "question-1",
        "wait": True,
        "questions": [
            {"question": "Should a lone CR also end a row?", "options": ["yes", "no, raise"]},
            {"question": "Keep the strict dialect?", "options": []},
        ],
    }
    _session(tmp_path, "asking-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/asking-AAA111")
    box = page.locator(".prompt-box")
    fields = box.locator("input.field")
    expect(fields).to_have_count(2)
    expect(fields.first).to_have_attribute("placeholder", "pick or type an answer…")
    expect(fields.last).to_have_attribute("placeholder", "type an answer…")
    first, last = fields.first.bounding_box(), fields.last.bounding_box()
    second = box.locator(".q").last.bounding_box()
    send = box.get_by_role("button", name="Submit all", exact=True).bounding_box()
    option = box.get_by_role("button", name="no, raise", exact=True).bounding_box()
    row = box.locator(".form-row").first.bounding_box()
    assert first and last and second and send and option and row
    assert second["y"] - (first["y"] + first["height"]) == 14
    assert send["y"] - (last["y"] + last["height"]) == 14
    if width > 780:
        assert first["y"] == option["y"]
    else:
        assert first["y"] > option["y"] + option["height"]
        assert first["width"] == row["width"]
    assert not errors


@pytest.mark.parametrize("asks", [False, True])
def test_a_phone_opens_a_run_at_its_tail_unless_a_prompt_waits(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    *,
    asks: bool,
) -> None:
    """A phone opens a long conversation at its end, and at the top when a prompt box waits.

    The page always scrolled to the conversation's end, about 2,500px under the approval box.
    """
    from playwright.sync_api import expect

    rows: list[dict[str, Any]] = []
    for i in range(40):
        rows.append(
            {"type": "tool.call", "name": "read_file", "args_preview": {"path": f"f{i}.py"}}
        )
        rows.append({"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"})
    if asks:
        rows.append({"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"})
    _session(tmp_path, "long-AAA111", rows, live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/long-AAA111")
    expect(page.locator(".card-conv .ci").filter(has_text="f39.py")).to_be_attached()
    if asks:
        assert page.evaluate("window.scrollY") == 0
        expect(page.locator(".prompt-box")).to_be_in_viewport()
    else:
        page.wait_for_function("window.scrollY > 0")
    assert not errors


def test_the_run_card_lines_up_the_judge_settings_and_last_call_with_its_rows(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The judge's reason, the settings and the last model call are rows of the run card.

    They were lines under the rows, starting at the card's edge in a larger size, and each
    setting's separator lost the space after it ("gpt-4.1-mini ·isolation full").
    """
    from playwright.sync_api import expect

    gate = {"command": ["pytest", "-q"], "source": "configured", "availability": "ready"}
    manifests = {
        "fan": {"comparison": {"group": "fan", "ranking": ["child"], "rationale": "covers it"}},
        "child": {
            "placement": {"group": "fan", "rank": 1, "of": 1, "winner": True},
            "models": {"driver": {"provider": "openai", "model": "gpt-4.1-mini"}},
            "policy": {"isolation": "full", "network": "session", "run_commands": "ask"},
            "harness": {"gate": gate},
        },
    }
    for sid, manifest in manifests.items():
        session = _session(tmp_path, sid, [{"type": "model.call", "role": "worker", "model": "m1"}])
        (session / "manifest.json").write_text(json.dumps({"mode": "run", **manifest}))
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/child")
    keys = page.locator(".card-head .kv .k")
    expect(keys).to_have_text(["task", "id", "status", "compare", "judge", "settings", "last call"])
    values = page.locator(".card-head .kv .v")
    expect(values.nth(4)).to_have_text("covers it")
    expect(values.nth(6)).to_have_text("worker / m1")
    settings = values.nth(5).evaluate("e => e.innerText")
    assert settings.startswith("gpt-4.1-mini · isolation full · network session · run_commands ask")
    assert len(set(values.evaluate_all("vs => vs.map(v => v.getBoundingClientRect().x)"))) == 1
    assert not errors


def test_a_live_frame_leaves_the_cards_a_reader_is_using(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A frame rebuilds only the cards it changes, and the event log keeps the reader's place.

    Every frame rebuilt every card: the event log snapped to its end, and a selection in the
    run card or a node in the tool calls went with the old nodes.
    """
    from playwright.sync_api import expect

    rows: list[dict[str, Any]] = []
    for i in range(60):
        rows.append({"type": "tool.call", "name": "read_file", "args_preview": {"path": f"f{i}"}})
        rows.append({"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"})
    session = _session(tmp_path, "live-AAA111", rows, live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/live-AAA111")
    lines = page.locator(".card-log .log > div")
    expect(lines).to_have_count(121)
    page.evaluate(
        """() => {
        document.querySelector('.card-log .card-body').scrollTop = 0;
        getSelection().selectAllChildren(document.querySelectorAll('.card-head .kv .v')[1]);
        document.querySelector('.card-tools .call').dataset.kept = 'yes';
    }"""
    )
    with (session / "logs.jsonl").open("a") as log:
        log.write(json.dumps({"type": "budget.update", "usd_total": 0.5}) + "\n")
    expect(lines).to_have_count(122)
    kept = page.evaluate(
        """() => [document.querySelector('.card-log .card-body').scrollTop,
        getSelection().toString(),
        document.querySelectorAll('.card-tools .call[data-kept]').length]"""
    )
    assert kept == [0, "live-AAA111", 1]
    assert not errors


def test_a_wrapped_event_log_row_hangs_under_its_event(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A wrapped row's later lines start under its event name, clear of the time column.

    They started at the row's left edge, under the times, and read as rows of their own.
    """
    from playwright.sync_api import expect

    argv = ["sh", "-c", "echo " + "word " * 40]
    call = {"type": "tool.call", "ts": "2026-11-13T10:00:00.000000+00:00", "name": "run_command"}
    _session(tmp_path, "wrap-AAA111", [{**call, "args_preview": {"argv": argv}}])
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/wrap-AAA111")
    row = page.locator(".card-log .log > div").filter(has_text="run_command")
    expect(row).to_be_visible()
    # The event name starts at the row's 15th character, after the time and two spaces.
    lines, last, event = row.evaluate(
        """e => { const text = e.firstChild, r = document.createRange();
        r.selectNodeContents(text); const lines = [...r.getClientRects()];
        r.setStart(text, 14); r.setEnd(text, 15);
        return [lines.length, lines.at(-1).left, r.getBoundingClientRect().left]; }"""
    )
    assert lines > 1
    assert last == pytest.approx(event, abs=0.5)  # layout rounds to 1/64 px
    assert not errors


def test_a_fan_outs_children_table_leaves_the_conversation_its_share(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Twenty children take at most 30% of the view, under left-aligned names, empty columns gone.

    The browser-default table centred bold names over left-aligned cells, kept model, usage and
    join columns no child filled, and grew until the conversation had no height left.
    """
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "verification": "passed", "reason": "finish_session"}
    for number in range(21):
        session = _session(tmp_path, f"fan-l{number}" if number else "fan", [stop])
        manifest: dict[str, Any] = {"version": 3, "session_id": session.name, "mode": "run"}
        if number:
            manifest["parallel"] = {"group": "fan", "number": number, "parent": "fan"}
        (session / "manifest.json").write_text(json.dumps(manifest))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/fan")
    card = page.locator(".card-children")
    expect(card.locator("tr")).to_have_count(21)
    expect(card.locator("th")).to_have_text(["id", "status"])
    name = card.locator("th").first.evaluate(
        "e => [getComputedStyle(e).textAlign, getComputedStyle(e).fontWeight]"
    )
    box = card.bounding_box()
    conversation = page.locator(".card-conv .conv-box").bounding_box()
    assert box and conversation
    assert name == ["left", "400"]
    assert box["height"] <= 900 * 0.3
    assert conversation["height"] > 300
    assert not errors


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
    _session(tmp_path, "drag-AAA111", rows)
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
    session = _session(tmp_path, "live-AAA111", [call], live=True)
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


def test_the_findings_card_spaces_its_parts_and_labels_its_filters(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """The findings card's parts stand 6px apart, its facts are rows, its selects take the skin.

    The counts touched the Receipt button, Resume to fix all open touched the review and merge
    lines it sat between, and the filters and the disposition were unlabelled native selects.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    body = page.locator(".card-findings .card-body")
    expect(body.locator(".kv .k")).to_have_text(["review", "merge"])
    expect(body.locator(".pickers .pick > span")).to_have_text(["status", "severity"])
    gaps = body.evaluate(
        """b => { const parts = [...b.children].filter(part => part.offsetParent);
        return parts.slice(1).map((part, i) => Math.round(
            part.getBoundingClientRect().top - parts[i].getBoundingClientRect().bottom)); }"""
    )
    assert set(gaps) == {6}
    body.locator(".finding").first.get_by_role("button", name="Resolve", exact=True).click()
    selects = body.locator("select:visible")
    expect(selects).to_have_count(3)
    assert selects.evaluate_all("all => all.map(s => s.className)") == ["field"] * 3
    assert not errors


def test_the_phone_widget_menu_closes_on_esc_an_outside_tap_and_a_pick(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The phone's widget menu is named Widgets and closes on Esc, a tap outside or a pick.

    The menu button was named "☰", and the menu stayed open over the page after Esc or a tap
    outside it. Then a menu left open while the window widened past 780px stayed in the page.
    """
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    _session(tmp_path, "menu-AAA111", [stop])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/menu-AAA111")
    button = page.get_by_role("button", name="Widgets", exact=True)
    menu = page.locator(".wmenu")
    for close in ("Escape", "tap"):
        button.click()
        expect(menu).to_be_visible()
        if close == "Escape":
            page.keyboard.press("Escape")
        else:
            page.mouse.click(120, 420)
        expect(menu).to_be_hidden()
    button.click()
    menu.get_by_role("button", name="Budget", exact=True).click()
    expect(menu).to_be_hidden()
    expect(page.locator(".card-budget")).to_be_visible()
    expect(menu.locator("[aria-current=true]")).to_have_text("Budget")
    button.click()
    page.set_viewport_size({"width": 1024, "height": 844})
    expect(menu).to_be_hidden()
    assert not errors


def test_the_phone_widget_menu_offers_only_the_widgets_a_run_has(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A run with no plan, review or background shell offers none of those widgets.

    The menu listed every widget, and picking one the run did not have showed an empty page.
    """
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    _session(tmp_path, "menu-AAA111", [stop])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/menu-AAA111")
    page.get_by_role("button", name="Widgets", exact=True).click()
    expect(page.locator(".wmenu button:visible")).to_have_text(
        [
            "Conversation",
            "Overview",
            "Findings",
            "Task graph",
            "Budget",
            "Tool calls",
            "Latest commit",
            "Event log",
        ]
    )
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
    _session(tmp_path, "wrap-AAA111", rows)
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
    _session(tmp_path, "deep-AAA111", [deep])
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


def test_a_task_title_wraps_and_a_skipped_task_strikes_only_its_title(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A long task title wraps under itself, and a skipped task's title is struck, not its glyph.

    A long title was cut at the drawer's edge with no way to read the rest, and the strike ran
    through the indent and the glyph while the title stayed plain.
    """
    from playwright.sync_api import expect

    title = "Handle a quoted field that spans a CRLF and keeps a doubled quote " * 2
    nodes = {
        "t1": {"title": title, "parent_id": None, "status": "skipped", "children": []},
    }
    _session(tmp_path, "tasks-AAA111", [{"type": "graph.update", "nodes": nodes, "cursor": None}])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/tasks-AAA111")
    node = page.locator(".card-tasks .tree > div").filter(has_text="Handle a quoted field")
    expect(node).to_be_visible()
    fits, lines = node.evaluate(
        "n => [n.scrollWidth <= n.clientWidth, Math.round(n.offsetHeight / 19.5)]"
    )
    struck = node.evaluate(
        """n => [...n.querySelectorAll('span')].map(s =>
        [s.textContent.trim().slice(0, 8), getComputedStyle(s).textDecorationLine])"""
    )
    assert fits
    assert lines > 1
    assert ["\u2013", "none"] in struck  # the skipped glyph
    assert ["Handle a", "line-through"] in struck
    assert not errors


def test_a_budget_bar_stands_apart_from_its_label_and_the_next_line(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A budget bar stands 4px under its label and 10px over the line under it.

    The bar touched its label above and the tokens line below.
    """
    from playwright.sync_api import expect

    _session(tmp_path, "budget-AAA111", [{"type": "budget.update", "usd_total": 0.5}])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/budget-AAA111")
    bar = page.locator(".card-budget .bar")
    expect(bar).to_be_visible()
    gaps = bar.evaluate(
        """b => { const r = e => e.getBoundingClientRect();
        const label = b.previousElementSibling, next = b.parentElement.nextElementSibling;
        return [r(b).top - r(label).bottom, r(next).top - r(b).bottom]; }"""
    )
    assert gaps == [4, 10]
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

    _session(tmp_path, "heads-AAA111", [{"type": "model.call", "role": "worker", "model": "m1"}])
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
    _session(tmp_path, "stale-AAA111", rows)
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


def test_the_latest_commit_card_shows_no_error_under_the_logs_diff(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """With no commit picked, a pruned commit leaves the log's diff and no error under it.

    The card painted the diff the log recorded, then added "no diff for <sha>" under it.
    """
    from playwright.sync_api import expect

    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    sha = "5" * 40
    rows = [
        {"type": "checkpoint.committed", "turn": 1, "sha": sha, "subject": "change"},
        {"type": "diff.updated", "patch": patch, "sha": sha},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    session = _session(tmp_path, "pruned-AAA111", rows)
    (session / "manifest.json").write_text(json.dumps({"version": 3, "mode": "run"}))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    with page.expect_response(lambda response: "/diff?sha=" in response.url) as answer:
        page.goto(f"http://127.0.0.1:{port}/#/session/pruned-AAA111")
    refusal = answer.value.json()["error"]
    card = page.locator(".card-diff .card-body")
    expect(card).to_contain_text("new")
    page.wait_for_timeout(300)  # the refusal's paint
    assert refusal.startswith("no diff for 5555")
    assert refusal not in card.inner_text()
    assert [error for error in errors if "Failed to load resource" not in error] == []


def test_a_finding_card_on_its_session_links_away_and_sizes_its_submit(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """A finding on its own session's page has no link back to that page, and Submit fits its label.

    Each card linked "Session findings" to the page it was on, and the Resolve form's submit
    button stretched across the card.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    finding = page.locator(".card-findings .finding").first
    expect(finding.get_by_role("link", name="Reviewed comparison")).to_be_visible()
    expect(finding.get_by_role("link", name="Session findings")).to_have_count(0)
    finding.get_by_role("button", name="Resolve", exact=True).click()
    form = finding.locator(".finding-form")
    submit = form.get_by_role("button", name="Submit resolution")
    expect(submit).to_be_visible()
    widths = [form.bounding_box(), submit.bounding_box()]
    assert widths[0] and widths[1]
    assert widths[1]["width"] < widths[0]["width"] / 2
    assert not errors


def test_the_receipt_hangs_its_wrapped_lines_in_a_box_and_folds_only_output(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
    tmp_path: pathlib.Path,
) -> None:
    """The receipt's lines sit in a bordered 12px box, a wrapped line hanging under its start.

    The receipt was 14px text with no box, its wrapped lines restarting at column 0, and a
    check's empty stream still had a fold that opened onto nothing.
    """
    from playwright.sync_api import expect

    sink = event_log.EventSink(
        layout.SessionLayout(paths.state_dir(tmp_path), "review-run").logs_path
    )
    sink.emit(
        journal.VerifyEnd(
            command=["pytest"],
            exit_code=1,
            stdout_tail="",
            stderr_tail="1 failed",
            stdout_truncated=False,
            stderr_truncated=False,
            scoped=False,
            selected_paths=[],
            outcome="failed",
            duration_s=0.2,
            timeout_s=60.0,
            baseline=False,
            reason="",
        )
    )
    port, _ = evidence
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    page.get_by_role("button", name="Receipt", exact=True).click()
    text = page.locator(".receipt > pre")
    expect(text).to_contain_text("Receipt review-run")
    size, border = text.evaluate(
        "e => [getComputedStyle(e).fontSize, getComputedStyle(e).borderTopWidth]"
    )
    hangs = text.evaluate(
        """e => [...e.children].map(line => { const r = document.createRange();
        r.selectNodeContents(line); const rects = [...r.getClientRects()];
        const tops = [...new Set(rects.map(x => Math.round(x.top)))];
        const left = top => Math.min(...rects.filter(x => Math.round(x.top) === top)
            .map(x => x.left));
        return tops.length > 1 ? left(tops[1]) - left(tops[0]) : null; })
        .filter(gap => gap !== null)"""
    )
    expect(page.locator(".receipt summary")).to_have_count(1)
    assert (size, border) == ("12px", "1px")
    assert hangs
    assert all(gap == pytest.approx(2 * 7.2, abs=1.5) for gap in hangs)  # 2ch of 12px mono
    assert page.locator(".receipt summary").inner_text().endswith("stderr")
    assert not errors


def test_the_receipt_says_it_is_loading_and_its_button_hides_it_again(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Receipt shows a loading line while it fetches, and a second press hides the receipt.

    The card showed nothing until the receipt landed, seconds on a busy machine, and a second
    press fetched and drew it again. Disabling the button while it fetched dropped the keyboard
    focus, so Enter could not hide the receipt.
    """
    from playwright.sync_api import Route, expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)  # answered once the loading line is seen

    page.route("**/receipt", hold)
    button = page.get_by_role("button", name="Receipt", exact=True)
    button.focus()
    page.keyboard.press("Enter")
    expect(page.locator(".receipt")).to_have_text("loading the receipt…")
    expect(button).to_be_focused()
    while not held:
        page.wait_for_timeout(50)
    held[0].continue_()
    expect(page.locator(".receipt > pre")).to_contain_text("Receipt review-run")
    expect(button).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Enter")
    expect(page.locator(".receipt")).to_be_hidden()
    expect(button).to_have_attribute("aria-expanded", "false")
    assert not errors


def test_a_picked_commit_says_its_diff_is_loading(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A picked commit's diff shows a loading line until it lands, then the outcome alone.

    The diff went blank until the picked commit's diff arrived.
    """
    from playwright.sync_api import Route, expect

    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    rows = [
        {"type": "checkpoint.committed", "turn": 1, "sha": "1" * 40, "subject": "first"},
        {"type": "checkpoint.committed", "turn": 2, "sha": "2" * 40, "subject": "second"},
        {"type": "diff.updated", "patch": patch, "sha": "2" * 40},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    session = _session(tmp_path, "picked-AAA111", rows)
    (session / "manifest.json").write_text(json.dumps({"version": 3, "mode": "run"}))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/picked-AAA111")
    expect(page.locator(".card-diff .card-body")).to_contain_text("new")
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)  # answered once the loading line is seen

    page.route(f"**/diff?sha={'1' * 40}*", hold)
    page.locator(".card-diff select").select_option("1" * 40)
    body = page.locator(".card-diff .card-body > div").last
    expect(body).to_have_text("loading the diff…")
    while not held:
        page.wait_for_timeout(50)
    held[0].continue_()
    expect(body).to_have_text(f"no diff for {'1' * 12} (pruned, or not a commit of this run)")
    assert [error for error in errors if "Failed to load resource" not in error] == []


@pytest.mark.parametrize("width", [390, 320])
def test_an_answer_field_beside_short_options_shows_its_whole_placeholder(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """On a phone, a field that would cut its placeholder beside yes and no takes its own row.

    Beside two short options the field kept 221px and cut "pick an option or type an answer…",
    and at 320px its own row was 10px short of it.
    """
    from playwright.sync_api import expect

    ask = {
        "type": "question.prompt",
        "id": "question-1",
        "wait": True,
        "questions": [{"question": "Keep the flaky test quarantined?", "options": ["yes", "no"]}],
    }
    _session(tmp_path, "asking-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/asking-AAA111")
    field = page.locator(".prompt-box input.field")
    expect(field).to_be_visible()
    need, width = field.evaluate(
        """i => { const s = getComputedStyle(i);
        const c = document.createElement('canvas').getContext('2d');
        c.font = s.fontSize + ' ' + s.fontFamily;
        return [c.measureText(i.placeholder).width + parseFloat(s.paddingLeft)
            + parseFloat(s.paddingRight) + 2 * parseFloat(s.borderLeftWidth), i.offsetWidth]; }"""
    )
    assert need <= width
    assert not errors


def test_a_control_tabbed_to_under_the_drawers_fold_shows_its_whole_ring(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Tab scrolls each drawer control into view with its focus ring clear of the drawer's edge.

    The drawer scrolled a focused control flush with its bottom edge and cut the ring's outer 4px.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 600})
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    expect(page.locator(".card-findings .finding").first).to_be_attached()
    page.get_by_role("button", name="Details", exact=True).focus()
    edges = []
    for _ in range(40):
        page.keyboard.press("Tab")
        edge = page.evaluate(
            """() => { const a = document.activeElement, d = a.closest('.drawer');
            if (!d) return null;
            const r = a.getBoundingClientRect(), s = getComputedStyle(a);
            const ring = parseFloat(s.outlineOffset) + parseFloat(s.outlineWidth);
            const top = d.getBoundingClientRect().top + d.clientTop;
            return [r.top - ring - top, top + d.clientHeight - r.bottom - ring, d.scrollTop]; }"""
        )
        if edge:
            edges.append(edge)
    assert max(scrolled for _, _, scrolled in edges) > 0
    assert min(min(above, below) for above, below, _ in edges) >= 0
    assert not errors


def test_details_and_resolve_say_whether_what_they_open_shows(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Details and a finding's Resolve carry aria-expanded as the drawer or the form shows.

    Details showed the drawer's state by its colour alone, and Resolve not at all.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    details = page.get_by_role("button", name="Details", exact=True)
    drawer = page.locator(".run-app .drawer")
    expect(details).to_have_attribute("aria-expanded", "true")
    details.click()
    expect(drawer).to_be_hidden()
    expect(details).to_have_attribute("aria-expanded", "false")
    details.click()
    finding = page.locator(".card-findings .finding").first
    resolve = finding.get_by_role("button", name="Resolve", exact=True)
    expect(resolve).to_have_attribute("aria-expanded", "false")
    resolve.click()
    expect(finding.locator(".finding-form")).to_be_visible()
    expect(resolve).to_have_attribute("aria-expanded", "true")
    resolve.click()
    expect(finding.locator(".finding-form")).to_be_hidden()
    expect(resolve).to_have_attribute("aria-expanded", "false")
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
    _session(tmp_path, "keys-AAA111", [call, result, call, result, call, result, stop])
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
    _session(tmp_path, "detail-AAA111", rows)
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
