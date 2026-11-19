"""The session page driven in a browser: its prompts, action bar, run card and phone pages."""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths
from tests.web import session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port

LONG_COMMAND = (
    "sh -c 'rm -rf build dist && python -m pip install --no-deps -e . && .venv/bin/python -m "
    'pytest tests/test_csv_parse.py -k "quoted and not slow" --maxfail=1 2>&1 | tee /tmp/LAST.log\''
)


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
    session_logs.write_run(tmp_path, "asking-AAA111", [ask], live=True)
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
    session_logs.write_run(tmp_path, "asking-AAA111", [ask], live=True)
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
    session_logs.write_run(tmp_path, "asking-AAA111", [ask], live=True)
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
    session_logs.write_run(tmp_path, "long-AAA111", rows, live=True)
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
        session = session_logs.write_run(
            tmp_path, sid, [{"type": "model.call", "role": "worker", "model": "m1"}]
        )
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
        session = session_logs.write_run(tmp_path, f"fan-l{number}" if number else "fan", [stop])
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


def test_a_directive_the_composer_runs_closes_its_popup(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """/shells and Enter emptied the entry and left the directive popup open over it."""
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    session_logs.write_run(tmp_path, "shells-AAA111", [stop])
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/shells-AAA111")
    entry = page.locator(".composer textarea")
    expect(entry).to_have_attribute("placeholder", re.compile("continue|next"))
    entry.fill("/shells")
    expect(page.locator(".ac-pop .ac-item")).to_have_count(1)
    entry.press("Enter")
    expect(page.locator(".toast")).to_contain_text("no background commands this run")
    expect(entry).to_have_value("")
    expect(page.locator(".ac-pop")).to_have_count(0)
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
    session_logs.write_run(tmp_path, "menu-AAA111", [stop])
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
    session_logs.write_run(tmp_path, "menu-AAA111", [stop])
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
    session_logs.write_run(tmp_path, "asking-AAA111", [ask], live=True)
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


def test_a_phone_draft_starts_its_conversation_under_the_top_bar(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """On a phone a machine draft's conversation starts right under the top bar.

    The draft's action bar held only Details, which a phone hides, and kept a 20px empty band.
    """
    from playwright.sync_api import expect

    draft = paths.state_dir(tmp_path) / "sessions" / "machines" / "brave-otter"
    draft.mkdir(parents=True)
    rows = [
        {"type": "session.start", "mode": "machine", "user_task": "Author a triage machine"},
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}},
    ]
    (draft / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/draft/brave-otter")
    card = page.locator(".card-conv")
    expect(card.locator(".ci").first).to_be_visible()
    page.evaluate("window.scrollTo(0, 0)")
    header, top = page.locator("header").bounding_box(), card.bounding_box()
    assert header and top
    assert top["y"] == header["y"] + header["height"]
    assert not errors
