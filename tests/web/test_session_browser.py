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


def _reads(count: int, start: int = 0) -> list[dict[str, Any]]:
    """Return `count` read_file calls and their results, of files f<start>.py onwards."""
    rows: list[dict[str, Any]] = []
    for i in range(start, start + count):
        rows.append(
            {"type": "tool.call", "name": "read_file", "args_preview": {"path": f"f{i}.py"}}
        )
        rows.append({"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"})
    return rows


def _append(session: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a run's log, as its worker writes them."""
    with (session / "logs.jsonl").open("a", encoding="utf-8") as log:
        log.write("".join(json.dumps(row) + "\n" for row in rows))


# Once painted: the gap over the prompt, whether its box shows it whole, and that box's scrollTop.
PROMPT_PLACE = """async p => {
  await new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
  const box = p.closest('.conv-box'), host = p.parentElement;
  const shown = [...host.parentElement.children].filter(e => e !== host && e.offsetParent);
  const line = shown.at(-1).lastElementChild.getBoundingClientRect();
  const r = p.getBoundingClientRect(), b = box.getBoundingClientRect();
  const whole = r.top >= b.top - 1 && r.bottom <= b.bottom + 1; // a scroll offset is whole pixels
  return {gap: Math.round(r.top - line.bottom), whole, top: box.scrollTop};
}"""


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


def test_allow_is_an_approvals_one_filled_answer(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Allow all was filled as Allow is, so the wider answer read as the expected one."""
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow run_command: make"}
    session_logs.write_run(tmp_path, "asking-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/asking-AAA111")
    answers = page.locator(".prompt-box button")
    expect(answers).to_have_text(["Allow", "Allow all", "Deny", "Deny all"])
    expect(page.locator(".prompt-box button.primary")).to_have_text(["Allow"])
    assert not errors


@pytest.mark.parametrize("width", [1280, 390])
def test_the_action_bar_stands_under_the_top_bar_while_a_prompt_waits(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """The action bar stands 14px under the top bar, in the page's gutter, over a waiting prompt.

    The prompt boxes stood over the bar and pushed it down. The bar was a scroll container, which
    cut its buttons' focus rings, and on a phone it sat 6px right of the cards.
    """
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow run_command: ls"}
    session_logs.write_run(tmp_path, "asking-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/asking-AAA111")
    expect(page.locator(".prompt-box")).to_be_visible()
    actions = page.locator(".run-app .actions")
    page.evaluate("window.scrollTo(0, 0)")
    header, bar = page.locator("header").bounding_box(), actions.bounding_box()
    first = actions.locator("button:visible").first.bounding_box()
    entry = page.locator(".composer textarea").bounding_box()
    assert header is not None and bar is not None and first is not None and entry is not None
    gap = bar["y"] - (header["y"] + header["height"])
    overflow = actions.evaluate("e => getComputedStyle(e).overflow")
    assert (gap, overflow, first["x"]) == (14, "visible", entry["x"])
    assert not errors


def test_a_prompt_lands_after_the_conversation_and_moves_neither_bar(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A prompt lands 10px under the conversation's last line, in view while the box follows it.

    The prompts stood over the action bar, so the bar and Stop jumped down as one arrived. Output
    that lands later goes above the prompt, and a reader scrolled up keeps their place.
    """
    from playwright.sync_api import expect

    session = session_logs.write_run(tmp_path, "arrive-AAA111", _reads(40), live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto(f"http://127.0.0.1:{port}/#/session/arrive-AAA111")
    box = page.locator(".card-conv .conv-box")
    expect(box.locator(".ci").filter(has_text="f39.py")).to_be_attached()
    bars = page.locator(".run-app .actions, .composer.dock")
    before = bars.evaluate_all("all => all.map(e => e.getBoundingClientRect().toJSON())")
    _append(session, [{"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}])
    prompt = box.locator(".prompts .prompt-box")
    expect(prompt).to_be_visible()
    place = prompt.evaluate(PROMPT_PLACE)
    assert (place["gap"], place["whole"]) == (10, True)
    assert bars.evaluate_all("all => all.map(e => e.getBoundingClientRect().toJSON())") == before
    _append(session, _reads(1, 40))
    expect(box.locator(".ci").filter(has_text="f40.py")).to_be_attached()
    assert prompt.evaluate(PROMPT_PLACE)["whole"]
    box.evaluate("b => { b.scrollTop = 0; }")
    _append(session, _reads(1, 41))
    expect(box.locator(".ci").filter(has_text="f41.py")).to_be_attached()
    place = prompt.evaluate(PROMPT_PLACE)
    assert (place["whole"], place["top"]) == (False, 0)
    assert not errors


def test_a_conversation_at_its_tail_keeps_its_prompt_in_view_as_the_drawer_opens(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Opening Details narrows the conversation, which stays at its tail with its prompt in view.

    The browser's scroll anchoring held a short line in place as the long ones wrapped, which
    scrolled the box off its tail: the box stopped following and the prompt dropped out of view.
    """
    from playwright.sync_api import expect

    said = "The parser keeps a quoted CRLF and drops the doubled quote before the line ending. " * 3
    rows = _reads(30)
    for i in range(8):
        reply = {"type": "model.result", "role": "worker", "ok": True, "text": f"{i}. {said}"}
        rows += [*_reads(1, 30 + i), {**reply, "model_call_id": f"c{i}"}]
    rows.append({"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"})
    session_logs.write_run(tmp_path, "drawer-AAA111", rows, live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 820, "height": 1180})
    page.goto(f"http://127.0.0.1:{port}/#/session/drawer-AAA111")
    prompt = page.locator(".card-conv .prompts .prompt-box")
    expect(page.locator(".card-conv .ci").filter(has_text="7. The parser")).to_be_attached()
    expect(prompt).to_be_visible()
    assert prompt.evaluate(PROMPT_PLACE)["whole"]
    details = page.get_by_role("button", name="Details", exact=True)
    expect(details).to_have_attribute("aria-expanded", "false")
    details.click()
    expect(page.locator(".drawer")).to_be_visible()
    assert prompt.evaluate(PROMPT_PLACE)["whole"]
    assert not errors


@pytest.mark.parametrize("width", [1280, 390])
@pytest.mark.parametrize("live", [True, False])
def test_an_ended_runs_bar_offers_only_what_it_can_do_and_ends_in_delete(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    live: bool,
    width: int,
) -> None:
    """An ended run's bar has no Stop now, Stop after turn or Compact context; Delete ends the bar.

    The three sat disabled on every ended run, and Delete stood between Merge and the free space.
    """
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session"}
    session_logs.write_run(tmp_path, "bar-AAA111", [] if live else [stop], live=live)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/bar-AAA111")
    bar = page.locator(".run-app .actions")
    delete = bar.get_by_role("button", name="Delete", exact=True)
    live_controls = [
        bar.get_by_role("button", name=name, include_hidden=True)
        for name in ("Stop now", "Stop after turn", "Compact context")
    ]
    if live:
        expect(delete).to_be_disabled()
        for button in live_controls:
            expect(button).to_be_visible()
            expect(button).to_be_enabled()
    else:
        expect(delete).to_be_enabled()
        for button in live_controls:
            expect(button).to_be_hidden()
    end, room = delete.bounding_box(), bar.bounding_box()
    assert end is not None and room is not None
    assert end["x"] + end["width"] == pytest.approx(room["x"] + room["width"], abs=0.5)
    assert not errors


def test_the_composer_keeps_the_focus_while_a_resume_is_in_flight(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Enter in an ended run's composer disabled the entry, and the focus fell to the page.

    A refused resume gave the entry back its text but not the focus.
    """
    from playwright.sync_api import Route, expect

    session_logs.write_run(tmp_path, "resume-AAA111", [{"type": "session.stop"}])
    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/resume", hold)
    page.goto(f"http://127.0.0.1:{port}/#/session/resume-AAA111")
    entry = page.locator(".run-app .composer textarea")
    expect(entry).to_have_attribute("placeholder", re.compile("next|continue"))  # resumes
    entry.fill("go on")
    entry.press("Enter")
    while not held:
        page.wait_for_timeout(50)
    expect(entry).to_be_focused()
    expect(entry).not_to_be_editable()
    held[0].fulfill(status=409, json={"error": "refused"})
    expect(page.locator(".toast.bad")).to_contain_text("refused")
    expect(entry).to_be_editable()
    expect(entry).to_be_focused()
    expect(entry).to_have_value("go on")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


@pytest.mark.parametrize(
    ("model", "line"), [("o/vendor/configured", "configured"), ("", "no model")]
)
def test_a_phones_resume_pickers_fold_behind_one_line_that_names_their_picks(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    line: str,
) -> None:
    """On a phone the resume's preset and model fold behind one line that names both.

    The model dropdown beside its label cut "openrouter/moonshotai/kimi-k2.6 (config de" mid-word.
    With no model the line read "preset: none · none". A desktop keeps the row.
    """
    from playwright.sync_api import expect

    if model:
        config = tmp_path / "config" / "agent6"
        config.mkdir(parents=True)
        (config / "config.toml").write_text(
            '[providers.o]\nbackend = "openai"\nbase_url = "https://x/v1"\n'
            f'[models.worker]\nmodel = "{model}"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config.parent))
    stop = {"type": "session.stop", "reason": "finish_session"}
    session_logs.write_run(tmp_path, "ended-AAA111", [stop])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/ended-AAA111")
    composer = page.locator(".composer")
    summary = composer.get_by_role("button", name=re.compile("^preset: "))
    row = composer.locator(".pickers")
    expect(summary).to_have_text("preset: none · " + line)
    expect(row).to_be_hidden()
    summary.click()
    expect(summary).to_have_attribute("aria-expanded", "true")
    expect(row).to_be_visible()
    page.set_viewport_size({"width": 1280, "height": 900})
    expect(summary).to_be_hidden()
    expect(row).to_be_visible()
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


def test_the_approval_a_run_waits_on_sits_under_its_queued_questions(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The approval a run waits on stays last, whole at the conversation's tail, as questions queue.

    The approval painted first, so a queued question under it cut its question off at the top of a
    short conversation.
    """
    from playwright.sync_api import expect

    def queued(qid: str, text: str) -> dict[str, Any]:
        ask = [{"question": text, "options": []}]
        return {"type": "question.prompt", "id": qid, "wait": False, "questions": ask}

    command = f"Allow run_command: {LONG_COMMAND}"
    approval = {"type": "approval.prompt", "id": "approval-1", "prompt": command}
    rows = [*_reads(30), queued("question-1", "Fixtures under tests/data?"), approval]
    session = session_logs.write_run(tmp_path, "order-AAA111", rows, live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto(f"http://127.0.0.1:{port}/#/session/order-AAA111")
    boxes = page.locator(".card-conv .prompts .prompt-box")
    expect(boxes).to_contain_text(["Fixtures under tests/data?", "Allow run_command"])
    _append(session, [queued("question-2", "Keep the BOM?")])
    expect(boxes).to_contain_text(["Fixtures", "Keep the BOM?", "Allow run_command"])
    assert boxes.last.evaluate(PROMPT_PLACE)["whole"]
    assert not errors


@pytest.mark.parametrize("children", [False, True])
@pytest.mark.parametrize("asks", [False, True])
def test_a_phone_opens_a_run_at_its_tail_where_a_prompt_waits(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    *,
    asks: bool,
    children: bool,
) -> None:
    """A phone opens a long conversation at its end, where a waiting prompt sits over the composer.

    With a prompt waiting the page opened at its top, about 2,500px above the conversation's end,
    and a fan-out's did still. A fan-out with no prompt waiting opens at its top, on its children.
    """
    from playwright.sync_api import expect

    rows = _reads(40)
    if asks:
        rows.append({"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"})
    session = session_logs.write_run(tmp_path, "long-AAA111", rows, live=True)
    if children:
        stop = {"type": "session.stop", "verification": "passed", "reason": "finish_session"}
        group = {"group": session.name, "parent": session.name}
        for number in (1, 2):
            child = session_logs.write_run(tmp_path, f"{session.name}-l{number}", [stop])
            manifest = {"version": 3, "session_id": child.name, "mode": "run"}
            (child / "manifest.json").write_text(
                json.dumps({**manifest, "parallel": {**group, "number": number}})
            )
        manifest = {"version": 3, "session_id": session.name, "mode": "run"}
        (session / "manifest.json").write_text(json.dumps(manifest))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/long-AAA111")
    expect(page.locator(".card-conv .ci").filter(has_text="f39.py")).to_be_attached()
    if children:
        expect(page.locator(".card-children tr")).to_have_count(3)
    if children and not asks:
        assert page.evaluate("window.scrollY") == 0
    else:
        page.wait_for_function("window.scrollY > 0")
    if asks:
        prompt = page.locator(".card-conv .prompt-box")
        expect(prompt).to_be_in_viewport(ratio=1)
        box, dock = prompt.bounding_box(), page.locator(".composer.dock").bounding_box()
        assert box is not None and dock is not None
        assert box["y"] + box["height"] <= dock["y"]
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
    assert name == ["left", "600"]  # the card-label style
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


def test_shells_opens_the_drawer_as_details_does(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """/shells opened a closed drawer behind Details' back: Details still read collapsed."""
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    session = session_logs.write_run(tmp_path, "shells-AAA111", [stop])
    shell = session / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(json.dumps({"command": "python -m http.server"}))
    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-drawer', 'closed')")
    page.goto(f"http://127.0.0.1:{port}/#/session/shells-AAA111")
    details = page.get_by_role("button", name="Details", exact=True)
    expect(details).to_have_attribute("aria-expanded", "false")
    entry = page.locator(".composer textarea")
    expect(entry).to_have_attribute("placeholder", re.compile("continue|next"))
    entry.fill("/shells")
    entry.press("Escape")
    entry.press("Enter")
    expect(page.locator(".card-shells")).to_be_visible()
    expect(details).to_have_attribute("aria-expanded", "true")
    assert page.evaluate("localStorage.getItem('a6-drawer')") == "open"
    assert not errors


def test_a_phone_shows_each_directives_whole_help(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """On a phone the directive popup cut every help line short with an ellipsis."""
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    session_logs.write_run(tmp_path, "help-AAA111", [stop])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/help-AAA111")
    entry = page.locator(".composer textarea")
    expect(entry).to_have_attribute("placeholder", re.compile("continue|next"))
    entry.fill("/")
    rows = page.locator(".ac-pop .ac-item")
    expect(rows.first).to_be_visible()
    assert rows.count() > 1
    assert rows.evaluate_all("rows => rows.every(r => r.scrollWidth <= r.clientWidth)")
    assert not errors


def test_the_phone_widget_menu_closes_on_esc_an_outside_tap_and_a_pick(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The phone's widget menu is named Widgets and closes on Esc, a tap outside or a pick.

    The menu button was named "☰", and the menu stayed open over the page after Esc or a tap
    outside it. Then a menu left open while the window widened past 780px stayed in the page.
    The button gave a pointer no title, as the theme button beside it does.
    """
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    session_logs.write_run(tmp_path, "menu-AAA111", [stop])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/menu-AAA111")
    button = page.get_by_role("button", name="Widgets", exact=True)
    expect(button).to_have_attribute("title", "Widgets")
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


def test_the_session_page_announces_its_run_waiting_and_ending(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The session page announced nothing when its run started waiting on the operator or ended.

    It announces both in the hub's words, and opening a waiting run replays nothing.
    """
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}
    session = session_logs.write_run(tmp_path, "say-AAA111", [ask], live=True)
    page, errors = open_page(port)
    page.add_init_script(
        """window.said = [];
        { const hear = rs => rs.forEach(r => r.target.id === 'toasts'
            && r.addedNodes.forEach(n => said.push(n.textContent)));
          new MutationObserver(hear).observe(document, { childList: true, subtree: true }); }"""
    )
    page.goto(f"http://127.0.0.1:{port}/#/session/say-AAA111")
    expect(page.locator(".prompt-box")).to_be_visible()
    # A line streamed in while the run still waits: the frames the page opened on came first.
    _append(session, _reads(1))
    expect(page.locator(".card-conv .ci").filter(has_text="f0.py")).to_be_attached()
    assert page.evaluate("said") == []
    answer = {"type": "approval.answer", "approved": True, "source": "web"}
    _append(session, [{**answer, "id": "approval-1"}])
    expect(page.locator(".prompt-box")).to_have_count(0)
    _append(session, [{**ask, "id": "approval-2"}])
    page.wait_for_function("said.length === 1")
    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    _append(session, [{**answer, "id": "approval-2"}, stop])
    page.wait_for_function("said.length === 2")
    assert page.evaluate("said") == ["say-AAA111 needs you", "say-AAA111 passed"]
    assert not errors


# A granted Notification that records the title of each one made.
NOTIFIED = """window.notes = [];
window.Notification = class { constructor(title) { notes.push(title); }
  static get permission() { return 'granted'; }
  static requestPermission() { return Promise.resolve('granted'); } };"""


def test_one_wait_raises_one_os_notification_with_the_hub_in_another_tab(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """With the hub in one tab and a session in another, one wait raised two OS notifications.

    The hub raises it; the session's page announces it as a toast only.
    """
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}
    session = session_logs.write_run(tmp_path, "two-AAA111", [ask], live=True)
    hub, errors = open_page(port)
    hub.context.add_init_script(NOTIFIED)
    hub.goto(f"http://127.0.0.1:{port}/#/")
    status = hub.locator(".item").filter(has_text="two-AAA111").locator(".pill")
    expect(status).to_contain_text("waiting")
    run = hub.context.new_page()
    run.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    run.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    run.goto(f"http://127.0.0.1:{port}/#/session/two-AAA111")
    expect(run.locator(".prompt-box")).to_be_visible()
    answer = {"type": "approval.answer", "id": "approval-1", "approved": True, "source": "web"}
    _append(session, [answer])
    expect(run.locator(".prompt-box")).to_have_count(0)
    expect(status).to_contain_text("running")
    _append(session, [{**ask, "id": "approval-2"}])
    expect(run.locator(".prompt-box")).to_be_visible()
    expect(run.locator("#toasts")).to_contain_text("two-AAA111 needs you")
    hub.wait_for_function("notes.length > 0")
    assert hub.evaluate("notes") + run.evaluate("notes") == ["agent6: two-AAA111 needs you"]
    assert not errors


def test_a_prompt_waiting_behind_another_widget_marks_the_widget_menu(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A prompt that landed behind another widget on a phone showed only as the top bar's state.

    The widget button and the menu's Conversation entry take the status pill's dot in the warning
    colour, the button's name says a prompt waits, and the page stays on the reader's widget.
    """
    from playwright.sync_api import expect

    session = session_logs.write_run(tmp_path, "mark-AAA111", _reads(2), live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/mark-AAA111")
    menu = page.locator(".wmenu")
    page.get_by_role("button", name="Widgets", exact=True).click()
    menu.get_by_role("button", name="Budget", exact=True).click()
    _append(session, [{"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}])
    button = page.get_by_role("button", name="Widgets: a prompt waits in Conversation", exact=True)
    expect(button).to_have_attribute("title", "Widgets: a prompt waits in Conversation")
    expect(page.locator(".card-budget")).to_be_visible()
    dot = "e => getComputedStyle(e, '::after').backgroundColor"
    warn = page.locator("#crumb > .pill.warn").evaluate(
        "e => getComputedStyle(e, '::before').backgroundColor"
    )
    assert button.evaluate(dot) == warn
    button.click()
    entries = menu.locator("button:visible")
    expect(entries.first).to_have_text("Conversation")
    dots = entries.evaluate_all("bs => bs.map(b => getComputedStyle(b, '::after').content)")
    assert dots[0] != "none" and set(dots[1:]) == {"none"}, dots
    assert entries.first.evaluate(dot) == warn
    entries.first.click()
    expect(page.get_by_role("button", name="Widgets", exact=True)).to_be_visible()
    assert page.locator(".wmenu-btn").evaluate("b => getComputedStyle(b, '::after').content") == (
        "none"
    )
    assert not errors


def test_the_phone_widget_button_draws_the_theme_buttons_icon_in_its_square(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The widget button drew the glyph ☰, lighter and smaller than the theme button's icon."""
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    session_logs.write_run(tmp_path, "menu-AAA111", [stop])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/menu-AAA111")
    button = page.get_by_role("button", name="Widgets", exact=True)
    expect(button).to_be_visible()
    boxes = button.evaluate(
        """b => [b, b.querySelector('svg.ic'), document.querySelector('header .theme-toggle'),
          document.querySelector('header .theme-toggle svg.ic')]
          .map(e => e && e.getBoundingClientRect().toJSON())"""
    )
    own, icon, theme, theme_icon = boxes
    assert icon is not None, "no line icon"
    assert button.text_content() == ""
    assert (own["width"], own["height"]) == (theme["width"], theme["height"])
    assert (icon["width"], icon["height"]) == (theme_icon["width"], theme_icon["height"])
    assert icon["x"] - own["x"] == theme_icon["x"] - theme["x"]
    assert icon["y"] - own["y"] == theme_icon["y"] - theme["y"]
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
