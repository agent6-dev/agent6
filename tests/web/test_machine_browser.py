# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine view driven in a browser: its layout, its lists, its titles and its composer."""

from __future__ import annotations

import json
import os
import pathlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths
from agent6.sessions import process
from tests.web import session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port

OpenPage = Callable[[int], tuple["Page", list[str]]]
TS = "2026-11-13T10:00:00+00:00"
AGENT = {"kind": "agent", "outcome": "ok", "reason": "finish_session"}
TOOL = {"kind": "tool", "exit_code": 0, "stdout": "", "timed_out": False}
# An agent state, a tool state, a wait and an end.
LOOP = """
machine = "loop"
version = 1
initial = "work"

[budget]
max_transitions = 500

[vars.agent]
report = { type = "report", default = {} }

[schemas.report]
done = { type = "bool" }

[states.work]
kind = "agent"
prompt = "Do the work."
output_schema = "report"
capture = { finish_json = "report" }
timeout_s = 600
on = { ok = "check", failed = "check", budget_exhausted = "done", timeout = "check" }

[states.check]
kind = "tool"
command = ["true"]
timeout_s = 60
on = { ok = "rest", nonzero = "work", timeout = "work" }

[states.rest]
kind = "wait"
delay_s = 86400
on = { tick = "work", signal = "work" }

[states.done]
kind = "terminal"
status = "ok"
reason = "done"
"""
COLLECT = "collect_merged_pull_requests_since_the_previous_release_tag_for_every_component"
GIVEUP = "giveup_because_the_collector_could_not_reach_the_forge_after_three_attempts"
# Two states whose names are longer than half a 1280px page.
LONG = f"""
machine = "long"
version = 1
initial = "{COLLECT}"

[budget]
max_transitions = 10

[states.{COLLECT}]
kind = "tool"
command = ["true"]
timeout_s = 60
on = {{ ok = "{GIVEUP}", nonzero = "{GIVEUP}", timeout = "{GIVEUP}" }}

[states.{GIVEUP}]
kind = "terminal"
status = "failed"
reason = "the collector could not reach the forge"
"""
STAGES = [f"stage_{i:02d}" for i in range(30)]


def _chain() -> str:
    """Return a spec of 30 tool states in a loop, which a failing state ends."""
    spec = ['machine = "chain"', "version = 1", 'initial = "stage_00"', "", "[budget]"]
    spec += ["max_transitions = 500", ""]
    for i, name in enumerate(STAGES):
        after = STAGES[(i + 1) % len(STAGES)]
        spec += [f"[states.{name}]", 'kind = "tool"', 'command = ["true"]', "timeout_s = 60"]
        spec += [f'on = {{ ok = "{after}", nonzero = "finished", timeout = "finished" }}', ""]
    spec += ["[states.finished]", 'kind = "terminal"', 'status = "ok"', 'reason = "done"']
    return "\n".join(spec) + "\n"


def _step(seq: int, state: str, label: str, goto: str, fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "step",
        "ts": TS,
        "seq": seq,
        "state": state,
        "label": label,
        "goto": goto,
        "fact": fact,
    }


def _chain_step(seq: int) -> dict[str, Any]:
    """Return the chain's step `seq`: each stage passes to the next, the last back to the first."""
    return _step(seq, STAGES[seq % 30], "ok", STAGES[(seq + 1) % 30], TOOL)


def _machine(
    root: pathlib.Path,
    name: str,
    spec: str,
    rows: list[dict[str, Any]],
    log: list[dict[str, Any]] | None = None,
    *,
    alive: bool = False,
) -> pathlib.Path:
    """Write an instance: its spec, its journal and the log of the agent state `work`."""
    machine = paths.state_dir(root) / "machines" / name
    machine.mkdir(parents=True)
    (machine / "machine.asm.toml").write_text(spec, encoding="utf-8")
    begin = {"type": "machine.begin", "ts": TS, "machine": name, "version": 1}
    journal = "".join(json.dumps(row) + "\n" for row in [begin, *rows])
    (machine / "journal.jsonl").write_text(journal, encoding="utf-8")
    if log is not None:
        state = machine / "states" / "0000-work"
        state.mkdir(parents=True)
        (state / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
        if alive:
            process.write_worker_pid(state, os.getpid())
    if alive:
        process.write_worker_pid(machine, os.getpid())
    return machine


def _talk(lines: int) -> list[dict[str, Any]]:
    """Return an agent state's log: its start and `lines` paragraphs from the model."""
    said = [
        {
            "type": "model.result",
            "role": "worker",
            "ok": True,
            "model_call_id": f"c{i}",
            "text": f"Step {i}: read the next file and noted what it does.",
        }
        for i in range(lines)
    ]
    return [{"type": "session.start", "mode": "run", "user_task": "Do the work."}, *said]


def _ended(root: pathlib.Path, lines: int = 3) -> None:
    """Write `ended`: its agent state talked, then the machine ended in `done`."""
    end = {
        "type": "machine.end",
        "ts": TS,
        "status": "ok",
        "reason": "done",
        "state": "done",
        "transitions": 1,
    }
    rows = [_step(0, "work", "budget_exhausted", "done", AGENT), end]
    _machine(root, "ended", LOOP, rows, _talk(lines))


def _parked(root: pathlib.Path) -> None:
    """Write `parked`: waiting in `rest` with no worker, so only a poke reaches it."""
    rows = [_step(0, "work", "ok", "check", AGENT), _step(1, "check", "ok", "rest", TOOL)]
    machine = _machine(root, "parked", LOOP, rows, _talk(2))
    wait = {"state": "rest", "wake_epoch": time.time() + 86400, "seq": 2}
    (machine / "wait.json").write_text(json.dumps(wait) + "\n", encoding="utf-8")


def _open(open_page: OpenPage, port: int, name: str) -> tuple[Page, list[str]]:
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/machine/{name}")
    page.locator(".tree .node").first.wait_for()
    return page, errors


def test_the_newest_conversation_lines_sit_above_the_composer(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The conversation box ran under the docked composer, which hid its newest lines."""
    from playwright.sync_api import expect

    _ended(tmp_path, lines=40)
    page, errors = _open(open_page, port, "ended")
    last = page.locator(".conv .ci").last
    expect(last).to_contain_text("Step 39")
    page.wait_for_timeout(300)  # the box follows its tail once the items paint
    dock = page.locator(".composer.dock").bounding_box()
    line = last.bounding_box()
    assert dock is not None and line is not None
    assert line["y"] + line["height"] <= dock["y"] + 1, (line, dock)
    assert page.evaluate("document.scrollingElement.scrollHeight <= innerHeight")
    assert not errors


def test_a_short_screen_scrolls_rather_than_fold_the_cards(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A phone held sideways folded States and Path to their headings and clipped the status."""
    _machine(tmp_path, "chain", _chain(), [_chain_step(seq) for seq in range(12)])
    page, errors = _open(open_page, port, "chain")
    page.set_viewport_size({"width": 844, "height": 390})
    shape = page.evaluate(
        """() => {
          const cards = [...document.querySelectorAll('.tree')].map(t => t.closest('.card'));
          const pill = document.querySelector('.machine-grid .pill').getBoundingClientRect();
          const dock = document.querySelector('.composer.dock').getBoundingClientRect();
          return {heights: cards.map(c => c.getBoundingClientRect().height),
                  pillShown: pill.bottom <= cards[0].getBoundingClientRect().bottom,
                  scrolls: document.scrollingElement.scrollHeight > innerHeight,
                  docked: dock.bottom === innerHeight};
        }"""
    )
    assert min(shape["heights"]) >= 120 and shape["pillShown"], shape
    assert shape["scrolls"] and shape["docked"], shape
    assert not errors


@pytest.mark.parametrize("width", [1440, 390])
def test_the_cards_stand_under_the_action_bar_as_the_run_views_panes_do(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path, width: int
) -> None:
    """The cards stood 10px under the action bar with no rule, the States and Path line in mid-air.

    The run view rules its panes off its bar, 14px under it, from 781px; a phone draws no rule.
    """
    _ended(tmp_path)
    session_logs.write_run(tmp_path, "ended-AAA111", [{"type": "session.stop"}])
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    under = """([bar, body]) => { const b = document.querySelector(bar).getBoundingClientRect();
      const p = document.querySelector(body);
      return [p.getBoundingClientRect().top - b.bottom, getComputedStyle(p).borderTopWidth]; }"""
    page.goto(f"http://127.0.0.1:{port}/#/session/ended-AAA111")
    page.locator(".card-conv").wait_for()
    run = page.evaluate(under, [".run-app > .actions", ".run-body"])
    page.goto(f"http://127.0.0.1:{port}/#/machine/ended")
    page.locator(".tree .node").first.wait_for()
    machine = page.evaluate(under, [".machine-app > .actions", ".machine-grid"])
    assert machine == run == [14, "1px" if width > 780 else "0px"]
    assert not errors


def test_states_and_path_share_one_row_height(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The Path card kept its own height, so the line between it and States stopped part way."""
    _ended(tmp_path)
    page, errors = _open(open_page, port, "ended")
    heights = page.evaluate(
        "() => [...document.querySelectorAll('.tree')].map(t => t.closest('.card')"
        ".getBoundingClientRect().height)"
    )
    assert len(heights) == 2 and heights[0] == heights[1], heights
    assert not errors


def test_a_machines_prompt_waits_at_its_conversations_end(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A pending prompt follows the conversation's last line, over the composer.

    The prompts stood over Stop, which moved down as each one arrived.
    """
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}
    _machine(tmp_path, "asking", LOOP, [], [*_talk(1), ask], alive=True)
    page, errors = _open(open_page, port, "asking")
    expect(page.locator(".conv .ci").last).to_contain_text("Step 0")
    box = page.locator(".conv-box .prompts .prompt-box")
    expect(box).to_have_count(1)
    expect(box).to_be_in_viewport(ratio=1)
    gap = box.evaluate(
        """p => { const host = p.parentElement;
        const shown = [...host.parentElement.children].filter(e => e !== host && e.offsetParent);
        return Math.round(p.getBoundingClientRect().top
          - shown.at(-1).lastElementChild.getBoundingClientRect().bottom); }"""
    )
    header, stop = page.locator("header").bounding_box(), page.locator(".actions").bounding_box()
    prompt, dock = box.bounding_box(), page.locator(".composer.dock").bounding_box()
    assert header is not None and stop is not None and prompt is not None and dock is not None
    assert stop["y"] - (header["y"] + header["height"]) == 14
    assert prompt["y"] + prompt["height"] <= dock["y"]
    assert gap == 10
    assert not errors


@pytest.mark.parametrize("streamed", [False, True])
def test_a_state_waiting_on_the_operator_shows_no_live_line(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path, *, streamed: bool
) -> None:
    """A state blocked on an approval showed a live line under a rule, over its prompt.

    The heartbeat goes quiet while the operator is asked, so the line stood empty; a thought the
    last call streamed showed as "thinking…" beside a prompt that blocked every next step.
    """
    from playwright.sync_api import expect

    call = {"role": "worker", "model_call_id": "c9"}
    log = [*_talk(2), {"type": "model.call", "model": "m", "provider": "p", **call}]
    if streamed:
        log.append({"type": "model.thinking_delta", "text": "I will run the check.", **call})
    log.append({"type": "model.result", "ok": True, **call})
    log.append({"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"})
    _machine(tmp_path, "asking", LOOP, [], log, alive=True)
    page, errors = _open(open_page, port, "asking")
    expect(page.locator(".conv-box .prompt-box")).to_be_visible()
    expect(page.locator(".conv-card h2")).to_contain_text("Current state")  # the first frame
    expect(page.locator(".conv-live")).to_be_hidden()
    assert not errors


def test_a_poll_that_answers_a_prompt_and_streams_on_leaves_the_conversation_at_its_tail(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """One poll takes an answered prompt's box away and streams new text: the box keeps its tail.

    The Path's follow laid the page out between the two, which clamped the conversation's offset,
    and the clamp's scroll event read the text under it as the reader scrolling up.
    """
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}
    machine = _machine(tmp_path, "asking", LOOP, [], [*_talk(40), ask], alive=True)
    page, errors = _open(open_page, port, "asking")
    box = page.locator(".conv-box")
    expect(box.locator(".prompt-box")).to_have_count(1)
    expect(box).to_have_class("conv-box tail scrolled")
    said = "The parser keeps a quoted CRLF and drops the doubled quote before the line ending. "
    call = {"role": "worker", "model_call_id": "n1"}
    rows = [
        {"type": "approval.answer", "id": "approval-1", "approved": True, "source": "web"},
        {"type": "model.call", "model": "m", "provider": "p", **call},
        {"type": "model.text_delta", "text": said * 6, **call},
    ]
    with (machine / "states" / "0000-work" / "logs.jsonl").open("a", encoding="utf-8") as log:
        log.write("".join(json.dumps(row) + "\n" for row in rows))
    expect(box.locator(".prompt-box")).to_have_count(0)
    expect(box.locator(".conv-live")).to_contain_text("The parser keeps")
    after = box.evaluate(
        """b => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(() =>
        done([b.className, b.scrollHeight - b.scrollTop - b.clientHeight]))))"""
    )
    assert after == ["conv-box tail scrolled", 0]
    assert not errors


def test_the_next_agent_state_is_followed_after_a_scroll_up_in_the_last(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A box the reader had scrolled up kept its place while the next state's first lines fit it.

    That state's lines then streamed past the box's end, and the box stayed at its top.
    """
    from playwright.sync_api import expect

    machine = _machine(tmp_path, "looping", LOOP, [], _talk(40), alive=True)
    page, errors = _open(open_page, port, "looping")
    expect(page.locator(".conv .ci").last).to_contain_text("Step 39")
    box = page.locator(".conv-box")
    expect(box).to_have_class("conv-box tail scrolled")
    box.evaluate("b => { b.scrollTop = 0; }")
    expect(box).to_have_class("conv-box")
    log = machine / "states" / "0002-work" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    talk = _talk(42)
    log.write_text("".join(json.dumps(e) + "\n" for e in talk[:3]))
    expect(page.locator(".conv .ci")).to_have_count(3)
    with log.open("a", encoding="utf-8") as more:
        more.write("".join(json.dumps(e) + "\n" for e in talk[3:]))
    expect(page.locator(".conv .ci").last).to_contain_text("Step 41")
    at_end = "b => b.scrollTop > 0 && b.scrollTop + b.clientHeight >= b.scrollHeight - 1"
    page.wait_for_function(f"({at_end})(document.querySelector('.conv-box'))", timeout=5000)
    assert not errors


def test_long_states_and_transitions_wrap_under_their_text(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A long state or transition was cut with an ellipsis, and its end could not be read.

    A wrapped line continues under the text after its mark or its `[seq] `.
    """
    failed = {
        "kind": "tool",
        "exit_code": 1,
        "stdout": "",
        "timed_out": False,
        "stderr": (
            "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='forge.example',"
            " port=443): Max retries exceeded with url: /api/v4/projects/platform%2Finfra"
        ),
    }
    end = {
        "type": "machine.end",
        "ts": TS,
        "status": "failed",
        "reason": "no forge",
        "state": GIVEUP,
        "transitions": 1,
    }
    _machine(tmp_path, "long", LONG, [_step(0, COLLECT, "nonzero", GIVEUP, failed), end])
    page, errors = _open(open_page, port, "long")
    # Each line's start is its leftmost fragment: a line can end in a fragment of spaces.
    lines = page.evaluate(
        """() => [...document.querySelectorAll('.tree .node')].map(node => {
          const range = document.createRange();
          range.selectNodeContents(node.lastChild);
          const starts = new Map();
          for (const r of range.getClientRects()) {
            const top = Math.round(r.top);
            starts.set(top, Math.min(starts.get(top) ?? Infinity, Math.round(r.left)));
          }
          return {text: node.textContent.slice(0, 30), cut: node.scrollWidth > node.clientWidth + 1,
                  lines: starts.size, lefts: [...new Set(starts.values())]};
        })"""
    )
    assert [line for line in lines if line["cut"]] == [], lines
    wrapped = [line for line in lines if line["lines"] > 1]
    assert len(wrapped) >= 3, lines  # both long states and the transition
    assert all(len(line["lefts"]) == 1 for line in wrapped), wrapped
    assert not errors


def test_the_current_state_and_the_newest_transition_open_in_view(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """States opened at its top with the current state below the fold, Path at its oldest line.

    Path follows each new transition while the reader stays at its end, and keeps the reader's
    place once they scroll up.
    """
    from playwright.sync_api import expect

    machine = _machine(tmp_path, "chain", _chain(), [_chain_step(seq) for seq in range(55)])
    page, errors = _open(open_page, port, "chain")
    in_view = """() => {
      const node = document.querySelector('.tree .cursor');
      const pane = node.parentElement.parentElement, a = node.getBoundingClientRect();
      const b = pane.getBoundingClientRect();
      return pane.scrollTop > 0 && a.top >= b.top && a.bottom <= b.bottom;
    }"""
    at_end = """() => {
      const pane = [...document.querySelectorAll('.tree')][1].parentElement;
      return pane.scrollHeight > pane.clientHeight
        && pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 1;
    }"""
    expect(page.locator(".tree .cursor")).to_contain_text("stage_25")
    assert page.evaluate(in_view)
    assert page.evaluate(at_end)

    def add(seq: int) -> None:
        with (machine / "journal.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(_chain_step(seq)) + "\n")
        expect(page.locator(".tree .node", has_text=f"[{seq}]")).to_have_count(1)

    page.evaluate(
        "() => { [...document.querySelectorAll('.tree')][1].parentElement.scrollTop = 0 }"
    )
    add(55)
    assert page.evaluate("[...document.querySelectorAll('.tree')][1].parentElement.scrollTop") == 0
    page.evaluate(
        "() => { const p = [...document.querySelectorAll('.tree')][1].parentElement;"
        " p.scrollTop = p.scrollHeight; }"
    )
    add(56)
    assert page.evaluate(at_end)
    expect(page.locator(".tree .cursor")).to_contain_text("stage_27")
    assert page.evaluate(in_view)
    assert not errors


@pytest.mark.parametrize("font", [16, 20])
def test_path_stays_at_its_end_when_a_prompt_or_the_entry_shrinks_it(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path, font: int
) -> None:
    """Path stopped short of its newest transition once a prompt or a growing entry shrank it.

    At a 20px browser font the dock grew after the first scroll to the end, and that scroll's
    event read as the reader leaving the end.
    """
    from playwright.sync_api import expect

    failed = {**TOOL, "exit_code": 1}
    rows = [
        _step(seq, "check", "nonzero", "work", failed)
        if seq % 2
        else _step(seq, "work", "ok", "check", AGENT)
        for seq in range(16)
    ]
    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}
    _machine(tmp_path, "asking", LOOP, rows, [*_talk(1), ask], alive=True)
    page, errors = open_page(port)
    cdp = page.context.new_cdp_session(page)
    cdp.send("Page.setFontSizes", {"fontSizes": {"standard": font}})
    page.goto(f"http://127.0.0.1:{port}/#/machine/asking")
    page.locator(".tree .node").first.wait_for()
    assert page.evaluate("getComputedStyle(document.documentElement).fontSize") == f"{font}px"
    at_end = """async () => {
      await new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
      const pane = [...document.querySelectorAll('.tree')][1].parentElement;
      return pane.scrollHeight > pane.clientHeight
        && pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 1;
    }"""
    expect(page.locator(".prompt-box")).to_have_count(1)
    assert page.evaluate(at_end)
    page.locator(".dock textarea").click()
    for line in range(5):
        page.keyboard.type(f"line {line}")
        page.keyboard.press("Shift+Enter")
    assert page.evaluate(at_end)
    assert not errors


def test_the_conversation_names_the_agent_state_it_shows(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The card read "Current state" over a past agent state's conversation in a wait state.

    A state entered again is current only once its new execution has a log.
    """
    from playwright.sync_api import expect

    _parked(tmp_path)
    _machine(tmp_path, "working", LOOP, [], _talk(2), alive=True)
    failed = {**TOOL, "exit_code": 1}
    rows = [_step(0, "work", "ok", "check", AGENT), _step(1, "check", "nonzero", "work", failed)]
    _machine(tmp_path, "again", LOOP, rows, _talk(2))
    page, errors = _open(open_page, port, "parked")
    title = page.locator(".conv-card h2")
    expect(title).to_have_text("Last agent state · work")
    page.goto(f"http://127.0.0.1:{port}/#/machine/working")
    expect(title).to_have_text("Current state · work")
    page.goto(f"http://127.0.0.1:{port}/#/machine/again")
    expect(title).to_have_text("Last agent state · work")
    assert not errors


def test_the_filled_button_is_the_verb_enter_sends(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A parked machine kept the fill on its refused Steer while Enter sent Message."""
    from playwright.sync_api import expect

    _parked(tmp_path)
    _machine(tmp_path, "working", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "parked")
    steer, message = (
        page.locator(".dock button", has_text="Steer"),
        page.locator(".dock button", has_text="Message"),
    )
    expect(message).to_be_enabled()
    expect(message).to_contain_class("primary")
    expect(steer).to_be_disabled()
    expect(steer).not_to_contain_class("primary")
    page.goto(f"http://127.0.0.1:{port}/#/machine/working")
    expect(steer).to_be_enabled()
    expect(steer).to_contain_class("primary")
    expect(message).not_to_contain_class("primary")
    assert not errors


def test_a_closed_entry_says_why_it_takes_no_input(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """An ended machine's disabled entry still invited "steer or message the machine…"."""
    from playwright.sync_api import expect

    _ended(tmp_path)
    page, errors = _open(open_page, port, "ended")
    entry = page.locator(".dock textarea")
    expect(entry).to_be_disabled()
    refusals = page.evaluate("getJSON('/api/machine/ended').then(m => m.refusals)")
    assert "already ended" in refusals["steer"] and "already ended" in refusals["poke"]
    expect(entry).to_have_attribute("placeholder", "")
    expect(page.get_by_text(refusals["steer"], exact=True)).to_be_visible()  # the end, said once
    expect(page.locator(".dock .hint")).to_be_hidden()
    assert not errors


def test_the_status_names_its_reason_and_a_parked_machine_says_when_it_wakes(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The page read a bare "waiting" where the list read "waiting · no worker".

    Nothing on the page said when a parked machine wakes; the Path's end says it, as
    `machine status` does.
    """
    from playwright.sync_api import expect

    _parked(tmp_path)
    page, errors = _open(open_page, port, "parked")
    expect(page.locator(".machine-grid .pill")).to_have_text("waiting · no worker")
    path = page.locator(".machine-grid > .card").nth(1)
    expect(path).to_contain_text("parked in 'rest' with no worker running (due at ")
    expect(path.locator("code")).to_have_text("agent6 machine run parked")
    assert not errors


def test_an_ended_machines_dock_stays_short_on_a_phone(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """An ended machine's dock said a long end reason twice, in full, over half a phone's screen."""
    from playwright.sync_api import expect

    reason = (
        "the collector exited nonzero three times: HTTPSConnectionPool(host='forge.example',"
        " port=443): Max retries exceeded with url: /api/v4/projects/platform%2Finfra"
        "/merge_requests?state=merged (Caused by NewConnectionError: Failed to establish a new"
        " connection: [Errno 111] Connection refused)"
    )
    end = {
        "type": "machine.end",
        "ts": TS,
        "status": "failed",
        "reason": reason,
        "state": "done",
        "transitions": 1,
    }
    rows = [_step(0, "work", "budget_exhausted", "done", AGENT), end]
    _machine(tmp_path, "ended", LOOP, rows, _talk(1))
    page, errors = _open(open_page, port, "ended")
    page.set_viewport_size({"width": 375, "height": 667})
    expect(page.locator(".dock .refusal")).to_contain_text("already ended")
    dock = page.locator(".composer.dock").bounding_box()
    assert dock is not None and dock["height"] < 667 / 4, dock
    assert not errors


@pytest.mark.parametrize("asks", [False, True])
def test_a_phone_opens_a_machine_at_its_waiting_prompt(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path, *, asks: bool
) -> None:
    """A phone opens a machine at its conversation's end while a prompt waits, else at its top.

    The page opened on States, about 2,600px above the prompt the machine waited on.
    """
    from playwright.sync_api import expect

    log = _talk(40)
    if asks:
        log.append({"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"})
    _machine(tmp_path, "asking", LOOP, [], log, alive=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/machine/asking")
    expect(page.locator(".conv .ci").filter(has_text="Step 39")).to_be_attached()
    expect(page.locator(".conv-card h2")).to_contain_text("Current state")  # the first frame
    if asks:
        prompt = page.locator(".conv-box .prompt-box")
        expect(prompt).to_be_in_viewport(ratio=1)
        box, dock = prompt.bounding_box(), page.locator(".composer.dock").bounding_box()
        assert box is not None and dock is not None
        assert box["y"] + box["height"] <= dock["y"]
    else:
        assert page.evaluate("window.scrollY") == 0
    assert not errors


def test_a_long_status_reason_leaves_the_states_list_its_room(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A long reason wrapped without limit over the States list and pushed it out of its card.

    The status shows the reason's first two lines and the whole label in its title; on a phone
    it keeps within the card's width.
    """
    from playwright.sync_api import expect

    reason = (
        "the collector exited nonzero three times: HTTPSConnectionPool(host='forge.example',"
        " port=443): Max retries exceeded with url: /api/v4/projects/platform%2Finfra"
        "/merge-requests?state=merged (Caused by NewConnectionError: Failed to establish a new"
        " connection: [Errno 111] Connection refused)"
    )
    end = {
        "type": "machine.end",
        "ts": TS,
        "status": "failed",
        "reason": reason,
        "state": "done",
        "transitions": 1,
    }
    _machine(tmp_path, "ended", LOOP, [_step(0, "work", "budget_exhausted", "done", AGENT), end])
    shape = """() => {
      const pill = document.querySelector('.machine-grid .pill');
      const card = pill.closest('.card'), body = card.querySelector('.card-body');
      const at = body.querySelector('.cursor').getBoundingClientRect();
      const box = body.getBoundingClientRect(), p = pill.getBoundingClientRect();
      return {inView: at.top >= box.top && at.bottom <= box.bottom,
              lines: Math.round(p.height / parseFloat(getComputedStyle(pill).lineHeight)),
              inside: p.right <= card.getBoundingClientRect().right};
    }"""
    for width, height, font in ((1024, 768, 16), (1440, 900, 20), (390, 844, 16)):
        page, errors = open_page(port)
        fonts = {"standard": font, "fixed": font * 13 // 16}
        page.context.new_cdp_session(page).send("Page.setFontSizes", {"fontSizes": fonts})
        page.set_viewport_size({"width": width, "height": height})
        page.goto(f"http://127.0.0.1:{port}/#/machine/ended")
        expect(page.locator(".machine-grid .pill")).to_have_attribute("title", f"failed · {reason}")
        got = page.evaluate(shape)
        assert got == {"inView": True, "lines": 2, "inside": True}, (width, font, got)
        assert not errors


def test_an_error_view_stands_under_the_name_it_is_about(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A machine that failed to open kept the previous machine's name in the top bar."""
    from playwright.sync_api import expect

    _ended(tmp_path)
    page, errors = _open(open_page, port, "ended")
    page.evaluate("location.hash = '#/machine/missing'")
    expect(page.locator(".card.problem")).to_contain_text("no machine 'missing'")
    expect(page.locator("#crumb")).to_have_text("missing")
    assert [e for e in errors if "404" not in e] == []


def test_a_notification_shows_as_a_toast_named_by_the_instance(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A notification stacked a banner over the cards, named by the spec, not the instance."""
    from playwright.sync_api import expect

    machine = _machine(tmp_path, "nightly", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "nightly")
    grid = page.locator(".machine-grid")
    before = grid.bounding_box()
    note = {"type": "machine.notify", "ts": TS, "state": "work", "message": "checkpoint committed"}
    with (machine / "journal.jsonl").open("a", encoding="utf-8") as journal:
        journal.write(json.dumps({**note, "level": "info"}) + "\n")
    expect(page.locator("#toasts .toast")).to_have_text("nightly · work: checkpoint committed")
    assert grid.bounding_box() == before
    assert not errors


def test_a_machine_whose_agent_state_never_ran_says_so(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A tool-only machine's conversation read "this session made no conversation"."""
    from playwright.sync_api import expect

    _machine(tmp_path, "tools", LONG, [])
    page, errors = _open(open_page, port, "tools")
    expect(page.locator(".conv-card .conv-empty")).to_have_text(
        "no agent state has run in this machine"
    )
    assert not errors


def test_a_warning_holds_until_dismissed_in_the_warning_colour(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A warn-level notification cleared itself after four seconds, as an info one does."""
    from playwright.sync_api import expect

    machine = _machine(tmp_path, "nightly", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "nightly")
    note = {"type": "machine.notify", "ts": TS, "state": "work", "message": "disk 91% full"}
    with (machine / "journal.jsonl").open("a", encoding="utf-8") as journal:
        journal.write(json.dumps({**note, "level": "warn"}) + "\n")
    toast = page.locator("#toasts .toast.warn")
    expect(toast).to_have_text("nightly · work: disk 91% full")
    expect(toast).to_have_attribute("role", "alert")
    toast.get_by_role("button", name="Dismiss").click()
    expect(toast).to_have_count(0)
    assert not errors


def test_only_the_three_newest_held_notifications_stay(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Every error-level notification held until dismissed, a stack without end."""
    from playwright.sync_api import expect

    machine = _machine(tmp_path, "nightly", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "nightly")
    with (machine / "journal.jsonl").open("a", encoding="utf-8") as journal:
        for n, level in enumerate(["error", "warn", "error", "warn", "error"]):
            note = {"type": "machine.notify", "ts": TS, "state": "work", "message": f"check {n}"}
            journal.write(json.dumps({**note, "level": level}) + "\n")
    expect(page.locator("#toasts .toast")).to_have_text(
        [f"nightly · work: check {n}" for n in (4, 3, 2)]
    )
    assert not errors


def test_a_warning_repeated_four_times_stays_showing(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The cap counted a repeated warning's one message once per repeat and removed it."""
    from playwright.sync_api import expect

    machine = _machine(tmp_path, "nightly", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "nightly")
    with (machine / "journal.jsonl").open("a", encoding="utf-8") as journal:
        for n, message in enumerate(["disk low"] * 4 + ["budget nearly spent"]):
            note = {"type": "machine.notify", "ts": f"2026-11-13T10:00:0{n}+00:00", "state": "work"}
            journal.write(json.dumps({**note, "message": message, "level": "warn"}) + "\n")
    expect(page.locator("#toasts .toast")).to_have_text(
        ["nightly · work: budget nearly spent", "nightly · work: disk low"]
    )
    assert not errors


def test_the_palette_offers_the_machine_views_actions(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The command palette offered none of the machine view's controls, Stop among them."""
    from playwright.sync_api import expect

    _machine(tmp_path, "working", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "working")
    page.keyboard.press("Control+k")
    expect(page.locator("dialog[open] input")).to_be_focused()
    page.keyboard.type(">this page")
    expect(page.locator("dialog[open] .ac-item")).to_have_text(
        ["Stop · this page", "Notifications blocked · this page"]
    )
    page.keyboard.press("Enter")
    question = page.get_by_role("alertdialog", name="Stop working?")
    question.get_by_role("button", name="Cancel").click()
    expect(page.locator("dialog[open]")).to_have_count(0)
    assert not errors


def test_stop_asks_by_the_instance_name(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Stop's confirm said "this machine" and showed a command in Markdown backticks.

    The button's name led with its glyph: "■ Stop".
    """
    from playwright.sync_api import expect

    _machine(tmp_path, "working", LOOP, [], _talk(2), alive=True)
    page, errors = _open(open_page, port, "working")
    stop = page.get_by_role("button", name="Stop", exact=True)
    stop.click()
    question = page.get_by_role("alertdialog", name="Stop working?")
    text = question.locator("p")
    expect(text).to_have_text(
        "It parks at its next transition, and agent6 machine run working resumes it."
    )
    expect(text.locator("code")).to_have_text("agent6 machine run working")
    question.get_by_role("button", name="Cancel").click()
    expect(stop).to_be_focused()
    assert not errors


def test_a_dimmed_verb_says_why_on_a_phone(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A refused verb's reason was only a hover title, and a phone hid the hint line too."""
    from playwright.sync_api import expect

    _parked(tmp_path)
    page, errors = _open(open_page, port, "parked")
    page.set_viewport_size({"width": 390, "height": 844})
    refusals = page.evaluate("getJSON('/api/machine/parked').then(m => m.refusals)")
    expect(page.get_by_role("button", name="Steer")).to_be_disabled()
    expect(page.get_by_text(refusals["steer"], exact=True)).to_be_visible()
    assert not errors


def test_every_transition_starts_in_one_column(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """From [10] on, a transition started one character right of the ones before it."""
    _machine(tmp_path, "chain", _chain(), [_chain_step(seq) for seq in range(12)])
    page, errors = _open(open_page, port, "chain")
    starts = page.evaluate(
        """() => [...[...document.querySelectorAll('.tree')][1].querySelectorAll('.node')]
          .map(node => {
            const range = document.createRange();
            range.setStart(node.lastChild, 0);
            range.setEnd(node.lastChild, 1);
            return Math.round(range.getBoundingClientRect().left);
          })"""
    )
    assert len(starts) == 12 and len(set(starts)) == 1, starts
    assert not errors


def test_the_summary_and_the_end_note_take_their_neighbours_type(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The summary stood in 14px beside its 13px status, the end note in 14px sans under mono."""
    _ended(tmp_path)
    page, errors = _open(open_page, port, "ended")
    fonts = page.evaluate(
        """() => {
          const font = (e) => getComputedStyle(e).fontSize + ' ' + getComputedStyle(e).fontFamily;
          const path = [...document.querySelectorAll('.tree')][1].closest('.card');
          const note = [...path.querySelectorAll('div')]
            .find(e => e.textContent.startsWith('ended:'));
          return [font(document.querySelector('.machine-grid .row.sub > .muted')),
                  font(document.querySelector('.machine-grid .pill')),
                  font(note), font(path.querySelector('.node'))];
        }"""
    )
    summary, status, note, line = fonts
    assert summary.split()[0] == status.split()[0], fonts
    assert note == line, fonts
    assert not errors
