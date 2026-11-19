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
from tests.web import test_web_browser

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


def test_a_machines_first_prompt_stands_under_the_top_bar(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """A pending prompt sat flush against the top bar."""
    from playwright.sync_api import expect

    ask = {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow it?"}
    _machine(tmp_path, "asking", LOOP, [], [*_talk(1), ask], alive=True)
    page, errors = _open(open_page, port, "asking")
    box = page.locator(".prompt-box")
    expect(box).to_have_count(1)
    above, under = page.locator("header").bounding_box(), box.bounding_box()
    assert above is not None and under is not None
    assert under["y"] - (above["y"] + above["height"]) == 12
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
