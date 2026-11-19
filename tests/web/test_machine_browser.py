# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine view driven in a browser: its layout, its lists, its titles and its composer."""

from __future__ import annotations

import json
import os
import pathlib
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
