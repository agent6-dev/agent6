"""The session page driven in a browser: its prompts, run card, drawer cards and conversation."""

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
