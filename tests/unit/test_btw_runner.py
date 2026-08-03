# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `/btw` runner returns at once and delivers the answer later."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from agent6.ui.cli._btw import make_btw_runner
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._steer_menu import (
    MENU_COMMANDS,
    _run_info_command,  # pyright: ignore[reportPrivateUsage]
)


def _answered_ask(root: Path, name: str, answer: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "ask"}), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "run.start", "user_task": "q"}),
                json.dumps({"type": "role.result", "text": answer}),
                json.dumps({"type": "run.end", "reason": "answered", "all_passed": True}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return d


def test_the_run_is_never_blocked_and_the_answer_arrives_later(tmp_path: Path) -> None:
    """The point of asking beside a run: `/btw` returns immediately, and the
    answer lands on the view at the next turn boundary."""
    asks = tmp_path / "asks"
    asks.mkdir()
    out = io.StringIO()
    view = ConsoleView(out, color=False)

    def launch(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        _answered_ask(asks, "quiet-fox-AAAAAA", "use ffmpeg -c:v libx265")
        return ""

    runner = make_btw_runner(
        "parent-BBBBBB",
        launch=launch,
        list_asks=lambda: [d for d in asks.iterdir() if d.is_dir()],
        console_view=lambda: view,
    )
    started = time.monotonic()
    line = runner("why h265", tmp_path)
    assert time.monotonic() - started < 2.0  # returned, did not wait for an answer
    assert "quiet-fox-AAAAAA" in line

    deadline = time.monotonic() + 15.0
    while "libx265" not in out.getvalue() and time.monotonic() < deadline:
        view.feed({"type": "role.result"})  # turn boundaries keep arriving
        time.sleep(0.2)
    text = out.getvalue()
    assert "--- btw: why h265" in text
    assert "use ffmpeg -c:v libx265" in text
    assert "agent6 resume quiet-fox-AAAAAA" in text


def test_btw_is_offered_in_the_menu() -> None:
    assert "/btw" in MENU_COMMANDS


def test_an_unwired_btw_says_so_rather_than_failing_obscurely(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached run has no console view to print an answer into."""
    _run_info_command("/btw why", tmp_path, None)
    assert "needs a live run" in capsys.readouterr().out


def test_a_bare_btw_asks_for_a_question(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Opening an empty session would be worse than saying nothing was asked."""
    _run_info_command("/btw", tmp_path, None)
    assert "ask something" in capsys.readouterr().out
