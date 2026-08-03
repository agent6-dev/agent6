# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions` (list): the winner marker on fan-out compare winners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.ui.cli._common import _runs_dir  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.sessions_cmds import _cmd_list  # pyright: ignore[reportPrivateUsage]


def _run(runs: Path, session_id: str, *, winner: bool | None = None) -> None:
    d = runs / session_id
    d.mkdir(parents=True)
    manifest: dict[str, object] = {"mode": "run"}
    if winner is not None:
        rank = 1 if winner else 2
        manifest["compare"] = {"group": "fan", "rank": rank, "of": 2, "winner": winner}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": session_id})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_run"})
        + "\n",
        encoding="utf-8",
    )


def test_runs_list_marks_the_fan_out_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    _run(runs, "fan-l1", winner=False)
    _run(runs, "fan-l2", winner=True)
    _run(runs, "solo")  # a run outside any fan-out: no marker

    assert _cmd_list() == 0
    out = capsys.readouterr().out
    assert "fan-l2 ★" in out  # the winner id carries the ★
    assert "fan-l1 ★" not in out and "solo ★" not in out  # losers / non-lanes do not


def test_runs_list_marks_a_partial_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cost the scanner knows is a lower bound (unpriced model in some leg)
    renders with the '~' marker in the listing, matching `runs show`."""
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    d = runs / "unpriced"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"mode": "run"}), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "budget.update", "usd_total": 0.0123, "usd_partial": True})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_run"})
        + "\n",
        encoding="utf-8",
    )
    assert _cmd_list() == 0
    assert "~$0.0123" in capsys.readouterr().out


def test_styled_status_colors_stale_red_and_parked_yellow() -> None:
    """The CLI status colors mirror the TUI/web: a lost worker (stale) is red and
    a parked submission (needs a resume) is yellow, not the old dim/uncolored that
    let a dead or unstarted run read as neutral in `agent6 sessions`."""
    from agent6.ui.cli.sessions_cmds import _styled_status  # pyright: ignore[reportPrivateUsage]

    stale, _ = _styled_status("stale", "", color=True)
    assert "\x1b[31m" in stale  # red, like the run header + web pill
    parked, _ = _styled_status("parked", "resume to start", color=True)
    assert "\x1b[33m" in parked  # yellow: attention, not a neutral done
