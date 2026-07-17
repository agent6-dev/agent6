# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 runs` (list): the winner marker on fan-out compare winners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.ui.cli._common import _runs_dir  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.runs_cmds import _cmd_list  # pyright: ignore[reportPrivateUsage]


def _run(runs: Path, run_id: str, *, winner: bool | None = None) -> None:
    d = runs / run_id
    d.mkdir(parents=True)
    manifest: dict[str, object] = {"mode": "run"}
    if winner is not None:
        rank = 1 if winner else 2
        manifest["compare"] = {"group": "fan", "rank": rank, "of": 2, "winner": winner}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "run.start", "mode": "run", "user_task": run_id})
        + "\n"
        + json.dumps({"type": "run.end", "all_passed": True, "reason": "finish_run"})
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
