# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 runs show`: one-shot liveness + progress of a run from its run dir."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from agent6.ui.bridge.approval import worker_is_alive, write_worker_pid
from agent6.ui.cli._common import _runs_dir  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.plan_watch import _cmd_status  # pyright: ignore[reportPrivateUsage]


def _ts(off_s: float) -> str:
    return (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=off_s)).isoformat()


def _make_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, object]]
) -> Path:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = _runs_dir(repo)
    runs.mkdir(parents=True, exist_ok=True)
    d = runs / "winsome-dawn-YWH5ZS"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "models": {"worker": {"model": "claude-opus-4-8"}}})
    )
    (d / "logs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return d


def test_status_running_with_live_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start", "mode": "run"},
            {"ts": _ts(3), "type": "loop.tool.call", "iteration": 3},
        ],
    )
    write_worker_pid(d, os.getpid())  # this test process is genuinely alive
    assert worker_is_alive(d)
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "running" in out
    assert "claude-opus-4-8" in out
    assert "iteration:  3" in out


def test_status_json_is_machine_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "run.start", "mode": "run"}])
    write_worker_pid(d, os.getpid())
    assert _cmd_status("", as_json=True) == 0  # "" -> most recent run
    obj = json.loads(capsys.readouterr().out)
    assert obj["run_id"] == "winsome-dawn-YWH5ZS"
    assert obj["alive"] is True
    assert obj["state"] == "running"


def test_status_crashed_when_pid_dead_and_no_run_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start"},
            {"ts": _ts(30), "type": "loop.tool.call", "iteration": 1},
        ],
    )
    (d / "worker.pid").write_text("999999")  # almost certainly not a live pid
    assert not worker_is_alive(d)
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "likely crashed or killed" in capsys.readouterr().out


def test_status_finished_reports_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start"},
            {"ts": _ts(1), "type": "run.end", "reason": "completed"},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "finished (completed)" in capsys.readouterr().out


def test_status_no_such_run_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert _cmd_status("nope") == 2


def test_worker_pid_clear(tmp_path: Path) -> None:
    from agent6.ui.bridge.approval import clear_worker_pid, read_worker_pid

    write_worker_pid(tmp_path, os.getpid())
    assert read_worker_pid(tmp_path) == os.getpid()
    clear_worker_pid(tmp_path)
    assert read_worker_pid(tmp_path) is None
    assert not worker_is_alive(tmp_path)


def test_status_shows_usage_from_budget_update_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `runs show` must read budget.update (the authoritative post-call totals),
    # not loop.budget (emitted BEFORE each call: lags one call, 0 on iter 1).
    # A stray loop.budget must NOT override the real usage.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start"},
            {
                "ts": _ts(30),
                "type": "loop.budget",
                "iteration": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            {
                "ts": _ts(20),
                "type": "budget.update",
                "input_total": 1000,
                "output_total": 200,
                "usd_total": 0.0123,
                "usd_partial": False,
            },
            {
                "ts": _ts(5),
                "type": "budget.update",
                "input_total": 4200,
                "output_total": 800,
                "usd_total": 0.0456,
                "usd_partial": False,
            },
        ],
    )
    write_worker_pid(d, os.getpid())
    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "in=4200 out=800" in out  # latest budget.update wins, not the 0/0 loop.budget
    assert "$0.0456" in out
    # json carries the same
    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["input_tokens"] == 4200 and obj["cost_usd"] == 0.0456
