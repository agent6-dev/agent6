# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 runs show`: one-shot liveness + progress of a run from its run dir."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from agent6.runs.ipc import worker_is_alive, write_worker_pid
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


def test_status_waiting_when_blocked_on_an_operator_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run blocked on an unanswered approval/question must read
    "waiting (needs answer)" -- the same first-class status `agent6 runs`
    gives it -- not "running (long step, likely a provider call)", which sent
    the operator off to wait on a provider while the run sat blocked on THEM."""
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(400), "type": "run.start", "mode": "run"},
            {"ts": _ts(300), "type": "approval.prompt", "id": "approval-1", "prompt": "rm -rf?"},
        ],
    )
    write_worker_pid(d, os.getpid())
    assert _cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "waiting" in out and "needs answer" in out
    assert "provider call" not in out
    assert _cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["state"].startswith("waiting")


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


def test_status_leads_with_the_listing_word_then_the_raw_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `runs show` must agree with `runs list`: a finish_run+all_passed run reads
    # "passed", not the opposite "finished" the raw reason alone used to print.
    # The raw reason stays in parens as the diagnostic.
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start"},
            {"ts": _ts(1), "type": "run.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "passed (finish_run)" in capsys.readouterr().out


def test_status_finish_without_all_passed_reads_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start"},
            {"ts": _ts(1), "type": "run.end", "reason": "finish_run", "all_passed": False},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "finished (finish_run)" in capsys.readouterr().out


def test_status_error_reason_reads_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "run.start"},
            {"ts": _ts(1), "type": "run.end", "reason": "provider_error", "all_passed": False},
        ],
    )
    _cmd_status("winsome-dawn-YWH5ZS")
    assert "failed (provider_error)" in capsys.readouterr().out


def test_status_no_such_run_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert _cmd_status("nope") == 2


def test_status_shows_fan_out_compare_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`runs show` prints where a lane placed in its fan-out (+ the judge's
    rationale), and the JSON carries the raw compare block."""
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "run.start", "mode": "run"}])
    manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    manifest["compare"] = {
        "group": "fan", "rank": 1, "of": 2, "winner": True,
        "ranked_by": "judge", "rationale": "cleanest diff, all tests pass",
    }  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "compare:    rank 1/2 · winner · judge" in out
    assert "judge: cleanest diff, all tests pass" in out

    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["compare"]["winner"] is True and obj["compare"]["rank"] == 1


def test_worker_pid_clear(tmp_path: Path) -> None:
    from agent6.runs.ipc import clear_worker_pid, read_worker_pid

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


def test_status_cost_cumulative_and_unfinished_across_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A resume leg restarts the budget from 0 and un-finishes the run; `runs
    # show` banks legs (same rule as `runs list` and the run view) and must
    # not report leg 1's run.end for a run that is live again. A valid-JSON
    # non-object line is skipped, not a crash.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(60), "type": "run.start"},
            {
                "ts": _ts(50),
                "type": "budget.update",
                "input_total": 1000,
                "output_total": 200,
                "usd_total": 0.02,
                "usd_partial": True,
            },
            {"ts": _ts(40), "type": "run.end", "reason": "finish_run", "all_passed": True},
            {"ts": _ts(30), "type": "loop.resume.start", "iteration": 4},
            {
                "ts": _ts(5),
                "type": "budget.update",
                "input_total": 300,
                "output_total": 50,
                "usd_total": 0.005,
                "usd_partial": False,
            },
        ],
    )
    logs = d / "logs.jsonl"
    logs.write_text(logs.read_text(encoding="utf-8") + "42\n", encoding="utf-8")
    write_worker_pid(d, os.getpid())
    _cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["cost_usd"] == pytest.approx(0.025)
    assert obj["usd_partial"] is True  # sticky: leg 1's unpriced spend
    assert obj["state"] == "running"  # not leg 1's "passed (finish_run)"
    assert obj["input_tokens"] == 300  # token gauges stay per-leg


def test_status_missing_id_and_empty_state_speak_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A bad id names itself and where it looked, without leaking the
    # (runs|asks|machine-drafts) layout alternation; an empty state dir gets
    # the same first-contact copy as `runs`.
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert _cmd_status("zzz", as_json=False) == 2
    err = capsys.readouterr().err
    assert "no run matches 'zzz'" in err and "machine-drafts" not in err
    assert _cmd_status("", as_json=False) == 2
    assert 'no runs yet. Start one with `agent6 run "<task>"`.' in capsys.readouterr().err


def test_status_text_labels_leg_scoped_figures_on_a_resumed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Cost is banked across legs, token counters are the latest leg's; the
    # usage line must say which scope each figure describes once they differ.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(90), "type": "run.start", "mode": "run", "user_task": "t"},
            {
                "ts": _ts(80),
                "type": "budget.update",
                "input_total": 9000,
                "output_total": 500,
                "usd_total": 0.02,
            },
            {"ts": _ts(70), "type": "run.end", "reason": "finish_run", "all_passed": True},
            {"ts": _ts(60), "type": "loop.resume.start", "iteration": 4},
            {
                "ts": _ts(10),
                "type": "budget.update",
                "input_total": 300,
                "output_total": 50,
                "usd_total": 0.005,
            },
            {"ts": _ts(5), "type": "run.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    write_worker_pid(d, 999999999)
    _cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "in=300 out=50 (latest leg)" in out
    assert "cost $0.0250 (all 2 legs)" in out


def test_worker_is_alive_reads_a_foreign_owned_pid_as_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker is always spawned by the probing user, so PermissionError on
    the recorded pid means the worker died and the kernel reused the number
    for another user's process. Reading it as alive rendered a crashed run
    "running" forever and hung the /parallel lane await permanently."""
    if os.geteuid() == 0:
        pytest.skip("root can signal any pid; the foreign-owner probe needs a non-root euid")
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    d = tmp_path / "run"
    d.mkdir()
    write_worker_pid(d, 1)  # init: exists, foreign-owned -> PermissionError
    assert not worker_is_alive(d)
    write_worker_pid(d, os.getpid())
    assert worker_is_alive(d)
