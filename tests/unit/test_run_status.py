# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions show`: one-shot liveness + progress of a run from its run dir."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import time
from typing import Any

import pytest

from agent6 import portable
from agent6.sessions import ipc as sessions_ipc
from agent6.sessions import journal, process
from agent6.ui.cli import common, sessions_show
from agent6.vcs import git_read


def _ts(off_s: float) -> str:
    return (datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=off_s)).isoformat()


def _make_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, object]]
) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = common._runs_dir(repo)
    runs.mkdir(parents=True, exist_ok=True)
    d = runs / "winsome-dawn-YWH5ZS"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "models": {"driver": {"model": "claude-opus-4-8"}}})
    )
    (d / "logs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return d


def test_sessions_show_uses_the_verification_marker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sessions show reads the same waiting, running and result lines as the transcript."""
    from tests.unit import test_transcript_verify

    rows = test_transcript_verify.attempt()
    run = _make_run(tmp_path, monkeypatch, [])
    process.write_worker_pid(run, os.getpid())
    for length, text in [
        (2, "verify: waiting for approval"),
        (3, "verify: running `pytest -q`"),
        (4, "verify: ✓ pass · 0.2s"),
    ]:
        (run / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:length]))
        assert sessions_show._cmd_status(run.name) == 0
        assert capsys.readouterr().out.count(text) == 1


def test_status_running_with_live_pid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {"ts": _ts(3), "type": "tool.call", "turn": 3},
        ],
    )
    # this test process is genuinely alive
    process.write_worker_pid(d, os.getpid())
    assert process.worker_is_alive(d)
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "running" in out
    assert "claude-opus-4-8" in out
    assert "turn:       3" in out


def test_status_json_is_machine_readable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    process.write_worker_pid(d, os.getpid())
    assert sessions_show._cmd_status("", as_json=True) == 0  # "" -> most recent run
    obj = json.loads(capsys.readouterr().out)
    assert obj["session_id"] == "winsome-dawn-YWH5ZS"
    assert obj["alive"] is True
    assert obj["status"] == "running"
    # A live run keeps elapsing while it waits (its last event is 5 s old and
    # it wrote nothing since); a dead one stops at its last event.
    assert obj["elapsed_s"] >= 4


def test_status_elapsed_of_a_fork_execution_runs_from_its_first_event(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork execution's elapsed time runs from its first event.

    A fork's log opens with `session.resume` and never carries a `session.start`; without one the
    scan's start is the first event's timestamp, not `elapsed: -`.
    """
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"type": "session.resume", "ts": "2026-01-01T00:00:00+00:00", "turn": 2},
            {"type": "tool.call", "ts": "2026-01-01T00:00:10+00:00", "turn": 2},
            {
                "type": "session.stop",
                "ts": "2026-01-01T00:00:30+00:00",
                "reason": "finish_session",
                "verification": "passed",
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["elapsed_s"] == 30.0
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "elapsed:    30s" in capsys.readouterr().out


def test_a_finished_runs_elapsed_time_stops_at_its_receipt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A worker can remain alive during teardown after `session.stop` is written.

    The finished receipt fixes the run's elapsed time at its end event; worker
    teardown must not make that completed duration keep growing.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"type": "session.start", "ts": "2026-01-01T00:00:00+00:00"},
            {
                "type": "session.stop",
                "ts": "2026-01-01T00:00:10+00:00",
                "reason": "finish_session",
                "turns": 1,
                "verification": "passed",
            },
        ],
    )
    process.write_worker_pid(d, os.getpid())

    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["elapsed_s"] == 10.0


def test_status_waiting_when_blocked_on_an_operator_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run blocked on an unanswered prompt reads "waiting (needs answer)".

    The same first-class status `agent6 sessions` gives it, never "running (long step, likely a
    provider call)", which sends the operator off to wait on a provider while the run waits on them.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(400), "type": "session.start", "mode": "run"},
            {"ts": _ts(300), "type": "approval.prompt", "id": "approval-1", "prompt": "rm -rf?"},
        ],
    )
    process.write_worker_pid(d, os.getpid())
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "waiting" in out and "needs answer" in out
    assert "provider call" not in out
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["status"], obj["detail"]) == ("waiting", "needs answer; attach to respond")


@pytest.mark.parametrize(
    ("last", "detail"),
    [
        (
            {"type": "verify.start", "command": ["pytest"], "timeout_s": 1800},
            "verify command running",
        ),
        ({"type": "tool.call", "name": "run_command", "args_preview": {}}, "tool call running"),
        ({"type": "model.call", "role": "worker"}, "provider call in flight"),
        ({"type": "model.text_delta", "role": "worker", "text": "x"}, "provider call in flight"),
        ({"type": "checkpoint.committed"}, "long step"),
    ],
)
def test_status_names_what_a_long_step_waits_on(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    last: dict[str, object],
    detail: str,
) -> None:
    """A step over two minutes old names what the journal's last event started.

    A twenty-minute verify read "long step, likely a provider call", sending the operator
    to wait on the wrong thing.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [{"ts": _ts(400), "type": "session.start", "mode": "run"}, {"ts": _ts(300), **last}],
    )
    process.write_worker_pid(d, os.getpid())
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["status"], obj["detail"]) == ("running", detail)


def test_status_crashed_when_pid_dead_and_no_run_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead worker without a session.stop leads with the hub's word "stale".

    This surface adds its diagnostic detail after it. "stopped" is the hub's word for an operator
    stop, so leading with it reads a crash as deliberate.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(30), "type": "tool.call", "turn": 1},
        ],
    )
    (d / "worker.pid").write_text("999999")  # almost certainly not a live pid
    assert not process.worker_is_alive(d)
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "status:     stale (worker exited without finishing (crashed or killed))" in out
    assert "stopped" not in out


@pytest.mark.parametrize("pid_file", [False, True])
def test_show_stale_run_offers_resume(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pid_file: bool,
) -> None:
    """A worker that died without an end must leave its resume command on the show page."""
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {"ts": _ts(30), "type": "tool.call", "turn": 1},
        ],
    )
    (d / "loop_state.json").write_text("{}", encoding="utf-8")  # a turn ran: a resume can start
    if pid_file:
        process.write_worker_pid(d, 999999999)

    assert sessions_show._cmd_status("winsome-dawn") == 0

    out = capsys.readouterr().out
    assert "status:     stale (worker exited without finishing (crashed or killed))" in out
    assert f"resume with:  agent6 resume {d.name}\n" in out


def test_status_words_lead_with_the_listing_word_in_every_state(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `sessions show --json` status is exactly the hub row's word for the same dir.

    For every state without a session.stop: "created", "starting", "waiting", "stale". One decision,
    `status_for_session_dir`, so the two cannot drift.
    """
    from agent6.viewmodel import listing

    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])

    def state_word() -> str:
        assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
        word = json.loads(capsys.readouterr().out)["status"]
        assert word == listing.summarize_session_dir(d).status
        return word

    process.write_worker_pid(d, os.getpid())
    assert state_word() == "running"
    (d / "logs.jsonl").unlink()
    assert state_word() == "starting"
    # A dead pid file with no session.start: a worker died LAUNCHING (the pid
    # survives a kill; a clean refusal clears it) -- not the never-ran word.
    (d / "worker.pid").write_text("999999999", encoding="utf-8")
    assert state_word() == "stale"
    (d / "worker.pid").unlink()
    assert state_word() == "created"
    (d / "logs.jsonl").write_text(
        json.dumps({"ts": _ts(30), "type": "session.start", "mode": "run"})
        + "\n"
        + json.dumps({"ts": _ts(9), "type": "approval.prompt", "id": "a1", "prompt": "ok?"})
        + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(d, os.getpid())
    assert state_word() == "waiting"
    (d / "worker.pid").write_text("999999999", encoding="utf-8")
    assert state_word() == "stale"


def test_status_prints_the_final_turn_count_from_the_receipt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The terminal receipt's `session.stop.turns` is the final count.

    A run that ended before another turn-stamped event printed `-`, even
    though its journal recorded the figure the command promises to show.
    """
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "finish_session",
                "turns": 7,
                "verification": "passed",
            },
        ],
    )

    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "turn:       7\n" in capsys.readouterr().out
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["turn"] == 7


@pytest.mark.parametrize("reason", ["steer_abort", "finish_session"])
def test_show_words_the_end_instead_of_the_reason_token(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
) -> None:
    """Text shows the end sentence while JSON retains the raw reason for scripts."""
    summary = "operator stopped the run after turn 4" if reason == "steer_abort" else "added mul"
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"type": "session.start", "mode": "run"},
            {
                "type": "session.stop",
                "reason": reason,
                "summary": summary,
                "verification": "passed",
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    text = capsys.readouterr().out
    assert f"ended:      {summary}\n" in text and reason not in text
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["end_summary"] == summary and obj["stop_reason"] == reason


@pytest.mark.parametrize("with_output", [True, False])
def test_show_prints_a_setup_failure_once(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    with_output: bool,
) -> None:
    """A failed setup's diagnostic prints once: the sentence on `ended:`, its output on `setup:`."""
    sentence = "the setup command failed (exit 1): uv sync --offline"
    output = (
        "    no cached package\n"
        "  the command ran with no route off this machine (sandbox.network), so it has to"
        " work offline. See docs/security.md."
    )
    problem = f"{sentence}\n{output}" if with_output else sentence
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"type": "session.start", "mode": "run"},
            {"type": "workspace.setup", "command": ["uv", "sync", "--offline"], "exit_code": 1},
            {
                "type": "session.stop",
                "reason": "setup_command_failed",
                "verification": "unverified",
                "summary": problem,
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert out.count(sentence) == 1
    assert f"ended:      {sentence}\n" in out
    if with_output:
        assert (
            "setup:      no cached package\n"
            "            the command ran with no route off this machine (sandbox.network),"
            " so it has to work offline. See docs/security.md.\n"
        ) in out
    else:
        assert "setup:" not in out
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["end_summary"] == problem and obj["setup_output"] == problem


def test_status_leads_with_the_listing_word_then_the_raw_reason(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `sessions show` must agree with `sessions list`: a finish_session+verification run reads
    # "passed", not the opposite "finished" the raw reason alone used to print.
    # The raw reason stays in parens as the diagnostic.
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
        ],
    )
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    assert "passed (finish_session)" in capsys.readouterr().out


def test_status_of_a_scoped_green_names_the_scoped_gate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass certified by a scoped gate reads `passed · scoped gate`, as on every listing.

    The label is `status_label`'s; the raw reason stays in parentheses.
    """
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
                "scoped": True,
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["status"], obj["label"], obj["detail"]) == (
        "passed",
        "passed · scoped gate",
        "finish_session",
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "status:     passed · scoped gate (finish_session)\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reason", "label"),
    [
        ("silent_finish", "finished · no finish call"),
        ("metric_plateau", "finished · metric plateau"),
    ],
)
def test_sessions_and_show_name_a_harness_end_over_a_green_gate(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reason: str,
    label: str,
) -> None:
    """The sessions listing and show must not present a harness end as a finish call's pass."""
    from agent6.ui.cli import sessions_cmds

    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {"ts": _ts(1), "type": "session.stop", "reason": reason, "verification": "passed"},
        ],
    )
    assert sessions_cmds._cmd_list(as_json=False) == 0
    assert label in capsys.readouterr().out
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert f"status:     {label} ({reason})\n" in capsys.readouterr().out


def test_status_finish_without_all_passed_reads_finished(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(2), "type": "verify.end", "command": ["pytest"], "exit_code": 1},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "failed",
            },
        ],
    )
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    assert "finished · gate red (finish_session)" in capsys.readouterr().out


def test_status_error_reason_reads_failed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "provider_error",
                "verification": "unverified",
            },
        ],
    )
    # The label already carries the reason; the parenthetical is not repeated.
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "status:     failed · provider error\n" in capsys.readouterr().out


def test_status_of_a_provider_failure_keeps_the_providers_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run ended by a provider error shows the error its driving call journaled.

    The detail was empty, so a 401 read only `failed · provider error`; a side call
    failing after it is not the reason.
    """
    error = (
        'Anthropic API error 401: {"type":"error","error":{"type":"authentication_error",'
        '"message":"API key is invalid."},"request_id":null}'
    )
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {"ts": _ts(4), "type": "model.call", "role": "worker", "model": "claude-sonnet-4-5"},
            {"ts": _ts(3), "type": "model.result", "role": "worker", "ok": False, "error": error},
            {"ts": _ts(3), "type": "provider.fatal", "status_code": 401, "error": error},
            {"ts": _ts(2), "type": "model.result", "role": "reviewer", "ok": False, "error": "x"},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "provider_error",
                "turns": 1,
                "verification": "unverified",
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["label"], obj["detail"]) == ("failed · provider error", error)
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert f"status:     failed · provider error ({error})\n" in capsys.readouterr().out


def test_status_shows_fan_out_compare_outcome(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` prints where a lane placed in its fan-out, with the judge's rationale.

    The JSON carries the raw compare block.
    """
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    manifest["placement"] = {
        "group": "fan", "rank": 1, "of": 2, "winner": True,
    }  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    coordinator = d.parent / "fan"
    coordinator.mkdir()
    (coordinator / "manifest.json").write_text(
        json.dumps(
            {
                "comparison": {
                    "group": "fan",
                    "ranking": [d.name],
                    "ranked_by": "judge",
                    "rationale": "cleanest diff, all tests pass",
                }
            }
        ),
        encoding="utf-8",
    )

    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "compare:    rank 1/2 · winner · judge" in out
    assert "judge: cleanest diff, all tests pass" in out

    sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["placement"]["winner"] is True and obj["placement"]["rank"] == 1


def test_worker_pid_clear(tmp_path: pathlib.Path) -> None:
    process.write_worker_pid(tmp_path, os.getpid())
    assert process.read_worker_pid(tmp_path) == os.getpid()
    process.clear_worker_pid(tmp_path)
    assert process.read_worker_pid(tmp_path) is None
    assert not process.worker_is_alive(tmp_path)


def test_status_shows_usage_from_budget_update_event(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `sessions show` must read budget.update (the authoritative post-call totals),
    # not budget.turn (emitted BEFORE each call: lags one call, 0 on turn 1).
    # A stray budget.turn must NOT override the real usage.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {
                "ts": _ts(30),
                "type": "budget.turn",
                "turn": 1,
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
                "cache_read_total": 42486,
                "cache_write_total": 22617,
                "usd_total": 0.0456,
                "usd_partial": False,
            },
        ],
    )
    process.write_worker_pid(d, os.getpid())
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "in=4200 out=800 cache_r=42486 cache_w=22617" in out  # the latest update wins
    assert "$0.05" in out
    # json carries the same
    sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["input_tokens"] == 4200 and obj["cost_usd"] == 0.0456
    assert (obj["cache_read_tokens"], obj["cache_write_tokens"]) == (42486, 22617)


def test_status_names_the_pins_in_force(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The show page names a run's pinned instructions, one per line; the JSON carries the list.

    The execution-start announcement replaces the list and `/pin` appends, the fold's rule.
    """
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start"},
            {"ts": _ts(30), "type": "pin.restored", "pins": ["never touch tests"]},
            {"ts": _ts(20), "type": "pin.added", "text": "keep the API stable"},
        ],
    )
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "pins:       never touch tests\n            keep the API stable\n" in out
    sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    assert json.loads(capsys.readouterr().out)["pins"] == [
        "never touch tests",
        "keep the API stable",
    ]


def test_status_cost_cumulative_and_unfinished_across_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A resume execution restarts the budget from 0 and un-finishes the run; `runs
    # show` banks executions (same rule as `sessions list` and the run view) and must
    # not report execution 1's session.stop for a run that is live again. A valid-JSON
    # non-object line is skipped, not a crash.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(60), "type": "session.start"},
            {
                "ts": _ts(50),
                "type": "budget.update",
                "input_total": 1000,
                "output_total": 200,
                "cache_read_total": 9000,
                "cache_write_total": 100,
                "usd_total": 0.02,
                "usd_partial": True,
            },
            {
                "ts": _ts(40),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
            {"ts": _ts(30), "type": "session.resume", "turn": 4},
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
    process.write_worker_pid(d, os.getpid())
    sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True)
    obj = json.loads(capsys.readouterr().out)
    assert obj["cost_usd"] == pytest.approx(0.025)
    assert obj["usd_partial"] is True  # sticky: execution 1's unpriced spend
    assert obj["status"] == "running"  # not execution 1's "passed (finish_session)"
    assert obj["input_tokens"] == 300  # token gauges stay per-execution
    assert obj["cache_read_tokens"] is None  # execution 1's cached side does not carry over


def test_status_missing_id_and_empty_state_speak_human(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A bad id names itself and where it looked, without leaking the
    # bucket alternation under sessions/; an empty state dir gets
    # the same first-contact copy as `runs`.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert sessions_show._cmd_status("zzz", as_json=False) == 2
    err = capsys.readouterr().err
    assert "no session matches 'zzz'" in err and "machines" not in err
    assert sessions_show._cmd_status("", as_json=False) == 2
    assert 'no sessions yet. Start one with `agent6 run "<task>"`.' in capsys.readouterr().err


def test_status_text_labels_plan_points_on_a_resumed_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resumed plan run shows this execution's points against the cap and the session's cost.

    The usage line held no whole-session figure, and `elapsed` did not say it spans the
    time between executions.
    """
    plan = {"plan_used_percent": 40.0, "plan_cap": 2.0, "input_total": 10, "output_total": 5}
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(90), "type": "session.start", "mode": "run", "user_task": "t"},
            {"ts": _ts(80), "type": "budget.update", **plan, "plan_observed_points": 1.0},
            {
                "ts": _ts(70),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
            {"ts": _ts(60), "type": "session.resume", "turn": 4},
            {"ts": _ts(10), "type": "budget.update", **plan, "plan_observed_points": 0.0},
            {
                "ts": _ts(5),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
        ],
    )
    process.write_worker_pid(d, 999999999)
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "plan moved=0/2pt (latest execution)" in out
    assert "usage 1pt (all 2 executions)" in out
    assert "elapsed:    1m25s (wall, 2 executions)" in out


def test_status_text_labels_execution_scoped_figures_on_a_resumed_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Cost is banked across executions, token counters are the latest execution's; the
    # usage line must say which scope each figure describes once they differ.
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(90), "type": "session.start", "mode": "run", "user_task": "t"},
            {
                "ts": _ts(80),
                "type": "budget.update",
                "input_total": 9000,
                "output_total": 500,
                "usd_total": 0.02,
            },
            {
                "ts": _ts(70),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
            {"ts": _ts(60), "type": "session.resume", "turn": 4},
            {
                "ts": _ts(10),
                "type": "budget.update",
                "input_total": 300,
                "output_total": 50,
                "usd_total": 0.005,
            },
            {
                "ts": _ts(5),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
        ],
    )
    process.write_worker_pid(d, 999999999)
    sessions_show._cmd_status("winsome-dawn-YWH5ZS")
    out = capsys.readouterr().out
    assert "in=300 out=50 (latest execution)" in out
    assert "usage $0.03 (all 2 executions)" in out


def test_worker_is_alive_reads_a_foreign_owned_pid_as_dead(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded pid owned by another user reads as dead.

    A worker is always spawned by the probing user, so PermissionError means the worker died and the
    kernel reused the number. Reading it as alive renders a crashed run "running" forever and hangs
    the `/parallel` lane await.
    """
    if os.geteuid() == 0:
        pytest.skip("root can signal any pid; the foreign-owner probe needs a non-root euid")
    if pathlib.Path("/proc/1").stat().st_uid == os.geteuid():
        pytest.skip("PID 1 belongs to the test user; the probe needs a foreign-owned pid")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    d = tmp_path / "run"
    d.mkdir()
    # init: exists, foreign-owned -> PermissionError
    process.write_worker_pid(d, 1)
    assert not process.worker_is_alive(d)
    process.write_worker_pid(d, os.getpid())
    assert process.worker_is_alive(d)


def test_concurrent_answer_writers_do_not_race_on_the_temp(tmp_path: pathlib.Path) -> None:
    """Two front-ends answering the same prompt at once do not race on one temp file.

    A shared sibling `.tmp` made the loser hit FileNotFoundError after the winner's rename, a 500 on
    an answer that landed; the durable write uses a unique mkstemp temp per call.
    """
    import threading

    d = tmp_path / "run"
    d.mkdir()
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _answer() -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(50):
                sessions_ipc.write_answer(d, "approval-1", "yes")
        except Exception as exc:
            errors.append(exc)

    t1, t2 = threading.Thread(target=_answer), threading.Thread(target=_answer)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert errors == []
    assert (d / "approvals" / "approval-1.answer").read_text(encoding="utf-8") == "yes"


def test_status_ambiguous_prefix_names_the_candidates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ambiguous id prefix must say so and name the matches, as `attach` and `stop` do.

    `sessions show` swallowed the resolver's error and printed "no session matches 't'", telling
    the operator no such run exists while two did.
    """
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    sibling = d.parent / "winsome-dusk-AAAAAA"
    sibling.mkdir()
    (sibling / "logs.jsonl").write_text(
        json.dumps({"ts": _ts(9), "type": "session.start", "mode": "run"}) + "\n", encoding="utf-8"
    )

    assert sessions_show._cmd_status("winsome-d") == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "winsome-dawn-YWH5ZS" in err and "winsome-dusk-AAAAAA" in err
    assert "no session matches" not in err


def test_a_nonpositive_recorded_pid_never_reads_alive(tmp_path: pathlib.Path) -> None:
    """A worker.pid holding 0 or -1 never reads alive.

    `os.kill(0, 0)` signals the process group and `os.kill(-1, 0)` every process, so both succeed;
    read as alive, such a pid refuses resume and hangs the `/parallel` lane await forever. The
    worker probe guards it like the front-end probe.
    """
    for junk in ("0", "-1"):
        (tmp_path / "worker.pid").write_text(junk, encoding="utf-8")
        assert process.worker_is_alive(tmp_path) is False, junk


def test_pid_alive_rejects_nonpositive_values_without_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID 0 and -1 are process-group selectors on hosts without /proc too."""
    monkeypatch.setattr(process, "_HAS_PROC", False)
    assert process.pid_alive(0) is False
    assert process.pid_alive(-1) is False


@pytest.mark.skipif(not pathlib.Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")
def test_a_zombie_worker_is_not_alive(tmp_path: pathlib.Path) -> None:
    """A worker that exited but has not been reaped is dead, despite kill-0."""
    proc = subprocess.Popen(["/bin/true"])
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stat = pathlib.Path(f"/proc/{proc.pid}/stat").read_text(encoding="ascii")
            if stat.rpartition(")")[2].split()[0] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("child did not become a zombie")
        process.write_worker_pid(tmp_path, proc.pid)
        assert process.worker_is_alive(tmp_path) is False
    finally:
        proc.wait()


@pytest.mark.skipif(not pathlib.Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")
def test_a_zombie_is_not_alive_without_proc_either(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `/proc`, the `ps` fallback catches a zombie too.

    A zombie still answers kill-0, so a host that skips the state check reads it alive forever, the
    stale-`running` symptom the `/proc` branch exists to kill.
    """
    monkeypatch.setattr(process, "_HAS_PROC", False)
    proc = subprocess.Popen(["/bin/true"])
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stat = pathlib.Path(f"/proc/{proc.pid}/stat").read_text(encoding="ascii")
            if stat.rpartition(")")[2].split()[0] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("child did not become a zombie")
        assert process.pid_alive(proc.pid) is False
    finally:
        proc.wait()


def test_worker_pid_is_published_atomically(tmp_path: pathlib.Path) -> None:
    """The last polled state file written with plain write_text.

    It truncates, then writes, so a reader in that window sees a PREFIX of the pid with the start-
    time identity stripped, and a prefix that happens to name a live process you own reads alive
    with nothing left to refute it, which is the recycled-pid lie the identity was added to kill.
    """
    seen: list[str] = []
    real = portable.atomic_write

    def spy(path: pathlib.Path, text: str) -> None:
        seen.append(path.name)
        real(path, text)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(portable, "atomic_write", spy)
    try:
        process.write_worker_pid(tmp_path, os.getpid())
    finally:
        monkey.undo()
    assert seen == ["worker.pid"]
    # still a valid record
    assert process.worker_is_alive(tmp_path) is True


def test_a_started_session_with_no_pid_file_is_not_running(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A started session with no pid file is not running.

    A worker writes its pid before session.start, so a missing pid file means the worker cleared it
    on the way out, whatever the log says. Reachable by `agent6 run ... | head`: SIGPIPE unwinds
    through the finally, which clears the pid but writes no session.stop. Treating the absence as
    weaker evidence than a dead pid inverts the two: `kill -9` leaves the pid file and reads "stale"
    at once, while the tidier death reads "running" for the whole 600s silence window.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(20), "type": "session.start", "mode": "run"},
            {"ts": _ts(2), "type": "tool.call", "turn": 2},
        ],
    )
    assert not (d / "worker.pid").exists()

    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "running" not in out, "a session with no live worker must never render as running"
    assert "stale" in out


def test_a_start_event_is_never_readable_before_the_pid_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The invariant `_running_is_stale` rests on.

    No pid file under a started session means the worker cleared it, not that it has not written it
    yet. Emitting session.start first opens a window where a live session reads as dead on every
    surface. The start emitter owns the order, so no entry point (the loop's two starts, machine
    create's header) can invert it.
    """
    from agent6.sessions import event_log

    sdir = tmp_path / "sess"
    sdir.mkdir()
    pid_present_at_emit: list[bool] = []
    real_emit = event_log.EventSink.emit

    def spy(self: event_log.EventSink, event: journal.Event, /) -> None:
        pid_present_at_emit.append((sdir / "worker.pid").exists())
        real_emit(self, event)

    monkeypatch.setattr(event_log.EventSink, "emit", spy)
    process.emit_session_start(
        event_log.EventSink(sdir / "logs.jsonl"),
        sdir,
        journal.SessionStart(mode="run", user_task="task", execution_id="execution"),
    )
    assert pid_present_at_emit == [True]


def _stamp_manifest(d: pathlib.Path, **fields: Any) -> None:
    (d / "manifest.json").write_text(
        json.dumps({"mode": "run", "models": {"driver": {"model": "m"}}, **fields})
    )


def test_status_says_where_the_changes_are(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions show` says where the run's changes are.

    The branch fact the web header shows and the end-of-run footer said: on the run branch awaiting
    its merge (with the command), merged into the base, or a branch no commit reached. The JSON
    carries run_branch, base_branch and merged_into.
    """
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=pathlib.Path.cwd(), check=True)
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "base"],
        cwd=pathlib.Path.cwd(),
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    branch = "agent6/winsome-dawn-YWH5ZS"

    def show() -> str:
        assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
        return capsys.readouterr().out

    _stamp_manifest(d, run_branch=branch, base_branch="main")
    # No commit was made, so the branch does not exist and the row names none.
    assert "changes:    none (the run made no commit)" in show()
    subprocess.run(["git", "branch", branch], cwd=pathlib.Path.cwd(), check=True)
    assert (
        f"changes:    {branch} → merges into main; merge with: agent6 sessions merge"
        " winsome-dawn-YWH5ZS"
    ) in show()
    tip = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=pathlib.Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _stamp_manifest(
        d, run_branch=branch, base_branch="main", merged={"into": "main", "sha": tip, "tip": tip}
    )
    assert f"changes:    {branch} (merged into main)" in show()
    # A stamp that names no base covers nothing (one rule: `covering_stamp`);
    # show used to read it as merged into the base branch.
    _stamp_manifest(
        d, run_branch=branch, base_branch="main", merged={"into": "", "sha": tip, "tip": tip}
    )
    assert f"changes:    {branch} → merges into main; merge with:" in show()
    _stamp_manifest(
        d, run_branch=branch, base_branch="main", merged={"into": "main", "sha": tip, "tip": tip}
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["run_branch"], obj["base_branch"], obj["merged_into"]) == (branch, "main", "main")
    _stamp_manifest(d, mode="ask")
    assert "changes:" not in show()
    # The branch deleted (no merge stamp) while the chain ref keeps the
    # commits: the ref is named, and the merge still offered.
    _stamp_manifest(d, run_branch=branch, base_branch="main")
    chain = "refs/agent6/winsome-dawn-YWH5ZS/head"
    subprocess.run(["git", "update-ref", chain, tip], cwd=pathlib.Path.cwd(), check=True)
    subprocess.run(
        ["git", "branch", "-D", branch], cwd=pathlib.Path.cwd(), check=True, capture_output=True
    )
    out = show()
    assert "changes:    refs/agent6/winsome-dawn-YWH5ZS/head (" in out
    assert "is gone; the commits are kept); merge with: agent6 sessions merge" in out


def test_status_of_a_run_cut_from_a_detached_head_offers_merge_into(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run started on a detached HEAD has no base branch, so the merge hint asks for one.

    The bare `agent6 sessions merge ID` it offered fails there: the target has to be named.
    """
    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "run"}])
    cwd = pathlib.Path.cwd()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=cwd, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=cwd, check=True)
    subprocess.run(["git", "checkout", "-q", "--detach"], cwd=cwd, check=True)
    branch = "agent6/winsome-dawn-YWH5ZS"
    subprocess.run(["git", "branch", branch], cwd=cwd, check=True)
    # What the preflight records as the base on this checkout.
    _stamp_manifest(d, run_branch=branch, base_branch=git_read.status(cwd).branch)

    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "; merge with: agent6 sessions merge winsome-dawn-YWH5ZS --into BRANCH\n" in out


def test_status_of_an_undone_run_does_not_offer_a_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The changes line of an undone run agrees with the listing.

    The branch was taken back by /undo, so no merge is offered. It read the end reason not at all
    and offered `merge with: agent6 sessions merge <id>`.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(5), "type": "session.start", "mode": "run"},
            {
                "ts": _ts(1),
                "type": "session.stop",
                "reason": "undone",
                "verification": "unverified",
            },
        ],
    )
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=pathlib.Path.cwd(), check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "b",
        ],
        cwd=pathlib.Path.cwd(),
        check=True,
    )
    branch = "agent6/winsome-dawn-YWH5ZS"
    subprocess.run(["git", "branch", branch], cwd=pathlib.Path.cwd(), check=True)
    _stamp_manifest(d, run_branch=branch, base_branch="main")

    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "status:     undone" in out
    assert f"changes:    {branch} (taken back by /undo)" in out
    assert "merge with:" not in out


def test_status_of_an_ask_does_not_repeat_its_word(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ask ends with reason "answered", the listing word too: never "answered (answered)"."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(9), "type": "session.start", "mode": "ask"},
            {
                "ts": _ts(5),
                "type": "session.stop",
                "reason": "answered",
                "verification": "unverified",
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "status:     answered\n" in capsys.readouterr().out


def test_show_json_label_matches_the_listing_cell(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One vocabulary across the two surfaces.

    `show --json`'s label was the bare word while the listing folds the mode in, so a script could
    not match a run's row to its detail view.
    """
    from agent6.viewmodel import format, listing

    d = _make_run(tmp_path, monkeypatch, [{"ts": _ts(5), "type": "session.start", "mode": "plan"}])
    (d / "manifest.json").write_text(json.dumps({"mode": "plan"}), encoding="utf-8")
    process.write_worker_pid(d, os.getpid())

    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0

    obj = json.loads(capsys.readouterr().out)
    s = listing.summarize_session_dir(d)
    assert obj["label"] == format.listing_status_label(
        s.mode, s.status, s.detail, unmerged=s.unmerged
    )
    assert obj["label"] == "plan · running", "the mode is folded in, as the listing folds it"


def test_status_prints_the_task_and_names_a_plans_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The show page prints the task, and a plan's show names its plans page.

    Text and JSON keep the whole task, including heading marks and body lines.
    """
    d = _make_run(
        tmp_path,
        monkeypatch,
        [{"ts": _ts(5), "type": "session.start", "mode": "run", "user_task": "# add a flag\nmore"}],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "task:       # add a flag\n            more\n" in out
    assert "plan:" not in out
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["task"] == "# add a flag\nmore"

    (d / "manifest.json").write_text(json.dumps({"mode": "plan"}))
    (d / "plan.md").write_text("# Plan\n", encoding="utf-8")
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    assert "plan:       agent6 plan show winsome-dawn-YWH5ZS" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("verification", "line"),
    [
        ("passed", "verify:     green on the final tree"),
        ("failed", "verify:     red on the final tree"),
        ("unverified", "verify:     not verified: no verify covered the final tree"),
        ("not_applicable", "verify:     no gate"),
        (None, None),
    ],
)
def test_show_says_whether_the_final_tree_was_verified(
    verification: str | None,
    line: str | None,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`sessions show` had no verify line, so a stopped run said nothing about its tree."""
    end: dict[str, object] = {
        "ts": _ts(3),
        "type": "session.stop",
        "reason": "steer_abort",
    }
    if verification is not None:
        end["verification"] = verification
    _make_run(tmp_path, monkeypatch, [{"ts": _ts(40), "type": "session.start", "mode": "run"}, end])
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert [ln for ln in out.splitlines() if ln.startswith("verify:")] == ([line] if line else [])
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["verification"] == verification


def test_show_says_a_timed_out_gate_timed_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `verify:` line reads "timed out" under a "gate timed out" headline."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {
                "ts": _ts(9),
                "type": "verify.end",
                "exit_code": 124,
                "outcome": "timed_out",
                "timeout_s": 300.0,
            },
            {
                "ts": _ts(3),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "failed",
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS") == 0
    out = capsys.readouterr().out
    assert "status:     finished · gate timed out" in out
    assert "verify:     timed out on the final tree" in out
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["verification"] == "failed"


def test_show_says_why_a_stale_run_with_no_snapshot_cannot_be_resumed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that never reached a turn gets the reason, not a resume command resume refuses."""
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {"ts": _ts(30), "type": "tool.call", "turn": 1},
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn") == 0
    out = capsys.readouterr().out
    assert "resume with:" not in out
    assert "resume:     it never reached its first turn" in out


def test_show_json_carries_the_points_earlier_executions_banked(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`plan_observed_points` sums every execution and `plan_cap` is the latest one's.

    The banked points ride beside them, so a script can tell this execution's spend.
    """
    _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(40), "type": "session.start", "mode": "run"},
            {
                "ts": _ts(35),
                "type": "budget.update",
                "usd_total": 0.0,
                "plan_observed_points": 3.0,
                "plan_cap": 10.0,
                "plan_used_percent": 5.0,
            },
            {
                "ts": _ts(30),
                "type": "session.stop",
                "reason": "steer_abort",
                "verification": "unverified",
            },
            {"ts": _ts(20), "type": "session.resume", "mode": "run"},
            {
                "ts": _ts(10),
                "type": "budget.update",
                "usd_total": 0.0,
                "plan_observed_points": 2.0,
                "plan_cap": 10.0,
                "plan_used_percent": 7.0,
            },
        ],
    )
    assert sessions_show._cmd_status("winsome-dawn", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert (obj["plan_observed_points"], obj["plan_prior_executions"], obj["plan_cap"]) == (
        5.0,
        3.0,
        10.0,
    )


def test_a_live_run_asking_after_an_earlier_end_reads_waiting(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resume that asks before its execution begins is waiting, not the earlier end's word.

    It read "passed (worker pid N alive)", and `agent6 answer` saved the answer for a resume
    instead of handing it to the live worker.
    """
    from agent6.viewmodel import listing

    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(5), "type": "session.start", "mode": "run"},
            {
                "ts": _ts(10),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
            {
                "ts": _ts(20),
                "type": "question.prompt",
                "id": "q1",
                "prompt": "continue?",
                "wait": True,
            },
        ],
    )
    process.write_worker_pid(d, os.getpid())
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "waiting"
    assert listing.summarize_session_dir(d).status == "waiting"
    assert listing.session_is_live(d)


def test_a_record_a_newer_agent6_wrote_reads_unreadable_with_the_upgrade(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An older agent6 says the record is newer than it reads, not that the worker died."""
    from agent6.sessions import manifest as sessions_manifest

    d = _make_run(
        tmp_path,
        monkeypatch,
        [
            {"ts": _ts(5), "type": "session.start", "mode": "run"},
            {
                "ts": _ts(10),
                "type": "session.stop",
                "reason": "finish_session",
                "verification": "passed",
            },
        ],
    )
    record = json.loads((d / "manifest.json").read_text())
    record["version"] = sessions_manifest.MANIFEST_VERSION + 1
    (d / "manifest.json").write_text(json.dumps(record))
    assert sessions_show._cmd_status("winsome-dawn-YWH5ZS", as_json=True) == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["status"] == "unreadable"
    assert "newer record" in obj["detail"] and "upgrade agent6" in obj["detail"]
