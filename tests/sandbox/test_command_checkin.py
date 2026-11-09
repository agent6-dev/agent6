# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A model's command is handed back rather than killed at its check-in.

A wall-clock timeout has to answer a question it cannot: whether a command that
has run twenty minutes is stuck or working. `[harness].command_checkin_s`
replaces the kill with a hand-back, so the judgement goes to the model (or the
operator), and the command keeps running either way.

The operator's gate (`run_verify_command`) is deliberately NOT part of this: the
loop needs a verdict from it, not a handle.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from typing import Any

import pytest

from agent6 import kinds
from agent6.config import Config
from agent6.sandbox import jail, jail_session
from agent6.tools import dispatch, tool_errors


def _dispatcher(tmp_path: pathlib.Path, checkin: float) -> dispatch.ToolDispatcher:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "allow"},
            "harness": {"command_checkin_s": checkin},
        }
    )
    return dispatch.ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
    )


def _run(d: dispatch.ToolDispatcher, script: str) -> dict[str, Any]:
    return d.dispatch("run_command", {"argv": ["/bin/sh", "-c", script]}).to_wire()


def test_a_command_that_finishes_is_an_ordinary_result(tmp_path: pathlib.Path) -> None:
    d = _dispatcher(tmp_path, checkin=30.0)
    try:
        out = _run(d, "echo fast; exit 2")
    finally:
        d.close()
    assert out["returncode"] == 2
    assert out["stdout"] == "fast\n"
    assert "still_running" not in out
    assert "background_id" not in out


def test_a_command_outliving_the_checkin_comes_back_as_a_background_job(
    tmp_path: pathlib.Path,
) -> None:
    """One ExecResult shape either way: `returncode` null and a `background_id` naming the command.

    Nothing has to branch on "a result or a handle".
    """
    d = _dispatcher(tmp_path, checkin=0.5)
    try:
        started = time.monotonic()
        out = _run(d, "echo starting; sleep 30; echo never")
        elapsed = time.monotonic() - started

        assert out["returncode"] is None
        assert out["still_running"] is True
        assert out["background_id"] == "bg1"
        # The output from before the hand-back comes back with it.
        assert out["stdout"] == "starting\n"
        assert elapsed < 10, "the hand-back waited for the command instead of returning"

        # It really is still running, and it is an ordinary background job now.
        read = d.dispatch("read_background", {"id": "bg1"}).to_wire()
        assert "running" in str(read["shells"])
        assert "starting" in str(read["output"])
        stopped = d.dispatch("stop_background", {"id": "bg1"}).to_wire()
        assert "stopped" in str(stopped["shells"])
    finally:
        d.close()


def test_a_zero_checkin_waits_for_the_command(tmp_path: pathlib.Path) -> None:
    """`0` disables the hand-back.

    Correct when a human is watching and can interrupt, and the path a run with no background
    roster falls back to.
    """
    d = _dispatcher(tmp_path, checkin=0.0)
    try:
        out = _run(d, "sleep 1; echo waited")
    finally:
        d.close()
    assert out["returncode"] == 0
    assert out["stdout"] == "waited\n"
    assert "background_id" not in out


def test_the_verify_gate_is_never_handed_back(tmp_path: pathlib.Path) -> None:
    """The operator's gate returns a verdict; a handle would leave the loop nothing to decide on."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "allow"},
            "harness": {
                "command_checkin_s": 0.5,
                "verify_command": ["/bin/sh", "-c", "sleep 2; echo verified"],
                "verify_timeout_s": 30.0,
            },
        }
    )
    d = dispatch.ToolDispatcher(
        root=root, config=cfg, isolation="none", session_dir=session_dir, use_jail_session=True
    )
    try:
        out = d.dispatch("run_verify_command", {}).to_wire()
    finally:
        d.close()
    assert out["returncode"] == 0
    assert "verified" in out["stdout"]
    assert "background_id" not in out


@pytest.mark.parametrize("checkin", [0.5, 0.0])
def test_nothing_a_handed_back_command_started_outlives_the_run(
    tmp_path: pathlib.Path, checkin: float
) -> None:
    """Teardown stops the roster, so a handed-back command dies with the run like any other."""
    d = _dispatcher(tmp_path, checkin=checkin)
    marker = tmp_path / "pid"
    try:
        _run(d, f"echo $$ > {marker}; sleep 0.2")
    finally:
        d.close()
    assert marker.exists()
    pid = int(marker.read_text().strip())
    try:
        state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        state = "gone"
    assert state in ("gone", "Z")


# --- one exec tool -----------------------------------------------------------


def test_background_true_returns_the_same_shape_immediately(tmp_path: pathlib.Path) -> None:
    """`background: true` is a check-in of zero: one parameter, not a second tool and shape."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        out = d.dispatch(
            "run_command", {"argv": ["/bin/sh", "-c", "echo up; sleep 30"], "background": True}
        ).to_wire()
        assert out["returncode"] is None
        assert out["still_running"] is True
        assert out["background_id"] == "bg1"
        # The command has only just started, so poll briefly for its first
        # line rather than assuming it has already reached the log.
        deadline = time.monotonic() + 5.0
        output = ""
        while time.monotonic() < deadline and "up" not in output:
            output = str(
                d.dispatch("read_background", {"id": "bg1", "wait_s": 0}).to_wire()["output"]
            )
            if "up" not in output:
                time.sleep(0.1)
        assert "up" in output
        d.dispatch("stop_background", {"id": "bg1"})
    finally:
        d.close()


def test_the_background_flag_replaced_the_second_tool(tmp_path: pathlib.Path) -> None:
    """One exec tool plus read + kill, matching what models are trained on."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        names = d.available_tool_names()
        assert "run_background" not in names
        assert {"run_command", "read_background", "stop_background"} <= set(names)
    finally:
        d.close()


def test_a_read_only_mode_cannot_background(tmp_path: pathlib.Path) -> None:
    """Only a session that edits owns a background command's lifetime.

    Every other mode is a short read-only pass and would kill it at the end; derived from the
    same tool set that withholds read_background there.
    """
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate({"sandbox": {"run_commands": "allow"}})
    d = dispatch.ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        mode="ask",
        session_dir=session_dir,
        use_jail_session=True,
    )
    try:
        with pytest.raises(tool_errors.ToolError, match="not available in ask mode"):
            d.dispatch("run_command", {"argv": ["/bin/echo", "hi"], "background": True})
    finally:
        d.close()


def test_an_operator_stop_cuts_a_wait_short(tmp_path: pathlib.Path) -> None:
    """A Stop pressed during a check-in wait ends the wait.

    Stop is a marker file polled at a turn boundary, and a tool call in flight reaches no
    boundary; unread, a Stop sits for the whole wait (measured: the full 10s of a 10s wait,
    and the default wait is 900).
    """
    from agent6.sessions import ipc

    d = _dispatcher(tmp_path, checkin=900.0)
    session_dir = tmp_path / "session"
    try:
        d.dispatch("run_command", {"argv": ["/bin/sh", "-c", "sleep 60"], "background": True})
        ipc.request_stop(session_dir)
        started = time.monotonic()
        d.dispatch("read_background", {"id": "bg1", "wait_s": 10})
        waited = time.monotonic() - started
        assert waited < 3.0, f"the stop was ignored for {waited:.1f}s of a 10s wait"
        d.dispatch("stop_background", {"id": "bg1"})
    finally:
        d.close()


def test_a_wait_still_waits_when_nobody_asked_to_stop(tmp_path: pathlib.Path) -> None:
    """The negative control: without a stop marker the wait runs to the command's end."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        d.dispatch("run_command", {"argv": ["/bin/sh", "-c", "sleep 2"], "background": True})
        started = time.monotonic()
        d.dispatch("read_background", {"id": "bg1", "wait_s": 30})
        waited = time.monotonic() - started
        assert 1.0 < waited < 15.0, f"waited {waited:.1f}s; expected to wait for the ~2s command"
    finally:
        d.close()


@pytest.mark.parametrize("stop", ["stop now", "lane stop"])
def test_an_immediate_stop_ends_a_running_command_at_once(
    tmp_path: pathlib.Path, stop: str
) -> None:
    """An immediate stop ends a foreground command without leaving a live handle in its result.

    `agent6 stop` writes the abort steer; a fan-out stops its lanes with the marker's `now` urgency.
    """
    from agent6.sessions import ipc

    d = _dispatcher(tmp_path, checkin=900.0)
    session_dir = tmp_path / "session"
    try:
        if stop == "stop now":
            ipc.submit_steer(session_dir, "abort", now=True)
        else:
            ipc.request_stop(session_dir, now=True)
        started = time.monotonic()
        out = _run(d, "sleep 30")
        waited = time.monotonic() - started
        assert waited < 5.0, f"the stop was ignored for {waited:.1f}s of a 900s check-in"
        assert out["stopped_by_operator"] is True and out["returncode"] != 0
        assert "still_running" not in out and "background_id" not in out
        read = d.dispatch("read_background", {"id": "bg1", "wait_s": 0}).to_wire()
        assert "stopped" in str(read["shells"]) and "running" not in str(read["shells"])
    finally:
        d.close()


def test_a_stop_after_the_step_lets_the_command_finish(tmp_path: pathlib.Path) -> None:
    """The stop marker alone leaves a running command to finish and return its own result.

    `stop --after-turn` handed the turn's command back as a background job on the first poll, and
    the end of the execution then killed it: the step the operator asked to let finish was cut.
    """
    from agent6.sessions import ipc

    d = _dispatcher(tmp_path, checkin=900.0)
    session_dir = tmp_path / "session"
    try:
        ipc.request_stop(session_dir)
        out = _run(d, "sleep 1; echo done")
        assert out["returncode"] == 0
        assert "done" in out["stdout"]
        assert "background_id" not in out and "still_running" not in out
    finally:
        d.close()


def test_a_command_runs_to_the_end_when_nobody_asked_to_stop(tmp_path: pathlib.Path) -> None:
    """The negative control: no marker, so the same command returns its own result and no handle."""
    d = _dispatcher(tmp_path, checkin=900.0)
    try:
        started = time.monotonic()
        out = _run(d, "sleep 2; echo done")
        waited = time.monotonic() - started
        assert out["returncode"] == 0
        # A finished command carries no handle at all, not an empty one.
        assert "background_id" not in out and "still_running" not in out
        assert "done" in out["stdout"]
        assert waited > 1.5, f"returned in {waited:.1f}s; the command sleeps 2"
    finally:
        d.close()


def test_a_plan_or_ask_command_runs_bounded_instead_of_handing_back(tmp_path: pathlib.Path) -> None:
    """Where the hand-back is unusable the command runs bounded.

    Plan and ask permit `run_command` but withhold `read_background` and `stop_background`,
    so a hand-back there would leave the model a handle it could neither poll nor stop.
    """
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "allow"}, "harness": {"command_checkin_s": 0.3}}
    )
    d = dispatch.ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
        mode="plan",
    )
    out = _run(d, "sleep 1; echo done")
    assert out.get("background_id") is None and not out.get("still_running"), out
    assert out["returncode"] == 0 and "done" in out["stdout"]


_LEVELS = ["none", "partial", pytest.param("full", marks=pytest.mark.needs_namespaces)]


@contextlib.contextmanager
def _supervisor(
    cwd: pathlib.Path, isolation: kinds.IsolationLevel = "none"
) -> Generator[subprocess.Popen[bytes]]:
    """Drive the real launcher using only its standard streams."""
    proc = subprocess.Popen(
        [str(jail.require_jail_binary())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _send(proc, {"cwd": str(cwd), "argv": [], "isolation": isolation, "mode": "serve"})
        _answer(proc)  # Readiness is pinned separately; cancellation must fail on its own.
        yield proc
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()


def _send(proc: subprocess.Popen[bytes], frame: dict[str, object]) -> None:
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(frame) + "\n").encode())
    proc.stdin.flush()


def _answer(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    assert proc.stdout is not None
    deadline = time.monotonic() + 5
    line = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0 and select.select([proc.stdout], [], [], remaining)[0], (
            "supervisor did not answer within 5s"
        )
        chunk = os.read(proc.stdout.fileno(), 65536)
        assert chunk, "supervisor closed the response channel before a complete frame"
        line.extend(chunk)
        if b"\n" in chunk:
            return json.loads(line)


def test_the_response_deadline_includes_the_final_newline() -> None:
    """A readable partial response must not turn the helper's deadline into an unbounded read."""
    script = (
        "import sys,time; sys.stdout.write('{\"id\":1'); sys.stdout.flush(); "
        "time.sleep(6); print('}', flush=True)"
    )
    with subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE) as proc:
        try:
            with pytest.raises(AssertionError, match="supervisor did not answer within 5s"):
                _answer(proc)
        finally:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize("isolation", _LEVELS)
@pytest.mark.parametrize("handback", [False, True])
def test_command_channel_cancel_precedes_the_checkin(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel, handback: bool
) -> None:
    """Cancellation needs no inherited fd and stops when hand-back is unavailable."""
    with _supervisor(tmp_path, isolation) as proc:
        _send(
            proc,
            {
                "kind": "run",
                "id": 1,
                "argv": ["sleep", "30"],
                "timeout_s": 20,
                "checkin_s": 900 if handback else 0,
                "log_dir": str(tmp_path),
            },
        )
        started = time.monotonic()
        _send(proc, {"kind": "cancel", "id": 1})
        answer = _answer(proc)
        assert time.monotonic() - started < 5
        assert answer["id"] == 1
        if handback:
            assert answer["backgrounded"] is True
            _send(proc, {"kind": "stop", "id": 2, "pid": answer["pid"]})
            assert _answer(proc)["stopped"] is True
        else:
            assert answer["returncode"] == 137 and not answer.get("backgrounded")


@pytest.mark.parametrize("delay", [0.0, 0.08, 0.2])
def test_completion_racing_cancel_cannot_cancel_the_next_request(
    tmp_path: pathlib.Path, delay: float
) -> None:
    """An old ID is harmless whether cancellation or completion wins."""
    with _supervisor(tmp_path) as proc:
        _send(proc, {"kind": "run", "id": 1, "argv": ["sh", "-c", "sleep 0.08; exit 7"]})
        time.sleep(delay)
        _send(proc, {"kind": "cancel", "id": 1})
        answer = _answer(proc)
        assert answer["id"] == 1 and answer["returncode"] in (7, 137)
        _send(proc, {"kind": "cancel", "id": 1})
        _send(proc, {"kind": "run", "id": 2, "argv": ["sh", "-c", "sleep 0.3; echo next"]})
        _send(proc, {"kind": "cancel", "id": 1})
        after = _answer(proc)
        assert after["id"] == 2 and after["returncode"] == 0
        assert after["stdout"] == "next\n"


@pytest.mark.parametrize("isolation", _LEVELS)
def test_incomplete_cancellation_does_not_block_command_polling(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """An unfinished frame preserves the command's timeout and the next request's framing."""
    marker = tmp_path / "started"
    with _supervisor(tmp_path, isolation) as proc:
        _send(
            proc,
            {
                "kind": "run",
                "id": 1,
                "argv": ["sh", "-c", f"echo $$ > {marker}; exec sleep 30"],
                "timeout_s": 1,
            },
        )
        deadline = time.monotonic() + 3
        while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "foreground command never started"
        assert proc.stdin is not None
        proc.stdin.write(b'{"kind":"cancel","id":')
        proc.stdin.flush()
        answer = _answer(proc)
        assert answer["id"] == 1 and answer["timed_out"] and answer["returncode"] == 124
        proc.stdin.write(b"1}\n")
        _send(proc, {"kind": "run", "id": 2, "argv": ["echo", "next"]})
        after = _answer(proc)
        assert after["id"] == 2 and after["returncode"] == 0 and after["stdout"] == "next\n"


@pytest.mark.parametrize("isolation", _LEVELS)
@pytest.mark.parametrize(
    ("frame", "close_input", "error"),
    [
        pytest.param(b"", True, None, id="eof"),
        pytest.param(b'{"kind":"cancel","id":', True, b"EOF in protocol frame", id="partial-eof"),
        pytest.param(b'{"kind":oops}\n', False, b"expected value", id="malformed"),
    ],
)
def test_invalid_input_stops_a_foreground_command_without_waiting_for_its_deadline(
    tmp_path: pathlib.Path,
    isolation: kinds.IsolationLevel,
    frame: bytes,
    close_input: bool,
    error: bytes | None,
) -> None:
    """The active request handles EOF and malformed frames and reaps its command."""
    marker = tmp_path / "started"
    with _supervisor(tmp_path, isolation) as proc:
        _send(
            proc,
            {"kind": "run", "id": 1, "argv": ["sh", "-c", f"echo $$ > {marker}; exec sleep 30"]},
        )
        deadline = time.monotonic() + 3
        while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "foreground command never started"
        assert proc.stdin is not None
        proc.stdin.write(frame)
        proc.stdin.flush()
        if close_input:
            proc.stdin.close()
        proc.wait(timeout=5)
        assert proc.returncode == (2 if error else 0)
        if error:
            assert proc.stderr is not None
            assert error in proc.stderr.read()
        if isolation != "full":
            pid = int(marker.read_text())
            assert not pathlib.Path(f"/proc/{pid}").exists(), "command was not reaped"


def test_output_flooding_does_not_block_cancellation(tmp_path: pathlib.Path) -> None:
    """Command output never owns the supervisor's control stream."""
    with _supervisor(tmp_path) as proc:
        _send(proc, {"kind": "run", "id": 1, "argv": ["sh", "-c", "yes x & yes y >&2 & wait"]})
        time.sleep(0.2)
        started = time.monotonic()
        _send(proc, {"kind": "cancel", "id": 1})
        answer = _answer(proc)
        assert time.monotonic() - started < 5
        assert answer["id"] == 1 and answer["returncode"] == 137
        assert len(answer["stdout"]) < 9 * 1024 * 1024
        assert len(answer["stderr"]) < 9 * 1024 * 1024


def test_session_stop_without_handback_is_immediate(tmp_path: pathlib.Path) -> None:
    """Python sends cancellation even when the request cannot become a background job."""
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    try:
        started = time.monotonic()
        answer = session.run(("sleep", "10"), timeout_s=3, interrupted=lambda: True)
        assert time.monotonic() - started < 2
        assert isinstance(answer, kinds.CommandResult) and answer.returncode == 137
    finally:
        session.close()


def test_supervisor_bounds_request_frames(tmp_path: pathlib.Path) -> None:
    """The launcher rejects an oversized frame before parsing or executing it."""
    with _supervisor(tmp_path) as proc:
        with contextlib.suppress(BrokenPipeError):
            _send(proc, {"kind": "run", "id": 1, "argv": ["echo", "x" * 1048576]})
        proc.wait(timeout=5)
        assert proc.stderr is not None
        assert b"frame exceeds" in proc.stderr.read()
        assert proc.returncode == 2
