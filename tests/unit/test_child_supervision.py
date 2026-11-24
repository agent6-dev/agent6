# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child runs under a deadline and a turn cap, and its process is owned until it is reaped."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any

import pytest

from agent6 import child_records, paths
from agent6.app.parallel import children as app_children
from agent6.app.parallel import launch as child_launch
from agent6.app.parallel import relay as child_watch
from agent6.app.parallel import wait as child_wait
from agent6.config import Config
from agent6.sessions import ipc, journal, process
from agent6.tools import operator_prompts
from agent6.ui import spawn
from agent6.ui.cli import child_agent, interact, parallel

# Each child reports ready through a file, so a test never signals it before its handler is in.
_DEAF = (
    "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
    " open(sys.argv[1], 'w').close(); time.sleep(60)"
)
_POLITE = "import sys, time; open(sys.argv[1], 'w').close(); time.sleep(60)"


def _child(script: str, tmp_path: pathlib.Path) -> subprocess.Popen[bytes]:
    ready = tmp_path / f"ready-{time.monotonic_ns()}"
    proc = subprocess.Popen([sys.executable, "-c", script, str(ready)], start_new_session=True)
    deadline = time.monotonic() + 10.0
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), "the child never reported ready"
    return proc


def _reaped(pid: int) -> bool:
    """Return whether the pid is no longer a child of ours: reaped, or never ours."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    return False


def _live_forever(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """A child session dir that reads running for as long as this test process lives."""
    d = tmp_path / "child-state" / "sessions" / "runs" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 0, "session_id": name, "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(d, os.getpid())
    return d


@pytest.mark.parametrize(
    ("configured", "cap", "expected"),
    [(50, 200, 50), (500, 200, 200), (-1, 200, 200), (200, 200, 200), (-1, None, -1), (7, None, 7)],
)
def test_the_turn_cap_clamps_and_unlimited_cannot_lift_it(
    configured: int, cap: int | None, expected: int
) -> None:
    assert child_records.ChildLimits(max_turns=cap).clamp_turns(configured) == expected


def test_apply_limits_clamps_the_config_and_leaves_the_rest(tmp_path: pathlib.Path) -> None:
    cfg = Config.model_validate({"harness": {"max_turns": -1}})
    clamped = child_launch.apply_limits(cfg, child_records.ChildLimits(max_turns=200))
    assert clamped.harness.max_turns == 200
    assert child_launch.apply_limits(cfg, None) is cfg
    assert child_launch.apply_limits(cfg, child_records.ChildLimits(deadline_at=1.0)) == cfg


def test_a_launch_round_trips_its_limits(tmp_path: pathlib.Path) -> None:
    launch = child_launch.ChildLaunch(
        worktree=tmp_path / "child-1",
        record_state=tmp_path / "state",
        session_id="fan-l1",
        task="t",
        config=tmp_path / "launch" / "child-1-config.toml",
        parent="co",
        group="fan",
        number=1,
        limits=child_records.ChildLimits(max_turns=200, deadline_at=1_800_000.5),
    )
    path = child_launch.write(tmp_path / "launch", launch)
    assert child_launch.read_file(path).limits == launch.limits


def test_await_child_ends_at_the_deadline(tmp_path: pathlib.Path) -> None:
    """The await returns False once the clock passes the deadline, a stop's shape."""
    d = _live_forever(tmp_path, "deadline-l1")
    res = child_records.ChildResult(
        spec=child_records.ChildSpec(1, "deadline-l1", tmp_path / "w", None),
        session_dir=d,
        branch="agent6/x",
        ok=True,
        error="",
    )
    ticks = iter([0.0, 1.0, 2.0, 10.0, 11.0])
    assert (
        child_wait.await_child(
            res, poll_interval_s=0.001, deadline_at=5.0, clock=lambda: next(ticks)
        )
        is False
    )
    # No deadline: the await runs until the stop check says so.
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    assert child_wait.await_child(res, poll_interval_s=0.001, should_stop=stop_after_two) is False


def test_stop_child_terminates_then_kills_and_reaps(tmp_path: pathlib.Path) -> None:
    """A child that ignores TERM is KILLed after the grace and reaped; a polite one just stops."""
    deaf = _child(_DEAF, tmp_path)
    assert child_wait.stop_child(deaf.pid, grace_s=0.3) == "killed"
    assert not process.pid_alive(deaf.pid)
    assert _reaped(deaf.pid)
    polite = _child(_POLITE, tmp_path)
    assert child_wait.stop_child(polite.pid, grace_s=3.0) == "stopped"
    assert not process.pid_alive(polite.pid)
    assert _reaped(polite.pid)
    assert child_wait.stop_child(0, grace_s=0.1) == "gone"
    assert child_wait.stop_child(polite.pid, grace_s=0.1) == "gone"


_PARENT = """
import os, sys, time
from pathlib import Path
from agent6.ui import spawn
found, pid, err = spawn.spawn_child(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    Path(sys.argv[1]), ready=lambda pid: Path(sys.argv[1]),
    stop_reason=lambda _: "", env=dict(os.environ),
)
print(pid, flush=True)
time.sleep(60)
"""


def test_spawn_child_ties_the_child_to_its_parent(tmp_path: pathlib.Path) -> None:
    """Killing the parent ends its child: the child carries PDEATHSIG."""
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT, str(tmp_path)],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert parent.stdout is not None
    try:
        child_pid = int(parent.stdout.readline().decode().strip())
        assert process.pid_alive(child_pid)
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5.0
        while process.pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process.pid_alive(child_pid), "the child outlived its dead parent"
    finally:
        parent.kill()
        parent.wait(timeout=5)


_FRONT_END = """
import os, sys, time
from pathlib import Path
from agent6.ui import spawn
home = Path(sys.argv[1])
session = (
    "import json, os, time, pathlib; d = pathlib.Path(%r) / 'session'; d.mkdir();"
    " (d / 'pid').write_text(str(os.getpid()));"
    " (d / 'logs.jsonl').write_text(json.dumps("
    "{'type': 'session.start', 'worker_pid': os.getpid()}) + chr(10)); time.sleep(60)"
) % str(home)
found, err = spawn.spawn_and_locate(
    [sys.executable, "-c", session], home, before=set(),
    list_dirs=lambda: [d for d in home.iterdir() if d.is_dir()],
)
print(found, err, flush=True)
time.sleep(60)
"""


def test_a_session_a_front_end_starts_outlives_the_front_end(tmp_path: pathlib.Path) -> None:
    """A run started from the web UI or the hub ended when that front-end's process died.

    The parent tie is for a run's own children; a front-end's launch carried it too.
    """
    front_end = subprocess.Popen(
        [sys.executable, "-c", _FRONT_END, str(tmp_path)],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert front_end.stdout is not None
    session_pid = 0
    try:
        assert front_end.stdout.readline().decode().split()[0] == str(tmp_path / "session")
        session_pid = int((tmp_path / "session" / "pid").read_text())
        front_end.kill()
        front_end.wait(timeout=5)
        time.sleep(1.0)
        assert process.pid_alive(session_pid), "the session ended with its front-end"
    finally:
        front_end.kill()
        front_end.wait(timeout=5)
        if session_pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(session_pid, signal.SIGKILL)


def _spec(tmp_path: pathlib.Path, name: str) -> child_records.ChildSpec:
    return child_records.ChildSpec(
        number=1, session_id=name, worktree=tmp_path / "w", settings=None
    )


def test_a_child_that_ended_during_the_last_poll_can_import(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _live_forever(tmp_path, "late-l1")
    res = child_records.ChildResult(_spec(tmp_path, d.name), d, "agent6/late-l1", True, "")
    now = 0.0

    def late_wakeup(_interval: float) -> None:
        nonlocal now
        with (d / "logs.jsonl").open("a", encoding="utf-8") as log:
            log.write('{"type":"session.stop","reason":"operator_stop"}\n')
        (d / "worker.pid").unlink()
        now += child_wait.STOP_GRACE_S + 1

    monkeypatch.setattr(
        child_wait, "time", types.SimpleNamespace(monotonic=lambda: now, sleep=late_wakeup)
    )
    assert child_wait.drain_child(res, poll_interval_s=1.0, hard_stop=None)


def test_a_child_past_its_deadline_is_stopped_and_reported_failed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent ends a child at its deadline, kills what ignores the stop, and says so."""
    monkeypatch.setattr(child_wait, "STOP_GRACE_S", 0.05)
    monkeypatch.setattr(
        child_wait, "stop_child", functools.partial(child_wait.stop_child, grace_s=0.05)
    )
    d = _live_forever(tmp_path, "late-l1")
    deaf = _child(_DEAF, tmp_path)
    spec = _spec(tmp_path, "late-l1")

    def spawner(spec: child_records.ChildSpec, task: str) -> child_records.ChildResult:
        return child_records.ChildResult(spec, d, "agent6/x", True, "", pid=deaf.pid)

    res = app_children.run_child_to_completion(
        spec,
        "t",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=tmp_path,
        record_state=tmp_path / "origin-state",
        group="fan",
        parent="co",
        runtime=parallel.child_runtime(),
        spawner=spawner,
        poll_interval_s=0.01,
        limits=child_records.ChildLimits(deadline_at=time.time() - 1.0),
    )
    assert not res.ok
    assert "deadline" in res.error and "killed" in res.error
    assert "not imported" in res.error
    assert ipc.stop_request_is_now(d)
    assert not process.pid_alive(deaf.pid) and _reaped(deaf.pid)


def test_a_child_is_stopped_once_the_runs_budget_is_spent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent ends a child when its watch says the run's budget is spent, and says so.

    A child ran on to its own share's end whatever its siblings had spent.
    """
    monkeypatch.setattr(child_wait, "STOP_GRACE_S", 0.05)
    monkeypatch.setattr(
        child_wait, "stop_child", functools.partial(child_wait.stop_child, grace_s=0.05)
    )
    d = _live_forever(tmp_path, "spent-l1")
    deaf = _child(_DEAF, tmp_path)
    spec = _spec(tmp_path, "spent-l1")
    polled: list[tuple[pathlib.Path, str]] = []

    def spawner(spec: child_records.ChildSpec, task: str) -> child_records.ChildResult:
        return child_records.ChildResult(spec, d, "agent6/x", True, "", pid=deaf.pid)

    def watch(
        child_dir: pathlib.Path,
        label: str,
        _status: str | child_watch.ChildStartup,
        _interrupted: Callable[[], bool],
    ) -> bool:
        polled.append((child_dir, label))
        return True

    res = app_children.run_child_to_completion(
        spec,
        "t",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=tmp_path,
        record_state=tmp_path / "origin-state",
        group="fan",
        parent="co",
        runtime=parallel.child_runtime(),
        spawner=spawner,
        poll_interval_s=0.01,
        watch=watch,
    )
    assert polled == [(d, "attempt 1 spent-l1")]
    assert not res.ok
    assert res.error.startswith("stopped at the run's budget; child") and "killed" in res.error
    assert ipc.stop_request_is_now(d)
    assert not process.pid_alive(deaf.pid) and _reaped(deaf.pid)


@pytest.mark.parametrize("end", ["stop", "teardown", "deadline", "budget"])
def test_a_child_can_be_stopped_before_its_start_report(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, end: str
) -> None:
    """Dispatch returns a handle while the child is starting; its watch still owns the stop."""
    from tests.unit import test_children_background as background

    repo = tmp_path / "repo"
    background._init_repo(repo)
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "clones")}}})
    monkeypatch.setattr(
        child_launch,
        "bootstrap_argv",
        lambda: [sys.executable, "-c", "import time; time.sleep(15)"],
    )
    waiting, spent = threading.Event(), threading.Event()

    def watch(
        _directory: pathlib.Path,
        _label: str,
        status: str | child_watch.ChildStartup,
        _interrupted: Callable[[], bool],
    ) -> bool:
        assert isinstance(status, child_watch.ChildStartup)
        waiting.set()
        return spent.is_set()

    group = app_children.build_child_spawner(
        cfg,
        repo,
        paths.state_dir(repo),
        explicit_leaves=frozenset(),
        parent_session_id="co",
        runtime=parallel.child_runtime(),
        watch=watch,
    )
    limits = child_records.ChildLimits(deadline_at=time.time() if end == "deadline" else None)
    (child,) = group([child_records.ChildTask(task="t", model=None, limits=limits)], "p1")
    try:
        if end != "deadline":
            assert waiting.wait(5) and not child.result.done()
        if end == "stop":
            child.stop()
        elif end == "teardown":
            group.close()
        elif end == "budget":
            spent.set()
        res = child.result.result(timeout=5)
    finally:
        group.close()
    reason = {"deadline": "deadline", "budget": "budget"}.get(end, "stopped by its parent")
    assert not res.ok and reason in res.error
    assert res.pid > 0 and not process.pid_alive(res.pid) and _reaped(res.pid)
    assert not res.session_dir.exists() and res.spec.worktree.is_dir()


def test_stop_does_not_wait_for_a_preflight_approval(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child parked at setup must stop without anyone answering its parent's prompt."""
    from tests.unit import test_children_background as background

    repo = tmp_path / "repo"
    background._init_repo(repo)
    state = paths.state_dir(repo)
    parent = child_wait.child_dir(state, "co", mode="run")
    parent.mkdir(parents=True)
    ipc.set_away_mode(parent, "wait")
    directory = parent.with_name("co-p1-l1")
    bootstrap = """\
import json, os, sys, time
from pathlib import Path
child = Path(sys.argv[1]); child.mkdir()
(child / 'worker.pid').write_text(str(os.getpid()))
row = {'type': 'approval.prompt', 'id': 'approval-1', 'prompt': 'Allow setup?', 'scope': 'command'}
(child / 'logs.jsonl').write_text(json.dumps(row) + '\\n')
time.sleep(60)
"""
    monkeypatch.setattr(
        child_launch, "bootstrap_argv", lambda: [sys.executable, "-c", bootstrap, str(directory)]
    )
    opened = threading.Event()

    def emit(row: journal.Event) -> None:
        if isinstance(row, journal.ApprovalPrompt):
            opened.set()

    prompts = operator_prompts.OperatorPrompts(
        session_dir=parent, approver=interact.build_approver(parent), journal=emit
    )
    group = app_children.build_child_spawner(
        Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "clones")}}}),
        repo,
        state,
        explicit_leaves=frozenset(),
        parent_session_id=parent.name,
        runtime=parallel.child_runtime(),
        watch=child_watch.ChildWatch(prompts, parent),
    )
    (child,) = group([child_records.ChildTask(task="t", model=None)], "p1")
    try:
        assert opened.wait(5)
        child.stop()
        result = child.result.result(timeout=5)
        assert not result.ok and "stopped by its parent" in result.error
        assert not process.pid_alive(result.pid) and _reaped(result.pid)
        assert not ipc.answer_written(directory, "approval-1")
    finally:
        if not child.result.done():
            ipc.write_answer(parent, "approval-1", "no")
        group.close()
        child.result.result(timeout=5)


def test_a_report_during_a_stop_poll_still_starts_the_child(tmp_path: pathlib.Path) -> None:
    """A start reported while checking for a stop belongs to the normal result path."""
    log = tmp_path / "logs.jsonl"
    bootstrap = """\
from pathlib import Path
import time
while not Path('report').exists(): time.sleep(0.01)
Path('logs.jsonl').write_text('{"type":"session.start"}\\n')
time.sleep(10)
"""

    def stop_reason(_pid: int) -> str:
        (tmp_path / "report").touch()
        until = time.monotonic() + 5
        while not log.exists() and time.monotonic() < until:
            time.sleep(0.01)
        assert log.exists()
        return "stopped by its parent"

    found, pid, error = spawn.spawn_child(
        [sys.executable, "-c", bootstrap],
        tmp_path,
        ready=lambda _: tmp_path if log.exists() and log.read_text() else None,
        stop_reason=stop_reason,
        env=dict(os.environ),
    )
    try:
        assert found == tmp_path and not error
        assert process.pid_alive(pid)
    finally:
        child_wait.stop_child(pid)


def test_arm_deadline_writes_the_stop_at_the_deadline(tmp_path: pathlib.Path) -> None:
    """The child's own backstop asks itself to stop at the deadline; the kill waits the grace."""
    d = tmp_path / "sessions" / "runs" / "armed-l1"
    d.mkdir(parents=True)
    timers = child_agent.arm_deadline(d, time.time() + 0.2, grace_s=60.0)
    try:
        deadline = time.monotonic() + 5.0
        while not ipc.stop_request_pending(d) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ipc.stop_request_is_now(d)
        assert ipc.steer_answer_is_abort(d)
        assert all(t.daemon for t in timers)
    finally:
        for t in timers:
            t.cancel()


def test_a_real_child_ends_by_its_deadline(
    short_repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A child whose model keeps running commands ends at its deadline, not as a success."""
    from agent6.app import worktrees
    from agent6.viewmodel.session import log as session_log
    from tests import launch_env

    clone = short_repo / "clone"
    clone.mkdir()
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@t"),
        ("config", "user.name", "t"),
    ):
        subprocess.run(["git", "-C", str(clone), *args], check=True, capture_output=True)
    (clone / "README.md").write_text("deadline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(clone), "commit", "-q", "-m", "init"], check=True, capture_output=True
    )
    replies: list[tuple[str, dict[str, Any]]] = [("run_command", {"argv": ["sleep", "30"]})]
    with launch_env.scripted_provider(replies) as (base_url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/base"}},
                "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "allow"},
                "harness": {"verify": {"when": "never"}, "max_turns": 50},
                "review": {"when": "never"},
            }
        )
        into = child_launch.launch_dir(paths.state_dir(short_repo), "deadline")
        spec = child_records.ChildSpec(
            number=1, session_id="deadline-l1", worktree=clone, settings=None
        )
        config = worktrees.write_child_config(
            cfg,
            spec,
            explicit_leaves=frozenset(
                {"sandbox.isolation", "sandbox.protect_git", "sandbox.run_commands"}
            ),
            into=into,
        )
        path = child_launch.write(
            into,
            child_launch.ChildLaunch(
                worktree=clone,
                clone=child_launch.CloneReceipt.capture(clone),
                record_state=paths.state_dir(clone),
                session_id="deadline-l1",
                task="keep running",
                config=config.path,
                notes=config.notes,
                parent="deadline",
                group="deadline",
                number=1,
                limits=child_records.ChildLimits(max_turns=200, deadline_at=time.time() + 3.0),
            ),
        )
        env = {
            **os.environ,
            child_launch.ENV: str(path),
            "AGENT6_CHILD": "1",
            "AGENT6_STREAM_TO_LOG": "1",
            "AGENT6_ALLOW_ROOT": "1",
        }
        started = time.monotonic()
        proc = subprocess.run(
            child_launch.bootstrap_argv(),
            cwd=str(clone),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert proc.returncode != 0, proc.stdout
    assert time.monotonic() - started < 30.0
    log = paths.state_dir(clone) / "sessions" / "runs" / "deadline-l1" / "logs.jsonl"
    scan = session_log.scan_session_log(log)
    assert scan.stop_reason not in ("finish_session", "max_turns"), scan.stop_reason
