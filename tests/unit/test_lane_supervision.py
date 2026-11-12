# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A lane runs under a deadline and a turn cap, and its process is owned until it is reaped."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import pytest

from agent6 import lane_records
from agent6.app import lane_launch, lane_watch
from agent6.app import lanes as app_lanes
from agent6.config import Config
from agent6.sessions import ipc, process
from agent6.ui.cli import lane_agent, parallel

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
    """A lane run dir that reads running for as long as this test process lives."""
    d = tmp_path / "lane-state" / "sessions" / "runs" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": name, "mode": "run", "user_task": "t"}) + "\n",
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
    assert lane_records.LaneLimits(max_turns=cap).clamp_turns(configured) == expected


def test_apply_limits_clamps_the_config_and_leaves_the_rest(tmp_path: pathlib.Path) -> None:
    cfg = Config.model_validate({"harness": {"max_turns": -1}})
    clamped = lane_launch.apply_limits(cfg, lane_records.LaneLimits(max_turns=200))
    assert clamped.harness.max_turns == 200
    assert lane_launch.apply_limits(cfg, None) is cfg
    assert lane_launch.apply_limits(cfg, lane_records.LaneLimits(deadline_at=1.0)) == cfg


def test_a_launch_round_trips_its_limits(tmp_path: pathlib.Path) -> None:
    launch = lane_launch.LaneLaunch(
        workdir=tmp_path / "lane-1",
        session_id="fan-l1",
        task="t",
        config=tmp_path / "launch" / "lane-1-config.toml",
        coordinator="co",
        group="fan",
        lane=1,
        limits=lane_records.LaneLimits(max_turns=200, deadline_at=1_800_000.5),
    )
    path = lane_launch.write(tmp_path / "launch", launch)
    assert lane_launch.read_file(path).limits == launch.limits


def test_await_lane_ends_at_the_deadline(tmp_path: pathlib.Path) -> None:
    """The await returns False once the clock passes the deadline, a stop's shape."""
    d = _live_forever(tmp_path, "deadline-l1")
    res = lane_records.LaneResult(
        spec=lane_records.LaneSpec(1, "deadline-l1", tmp_path / "w", None),
        session_dir=d,
        branch="agent6/x",
        ok=True,
        error="",
    )
    ticks = iter([0.0, 1.0, 2.0, 10.0, 11.0])
    assert (
        lane_watch.await_lane(
            res, poll_interval_s=0.001, deadline_at=5.0, clock=lambda: next(ticks)
        )
        is False
    )
    # No deadline: the await runs until the stop check says so.
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    assert lane_watch.await_lane(res, poll_interval_s=0.001, should_stop=stop_after_two) is False


def test_stop_child_terminates_then_kills_and_reaps(tmp_path: pathlib.Path) -> None:
    """A child that ignores TERM is KILLed after the grace and reaped; a polite one just stops."""
    deaf = _child(_DEAF, tmp_path)
    assert lane_watch.stop_child(deaf.pid, grace_s=0.3) == "killed"
    assert not process.pid_alive(deaf.pid)
    assert _reaped(deaf.pid)
    polite = _child(_POLITE, tmp_path)
    assert lane_watch.stop_child(polite.pid, grace_s=3.0) == "stopped"
    assert not process.pid_alive(polite.pid)
    assert _reaped(polite.pid)
    assert lane_watch.stop_child(0, grace_s=0.1) == "gone"
    assert lane_watch.stop_child(polite.pid, grace_s=0.1) == "gone"


_PARENT = """
import os, sys, time
from pathlib import Path
from agent6.ui import spawn
found, pid, err = spawn.spawn_lane(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    Path(sys.argv[1]), before=set(), list_dirs=list, env=dict(os.environ), timeout_s=0.3,
)
print(pid, flush=True)
time.sleep(60)
"""


def test_spawn_lane_ties_the_child_to_its_parent(tmp_path: pathlib.Path) -> None:
    """Killing the coordinator ends its lane: the child carries PDEATHSIG."""
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
        assert not process.pid_alive(child_pid), "the lane outlived its dead coordinator"
    finally:
        parent.kill()
        parent.wait(timeout=5)


def _spec(tmp_path: pathlib.Path, name: str) -> lane_records.LaneSpec:
    return lane_records.LaneSpec(lane=1, session_id=name, workdir=tmp_path / "w", settings=None)


def test_a_lane_past_its_deadline_is_stopped_and_reported_failed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator ends a lane at its deadline, kills what ignores the stop, and says so."""
    monkeypatch.setattr(lane_watch, "STOP_GRACE_S", 0.2)
    d = _live_forever(tmp_path, "late-l1")
    deaf = _child(_DEAF, tmp_path)
    spec = _spec(tmp_path, "late-l1")

    def spawner(spec: lane_records.LaneSpec, task: str) -> lane_records.LaneResult:
        return lane_records.LaneResult(spec, d, "agent6/x", True, "", pid=deaf.pid)

    res = app_lanes.run_lane_to_completion(
        spec,
        "t",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=tmp_path,
        origin_state=tmp_path / "origin-state",
        group="fan",
        coordinator="co",
        runtime=parallel.lane_runtime(),
        spawner=spawner,
        poll_interval_s=0.01,
        limits=lane_records.LaneLimits(deadline_at=time.time() - 1.0),
    )
    assert not res.ok
    assert "deadline" in res.error and "killed" in res.error
    assert "not imported" in res.error
    assert ipc.stop_request_is_now(d)
    assert not process.pid_alive(deaf.pid) and _reaped(deaf.pid)


def test_a_lane_is_stopped_once_the_runs_budget_is_spent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator ends a lane when its watch says the run's budget is spent, and says so.

    A lane ran on to its own share's end whatever its siblings had spent.
    """
    monkeypatch.setattr(lane_watch, "STOP_GRACE_S", 0.2)
    d = _live_forever(tmp_path, "spent-l1")
    deaf = _child(_DEAF, tmp_path)
    spec = _spec(tmp_path, "spent-l1")
    polled: list[tuple[pathlib.Path, str]] = []

    def spawner(spec: lane_records.LaneSpec, task: str) -> lane_records.LaneResult:
        return lane_records.LaneResult(spec, d, "agent6/x", True, "", pid=deaf.pid)

    def watch(child_dir: pathlib.Path, label: str, _status: str) -> bool:
        polled.append((child_dir, label))
        return True

    res = app_lanes.run_lane_to_completion(
        spec,
        "t",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=tmp_path,
        origin_state=tmp_path / "origin-state",
        group="fan",
        coordinator="co",
        runtime=parallel.lane_runtime(),
        spawner=spawner,
        poll_interval_s=0.01,
        watch=watch,
    )
    assert polled == [(d, "lane 1 spent-l1")]
    assert not res.ok
    assert res.error.startswith("stopped at the run's budget; lane") and "killed" in res.error
    assert ipc.stop_request_is_now(d)
    assert not process.pid_alive(deaf.pid) and _reaped(deaf.pid)


def test_a_lane_that_never_reported_is_killed_not_left_running(
    tmp_path: pathlib.Path,
) -> None:
    """A spawn that timed out before a session dir appeared still owns a process: it is stopped."""
    polite = _child(_POLITE, tmp_path)
    spec = _spec(tmp_path, "mute-l1")

    def spawner(spec: lane_records.LaneSpec, task: str) -> lane_records.LaneResult:
        return lane_records.LaneResult(
            spec, tmp_path / "none", "agent6/x", False, "has not reported starting", pid=polite.pid
        )

    res = app_lanes.run_lane_to_completion(
        spec,
        "t",
        explicit_leaves=frozenset(),
        cfg=Config(),
        origin=tmp_path,
        origin_state=tmp_path / "origin-state",
        group="fan",
        coordinator="co",
        runtime=parallel.lane_runtime(),
        spawner=spawner,
        poll_interval_s=0.01,
    )
    assert not res.ok
    assert "has not reported starting" in res.error and "stopped" in res.error
    assert not process.pid_alive(polite.pid) and _reaped(polite.pid)


def test_arm_deadline_writes_the_stop_at_the_deadline(tmp_path: pathlib.Path) -> None:
    """The lane's own backstop asks itself to stop at the deadline; the kill waits the grace."""
    d = tmp_path / "sessions" / "runs" / "armed-l1"
    d.mkdir(parents=True)
    timers = lane_agent.arm_deadline(d, time.time() + 0.2, grace_s=60.0)
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


def test_a_real_lane_ends_by_its_deadline(short_repo: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A lane whose model keeps running commands ends at its deadline, not as a success."""
    from agent6 import paths
    from agent6.app import workdirs
    from agent6.viewmodel.session import log as session_log
    from tests import lane_env

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
    with lane_env.scripted_provider(replies) as (base_url, _calls):
        cfg = Config.model_validate(
            {
                "providers": {"p": {"backend": "openai", "base_url": base_url}},
                "models": {"default": {"model": "p/base"}},
                "sandbox": {"isolation": "none", "protect_git": "off", "run_commands": "allow"},
                "harness": {"verify": {"when": "never"}, "max_turns": 50},
                "review": {"when": "never"},
            }
        )
        into = lane_launch.launch_dir(paths.state_dir(short_repo), "deadline")
        spec = lane_records.LaneSpec(lane=1, session_id="deadline-l1", workdir=clone, settings=None)
        config = workdirs.write_lane_config(
            cfg,
            spec,
            explicit_leaves=frozenset(
                {"sandbox.isolation", "sandbox.protect_git", "sandbox.run_commands"}
            ),
            into=into,
        )
        path = lane_launch.write(
            into,
            lane_launch.LaneLaunch(
                workdir=clone,
                session_id="deadline-l1",
                task="keep running",
                config=config.path,
                notes=config.notes,
                coordinator="deadline",
                group="deadline",
                lane=1,
                limits=lane_records.LaneLimits(max_turns=200, deadline_at=time.time() + 3.0),
            ),
        )
        env = {
            **os.environ,
            lane_launch.ENV: str(path),
            "AGENT6_SUBRUN": "1",
            "AGENT6_STREAM_TO_LOG": "1",
            "AGENT6_ALLOW_ROOT": "1",
        }
        started = time.monotonic()
        proc = subprocess.run(
            lane_launch.bootstrap_argv(),
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
