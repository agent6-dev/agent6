from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import select
import shlex
import signal
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_session
from agent6.tools import background


class Peer:
    """Drive the real launcher without Python's process cleanup."""

    def __init__(self, cwd: pathlib.Path, isolation: str = "none") -> None:
        self.proc = subprocess.Popen(
            [str(jail.require_jail_binary())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.seq = 0
        self.send({"cwd": str(cwd), "argv": [], "isolation": isolation, "mode": "serve"})
        assert self.read()["ready"]

    def send(self, value: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(value) + "\n").encode())
        self.proc.stdin.flush()

    def read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        assert select.select([self.proc.stdout], [], [], 15)[0], "launcher did not answer"
        return json.loads(self.proc.stdout.readline())

    def ask(self, kind: str, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        self.send({"kind": kind, "id": self.seq, **fields})
        answer = self.read()
        assert answer.pop("id") == self.seq
        return answer

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)
            for pipe in (self.proc.stdout, self.proc.stderr):
                if pipe is not None:
                    pipe.close()


def job_field(answer: dict[str, Any]) -> dict[str, Any]:
    return {"handle": answer["handle"]}


def wait_file(path: pathlib.Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.02)
    pytest.fail(f"descendant never wrote {path}")


def alive(pid: int) -> bool:
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.fixture(params=["none", "partial"])
def peer(tmp_path: pathlib.Path, request: pytest.FixtureRequest) -> Iterator[Peer]:
    owner = Peer(tmp_path, request.param)
    try:
        yield owner
    finally:
        owner.close()
        for marker in tmp_path.glob("*.pid"):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(marker.read_text()), signal.SIGKILL)


def daemon(marker: pathlib.Path, *, hold: bool, double_fork: bool) -> list[str]:
    # The intermediate exits after the grandchild has detached and reported readiness.
    script = (
        "import os,time; from pathlib import Path; "
        "child=os.fork(); "
        "os._exit(0) if child else None; os.setsid(); "
        + ("child=os.fork(); os._exit(0) if child else None; " if double_fork else "")
        + f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    gate = f"while [ ! -s '{marker}' ]; do sleep 0.01; done; "
    return [
        "/bin/sh",
        "-c",
        f"/usr/bin/python3 -c {shlex.quote(script)}; " + gate + ("sleep 60" if hold else "exit 23"),
    ]


@pytest.mark.parametrize("double_fork", [False, True])
def test_foreground_completion_reaps_detached_descendants(
    peer: Peer, tmp_path: pathlib.Path, double_fork: bool
) -> None:
    marker = tmp_path / "fg.pid"
    answer = peer.ask("run", argv=daemon(marker, hold=False, double_fork=double_fork))
    assert answer["returncode"] == 23
    pid = wait_file(marker)
    assert not pathlib.Path(f"/proc/{pid}").exists(), (
        "foreground response left an unreaped descendant"
    )


@pytest.mark.parametrize("double_fork", [False, True])
def test_background_stop_reaps_detached_descendants(
    peer: Peer, tmp_path: pathlib.Path, double_fork: bool
) -> None:
    marker = tmp_path / "bg.pid"
    job = peer.ask("background", argv=daemon(marker, hold=True, double_fork=double_fork))
    pid = wait_file(marker)
    assert alive(pid)
    assert peer.ask("stop", **job_field(job))["stopped"]
    assert not pathlib.Path(f"/proc/{pid}").exists(), "stop confirmed with an unreaped descendant"


@pytest.mark.parametrize("foreground", [False, True])
def test_cleanup_does_not_kill_a_sibling_that_joined_the_group(
    peer: Peer, tmp_path: pathlib.Path, foreground: bool
) -> None:
    group = tmp_path / "group.pid"
    joined = tmp_path / "joined.pid"
    joiner = (
        "import os,time; from pathlib import Path; "
        f"group=Path({str(group)!r}); joined=Path({str(joined)!r})\n"
        "while not group.exists() or not group.read_text().strip(): time.sleep(.01)\n"
        "os.setpgid(0,int(group.read_text())); joined.write_text(str(os.getpid())); time.sleep(60)"
    )
    sibling = peer.ask("background", argv=["/usr/bin/python3", "-c", joiner])
    leader = peer.ask(
        "run" if foreground else "background",
        argv=[
            "/bin/sh",
            "-c",
            f"echo $$ > '{group}'; while [ ! -s '{joined}' ]; do sleep .01; done; "
            + ("exit 0" if foreground else "sleep 60"),
        ],
    )
    wait_file(joined)
    if not foreground:
        peer.ask("stop", **job_field(leader))
    assert peer.ask("status", **job_field(sibling))["running"], "cleanup killed another job"


def test_repeated_stop_retains_the_stop_outcome(peer: Peer) -> None:
    job = peer.ask("background", argv=["/bin/sleep", "60"])
    first = peer.ask("stop", **job_field(job))
    assert first["returncode"] == 137
    assert peer.ask("stop", **job_field(job)) == first


def test_status_then_repeated_stop_retains_the_observed_outcome(peer: Peer) -> None:
    job = peer.ask("background", argv=["/bin/sh", "-c", "exit 23"])
    deadline = time.monotonic() + 5
    status = peer.ask("status", **job_field(job))
    while status["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
        status = peer.ask("status", **job_field(job))
    assert status["returncode"] == 23
    assert peer.ask("status", **job_field(job))["returncode"] == 23
    first = peer.ask("stop", **job_field(job))
    assert first["returncode"] == 23
    assert peer.ask("stop", **job_field(job)) == first


def test_handles_are_instance_scoped_and_reject_foreign_or_stale_values(
    peer: Peer, tmp_path: pathlib.Path
) -> None:
    other = Peer(tmp_path)
    try:
        a = peer.ask("background", argv=["/bin/sleep", "60"])
        b = other.ask("background", argv=["/bin/sleep", "60"])
        assert isinstance(a.get("handle"), str), "a public pid is not a job handle"
        assert a["handle"] != b["handle"]
        for handle in (a["handle"], "stale:1", str(os.getpid()), "-1"):
            answer = other.ask("stop", handle=handle)
            assert answer.get("error") and not answer.get("stopped")
        assert other.ask("status", handle=b["handle"])["running"]
        assert peer.ask("status", handle=a["handle"])["running"]
    finally:
        other.close()


def test_two_launchers_do_not_reap_each_others_late_orphans(
    peer: Peer, tmp_path: pathlib.Path
) -> None:
    other = Peer(tmp_path)
    try:
        first = peer.ask(
            "background", argv=daemon(tmp_path / "first.pid", hold=True, double_fork=True)
        )
        second = other.ask(
            "background", argv=daemon(tmp_path / "second.pid", hold=True, double_fork=True)
        )
        a, b = wait_file(tmp_path / "first.pid"), wait_file(tmp_path / "second.pid")
        assert peer.ask("stop", **job_field(first))["stopped"]
        assert not pathlib.Path(f"/proc/{a}").exists()
        assert alive(b)
        assert other.ask("status", **job_field(second))["running"]
    finally:
        other.close()


def test_a_late_double_fork_stays_with_its_job(peer: Peer, tmp_path: pathlib.Path) -> None:
    gate = tmp_path / "go"
    marker = tmp_path / "late.pid"
    argv = daemon(marker, hold=True, double_fork=True)
    argv[-1] = f"while [ ! -e '{gate}' ]; do sleep 0.01; done; " + argv[-1]
    older = peer.ask("background", argv=argv)
    younger = peer.ask("background", argv=["/bin/sleep", "60"])
    gate.touch()
    pid = wait_file(marker)
    assert peer.ask("stop", **job_field(younger))["stopped"]
    assert alive(pid), "a later job's stop took the older job's orphan"
    assert peer.ask("stop", **job_field(older))["stopped"]
    assert not pathlib.Path(f"/proc/{pid}").exists()


def test_job_reaper_cleans_up_after_session_supervisor_loss(
    peer: Peer, tmp_path: pathlib.Path
) -> None:
    marker = tmp_path / "loss.pid"
    peer.ask("background", argv=daemon(marker, hold=True, double_fork=True))
    pid = wait_file(marker)
    peer.proc.kill()
    peer.proc.wait(timeout=5)
    deadline = time.monotonic() + 7
    while pathlib.Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not pathlib.Path(f"/proc/{pid}").exists(), "private job channel loss left a descendant"


def test_eof_cleans_owned_descendants_without_being_pid_one(tmp_path: pathlib.Path) -> None:
    owner = Peer(tmp_path)
    marker = tmp_path / "eof.pid"
    pid = 0
    try:
        owner.ask("background", argv=daemon(marker, hold=False, double_fork=True))
        pid = wait_file(marker)
        assert owner.proc.pid != 1 and alive(pid)
        owner.close()
        assert not pathlib.Path(f"/proc/{pid}").exists(), "EOF left an unreaped descendant"
    finally:
        if owner.proc.poll() is None:
            owner.close()
        if pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.needs_namespaces
@pytest.mark.parametrize("backgrounded", [False, True])
def test_full_isolation_uses_the_same_descendant_lifecycle(
    tmp_path: pathlib.Path, backgrounded: bool
) -> None:
    owner = Peer(tmp_path, "full")
    marker = tmp_path / "full.pid"
    try:
        job = owner.ask(
            "background" if backgrounded else "run",
            argv=daemon(marker, hold=backgrounded, double_fork=True),
        )
        pid = wait_file(marker)
        if backgrounded:
            owner.ask("stop", **job_field(job))
        probe = owner.ask("run", argv=["/bin/test", "!", "-e", f"/proc/{pid}"])
        assert probe["returncode"] == 0, "completion or stop left an unreaped descendant"
    finally:
        owner.close()


def test_close_reports_unconfirmed_cleanup_after_supervisor_loss(tmp_path: pathlib.Path) -> None:
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none", network="host")
    )
    session.start_background(("/bin/sleep", "60"))
    proc = session._proc  # pyright: ignore[reportPrivateUsage]
    proc.kill()
    proc.wait(timeout=5)
    error = session.close()
    assert error and "unknown" in str(error), "transport exit is not confirmed job cleanup"
    assert session.close() == error


@pytest.mark.parametrize("isolation", ["none", "partial"])
def test_closing_a_hung_transport_leaves_job_reapers_alive(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation=isolation, network="host")
    )
    marker = tmp_path / "hung.pid"
    pid = 0
    try:
        session.start_background(tuple(daemon(marker, hold=True, double_fork=True)))
        pid = wait_file(marker)
        os.kill(session._proc.pid, signal.SIGSTOP)  # pyright: ignore[reportPrivateUsage]
        error = session.close()
        assert error and "unknown" in str(error)
        deadline = time.monotonic() + 7
        while pathlib.Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not pathlib.Path(f"/proc/{pid}").exists(), (
            "host teardown killed the private job reaper"
        )
    finally:
        session.close()
        if pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_client_loss_persists_unknown_not_an_invented_completion(tmp_path: pathlib.Path) -> None:
    shells = background.BackgroundShells(tmp_path / "shells")
    policy = kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none", network="host")
    session = jail_session.JailSession.open(policy)
    try:
        shells.start(
            ("/bin/sleep", "60"),
            lambda argv, _paths: dataclasses.replace(policy, argv=argv),
            session=session,
        )
        session.close()
        view = shells.roster()[0]
        assert view.returncode is None and view.state == "died"
        record = json.loads((tmp_path / "shells/bg1/result.json").read_text())
        assert record["returncode"] is None and record["error"]
        assert "unknown" in background.roster_from_dir(tmp_path / "shells")[0]
    finally:
        shells.stop_all()
        session.close()
