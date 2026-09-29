# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Background commands must never lie about being alive.

The failure these pin is the one that makes a background feature useless: a
command dies, nothing says so, and the agent either waits forever or keeps
reporting a shell that is already gone.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, cast

import pytest

from agent6 import kinds
from agent6.sandbox import jail
from agent6.tools import background

pytestmark = pytest.mark.needs_namespaces


@pytest.fixture
def shells(tmp_path: pathlib.Path) -> background.BackgroundShells:
    if jail.locate_jail_binary() is None:
        pytest.skip("no agent6-jail binary")
    return background.BackgroundShells(tmp_path / "shells")


def _policy_for(cwd: pathlib.Path, isolation: kinds.IsolationLevel = "hardened"):
    def build(argv: tuple[str, ...], rw: tuple[pathlib.Path, ...]) -> kinds.JailPolicy:
        return kinds.JailPolicy(
            cwd=cwd, argv=argv, isolation=isolation, extra_rw_paths=rw, timeout_s=60.0
        )

    return build


def _wait_state(
    shells: background.BackgroundShells, shell_id: str, state: str, timeout: float = 15.0
) -> str:
    """Poll until *shell_id* reaches *state*. Returns the state actually seen."""
    deadline = time.monotonic() + timeout
    seen = ""
    while time.monotonic() < deadline:
        seen = next(v.state for v in shells.roster() if v.id == shell_id)
        if seen == state:
            return seen
        time.sleep(0.05)
    return seen


def test_a_command_that_exits_on_its_own_reports_its_code(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A background command that ended reads as over, not "running".

    State comes from the process, the code from the launcher. The state at start is not
    pinned: a command this short can be gone before `start` samples it, and "exited" is then
    the true answer; the long-lived command below carries that half.
    """
    view = shells.start(("/bin/sh", "-c", "echo bye; exit 7"), _policy_for(tmp_path))
    assert _wait_state(shells, view.id, "exited") == "exited"
    after, output = shells.read(view.id, tail_lines=50)
    assert after.returncode == 7
    assert "bye" in output


def test_a_command_killed_from_outside_is_never_reported_running(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A crash agent6 did not ask for (an OOM kill, an operator's kill -9) reads as over."""
    view = shells.start(("/bin/sh", "-c", "echo up; sleep 300"), _policy_for(tmp_path))
    deadline = time.monotonic() + 15.0
    while "up" not in shells.read(view.id, tail_lines=10)[1]:
        assert time.monotonic() < deadline, "the command never started"
        time.sleep(0.05)
    pid = next(iter(_launcher_pids(shells, view.id)))
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    assert _wait_state(shells, view.id, "died") in {"died", "exited"}
    assert not any(v.state == "running" for v in shells.roster())


def test_reading_a_live_command_never_blocks(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A read of a command that will run for minutes returns immediately."""
    view = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    start = time.monotonic()
    for _ in range(5):
        shells.read(view.id, tail_lines=10)
        shells.roster()
    assert time.monotonic() - start < 2.0
    shells.stop(view.id)


def test_stopped_and_died_are_different_words(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A stop and a death read differently, so a disappearance never looks deliberate."""
    stopped = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    failed = shells.start(("/bin/sh", "-c", "exit 3"), _policy_for(tmp_path))
    assert shells.stop(stopped.id).state == "stopped"
    assert _wait_state(shells, failed.id, "exited") == "exited"
    words = {v.id: v.state for v in shells.roster()}
    assert words == {stopped.id: "stopped", failed.id: "exited"}


def test_the_roster_rides_on_every_answer(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """Reading one command reports them all, so a second one dying is seen unasked."""
    quiet = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    doomed = shells.start(("/bin/sh", "-c", "exit 1"), _policy_for(tmp_path))
    assert _wait_state(shells, doomed.id, "exited") == "exited"
    roster = shells.read(quiet.id, tail_lines=5)[0]
    assert roster.state == "running"
    assert [v.state for v in shells.roster() if v.id == doomed.id] == ["exited"]
    shells.stop_all()


def test_stop_all_kills_everything_and_the_processes_are_gone(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """Run-end teardown: nothing a run started may outlive it."""
    views = [
        shells.start(("/bin/sh", "-c", f"echo {i}; sleep 300"), _policy_for(tmp_path))
        for i in range(3)
    ]
    pids = {p for v in views for p in _launcher_pids(shells, v.id)}
    assert shells.stop_all()
    for pid in pids:
        deadline = time.monotonic() + 10.0
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid), f"launcher {pid} survived teardown"
    assert not any(v.state == "running" for v in shells.roster())
    assert shells.stop_all() == []  # idempotent


def test_an_unknown_id_names_what_exists(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    shells.start(("/bin/true",), _policy_for(tmp_path))
    with pytest.raises(background.BackgroundError, match="bg99"):
        shells.read("bg99", tail_lines=5)


def test_output_survives_the_command(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """The log is a file, so a command's output stays readable after it is gone."""
    view = shells.start(("/bin/sh", "-c", "echo first; echo second; exit 0"), _policy_for(tmp_path))
    assert _wait_state(shells, view.id, "exited") == "exited"
    _after, output = shells.read(view.id, tail_lines=50)
    assert "first" in output and "second" in output


@pytest.mark.skipif(shutil.which("unshare") is None, reason="needs userns for strict")
def test_strict_confines_a_background_command_too(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A detached command runs in the same jail as a foreground one."""
    view = shells.start(
        ("/bin/sh", "-c", "echo escaped > /etc/agent6-bg-escape"), _policy_for(tmp_path, "strict")
    )
    assert _wait_state(shells, view.id, "exited") == "exited"
    assert not pathlib.Path("/etc/agent6-bg-escape").exists()
    assert next(v for v in shells.roster() if v.id == view.id).returncode != 0


def test_a_foreground_commands_sweep_spares_a_background_one(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A background command survives the escapee sweep of every later command.

    The sweep kills whatever a jailed command leaves behind, and a background command has
    exactly that shape: a live jail nobody is waiting on. It is a deliberate child.
    """
    view = shells.start(("/bin/sh", "-c", "echo up; sleep 300"), _policy_for(tmp_path))
    deadline = time.monotonic() + 15.0
    while "up" not in shells.read(view.id, tail_lines=10)[1]:
        assert time.monotonic() < deadline, "the command never started"
        time.sleep(0.05)
    for _ in range(3):
        res = jail.run_in_jail(
            kinds.JailPolicy(
                cwd=tmp_path, argv=("/bin/true",), isolation="hardened", timeout_s=10.0
            )
        )
        assert res.returncode == 0
    assert next(v for v in shells.roster() if v.id == view.id).state == "running"
    shells.stop_all()


def test_a_command_started_mid_sweep_window_is_spared(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A background command started mid-foreground survives that command's sweep.

    It is not in the foreground command's before-snapshot, so only its registration as a
    live launcher keeps the sweep off it.
    """
    started: list[str] = []

    def start_midway() -> None:
        time.sleep(0.4)
        started.append(shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path)).id)

    thread = threading.Thread(target=start_midway)
    thread.start()
    res = jail.run_in_jail(
        kinds.JailPolicy(
            cwd=tmp_path, argv=("/bin/sh", "-c", "sleep 2"), isolation="hardened", timeout_s=20.0
        )
    )
    thread.join()
    assert res.returncode == 0
    assert next(v for v in shells.roster() if v.id == started[0]).state == "running"
    shells.stop_all()


def _launcher_pids(shells: background.BackgroundShells, shell_id: str) -> set[int]:
    shell = shells._get(shell_id)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(shell.job, jail.BackgroundJob), "these pin the per-command launcher"
    return {shell.job.pid}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_the_roster_is_readable_from_another_process(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """What each command was and how it ended is on disk, for surfaces in other processes."""
    live = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    done = shells.start(("/bin/sh", "-c", "exit 5"), _policy_for(tmp_path))
    assert _wait_state(shells, done.id, "exited") == "exited"

    lines = background.roster_from_dir(tmp_path / "shells")
    assert any(f"[{done.id}] exited 5" in line and "exit 5" in line for line in lines)
    assert any(f"[{live.id}] still running" in line for line in lines)
    shells.stop_all()


def test_an_empty_or_missing_dir_is_not_an_error(tmp_path: pathlib.Path) -> None:
    assert background.roster_from_dir(tmp_path / "nope") == []
    (tmp_path / "empty").mkdir()
    assert background.roster_from_dir(tmp_path / "empty") == []


@pytest.mark.parametrize("isolation", ["strict", "hardened"])
def test_a_detached_child_of_a_background_command_dies_with_the_run(
    shells: background.BackgroundShells, tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """`stop` sweeps the `setsid` child a background command left, on hardened too.

    `stop` kills the launcher's group, which by definition misses a child that left it, and
    `run_in_jail`'s sweep can never catch it either: by then it is not new.
    """
    beat = tmp_path / "beat"
    loop = f"for i in $(seq 60); do echo x >> {beat}; sleep 0.2; done"
    shells.start(
        ("/bin/sh", "-c", f"setsid /bin/sh -c {loop!r} </dev/null >/dev/null 2>&1 & sleep 30"),
        _policy_for(tmp_path, isolation),
    )
    deadline = time.monotonic() + 15.0
    while not beat.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    shells.stop_all()
    at_stop = beat.read_text(encoding="utf-8") if beat.exists() else ""
    time.sleep(3.0)
    later = beat.read_text(encoding="utf-8") if beat.exists() else ""
    assert later == at_stop, f"{isolation}: a detached child outlived the run ({later!r})"


def test_a_command_cannot_forge_its_own_exit_code_or_name(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A command cannot rewrite its own result or identity.

    Both live outside the directory the command holds read-write; an audit trail the audited
    party can rewrite is a suggestion box.
    """
    forge = (
        'd=$(awk "/agent6/ {print \\$2}" /proc/self/mounts | head -1); '
        'echo "{\\"returncode\\": 0}" > "$d/../result.json" 2>/dev/null; '
        'echo "{\\"command\\":\\"all green\\"}" > "$d/../meta.json" 2>/dev/null; '
        "exit 42"
    )
    import shlex

    argv = ("/bin/sh", "-c", forge)
    view = shells.start(argv, _policy_for(tmp_path, "strict"))
    assert _wait_state(shells, view.id, "exited") == "exited"
    after = next(v for v in shells.roster() if v.id == view.id)
    assert after.returncode == 42  # not the 0 it wrote
    assert after.command == shlex.join(argv)  # not the name it wrote
    assert any("exited 42" in line for line in background.roster_from_dir(tmp_path / "shells"))
    # The trail lives in the shell dir the command was never granted, so its
    # result.json carries the real 42 and no forged returncode=0 reached it.
    # BY ID, never next(iterdir()): the shells root also holds the logs/
    # sibling, and readdir order is filesystem-dependent -- picking blind made
    # this red on runners whose order put logs/ first.
    shell_dir = tmp_path / "shells" / view.id
    import json as _json

    result = _json.loads((shell_dir / "result.json").read_text(encoding="utf-8"))
    assert result["returncode"] == 42


def test_a_sweep_never_signals_a_process_group_it_does_not_own() -> None:
    """The sweep never signals a pgid it looked up earlier.

    A pgid is a leader's pid, reusable once that leader is reaped; under sudo an escapee
    exiting in the window would have root SIGKILL whatever group inherited the number.
    """
    import os
    import signal
    import subprocess

    # A group leader we hold: killing it by group is safe and takes the child.
    # New GROUP, same session, so a sibling can join it (setpgid is
    # session-scoped).
    leader = subprocess.Popen(["sleep", "30"], preexec_fn=lambda: os.setpgid(0, 0))  # noqa: PLW1509
    child = subprocess.Popen(["sleep", "30"], preexec_fn=lambda: os.setpgid(0, leader.pid))  # noqa: PLW1509
    try:
        assert os.getpgid(child.pid) == leader.pid
        jail.signal_group(leader.pid)
        assert leader.wait(timeout=5) != 0
        assert child.wait(timeout=5) != 0
    finally:
        for p in (leader, child):
            with contextlib.suppress(OSError):
                p.kill()

    # A non-leader: only it dies, never the group it happens to sit in.
    bystander = subprocess.Popen(["sleep", "30"], preexec_fn=lambda: os.setpgid(0, 0))  # noqa: PLW1509
    joiner = subprocess.Popen(["sleep", "30"], preexec_fn=lambda: os.setpgid(0, bystander.pid))  # noqa: PLW1509
    try:
        jail.signal_group(joiner.pid)
        assert joiner.wait(timeout=5) != 0
        assert bystander.poll() is None, "the sweep killed a group it did not lead"
    finally:
        for p in (bystander, joiner):
            with contextlib.suppress(OSError):
                os.kill(p.pid, signal.SIGKILL)
            p.wait(timeout=5)


def test_a_command_cannot_redirect_the_agent_at_another_file(tmp_path: pathlib.Path) -> None:
    """A command that symlinks its log at the operator's secrets gets nothing back.

    The jail holds RW (MakeSym included) on the log dir and `read` runs outside the jail as
    the operator, so the log is never opened by name. Proved under strict and hardened.
    """
    import os

    from agent6.config import Config
    from agent6.tools import policy

    secret = tmp_path / "vault"
    secret.mkdir()
    (secret / "secrets.toml").write_text('api_key = "sk-DECOY"\n', encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    shells = background.BackgroundShells(tmp_path / "shells")

    def _policy(argv: tuple[str, ...], rw: tuple[pathlib.Path, ...]) -> object:
        return policy.jail_policy(work, Config(), "none", argv, extra_rw_paths=rw)

    view = shells.start(("sh", "-c", "echo mine; sleep 30"), _policy)  # pyright: ignore[reportArgumentType]
    log = tmp_path / "shells" / "logs" / view.id / "out.log"
    for _ in range(100):
        if log.stat().st_size:
            break
        time.sleep(0.05)
    # What a jailed command can do inside its own granted dir:
    log.unlink()
    log.symlink_to(secret / "secrets.toml")

    _view, text = shells.read(view.id, tail_lines=50)
    shells.stop_all()

    assert "sk-DECOY" not in text, "the agent followed the command's symlink"
    assert "mine" in text, "the real output must still be readable"
    assert os.path.realpath(log) == str(secret / "secrets.toml")  # the swap did happen


def test_the_sweep_spares_a_session_the_agent_opened_on_purpose() -> None:
    """A `/btw` ask or a `/parallel` lane started later is not swept as an escapee.

    They are agent6's own children in their own session, exactly what an escapee looks like;
    the exclusion set is not frozen at the background command's start.
    """
    import subprocess

    ours = subprocess.Popen(["sleep", "30"], start_new_session=True)
    stranger = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        jail.keep_out_of_the_sweep(ours.pid)
        jail._kill_escapees(frozenset())
        assert ours.poll() is None, "the sweep killed a session the agent opened"
        assert stranger.poll() is not None, "a real escapee must still be swept"
    finally:
        for p in (ours, stranger):
            with contextlib.suppress(OSError):
                p.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)


def test_a_planted_symlink_cannot_redirect_the_log_directory(tmp_path: pathlib.Path) -> None:
    """A planted `<log_root>/bg<N>` symlink never places the log where a command says.

    Every command holds read-write on the shared log root; `mkdir(exist_ok=True)` and
    `is_dir()` both follow a symlink, and the agent creates the log unconfined as the
    operator. O_NOFOLLOW on the leaf does not protect the path above it.
    """
    shells = background.BackgroundShells(tmp_path / "shells")
    victim = tmp_path / "victim"
    victim.mkdir()
    (tmp_path / "shells" / "logs" / "bg1").symlink_to(victim)

    with pytest.raises(background.BackgroundError):
        shells.start(("/bin/true",), _policy_for(tmp_path))
    assert not list(victim.iterdir()), "the agent wrote through a command's symlink"


def test_a_stop_that_did_not_stop_says_so(tmp_path: pathlib.Path) -> None:
    """A stop the launcher could not confirm never renders as "stopped".

    A process wedged in uninterruptible I/O across the deadline, or a dead session, keeps its
    reason on the surface: "stopped" is the one word an operator acts on.
    """
    from typing import cast

    class _Unconfirmed:
        """A job whose stop the launcher could not confirm."""

        def status(self) -> jail.BackgroundStatus:
            return jail.BackgroundStatus(running=True, returncode=None, error="")

        def stop(self) -> str:
            return "pid 7 did not exit within 5s of SIGKILL"

    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    shell = shells._get(view.id)  # pyright: ignore[reportPrivateUsage]
    shell.job = cast("jail.BackgroundJob", _Unconfirmed())

    got = shells.stop(view.id)
    assert got.state != "stopped", f"a stop that failed rendered as {got.state!r}"
    assert "did not exit" in got.detail, got


def test_a_command_that_failed_to_start_is_not_listed_as_running(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command that never started is absent from meta.json.

    meta.json is what the out-of-process surfaces read; written before the start attempt, it
    listed the command as running while the roster and read_background knew no such id.
    """
    import agent6.tools.background as bg

    def refuses(*_a: object, **_k: object) -> jail.BackgroundJob:
        raise jail.JailUnavailableError("no launcher today")

    monkeypatch.setattr(jail, "start_in_jail", refuses)
    root = tmp_path / "shells"
    shells = background.BackgroundShells(root)

    with pytest.raises(background.BackgroundError):
        shells.start(("/bin/true",), _policy_for(tmp_path))
    assert shells.roster() == []
    assert bg.roster_from_dir(root) == [], "a command that never started is listed on disk"


def test_a_platform_without_proc_still_starts_a_background_command(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background command under `isolation = "none"` starts and is tracked without /proc.

    The escapee sweep is a /proc mechanism and macOS resolves to `none`; a /proc read while
    taking the descendant snapshot must not fail the start of a command already spawned.
    """
    import agent6.sandbox.jail as jail_mod

    real_iterdir = pathlib.Path.iterdir

    def no_proc(self: pathlib.Path):  # pyright: ignore[reportMissingParameterType]
        if str(self) == "/proc":
            raise FileNotFoundError(2, "No such file or directory", "/proc")
        return real_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", no_proc)
    assert jail_mod._own_children() == {}  # pyright: ignore[reportPrivateUsage]

    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("/bin/sh", "-c", "echo up; sleep 300"), _policy_for(tmp_path, "none"))
    try:
        assert view.state == "running", view
    finally:
        monkeypatch.undo()
        shells.stop_all()


def test_an_unsandboxed_commands_exit_code_reaches_another_process(tmp_path: pathlib.Path) -> None:
    """Under `none`, the observed exit code is written down for other processes.

    There is no launcher to write it, so the job records the code on the first observed
    exit; otherwise `/shells` elsewhere reads the command as still running for the run's life.
    """
    shells = background.BackgroundShells(tmp_path / "shells")
    done = shells.start(("/bin/sh", "-c", "exit 42"), _policy_for(tmp_path, "none"))
    live = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path, "none"))
    try:
        assert _wait_state(shells, done.id, "exited") == "exited"
        lines = background.roster_from_dir(tmp_path / "shells")
        assert any(f"[{done.id}] exited 42" in line for line in lines), lines
        assert any(f"[{live.id}] still running" in line for line in lines), lines
    finally:
        shells.stop_all()


def test_a_stopped_unsandboxed_command_records_its_ending(tmp_path: pathlib.Path) -> None:
    """A stop is an ending: another process must not read it as maybe still running."""
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path, "none"))
    assert _wait_state(shells, view.id, "running") == "running"
    shells.stop(view.id)
    lines = background.roster_from_dir(tmp_path / "shells")
    assert not any("still running" in line for line in lines), lines


def test_a_stopped_jailed_command_records_its_ending(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A stop of a jailed command is recorded for other processes, without inventing a code.

    The stop SIGKILLs the launcher before it can write the exit code, so the record says the
    command was stopped rather than claiming a number nobody saw.
    """
    view = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    assert _wait_state(shells, view.id, "running") == "running"
    shells.stop(view.id)
    lines = background.roster_from_dir(tmp_path / "shells")
    assert not any("still running" in line for line in lines), lines
    assert any(f"[{view.id}] stopped" in line for line in lines), lines


def test_stopping_an_already_exited_command_keeps_its_exit_code(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """The stop record never replaces a real exit code with "stopped".

    `stop_all` stops exited shells too (one can have left a detached child), so the record
    runs over a result the launcher already wrote.
    """
    view = shells.start(("/bin/sh", "-c", "exit 42"), _policy_for(tmp_path))
    assert _wait_state(shells, view.id, "exited") == "exited"
    shells.stop_all()
    lines = background.roster_from_dir(tmp_path / "shells")
    assert any(f"[{view.id}] exited 42" in line for line in lines), lines


def test_an_unreadable_result_is_never_clobbered_by_a_stop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop record writes only over a result it read as empty.

    A failed read says nothing about what is on disk, so writing anyway would replace a real
    exit code with "stopped" on the strength of that failure.
    """
    import agent6.sandbox.jail as jail_mod

    outcome = tmp_path / "bg"
    outcome.mkdir()
    result = outcome / "result.json"
    result.write_text('{"returncode": 42}', encoding="utf-8")

    real_read = pathlib.Path.read_text

    def _unreadable(self: pathlib.Path, *a: object, **k: object) -> str:
        if self == result:
            raise OSError(5, "Input/output error")
        return real_read(self, *a, **k)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(pathlib.Path, "read_text", _unreadable)
    jail_mod._write_stopped(outcome)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.undo()
    assert result.read_text(encoding="utf-8") == '{"returncode": 42}'


def test_settle_records_an_ending_nobody_asked_about(
    shells: background.BackgroundShells, tmp_path: pathlib.Path
) -> None:
    """A background command's ending is observed at the turn boundary.

    The ending is written down when someone observes it, and the model may never look again
    after starting one; without the settle, `/shells` read a finished command as running.
    """
    done = shells.start(("/bin/sh", "-c", "exit 7"), _policy_for(tmp_path))
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        shells.settle()
        if any(
            f"[{done.id}] exited 7" in line
            for line in background.roster_from_dir(tmp_path / "shells")
        ):
            return
        time.sleep(0.05)
    pytest.fail(
        f"settle never recorded the ending: {background.roster_from_dir(tmp_path / 'shells')}"
    )


def test_an_unsandboxed_background_command_survives_a_siblings_stop(
    tmp_path: pathlib.Path,
) -> None:
    """Stopping one unsandboxed background command leaves its sibling running.

    The sweep spares every launcher agent6 deliberately started, the unsandboxed branch's
    included; an unregistered sibling would be swept, and its next status would read
    poll() over a reaped pid, which CPython reports as returncode 0.
    """
    shells = background.BackgroundShells(tmp_path / "shells")
    first = shells.start(("/bin/sh", "-c", "echo one; sleep 300"), _policy_for(tmp_path, "none"))
    second = shells.start(("/bin/sh", "-c", "echo two; sleep 300"), _policy_for(tmp_path, "none"))
    try:
        assert _wait_state(shells, second.id, "running") == "running"
        shells.stop(first.id)
        states = {v.id: v.state for v in shells.roster()}
        assert states[second.id] == "running", f"a sibling's stop took it down: {states}"
    finally:
        shells.stop_all()


def test_stopping_an_already_exited_command_still_reads_exited(tmp_path: pathlib.Path) -> None:
    """A stop over a command that already exited leaves it "exited".

    The state word says how a command ended; `stop` and `stop_all` agree, so the run's roster
    and the on-disk one agree.
    """
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("/bin/sh", "-c", "exit 7"), _policy_for(tmp_path, "none"))
    assert _wait_state(shells, view.id, "exited") == "exited"

    stopped = shells.stop(view.id)
    assert (stopped.state, stopped.returncode) == ("exited", 7), stopped
    assert [v.state for v in shells.roster()] == ["exited"]
    assert any(
        f"[{view.id}] exited 7" in line for line in background.roster_from_dir(tmp_path / "shells")
    )


def test_the_disk_roster_is_in_start_order(tmp_path: pathlib.Path) -> None:
    """The shell listing sorts by number: bg2 precedes bg10 and bg11."""
    shells = background.BackgroundShells(tmp_path / "shells")
    try:
        started = [
            shells.start(("/bin/sh", "-c", "exit 0"), _policy_for(tmp_path, "none")).id
            for _ in range(11)
        ]
        listed = [
            line.split("]")[0].lstrip("[")
            for line in background.roster_from_dir(tmp_path / "shells")
        ]
        assert listed == started, listed
    finally:
        shells.stop_all()


class _HostSession:
    """The JailSession calls a SessionJob makes, over a real host pid."""

    def open_job(self, pid: int, before: kinds.ChildSnapshot) -> None:
        pass

    def status_background(self, pid: int) -> jail.BackgroundStatus:
        return jail.BackgroundStatus(running=True, returncode=None, error="")

    def stop_background(self, pid: int) -> jail.Stopped:
        os.kill(pid, signal.SIGKILL)
        return jail.Stopped(returncode=-9, survivors=frozenset())

    def sweep_for(self, pid: int, before: kinds.ChildSnapshot) -> frozenset[int]:
        return frozenset()


def test_adopt_stops_the_command_whose_log_it_cannot_open(tmp_path: pathlib.Path) -> None:
    """A hand-back `adopt` refuses is stopped, not left running.

    The launcher already started the command and this run owns it; a registration that
    raised without stopping it would leave a live process no roster or stop could reach.
    """
    proc = subprocess.Popen(["sleep", "60"])
    try:
        shells = background.BackgroundShells(tmp_path / "shells")
        handoff = kinds.BackgroundHandoff(
            argv=("sleep", "60"),
            pid=proc.pid,
            log=str(tmp_path / "logs" / "gone.log"),
            stdout="",
            stderr="",
            duration_s=900.0,
            before=kinds.ChildSnapshot(1, frozenset()),
        )
        with pytest.raises(background.BackgroundError):
            shells.adopt(handoff, session=cast(Any, _HostSession()))
        assert shells.roster() == []
        assert proc.wait(timeout=5) == -signal.SIGKILL
    finally:
        proc.kill()
        proc.wait()
