# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One jail process per run: its commands share the namespaces."""

from __future__ import annotations

import contextlib
import json
import pathlib
import threading
import time

import pytest

from agent6 import kinds
from agent6.app import session_runtime
from agent6.sandbox import jail, jail_session


def _session(cwd: pathlib.Path) -> jail_session.JailSession:
    return jail_session.JailSession.open(
        kinds.JailPolicy(cwd=cwd, argv=("true",), isolation="full", timeout_s=30.0)
    )


def _run(
    session: jail_session.JailSession, argv: tuple[str, ...], **kw: object
) -> kinds.CommandResult:
    """A run with no check-in always completes, so it is a CommandResult.

    The hand-off shape is exercised in test_jail_session_levels.py.
    """
    res = session.run(argv, **kw)  # pyright: ignore[reportArgumentType]
    assert isinstance(res, kinds.CommandResult), f"unexpected hand-off: {res}"
    return res


@pytest.mark.needs_namespaces
def test_the_session_netns_has_loopback_up(tmp_path: pathlib.Path) -> None:
    """An empty netns leaves `lo` down, so nothing inside reaches even itself.

    A shared address between a run's commands needs loopback up; in a namespace with no
    other interface it reaches nothing outside it.
    """
    session = _session(tmp_path)
    try:
        got = _run(session, ("ip", "link", "show", "lo"))
        assert got.returncode == 0, got.stderr
        assert "UP" in got.stdout, got.stdout
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_commands_in_one_session_share_a_tmp(tmp_path: pathlib.Path) -> None:
    """A run's commands see one private /tmp.

    The private /tmp is per launcher, so per-command launchers would give each command a
    fresh one.
    """
    session = _session(tmp_path)
    try:
        first = _run(session, ("sh", "-c", "echo shared > /tmp/marker; echo wrote"))
        assert first.returncode == 0, first.stderr
        second = _run(session, ("sh", "-c", "cat /tmp/marker"))
        assert second.returncode == 0, second.stderr
        assert "shared" in second.stdout
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_backgrounded_server_answers_the_next_command(tmp_path: pathlib.Path) -> None:
    """A server one command starts is reachable by the next.

    The point of one process per run: a per-command launcher puts each command in its own
    empty netns, and the escapee killpg takes the server down with its command.
    """
    session = _session(tmp_path)
    try:
        listener = (
            "import socket;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            "s.bind(('127.0.0.1',8731));s.listen(1);"
            "c,_=s.accept();c.sendall(b'alive');c.close()"
        )
        assert session.start_background(("python3", "-c", listener))
        probe = _run(
            session,
            (
                "python3",
                "-c",
                "import socket,time\n"
                "for _ in range(50):\n"
                "    try:\n"
                "        s=socket.create_connection(('127.0.0.1',8731),timeout=5)\n"
                "        print(s.recv(16).decode());break\n"
                "    except OSError:\n"
                "        time.sleep(0.1)\n",
            ),
            timeout_s=30.0,
        )
        assert probe.returncode == 0, probe.stderr + probe.stdout
        assert "alive" in probe.stdout
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_jailed_command_cannot_write_the_launchers_answer_pipe(tmp_path: pathlib.Path) -> None:
    """A command cannot open `/proc/1/fd/1` and write its own result.

    The serving launcher is PID 1 of the jail's PID namespace and answers every request on
    its stdout; a command that could write there would hand the agent model-authored JSON as
    its exit code, with every later answer one request behind (a verify gate handed the
    result of a command the model chose reports green on a broken tree, and the run
    auto-merges it). seccomp denying ptrace(2) does not cover this: reaching another
    process's /proc/<pid>/fd is a permission check (ptrace_may_access), not that syscall,
    and Landlock exempts a pipe reopened through /proc/<pid>/fd.
    """
    session = _session(tmp_path)
    try:
        answer = r'{"returncode":0,"stdout":"FORGED","stderr":""}'
        forge = f"printf '{answer}\\n' > /proc/1/fd/1; echo wrote"
        forged = _run(session, ("sh", "-c", forge))
        assert "FORGED" not in forged.stdout, "a command wrote the answer the agent read"
        after = _run(session, ("sh", "-c", "echo REAL; exit 3"))
        assert after.returncode == 3, f"the channel is desynced: {after}"
        assert "REAL" in after.stdout, f"the channel is desynced: {after}"
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_daemonizing_command_does_not_wedge_the_run(tmp_path: pathlib.Path) -> None:
    """A `setsid` grandchild holding the capture pipe does not hang the run.

    It leaves the command's process group, so the teardown killpg misses it and the reader
    threads would join forever: the launcher never answers, the request read above has
    nothing to time out, and one tool call stalls the whole run with no diagnostic. Any
    command that daemonizes by double-fork + setsid does this. Bounded here rather than by
    the suite: an unfixed hang fails this test, not every test after it.
    """
    session = _session(tmp_path)
    answered: list[kinds.CommandResult] = []

    def run_it() -> None:
        answered.append(
            _run(session, ("sh", "-c", "setsid sleep 300 & echo started"), timeout_s=5.0)
        )

    worker = threading.Thread(target=run_it, daemon=True)
    worker.start()
    worker.join(timeout=60.0)
    try:
        assert not worker.is_alive(), "the launcher never answered: the run is wedged"
        assert answered and answered[0].returncode == 0, answered
        assert "started" in answered[0].stdout, answered[0]
        after = _run(session, ("echo", "next"))
        assert after.returncode == 0 and "next" in after.stdout, after
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_command_that_cannot_be_executed_does_not_end_the_session(tmp_path: pathlib.Path) -> None:
    """A missing binary answers 127 from the session, like the per-command launcher.

    It is the model's typo, not a broken sandbox; one bad argv must not take the run's jail
    process down with every backgrounded server inside it.
    """
    session = _session(tmp_path)
    try:
        bad = _run(session, ("definitely-not-a-real-binary", "-q"))
        assert bad.returncode == 127, bad
        assert bad.exec_failed is True, bad
        assert "not found" in bad.stderr, bad.stderr
        after = _run(session, ("echo", "alive"))
        assert after.returncode == 0, after.stderr
        assert "alive" in after.stdout, "the session died with the bad command"
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_dead_session_refuses_with_its_own_error(tmp_path: pathlib.Path) -> None:
    """A dead launcher pipe surfaces as JailUnavailableError, and the whole group is killed.

    Every caller is written against JailUnavailableError; a raw OSError would escape the
    dispatcher's jail handler, SessionJob's status and ToolDispatcher.close() before teardown
    stopped the shells. Under `full` the request-serving process is a namespaced child the
    launcher forked, and it keeps the pipes if only its parent dies; normal teardown kills
    neither, since `close()` shuts stdin and the serve loop exits on EOF.
    """
    import os
    import signal

    session = _session(tmp_path)
    proc = session._proc  # pyright: ignore[reportPrivateUsage]
    try:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10.0)
        with pytest.raises(jail.JailUnavailableError):
            for _ in range(3):  # the first write can still buffer
                _run(session, ("echo", "hi"))
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_session_command_gets_the_configured_memory_cap(tmp_path: pathlib.Path) -> None:
    """Every request carries the run policy's memory cap.

    Sending only argv would leave every command on the launcher's own default, silently
    ignoring `[sandbox] memory_limit_mib`.
    """
    session = jail_session.JailSession.open(
        kinds.JailPolicy(
            cwd=tmp_path, argv=("true",), isolation="full", timeout_s=30.0, memory_limit_mib=256
        )
    )
    try:
        got = _run(session, ("python3", "-c", "bytearray(600 * 1024 * 1024)"))
        assert got.returncode != 0, got
        assert "MemoryError" in got.stderr, got.stderr
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_backgrounded_command_gets_the_same_memory_cap(tmp_path: pathlib.Path) -> None:
    """A backgrounded command honours `[sandbox] memory_limit_mib` like a foreground one.

    The detached spawn shares the capture transport's child setup, so the cap does not
    depend on the transport.
    """
    session = jail_session.JailSession.open(
        kinds.JailPolicy(
            cwd=tmp_path, argv=("true",), isolation="full", timeout_s=30.0, memory_limit_mib=256
        )
    )
    try:
        pid = session.start_background(
            (
                "python3",
                "-c",
                "import pathlib; bytearray(600 * 1024 * 1024);"
                " pathlib.Path('allocated').write_text('x')",
            )
        )
        deadline = time.monotonic() + 10.0
        status = session.status_background(pid)
        while status.running and time.monotonic() < deadline:
            time.sleep(0.05)
            status = session.status_background(pid)
        assert status.running is False, "allocator still alive past the cap"
        assert status.returncode != 0, status
        assert not (tmp_path / "allocated").exists()
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_backgrounded_command_stops_through_the_session(tmp_path: pathlib.Path) -> None:
    """Stop forwards the namespace-local pid to the launcher, which kills and reaps it.

    Only the launcher can signal it; unreaped, the pid stays a zombie and every liveness
    check still answers "running".
    """
    session = _session(tmp_path)
    try:
        marker = tmp_path / "command.pid"
        handle = session.start_background(("sh", "-c", f"echo $$ > {marker}; exec sleep 300"))
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pid = int(marker.read_text())
        alive = _run(session, ("sh", "-c", f"kill -0 {pid} && echo alive"))
        assert "alive" in alive.stdout, alive.stderr
        session.stop_background(handle)
        gone = _run(session, ("sh", "-c", f"kill -0 {pid} 2>/dev/null && echo alive || echo gone"))
        assert "gone" in gone.stdout, gone.stdout
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_the_session_reports_a_backgrounded_command_s_exit(tmp_path: pathlib.Path) -> None:
    """The exit code comes back over the launcher's channel; only it can wait on the command."""
    session = _session(tmp_path)
    try:
        pid = session.start_background(("sh", "-c", "exit 7"))
        status = session.status_background(pid)
        deadline = time.monotonic() + 5.0
        while status.running and time.monotonic() < deadline:
            time.sleep(0.05)
            status = session.status_background(pid)
        assert status.running is False, "the command never showed as exited"
        assert status.returncode == 7, status
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_closing_the_session_takes_the_namespace_down(tmp_path: pathlib.Path) -> None:
    """Closing the request channel ends the PID namespace and everything inside it."""
    session = _session(tmp_path)
    started = _run(session, ("sh", "-c", "(sleep 300 &) ; echo bg"))
    assert started.returncode == 0, started.stderr
    session.close()
    # The closed session refuses with its OWN error (a bare Exception would
    # swallow the raw-OSError failure mode the sibling test guards against).
    with pytest.raises(jail.JailUnavailableError):
        _run(session, ("true",))


@pytest.mark.needs_namespaces
def test_a_run_scoped_dispatcher_serves_its_commands_from_one_process(
    tmp_path: pathlib.Path,
) -> None:
    """A run's commands share one jail process; a bare dispatcher keeps the per-command launcher.

    The second command sees what the first left in the private /tmp; nothing outside a run
    changes.
    """
    from agent6.config import Config
    from agent6.tools import dispatch

    cfg = Config.model_validate({"sandbox": {"isolation": "full", "run_commands": "allow"}})
    scoped = dispatch.ToolDispatcher(
        root=tmp_path,
        config=cfg,
        isolation="full",
        command_access=session_runtime.prepare_local_execution(cfg, tmp_path, "full"),
    )
    try:
        first = scoped.dispatch(
            "run_command", {"argv": ["sh", "-c", "echo one > /tmp/marker; echo ok"]}
        ).to_wire()
        assert first["returncode"] == 0, first
        second = scoped.dispatch("run_command", {"argv": ["cat", "/tmp/marker"]}).to_wire()
        assert second["returncode"] == 0, second
        assert "one" in str(second["stdout"])
    finally:
        scoped.close()

    bare = dispatch.ToolDispatcher(root=tmp_path, config=cfg, isolation="full")
    try:
        wrote = bare.dispatch(
            "run_command", {"argv": ["sh", "-c", "echo two > /tmp/marker2; echo ok"]}
        ).to_wire()
        assert wrote["returncode"] == 0, wrote
        gone = bare.dispatch("run_command", {"argv": ["cat", "/tmp/marker2"]}).to_wire()
        assert gone["returncode"] != 0, "a bare dispatcher must not share a namespace"
    finally:
        bare.close()


@pytest.mark.needs_namespaces
def test_a_backgrounded_server_is_reachable_by_the_run_s_next_command(
    tmp_path: pathlib.Path,
) -> None:
    """A background dev server is reachable by a later `run_command` on loopback.

    What a run's own jail process is for: per-command launchers put each in its own empty
    netns, so the address would be unreachable however long the server ran.
    """
    from agent6.config import Config
    from agent6.tools import dispatch

    cfg = Config.model_validate({"sandbox": {"isolation": "full", "run_commands": "allow"}})
    d = dispatch.ToolDispatcher(
        root=tmp_path,
        config=cfg,
        isolation="full",
        command_access=session_runtime.prepare_local_execution(cfg, tmp_path, "full"),
        session_dir=tmp_path / "session",
        state_dir=tmp_path / "state",
    )
    try:
        listener = (
            "import socket;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            "s.bind(('127.0.0.1',8741));s.listen(1);"
            "c,_=s.accept();c.sendall(b'alive');c.close()"
        )
        started = d.dispatch(
            "run_command", {"argv": ["python3", "-c", listener], "background": True}
        ).to_wire()
        assert "running" in str(started), started
        probe = d.dispatch(
            "run_command",
            {
                "argv": [
                    "python3",
                    "-c",
                    "import socket,time\n"
                    "for _ in range(50):\n"
                    "    try:\n"
                    "        s=socket.create_connection(('127.0.0.1',8741),timeout=5)\n"
                    "        print(s.recv(16).decode());break\n"
                    "    except OSError:\n"
                    "        time.sleep(0.1)\n",
                ]
            },
        ).to_wire()
        assert probe["returncode"] == 0, probe
        assert "alive" in str(probe["stdout"]), probe
    finally:
        d.close()


@pytest.mark.needs_namespaces
def test_a_hung_command_times_out_without_ending_the_session(tmp_path: pathlib.Path) -> None:
    """One command's timeout does not cost the run its jail process.

    The launcher bounds each request itself, killing that command's group and answering
    124, so the next command runs in the same namespaces.
    """
    session = _session(tmp_path)
    try:
        _run(session, ("sh", "-c", "echo before > /tmp/timeout-marker"))
        hung = _run(session, ("sleep", "30"), timeout_s=1.0)
        assert hung.returncode == 124, hung
        after = _run(session, ("cat", "/tmp/timeout-marker"))
        assert after.returncode == 0, after.stderr
        assert "before" in after.stdout, "the session lost its namespaces"
    finally:
        session.close()


@pytest.mark.needs_namespaces
def test_a_clean_session_reports_no_startup_warning(tmp_path: pathlib.Path) -> None:
    """A clean jail leaves the setup-stderr field empty.

    open() reads the launcher's setup stderr at the ready handshake; only a degraded setup
    (a refused /proc mount under rootless podman, a skipped grant) fills it, and the
    dispatcher surfaces it once.
    """
    session = _session(tmp_path)
    try:
        assert _run(session, ("/bin/echo", "ok")).stdout.strip() == "ok"
        assert session.startup_stderr == "", session.startup_stderr
    finally:
        session.close()


def _peer(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """An executable protocol peer exercises the real pipe and process cleanup paths."""
    peer = tmp_path / "peer"
    peer.write_text(
        "#!/usr/bin/python3\nimport json, signal, sys\nsignal.alarm(5)\n"
        "sys.stdin.readline()\n" + body,
        encoding="utf-8",
    )
    peer.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(peer))


@pytest.mark.parametrize(
    "ready",
    [
        "not json",
        "[]",
        '{"ready":true}',
        '{"ready":false,"protocol":3}',
        '{"ready":true,"protocol":1}',
        '{"ready":true,"protocol":true}',
        '{"ready":true,"protocol":3,"extra":0}',
        pytest.param("x" * 4097, id="oversized"),
    ],
)
def test_session_rejects_malformed_or_incompatible_readiness(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, ready: str
) -> None:
    """A nonempty line is not proof of a compatible supervisor."""
    _peer(tmp_path, monkeypatch, f"print({ready!r}, flush=True)\nsys.stdin.read()\n")
    with pytest.raises(jail.JailUnavailableError, match=r"handshake|protocol|frame"):
        session = jail_session.JailSession.open(
            kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
        )
        session.close()


@pytest.mark.parametrize(
    "answer",
    [
        {"id": 9, "returncode": 0},
        {"returncode": 0},
        {"id": True, "returncode": 0},
    ],
)
def test_session_rejects_an_uncorrelated_answer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, answer: dict[str, object]
) -> None:
    """A stale response cannot serve the next request."""
    _peer(
        tmp_path,
        monkeypatch,
        'print(\'{"ready":true,"protocol":3}\', flush=True)\n'
        f"sys.stdin.readline()\nprint({json.dumps(answer)!r}, flush=True)\n",
    )
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    try:
        with pytest.raises(jail.JailUnavailableError, match="request ID"):
            session.run(("true",))
    finally:
        session.close()


@pytest.mark.parametrize("phase", ["setup", "runtime"])
def test_session_drains_runtime_stderr_independently(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """A transport's stderr cannot fill its pipe while Python waits for protocol stdout."""
    flood = "sys.stderr.write('x' * 1048576 + 'diagnostic tail\\n'); sys.stderr.flush()\n"
    ready = 'print(\'{"ready":true,"protocol":3}\', flush=True)\n'
    request = "request = json.loads(sys.stdin.readline())\n"
    _peer(
        tmp_path,
        monkeypatch,
        (flood + ready + request if phase == "setup" else ready + request + flood)
        + "print(json.dumps({'id': request.get('id'), 'returncode': 0, "
        "'stdout': 'answered', 'stderr': ''}), flush=True)\n",
    )
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    try:
        got = _run(session, ("true",))
        assert got.stdout == "answered"
        if phase == "setup":
            assert "diagnostic tail" in session.startup_stderr
            assert len(session.startup_stderr) < 70000
    finally:
        session.close()


def test_session_bounds_outgoing_requests(tmp_path: pathlib.Path) -> None:
    """An oversized request is refused locally without desynchronising the session."""
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    try:
        with pytest.raises(jail.JailUnavailableError, match=r"frame.*exceeds"):
            session.run(("echo", "x" * 1048576))
        assert _run(session, ("echo", "next")).stdout == "next\n"
    finally:
        session.close()


def test_session_bounds_response_frames(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The protocol reader refuses a response beyond the escaped capture budget."""
    _peer(
        tmp_path,
        monkeypatch,
        'print(\'{"ready":true,"protocol":3}\', flush=True)\n'
        "sys.stdin.readline()\n"
        'sys.stdout.write(\'{"id":1,"stdout":"\' + \'x\' * (128 * 1024 * 1024) '
        '+ \'","returncode":0}\\n\'); sys.stdout.flush()\n',
    )
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    try:
        with pytest.raises(jail.JailUnavailableError, match=r"frame.*exceeds"):
            session.run(("true",))
    finally:
        session.close()
