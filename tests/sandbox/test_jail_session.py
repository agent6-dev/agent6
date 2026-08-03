# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One jail process per run: its commands share the namespaces."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.sandbox.jail import JailSession
from agent6.types import JailPolicy

pytestmark = pytest.mark.needs_namespaces


def _session(cwd: Path) -> JailSession:
    return JailSession.open(JailPolicy(cwd=cwd, argv=("true",), isolation="strict", timeout_s=30.0))


def test_the_session_netns_has_loopback_up(tmp_path: Path) -> None:
    """An empty netns leaves `lo` DOWN, so nothing inside can reach even
    itself -- which is what a shared address between a run's commands needs.
    Loopback in a namespace with no other interface reaches nothing outside
    it."""
    session = _session(tmp_path)
    try:
        got = session.run(("ip", "link", "show", "lo"))
        assert got.returncode == 0, got.stderr
        assert "UP" in got.stdout, got.stdout
    finally:
        session.close()


def test_commands_in_one_session_share_a_tmp(tmp_path: Path) -> None:
    """The private /tmp is per-launcher, so per-command launchers gave every
    command a fresh one; a run's commands must see one."""
    session = _session(tmp_path)
    try:
        first = session.run(("sh", "-c", "echo shared > /tmp/marker; echo wrote"))
        assert first.returncode == 0, first.stderr
        second = session.run(("sh", "-c", "cat /tmp/marker"))
        assert second.returncode == 0, second.stderr
        assert "shared" in second.stdout
    finally:
        session.close()


def test_a_backgrounded_server_answers_the_next_command(tmp_path: Path) -> None:
    """The point of one process per run: a server one command starts is
    reachable by the next. A per-command launcher put each in its own empty
    netns, and the escapee killpg took the server down with its command."""
    session = _session(tmp_path)
    try:
        listener = (
            "import socket;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            "s.bind(('127.0.0.1',8731));s.listen(1);"
            "c,_=s.accept();c.sendall(b'alive');c.close()"
        )
        started = session.run(("python3", "-c", listener), background=True, timeout_s=30.0)
        assert started.returncode == 0, started.stderr
        probe = session.run(
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


def test_closing_the_session_takes_the_namespace_down(tmp_path: Path) -> None:
    """Nothing a run started outlives it: closing the request channel ends the
    PID namespace, and everything inside it goes."""
    session = _session(tmp_path)
    started = session.run(("sh", "-c", "(sleep 300 &) ; echo bg"))
    assert started.returncode == 0, started.stderr
    session.close()
    with pytest.raises(Exception):
        session.run(("true",))


def test_a_run_scoped_dispatcher_serves_its_commands_from_one_process(tmp_path: Path) -> None:
    """A run's commands share one jail process: the second sees what the first
    left in the private /tmp. A bare dispatcher (no run to scope it to) keeps
    the per-command launcher, so nothing outside a run changes."""
    from agent6.config import Config
    from agent6.tools.dispatch import ToolDispatcher

    cfg = Config.model_validate({"sandbox": {"isolation": "strict", "run_commands": "yes"}})
    scoped = ToolDispatcher(root=tmp_path, config=cfg, isolation="strict", use_jail_session=True)
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

    bare = ToolDispatcher(root=tmp_path, config=cfg, isolation="strict")
    try:
        wrote = bare.dispatch(
            "run_command", {"argv": ["sh", "-c", "echo two > /tmp/marker2; echo ok"]}
        ).to_wire()
        assert wrote["returncode"] == 0, wrote
        gone = bare.dispatch("run_command", {"argv": ["cat", "/tmp/marker2"]}).to_wire()
        assert gone["returncode"] != 0, "a bare dispatcher must not share a namespace"
    finally:
        bare.close()
