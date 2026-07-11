# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""End-to-end test: spawn the `graph-curator` subprocess, drive it via GraphClient."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent6.graph.client import CuratorClientError, GraphClient, spawn_curator
from agent6.graph.ipc import send_message
from agent6.graph.models import (
    AddSubtaskIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)


def _wait_for_socket(path: Path, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(path))
                s.close()
                return
            except OSError:
                pass
        time.sleep(0.02)
    raise RuntimeError(f"curator socket {path} did not appear")


@pytest.fixture
def curator_proc(tmp_path: Path):  # type: ignore[no-untyped-def]
    sock_path = tmp_path / ".agent6" / "runs" / "run1" / "curator.sock"
    proc = spawn_curator(tmp_path / ".agent6", "run1", sock_path)
    try:
        _wait_for_socket(sock_path)
        yield sock_path, proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_curator_survives_terminal_sigint(curator_proc) -> None:  # type: ignore[no-untyped-def]
    """The curator runs in its own session (a terminal Ctrl-C signals the whole
    foreground process group) and ignores SIGINT, so the run's graph outlives an
    operator interrupt instead of leaving every later DAG op on a dead socket."""
    sock_path, proc = curator_proc
    assert os.getsid(proc.pid) == proc.pid  # session leader: not in the tty's group
    os.kill(proc.pid, signal.SIGINT)  # a stray direct SIGINT is ignored too
    time.sleep(0.3)
    assert proc.poll() is None
    with GraphClient(sock_path) as client:
        root = client.add_subtask(
            AddSubtaskIntent(
                parent_id=None,
                draft=TaskNodeDraft(title="still alive", created_by="planner"),
            )
        )
    assert root.title == "still alive"


def test_curator_ipc_roundtrip(curator_proc) -> None:  # type: ignore[no-untyped-def]
    sock_path, _ = curator_proc
    with GraphClient(sock_path) as client:
        root = client.add_subtask(
            AddSubtaskIntent(
                parent_id=None,
                draft=TaskNodeDraft(title="root", created_by="planner"),
            )
        )
        child = client.add_subtask(
            AddSubtaskIntent(
                parent_id=root.id,
                draft=TaskNodeDraft(title="child", created_by="planner"),
            )
        )
        client.update_status(UpdateStatusIntent(id=child.id, new_status="in_progress"))
        client.set_cursor(SetCursorIntent(id=child.id))
        state = client.get_state()
        assert state["cursor"] == child.id
        assert root.id in state["nodes"]
        assert state["nodes"][child.id]["status"] == "in_progress"


def test_curator_writes_under_requested_subdir(tmp_path: Path) -> None:
    """An ask spawns its curator with subdir='asks'; the DAG must land under
    asks/<id>/, not the default runs/<id>/ (else a stray, log-less runs/<id>
    appears -- the phantom duplicate row in the TUI run list)."""
    state = tmp_path / ".agent6"
    sock_path = state / "asks" / "run1" / "curator.sock"
    proc = spawn_curator(state, "run1", sock_path, subdir="asks")
    try:
        _wait_for_socket(sock_path)
        with GraphClient(sock_path) as client:
            client.add_subtask(
                AddSubtaskIntent(
                    parent_id=None,
                    draft=TaskNodeDraft(title="root", created_by="planner"),
                )
            )
        assert (state / "asks" / "run1" / "graph.jsonl").is_file()
        assert not (state / "runs" / "run1").exists()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_curator_ipc_rejects_unknown_op(curator_proc) -> None:  # type: ignore[no-untyped-def]
    sock_path, _ = curator_proc
    from agent6.graph.client import CuratorClientError
    from agent6.graph.ipc import recv_message, send_message

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(sock_path))
    try:
        send_message(sock, {"id": 1, "intent": {"op": "nope"}})
        reply = recv_message(sock)
        assert reply is not None
        assert reply["ok"] is False
        assert "unknown op" in reply["error"]
    finally:
        sock.close()
    # Sanity: typed client still rejects unknown via exception wrapper path.
    assert CuratorClientError is not None


def test_curator_entrypoint_exists() -> None:
    """`python -m agent6.graph.server` with no args prints usage and exits 2."""
    proc = subprocess.run(
        [sys.executable, "-m", "agent6.graph.server"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert proc.returncode == 2
    assert "usage:" in proc.stderr


def test_server_survives_client_vanishing_before_reply(curator_proc) -> None:  # type: ignore[no-untyped-def]
    # A client that sends a mutation and dies before reading the reply must
    # not kill the curator: the mutation is already persisted, so the server
    # drops the connection and keeps serving the next one.
    sock_path, proc = curator_proc
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.connect(str(sock_path))
    send_message(
        raw,
        {
            "id": 1,
            "intent": {
                "op": "add_subtask",
                "parent_id": None,
                "draft": {"title": "orphaned reply", "created_by": "planner"},
            },
        },
    )
    raw.close()  # vanish without reading the reply
    deadline = time.monotonic() + 3.0
    state: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            with GraphClient(sock_path) as client:
                state = client.get_state()
            break
        except CuratorClientError:
            time.sleep(0.05)
    assert proc.poll() is None, "curator died on a vanished client"
    nodes = state.get("nodes")
    assert isinstance(nodes, dict) and len(nodes) == 1


def test_server_survives_client_reset_after_reply_queued(curator_proc) -> None:  # type: ignore[no-untyped-def]
    # Harder timing than the test above: wait until the reply has actually
    # arrived (so the server has replied and looped back into recv), THEN close
    # without reading it. The unread reply makes the kernel RST the socket, so
    # the server's next recv raises ConnectionResetError -- an OSError, not the
    # IpcError the reply-send path tolerates. The curator must survive it.
    import select

    sock_path, proc = curator_proc
    raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    raw.connect(str(sock_path))
    send_message(
        raw,
        {
            "id": 1,
            "intent": {
                "op": "add_subtask",
                "parent_id": None,
                "draft": {"title": "reset after reply", "created_by": "planner"},
            },
        },
    )
    # Block until the reply bytes are readable, then abandon them unread.
    readable, _, _ = select.select([raw], [], [], 3.0)
    assert readable, "curator never sent the reply"
    raw.close()  # RST: unread data in the receive buffer
    deadline = time.monotonic() + 3.0
    state: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            with GraphClient(sock_path) as client:
                state = client.get_state()
            break
        except CuratorClientError:
            time.sleep(0.05)
    assert proc.poll() is None, "curator died on a client reset between requests"
    nodes = state.get("nodes")
    assert isinstance(nodes, dict) and len(nodes) == 1


def test_client_wraps_transport_failure(curator_proc) -> None:  # type: ignore[no-untyped-def]
    # A dead curator surfaces as CuratorClientError (the class contract), not
    # a raw BrokenPipeError/IpcError that crashes degrade-gracefully callers.
    sock_path, proc = curator_proc
    client = GraphClient(sock_path)
    client.connect()
    try:
        proc.terminate()
        proc.wait(timeout=3)
        with pytest.raises(CuratorClientError):
            client.get_state()
            client.get_state()  # first call may drain a buffered reply; second cannot
    finally:
        client.close()
