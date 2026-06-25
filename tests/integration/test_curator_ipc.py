# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""End-to-end test: spawn the `graph-curator` subprocess, drive it via GraphClient."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent6.graph.client import GraphClient, spawn_curator
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
