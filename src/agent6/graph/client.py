# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Blocking client for the curator UDS server.

Used by the workflow process. Each call sends one request and blocks for one
response; serialization order matches the curator's single-threaded model.
"""

from __future__ import annotations

import itertools
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from agent6.graph.ipc import recv_message, send_message
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    NodeSnapshot,
    ObsoleteIntent,
    RecordCommitIntent,
    ReorderChildrenIntent,
    ResumeDiff,
    SetCursorIntent,
    SnapshotNodeIntent,
    TaskNode,
    UpdateStatusIntent,
)


class CuratorClientError(Exception):
    """The curator rejected an intent or the connection failed."""


class GraphClient:
    """Synchronous client; one instance per workflow."""

    def __init__(self, sock_path: Path) -> None:
        self._sock_path = sock_path
        self._sock: socket.socket | None = None
        self._ids = itertools.count(1)

    # ---- lifecycle --------------------------------------------------------

    def connect(self, *, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_exc: OSError | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self._sock_path))
                self._sock = sock
                return
            except OSError as exc:
                last_exc = exc
                time.sleep(0.02)
        raise CuratorClientError(f"could not connect to curator at {self._sock_path}: {last_exc}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> GraphClient:
        if self._sock is None:
            self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- transport --------------------------------------------------------

    def _call(self, intent: dict[str, Any]) -> Any:
        if self._sock is None:
            raise CuratorClientError("client not connected")
        req_id = next(self._ids)
        send_message(self._sock, {"id": req_id, "intent": intent})
        reply = recv_message(self._sock)
        if reply is None:
            raise CuratorClientError("curator closed the connection")
        if reply.get("id") != req_id:
            raise CuratorClientError(f"reply id mismatch: expected {req_id}, got {reply.get('id')}")
        if not reply.get("ok"):
            raise CuratorClientError(str(reply.get("error", "unknown error")))
        return reply.get("result")

    # ---- typed wrappers --------------------------------------------------

    def add_subtask(self, intent: AddSubtaskIntent) -> TaskNode:
        return TaskNode.model_validate(self._call(intent.model_dump(mode="json")))

    def update_status(self, intent: UpdateStatusIntent) -> TaskNode:
        return TaskNode.model_validate(self._call(intent.model_dump(mode="json")))

    def add_dependency(self, intent: AddDependencyIntent) -> TaskNode:
        return TaskNode.model_validate(self._call(intent.model_dump(mode="json")))

    def obsolete(self, intent: ObsoleteIntent) -> TaskNode:
        return TaskNode.model_validate(self._call(intent.model_dump(mode="json")))

    def reorder_children(self, intent: ReorderChildrenIntent) -> TaskNode:
        return TaskNode.model_validate(self._call(intent.model_dump(mode="json")))

    def record_commit(self, intent: RecordCommitIntent) -> TaskNode:
        return TaskNode.model_validate(self._call(intent.model_dump(mode="json")))

    def snapshot_node(self, intent: SnapshotNodeIntent) -> NodeSnapshot:
        return NodeSnapshot.model_validate(self._call(intent.model_dump(mode="json")))

    def set_cursor(self, intent: SetCursorIntent) -> None:
        self._call(intent.model_dump(mode="json"))

    def compute_resume_diff(self, run_id: str, repo_root: Path) -> ResumeDiff:
        result = self._call(
            {
                "op": "compute_resume_diff",
                "run_id": run_id,
                "repo_root": str(repo_root),
            }
        )
        return ResumeDiff.model_validate(result)

    def get_state(self) -> dict[str, Any]:
        result = self._call({"op": "get_state"})
        if not isinstance(result, dict):
            raise CuratorClientError(f"get_state: unexpected reply {result!r}")
        return result


# ---- subprocess spawn helper ---------------------------------------------


def spawn_curator(
    repo_root: Path,
    run_id: str,
    sock_path: Path,
) -> subprocess.Popen[bytes]:
    """Launch `agent6-curator` for one run and return the Popen.

    The caller is responsible for connecting (via `GraphClient`) and for
    terminating the process on shutdown.
    """
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent6.graph.server",
            str(repo_root),
            run_id,
            str(sock_path),
        ],
        stdin=subprocess.DEVNULL,
    )
