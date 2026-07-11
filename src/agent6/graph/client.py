# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Blocking client for the curator UDS server.

Used by the workflow process. Each call sends one request and blocks for one
response; serialization order matches the curator's single-threaded model.
"""

from __future__ import annotations

import contextlib
import itertools
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

from agent6.graph.ipc import IpcError, recv_message, send_message
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    ObsoleteIntent,
    RecordCommitIntent,
    ReorderChildrenIntent,
    SetCursorIntent,
    TaskNode,
    UpdateStatusIntent,
)

# Per-call send/recv deadline: a wedged curator must surface as an error, not a
# hang. Generous -- a real round-trip is milliseconds.
_CALL_TIMEOUT_S = 30.0


class CuratorClientError(Exception):
    """The curator rejected an intent or the connection failed."""


# Hard cap on connect patience while the curator process is verifiably alive.
# A loaded host can take well past the base timeout just to exec the
# interpreter (observed: 8 concurrent bench runs starved startup beyond 5s);
# only a genuinely wedged-but-alive curator hits this ceiling.
_STARTUP_CEILING_S = 60.0


class GraphClient:
    """Synchronous client; one instance per workflow."""

    def __init__(self, sock_path: Path, *, alive: Callable[[], bool] | None = None) -> None:
        self._sock_path = sock_path
        self._sock: socket.socket | None = None
        self._ids = itertools.count(1)
        # Liveness probe for the curator process (the spawner's Popen.poll).
        # None = liveness unknowable; connect() then keeps the base timeout.
        self._alive = alive

    # ---- lifecycle --------------------------------------------------------

    def connect(self, *, timeout_s: float = 5.0) -> None:
        """Wait for the curator socket. ``timeout_s`` bounds the wait when the
        curator's liveness is unknowable; with an ``alive`` probe, a process
        that is still running earns patience up to ``_STARTUP_CEILING_S``
        (condition-based, not a guess), while one that has exited fails
        immediately instead of burning the whole deadline."""
        start = time.monotonic()
        last_exc: OSError | None = None
        while True:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(str(self._sock_path))
                # Bound every later send/recv: a wedged curator (a stuck fsync,
                # flock contention) would otherwise block a DAG op forever. A
                # timeout raises socket.timeout (an OSError), which _call wraps as
                # CuratorClientError -- the already-handled "curator unavailable"
                # path, so the run degrades instead of hanging. 30s is far above a
                # local-socket + local-file round-trip; only a genuine wedge hits it.
                sock.settimeout(_CALL_TIMEOUT_S)
                self._sock = sock
                return
            except OSError as exc:
                last_exc = exc
            waited = time.monotonic() - start
            if self._alive is not None:
                if not self._alive():
                    raise CuratorClientError(
                        f"curator exited during startup (socket {self._sock_path}): {last_exc}"
                    )
                if waited >= _STARTUP_CEILING_S:
                    break
            elif waited >= timeout_s:
                break
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
        # Wrap transport faults so callers only ever see CuratorClientError,
        # as the class contract promises: a dead curator degrades the DAG
        # tools, it must not crash loop paths that catch CuratorClientError.
        try:
            send_message(self._sock, {"id": req_id, "intent": intent})
            reply = recv_message(self._sock)
        except (OSError, IpcError) as exc:
            # A timeout/fault can leave the socket mid-frame; every later _call
            # would then fail the reply-id check forever. Drop the socket so
            # subsequent calls fail cleanly ("client not connected") instead of
            # desyncing on a half-consumed frame.
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
            raise CuratorClientError(f"curator connection failed: {exc}") from exc
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

    def set_cursor(self, intent: SetCursorIntent) -> None:
        self._call(intent.model_dump(mode="json"))

    def get_state(self) -> dict[str, Any]:
        result = self._call({"op": "get_state"})
        if not isinstance(result, dict):
            raise CuratorClientError(f"get_state: unexpected reply {result!r}")
        return result


# ---- subprocess spawn helper ---------------------------------------------


def spawn_curator(
    state_dir: Path,
    run_id: str,
    sock_path: Path,
    *,
    subdir: str = "runs",
) -> subprocess.Popen[bytes]:
    """Launch the `graph-curator` subprocess for one run and return the Popen.

    ``state_dir`` is the resolved run-state base (see
    ``agent6.paths.state_dir``); the curator writes the run's graph under
    ``<state_dir>/<subdir>/<run_id>``. ``subdir`` MUST match the caller's
    ``RunLayout.subdir`` ("runs" for run/plan, "asks" for ask) or the curator
    writes the DAG to a different directory than the rest of the run state.
    The caller connects (via `GraphClient`) and terminates the process on
    shutdown.
    """
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent6.graph.server",
            str(state_dir),
            run_id,
            str(sock_path),
            subdir,
        ],
        stdin=subprocess.DEVNULL,
        # Its own session: a terminal Ctrl-C signals the whole foreground
        # process group, and a curator that died with it left every later DAG
        # op failing on a dead socket for the rest of the run. Lifecycle stays
        # with the caller (terminate + wait) and the server's orphan watchdog.
        start_new_session=True,
    )
