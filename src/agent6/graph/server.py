# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UDS server that hosts a single `GraphCurator` for one run id.

This is the `graph-curator` subprocess entrypoint, spawned as
`python -m agent6.graph.server <state-dir> <run-id> <sock-path>` by
`agent6.graph.client.spawn_curator`. It is *not* meant to be used directly from
agent6 application code, application code goes through
`agent6.graph.client.GraphClient`, which speaks the same wire protocol.

The server is single-threaded by design: every mutation already takes an
fcntl flock on the run directory, and the protocol overhead is negligible
compared to LLM round-trips. Concurrency would only add hazard surface.
"""

from __future__ import annotations

import contextlib
import socket
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.ipc import IpcError, recv_message, send_message
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    ObsoleteIntent,
    RecordCommitIntent,
    ReorderChildrenIntent,
    SetCursorIntent,
    SnapshotNodeIntent,
    UpdateStatusIntent,
)
from agent6.graph.storage import RunLayout

_INTENT_TABLE = {
    "add_subtask": AddSubtaskIntent,
    "update_status": UpdateStatusIntent,
    "add_dependency": AddDependencyIntent,
    "obsolete": ObsoleteIntent,
    "reorder_children": ReorderChildrenIntent,
    "record_commit": RecordCommitIntent,
    "snapshot_node": SnapshotNodeIntent,
    "set_cursor": SetCursorIntent,
}

# Ops that never mutate curator state (pure reads). Everything else -- the whole
# _INTENT_TABLE, plus an unknown/missing op -- is treated as mutating for the
# fail-safe in _serve_connection. A mutating handler updates self._nodes IN
# MEMORY *before* it calls write_node (see curator.py), so a non-validation
# fault on the write path can leave in-memory state ahead of disk; those must
# fail-safe (die -> respawn reloads consistent on-disk state), not stay alive.
_READ_ONLY_OPS = frozenset({"get_state", "compute_resume_diff"})


def _handle_one(curator: GraphCurator, intent_dict: dict[str, Any]) -> Any:
    op = intent_dict.get("op")
    if not isinstance(op, str):
        raise CuratorError(f"missing 'op' in intent: {intent_dict!r}")
    if op == "get_state":
        return {
            "nodes": {nid: n.model_dump(mode="json") for nid, n in curator.nodes().items()},
            "cursor": curator.cursor(),
            "graph_version": curator.graph_version,
        }
    if op == "compute_resume_diff":
        repo_root = intent_dict.get("repo_root")
        run_id = intent_dict.get("run_id")
        if not isinstance(repo_root, str) or not isinstance(run_id, str):
            raise CuratorError("compute_resume_diff requires str repo_root and run_id")
        diff = curator.compute_resume_diff(run_id, Path(repo_root))
        return diff.model_dump(mode="json")
    intent_cls = _INTENT_TABLE.get(op)
    if intent_cls is None:
        raise CuratorError(f"unknown op {op!r}")
    intent = intent_cls.model_validate(intent_dict)
    method_name = op
    method = getattr(curator, method_name)
    result = method(intent)
    if result is None:
        return None
    return result.model_dump(mode="json")


def serve(layout: RunLayout, sock_path: Path) -> None:
    """Run the request loop until the client disconnects."""
    if sock_path.exists():
        sock_path.unlink()
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    with contextlib.suppress(OSError):
        sock_path.chmod(0o600)
    srv.listen(1)
    try:
        curator = GraphCurator(layout)
        while True:
            conn, _ = srv.accept()
            with conn:
                _serve_connection(curator, conn)
    finally:
        srv.close()
        if sock_path.exists():
            sock_path.unlink()


def _serve_connection(curator: GraphCurator, conn: socket.socket) -> None:
    while True:
        try:
            msg = recv_message(conn)
        except IpcError as exc:
            send_message(conn, {"id": 0, "ok": False, "error": f"ipc: {exc}"})
            return
        if msg is None:
            return
        req_id = msg.get("id")
        intent_dict = msg.get("intent")
        if not isinstance(req_id, int) or not isinstance(intent_dict, dict):
            send_message(
                conn,
                {"id": 0, "ok": False, "error": "envelope must be {id, intent}"},
            )
            continue
        try:
            result = _handle_one(curator, intent_dict)
        except (CuratorError, ValidationError) as exc:
            # Clean validation rejections. Every curator mutation runs its
            # CuratorError/ValidationError checks BEFORE the in-memory
            # self._nodes write (see curator.py), so reaching here means nothing
            # was applied -- safe to report and stay alive.
            send_message(conn, {"id": req_id, "ok": False, "error": str(exc)})
            continue
        except OSError:
            # A disk fault mid-mutation (ENOSPC/EROFS/permission) can leave the
            # curator's in-memory graph ahead of what was persisted. Do NOT mask
            # it: let the subprocess die so the next spawn reloads consistent
            # on-disk state (the pre-resilience fail-safe). Masking here would let
            # the client observe a node that was never persisted.
            raise
        except Exception as exc:
            # Any other fault. For a MUTATING op the in-memory graph may already
            # be ahead of disk (e.g. a serialization TypeError or an
            # _ancestor_chain cycle ValueError surfacing from write_node, AFTER
            # self._nodes was updated), so a non-OSError write-path fault is just
            # as unsafe as the OSError case: fail-safe by dying so the next spawn
            # reloads consistent on-disk state. A read op cannot skew persisted
            # state, so it stays alive and reports the error in-band.
            if not _is_read_only(intent_dict):
                raise
            send_message(
                conn,
                {"id": req_id, "ok": False, "error": f"curator internal error: {exc}"},
            )
            continue
        send_message(conn, {"id": req_id, "ok": True, "result": result})


def _is_read_only(intent_dict: dict[str, Any]) -> bool:
    """True iff this intent is a pure read (cannot leave in-memory ahead of disk)."""
    return intent_dict.get("op") in _READ_ONLY_OPS


def main(argv: tuple[str, ...] | None = None) -> int:
    """`python -m agent6.graph.server <state-dir> <run-id> <sock-path> [subdir]` entrypoint."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (3, 4):
        print(
            "usage: python -m agent6.graph.server <state-dir> <run-id> <sock-path> [subdir]",
            file=sys.stderr,
        )
        return 2
    state_dir, run_id, sock_path = Path(args[0]), args[1], Path(args[2])
    subdir = args[3] if len(args) == 4 else "runs"
    layout = RunLayout(state_dir=state_dir, run_id=run_id, subdir=subdir)
    try:
        serve(layout, sock_path)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
