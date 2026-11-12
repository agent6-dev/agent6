# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The durable record of one lane dispatch: what was granted, launched, joined and settled.

Written atomically before any lane launches and advanced at each boundary, so a resumed
execution finishes or diagnoses a dispatch its predecessor left behind: one file per
dispatch under `<session_dir>/dispatch/`. The harness writes it; every other reader reads.
"""

from __future__ import annotations

import pathlib
import re
from typing import Literal

import pydantic

from agent6 import lane_records, portable

RECORD_VERSION = 4
DIR_NAME = "dispatch"
DispatchState = Literal["prepared", "running", "settling", "joining", "complete"]


class DispatchRecordError(Exception):
    """A record cannot be read: malformed, or written by a newer agent6."""


class SegmentRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """One requested task and its lane spec.

    Attributes:
        task: The task text.
        spec: The lane spec as given.
        lanes: How many lanes the segment expanded to.
        node_id: The DAG node the segment got, None without a graph.
    """

    task: str
    spec: str = ""
    lanes: int = 1
    node_id: str | None = None


class JoinRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """How one lane's work joined, with the chain tip on either side of the merge.

    Attributes:
        branch: The lane's branch.
        status: The join's outcome.
        sha: The merge commit, "" when none.
        detail: The failure's reason, or what stopped a lane whose work still joined; "" when
            neither.
        target_before: The chain tip before the join.
        target_after: The chain tip after it; equal to `target_before` when nothing merged.
    """

    branch: str = ""
    status: lane_records.JoinStatus
    sha: str = ""
    detail: str = ""
    target_before: str = ""
    target_after: str = ""


class UsageRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """What a lane used, as settled against the parent.

    Attributes:
        usd: The lane's metered spend.
        usd_partial: Whether unpriced calls make it a lower bound.
        tokens: Its fallback-ledger tokens.
    """

    usd: float
    usd_partial: bool = False
    tokens: int = 0


class ChildRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """One lane of the dispatch.

    Attributes:
        session_id: The lane's session id, fixed before launch.
        lane: Its number in the group.
        task: Its task text.
        model: Its selector, None for the worker's own settings.
        limits: Its turn cap and deadline, None for the config's own.
        join: How it joined, None until it did.
        report: What it said at its end, its finish summary; "" until it joined or when it
            gave none.
        usage: What it used, None until settled with known usage.
        settled: Whether its usage was charged; False while it is unknown.
    """

    session_id: str
    lane: int
    task: str
    model: str | None = None
    limits: lane_records.LaneLimits | None = None
    join: JoinRecord | None = None
    report: str = ""
    usage: UsageRecord | None = None
    settled: bool = False


class DispatchRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """The record of one dispatch.

    **Support: internal.** Written by the dispatching execution, read by a resume.

    Attributes:
        version: The record's shape; a newer one refuses the read.
        identity: The dispatch's id, the group id `<coordinator>-p<n>`.
        source: Who asked.
        join_policy: Ordered work merge or harness-owned record-only import.
        owner: The execution that made it.
        turn: The turn it was dispatched after.
        base: The chain tip the lanes were cut from.
        segments: The requested tasks.
        children: The lanes, in lane order.
        state: How far the dispatch got.
        message: The result delivered for it, "" until complete.
        delivered: Whether the result reached the conversation.
        written_at: When this version of the record was written, as epoch seconds.
    """

    version: int = RECORD_VERSION
    identity: str
    source: lane_records.DispatchSource
    join_policy: Literal["merge", "record_only"] = "merge"
    owner: str
    turn: int
    base: str
    segments: tuple[SegmentRecord, ...]
    children: tuple[ChildRecord, ...]
    state: DispatchState = "prepared"
    message: str = ""
    delivered: bool = False
    written_at: float = 0.0

    def child(self, session_id: str) -> ChildRecord:
        """Return the lane with this session id.

        Args:
            session_id: The lane's session id.

        Returns:
            Its record.

        Raises:
            KeyError: No lane has it.
        """
        for child in self.children:
            if child.session_id == session_id:
                return child
        raise KeyError(session_id)

    def with_child(self, child: ChildRecord) -> DispatchRecord:
        """Return the record with this lane's entry replaced."""
        return self.model_copy(
            update={
                "children": tuple(
                    child if c.session_id == child.session_id else c for c in self.children
                )
            }
        )

    def advanced(self, state: DispatchState, *, message: str | None = None) -> DispatchRecord:
        """Return the record in a later state, with its message when given."""
        update: dict[str, object] = {"state": state}
        if message is not None:
            update["message"] = message
        return self.model_copy(update=update)


def records_dir(session_dir: pathlib.Path) -> pathlib.Path:
    """Return the dir a session's dispatch records live in."""
    return session_dir / DIR_NAME


def record_path(session_dir: pathlib.Path, identity: str) -> pathlib.Path:
    """Return a dispatch's record file."""
    return records_dir(session_dir) / f"{identity}.json"


def write_record(session_dir: pathlib.Path, record: DispatchRecord, *, now: float) -> None:
    """Write the record atomically, stamped with the time.

    Args:
        session_dir: The dispatching session's dir.
        record: The record.
        now: The wall clock.

    Raises:
        OSError: The dir or file cannot be written; the caller refuses to launch.
    """
    path = record_path(session_dir, record.identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = record.model_copy(update={"written_at": now})
    portable.atomic_write(path, stamped.model_dump_json(indent=2) + "\n")


_GROUP_NUMBER = re.compile(r"-p(\d+)$")


def read_records(session_dir: pathlib.Path) -> tuple[DispatchRecord, ...]:
    """Return a session's dispatch records in dispatch order; none for a session without any.

    Args:
        session_dir: The session's dir; an older session has no records dir.

    Returns:
        The records, the earliest group first.

    Raises:
        DispatchRecordError: A record is malformed or written by a newer agent6.
    """
    directory = records_dir(session_dir)
    if not directory.is_dir():
        return ()
    records: list[DispatchRecord] = []
    for path in sorted(directory.glob("*.json")):
        records.append(read_record(path))
    return tuple(sorted(records, key=lambda record: order_key(record.identity)))


def read_record(path: pathlib.Path) -> DispatchRecord:
    """Return one record.

    Args:
        path: The record file.

    Returns:
        The record.

    Raises:
        DispatchRecordError: The file is malformed, unreadable or written by a newer agent6.
    """
    try:
        record = DispatchRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, pydantic.ValidationError) as exc:
        raise DispatchRecordError(f"invalid dispatch record {path}: {exc}") from exc
    if record.version > RECORD_VERSION:
        raise DispatchRecordError(
            f"dispatch record {path} is version {record.version}; this agent6 reads"
            f" version {RECORD_VERSION}"
        )
    return record


def order_key(identity: str) -> tuple[int, str]:
    """Return the sort key that puts dispatch ids in group order (`p2` before `p10`)."""
    match = _GROUP_NUMBER.search(identity)
    return (int(match.group(1)) if match else 0, identity)
