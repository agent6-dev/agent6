# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The durable record of one child dispatch: what was granted, launched, joined and settled.

Written atomically before any child launches and advanced at each boundary, so a resumed
execution finishes or diagnoses a dispatch its predecessor left behind: one file per
dispatch under `<session_dir>/dispatch/`. The harness writes it; every other reader reads.
"""

from __future__ import annotations

import pathlib
import re
from typing import Literal

import pydantic

from agent6 import child_records, portable

RECORD_VERSION = 9
DIR_NAME = "dispatch"
DispatchState = Literal["prepared", "running", "settling", "ranking", "joining", "complete"]


class DispatchRecordError(Exception):
    """A record cannot be read: malformed, or written by a newer agent6."""


class JoinRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """How one child's work joined, with the chain tip on either side of the merge.

    Attributes:
        branch: The child's branch.
        status: The join's outcome.
        sha: The merge commit, "" when none.
        detail: The failure's reason, a conflict's files, a ranked child's place, or what
            stopped a stopped child; "" when none.
        commits: The commits a stopped child's branch holds beyond its base; 0 otherwise.
        target_before: The chain tip before the join.
        target_after: The chain tip after it; equal to `target_before` when nothing merged.
    """

    branch: str = ""
    status: child_records.JoinStatus
    sha: str = ""
    detail: str = ""
    commits: int = 0
    target_before: str = ""
    target_after: str = ""


class MergeRequest(pydantic.BaseModel, frozen=True, extra="forbid"):
    """A stopped child's requested merge and its delivery state.

    Attributes:
        result: The merge outcome; None while the request awaits a turn boundary.
        delivered: Whether the parent's conversation holds the outcome.
    """

    result: JoinRecord | None = None
    delivered: bool = False


class UsageRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """What a child used, as settled against the parent.

    Attributes:
        usd: The child's metered spend.
        usd_partial: Whether unpriced calls make it a lower bound.
        tokens: Its fallback-ledger tokens.
    """

    usd: float
    usd_partial: bool = False
    tokens: int = 0


class RankRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """The order a rank join merged by, written before the first join so no resume ranks twice.

    Attributes:
        order: The ranked children's session ids, best first; empty when no child had a result.
        ranked_by: `judge` or `mechanical`.
        rationale: The judges' words; "" for the mechanical order.
        winner: The one child the join merges; "" when the order crowns none.
        fallback: Why the mechanical order ranked; "" when judges did.
    """

    order: tuple[str, ...]
    ranked_by: Literal["judge", "mechanical"] = "mechanical"
    rationale: str = ""
    winner: str = ""
    fallback: str = ""


class ChildRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """One child of the dispatch.

    Attributes:
        session_id: The child's session id, fixed before launch.
        number: Its number in the group.
        task: Its task text.
        model: Its selector, None for the worker's own settings.
        limits: Its turn cap and deadline, None for the config's own.
        join: How it joined, None until it did.
        report: What it said at its end, its finish summary; "" until its end was reported
            or when it gave none.
        reported: Whether its parent's conversation holds the notice of its end.
        usage: What it used, None until settled with known usage.
        settled: Whether its usage was charged; False while it is unknown.
        merge_request: The explicit request to merge stopped work; None before one.
    """

    session_id: str
    number: int
    task: str
    model: str | None = None
    limits: child_records.ChildLimits | None = None
    join: JoinRecord | None = None
    report: str = ""
    reported: bool = False
    usage: UsageRecord | None = None
    settled: bool = False
    merge_request: MergeRequest | None = None


class DispatchRecord(pydantic.BaseModel, frozen=True, extra="forbid"):
    """The record of one dispatch: one task's children, or one panel's reviewers.

    **Support: internal.** Written by the dispatching execution, read by a resume.

    Attributes:
        version: The record's shape; a newer one refuses the read.
        identity: The dispatch's id, the group id `<parent>-p<n>`.
        source: Who asked.
        join_policy: How the work merges: a task's one child merged, a panel's records
            imported and nothing merged, or the best-ranked of a task's attempts alone.
        ranking: A rank join's order; None before it ranked and under the other policies.
        owner: The execution that made it.
        turn: The turn it was dispatched after.
        base: The chain tip the children were cut from.
        spec: The task's child spec as given; "" for one default child and for a panel.
        parent_node: The DAG node the task was dispatched under; None without a graph and
            for a panel.
        node_id: The task's DAG node, added when its work merges or its dispatch fails; None
            until then, without a graph and for a panel.
        children: The children, in child order.
        state: How far the dispatch got; `ranking` while a task's judges run.
        message: The notice of what merged, "" until complete.
        delivered: Whether that notice reached the conversation.
        written_at: When this version of the record was written, as epoch seconds.
    """

    version: int = RECORD_VERSION
    identity: str
    source: child_records.DispatchSource
    join_policy: child_records.JoinPolicy = "merge"
    ranking: RankRecord | None = None
    owner: str
    turn: int
    base: str
    spec: str = ""
    parent_node: str | None = None
    node_id: str | None = None
    children: tuple[ChildRecord, ...]
    state: DispatchState = "prepared"
    message: str = ""
    delivered: bool = False
    written_at: float = 0.0

    def child(self, session_id: str) -> ChildRecord:
        """Return the child with this session id.

        Args:
            session_id: The child's session id.

        Returns:
            Its record.

        Raises:
            KeyError: No child has it.
        """
        for child in self.children:
            if child.session_id == session_id:
                return child
        raise KeyError(session_id)

    def with_child(self, child: ChildRecord) -> DispatchRecord:
        """Return the record with this child's entry replaced."""
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
