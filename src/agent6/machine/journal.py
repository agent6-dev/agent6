# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Append-only journal, blackboard snapshots, and the single-writer lock.

The journal is the source of truth (§5.1): every impure observation a state
makes, a tool's exit code and stdout, a wait's resolved wake instant, a
branch's chosen clause, is appended as a fact *before* the blackboard is
reduced. Replaying the journal therefore reproduces the exact path, because
the pure reducer reads recorded facts instead of re-touching the world.

Events are read back from disk, so they re-enter at a trust boundary and are
re-validated by pydantic (`extra="forbid", frozen=True`), exactly like the
machine spec itself. Snapshots are an optimisation for human inspection and
fast status; correctness depends only on the journal.

Layout under ``.agent6/machines/<id>/`` (§5.3)::

    machine.asm.toml     # the exact source the run was started from (for replay)
    journal.jsonl        # append-only, fsync'd, one event per line
    snapshots/<n>.json   # blackboard + current state, atomic temp+rename
    machine.lock         # flock single-writer guard
    signal               # optional operator poke consumed by a `wait` state
    wait.json            # persisted next-wake instant for --exit-on-wait mode
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent6.portable import lock_exclusive, unlock

__all__ = [
    "AgentFact",
    "BranchFact",
    "Fact",
    "JournalError",
    "JournalEvent",
    "MachineBegin",
    "MachineEnd",
    "MachineJournal",
    "PendingWait",
    "Snapshot",
    "StepEvent",
    "ToolFact",
    "WaitFact",
    "machine_lock",
    "read_source",
    "write_source",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class JournalError(Exception):
    """Raised when an on-disk journal or snapshot is missing or corrupt."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------
# Facts, the impure observation a single state execution produced.
# --------------------------------------------------------------------------


class ToolFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["tool"] = "tool"
    exit_code: int
    stdout: str
    timed_out: bool


class WaitFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["wait"] = "wait"
    wake_epoch: float
    woke_by: Literal["tick", "signal"]


class BranchFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["branch"] = "branch"
    clause_index: int = Field(ge=0)


class AgentFact(BaseModel):
    model_config = _MODEL_CONFIG

    kind: Literal["agent"] = "agent"
    outcome: Literal["ok", "failed", "budget_exhausted", "timeout"]
    reason: str
    payload: dict[str, Any] | None = None
    usd: float = 0.0
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


Fact = Annotated[ToolFact | WaitFact | BranchFact | AgentFact, Field(discriminator="kind")]


# --------------------------------------------------------------------------
# Events, one journal line each.
# --------------------------------------------------------------------------


class MachineBegin(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["machine.begin"] = "machine.begin"
    ts: str
    machine: str
    version: int


class StepEvent(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["step"] = "step"
    ts: str
    seq: int = Field(ge=0)
    state: str
    label: str
    goto: str
    fact: Fact


class MachineEnd(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["machine.end"] = "machine.end"
    ts: str
    status: Literal["ok", "failed"]
    reason: str
    state: str
    transitions: int = Field(ge=0)


JournalEvent = Annotated[MachineBegin | StepEvent | MachineEnd, Field(discriminator="type")]

_EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(JournalEvent)


# --------------------------------------------------------------------------
# Snapshot, blackboard + position, written after every transition.
# --------------------------------------------------------------------------


class Snapshot(BaseModel):
    model_config = _MODEL_CONFIG

    seq: int = Field(ge=0)
    state: str
    blackboard: dict[str, Any]


class PendingWait(BaseModel):
    """A `wait` armed by ``--exit-on-wait`` but not yet fired (§6).

    The absolute ``wake_epoch`` is computed once, when the wait is first
    reached, and persisted so that re-invocations by an external scheduler
    compare against the *same* instant rather than re-arming ``every_secs``
    from a fresh ``now`` each tick. Deleted once the wait fires.
    """

    model_config = _MODEL_CONFIG

    state: str
    wake_epoch: float


# --------------------------------------------------------------------------
# The journal directory.
# --------------------------------------------------------------------------


class MachineJournal:
    """Append-only event log plus snapshots for one machine instance."""

    def __init__(self, root: Path, *, snapshot_keep: int = 5) -> None:
        # Number of recent snapshots to retain (0 = keep all); see
        # `[machine] snapshot_keep` in the config.
        self.snapshot_keep = snapshot_keep
        self.root = root
        self.journal_path = root / "journal.jsonl"
        self.snapshots_dir = root / "snapshots"
        self.source_path = root / "machine.asm.toml"
        self.signal_path = root / "signal"
        self.wait_path = root / "wait.json"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.journal_path.is_file()

    def begin(self, *, machine: str, version: int) -> None:
        self.append(MachineBegin(ts=_now_iso(), machine=machine, version=version))

    def append(self, event: BaseModel) -> None:
        """Append one event as a JSON line, fsync'd."""
        line = event.model_dump_json()
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read(self) -> list[Any]:
        """Parse and validate every journal line in order."""
        if not self.journal_path.is_file():
            return []
        events: list[Any] = []
        for lineno, raw in enumerate(
            self.journal_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            try:
                events.append(_EVENT_ADAPTER.validate_json(raw))
            except ValidationError as exc:
                raise JournalError(
                    f"corrupt journal line {lineno} in {self.journal_path}: {exc}"
                ) from exc
        return events

    def write_snapshot(self, snapshot: Snapshot) -> None:
        """Write a snapshot atomically (temp file + rename), pruning old ones.

        Recovery only ever reads ``latest_snapshot`` and replay rebuilds from
        the journal, so old snapshots are dead weight: a 10-minute-loop machine
        would otherwise accumulate ~150k files a year. Keep a short fixed tail
        (paranoia against a corrupt latest) and delete the rest.
        """
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        dest = self.snapshots_dir / f"{snapshot.seq}.json"
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
        with tmp.open("r", encoding="utf-8") as fh:
            os.fsync(fh.fileno())
        tmp.rename(dest)
        if self.snapshot_keep <= 0:
            return
        with suppress(OSError):
            for entry in self.snapshots_dir.iterdir():
                if (
                    entry.suffix == ".json"
                    and entry.stem.isdigit()
                    and int(entry.stem) <= snapshot.seq - self.snapshot_keep
                ):
                    with suppress(OSError):
                        entry.unlink()

    def latest_snapshot(self) -> Snapshot | None:
        if not self.snapshots_dir.is_dir():
            return None
        best: int | None = None
        for entry in self.snapshots_dir.iterdir():
            if entry.suffix != ".json" or not entry.stem.isdigit():
                continue
            seq = int(entry.stem)
            if best is None or seq > best:
                best = seq
        if best is None:
            return None
        path = self.snapshots_dir / f"{best}.json"
        try:
            return Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise JournalError(f"corrupt snapshot {path}: {exc}") from exc

    def take_signal(self) -> bool:
        """Consume a pending operator poke, if any. Returns True if one was present."""
        if self.signal_path.exists():
            self.signal_path.unlink()
            return True
        return False

    def poke(self) -> None:
        """Drop a signal file so a blocked or armed `wait` wakes (§6 signal-poke)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.signal_path.write_text("", encoding="utf-8")

    def read_pending_wait(self) -> PendingWait | None:
        if not self.wait_path.is_file():
            return None
        try:
            return PendingWait.model_validate_json(self.wait_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise JournalError(f"corrupt pending wait {self.wait_path}: {exc}") from exc

    def write_pending_wait(self, pending: PendingWait) -> None:
        """Persist the armed next-wake instant atomically (temp file + rename)."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.wait_path.with_suffix(".json.tmp")
        tmp.write_text(pending.model_dump_json(indent=2) + "\n", encoding="utf-8")
        with tmp.open("r", encoding="utf-8") as fh:
            os.fsync(fh.fileno())
        tmp.rename(self.wait_path)

    def clear_pending_wait(self) -> None:
        self.wait_path.unlink(missing_ok=True)


@contextmanager
def machine_lock(root: Path) -> Generator[None]:
    """Single-writer guard for one machine id (§6). Refuses a second runner."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "machine.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            lock_exclusive(fd, blocking=False)
        except OSError as exc:
            raise JournalError(f"machine is already running (lock held): {lock_path}") from exc
        try:
            yield
        finally:
            unlock(fd)
    finally:
        os.close(fd)


def write_source(root: Path, text: str) -> None:
    """Persist the exact `.asm.toml` source the run started from (for replay)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "machine.asm.toml").write_text(text, encoding="utf-8")


def read_source(root: Path) -> str:
    path = root / "machine.asm.toml"
    if not path.is_file():
        raise JournalError(f"no persisted machine source at {path}")
    return path.read_text(encoding="utf-8")
