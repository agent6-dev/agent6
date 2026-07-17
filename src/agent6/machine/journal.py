# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Append-only journal, blackboard snapshots, and the single-writer lock for one
machine instance. The journal is the source of truth: every impure observation a
state makes is appended as a JournalEvent *before* the blackboard is reduced, so
replaying the events reproduces the exact path from the pure reducer.

The recorded observations are a tool's exit code and stdout, a wait's resolved
wake instant, and a branch's chosen clause (§5.1); the reducer reads them back
instead of re-touching the world.

Events are read back from disk, so they re-enter at a trust boundary and are
re-validated by pydantic (`extra="forbid", frozen=True`), exactly like the
machine spec itself. Snapshots are an optimisation for human inspection and
fast status; correctness depends only on the journal.

Layout under the per-repo state dir (``machines/<id>/``) (§5.3)::

    machine.asm.toml     # the exact source the run was started from (for replay)
    journal.jsonl        # append-only, fsync'd, one event per line
    snapshots/<n>.json   # blackboard + current state, atomic temp+rename
    machine.lock         # flock single-writer guard
    signal               # optional operator poke consumed by a `wait` state
    wait.json            # persisted next-wake instant for --exit-on-wait mode
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent6.machine.model import MachineError
from agent6.portable import atomic_write, lock_exclusive, unlock

__all__ = [
    "AgentFact",
    "BranchFact",
    "Fact",
    "JournalError",
    "JournalEvent",
    "MachineBegin",
    "MachineEnd",
    "MachineJournal",
    "MachineNotify",
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


class JournalError(MachineError):
    """Raised when on-disk journal state (journal, pending wait, source, lock) is
    missing, corrupt, or unusable.

    A `MachineError` subclass so every surface that degrades gracefully on a
    broken machine file (hub listing, machine page, SSE stream) degrades the
    same way on a broken journal instead of crashing.
    """

    def __init__(self, message: str) -> None:
        super().__init__([message])


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
    # ``None`` for a wait with no timer (parks until a `signal` poke, §4.3).
    wake_epoch: float | None = None
    woke_by: Literal["tick", "signal"]
    # The poke payload delivered by a `signal` wake, journaled so a replay
    # re-reads the identical input. ``None`` for a bare poke or a `tick`.
    payload: Any = None


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


class MachineNotify(BaseModel):
    """A state's `notify` message, journaled on entry (§4.3).

    Presentation only: it adds no edge and does not affect the reducer or
    routing. Front-ends render it as an ephemeral notification; the operator
    notify hook fires on it out-of-band.
    """

    model_config = _MODEL_CONFIG

    type: Literal["machine.notify"] = "machine.notify"
    ts: str
    state: str
    message: str
    level: Literal["info", "warn", "error"] = "info"


class MachineEnd(BaseModel):
    model_config = _MODEL_CONFIG

    type: Literal["machine.end"] = "machine.end"
    ts: str
    status: Literal["ok", "failed"]
    reason: str
    state: str
    transitions: int = Field(ge=0)


JournalEvent = Annotated[
    MachineBegin | StepEvent | MachineNotify | MachineEnd, Field(discriminator="type")
]

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
    # ``None`` for a wait with no timer: it fires only on a `signal` poke, never
    # on a wake instant, so ``--exit-on-wait`` parks it until the operator pokes.
    wake_epoch: float | None = None


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
        """Append one event as a JSON line, fsync'd.

        Heals a torn previous append first: a committed line always ends in
        ``\\n``, so a file that does not is a crash mid-write. Truncating the
        partial line off keeps this event on its own line instead of
        concatenating onto the fragment (which `read` would then reject).
        """
        self._heal_torn_tail()
        line = event.model_dump_json()
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _heal_torn_tail(self) -> None:
        if not self.journal_path.is_file():
            return
        # Cheap common path: peek the last byte only.
        with self.journal_path.open("rb") as fh:
            if fh.seek(0, os.SEEK_END) == 0:
                return
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return
        raw = self.journal_path.read_bytes()
        last_nl = raw.rfind(b"\n")
        self.journal_path.write_bytes(raw[: last_nl + 1])

    def read(self) -> list[Any]:
        """Parse and validate every journal line in order."""
        if not self.journal_path.is_file():
            return []
        raw_lines = self.journal_path.read_bytes().split(b"\n")
        # split(b"\n"), NOT splitlines(): splitlines() also breaks on U+2028 /
        # U+2029 / U+0085 after decode, which `model_dump_json` writes literally
        # inside JSON strings, so a captured value containing one would shred a
        # single line into unparseable fragments and brick the instance.
        #
        # Split bytes before decoding: a crash can tear the final line in the
        # middle of a multibyte UTF-8 sequence. Dropping that byte tail first
        # keeps the committed prefix readable.
        if raw_lines and raw_lines[-1] != b"":
            raw_lines.pop()
        events: list[Any] = []
        for lineno, raw in enumerate(raw_lines, start=1):
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
        atomic_write(dest, snapshot.model_dump_json(indent=2) + "\n")
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
        """The newest readable snapshot, falling back through the retained tail.

        Snapshots are an inspection optimization (the journal is authoritative),
        and `write_snapshot` keeps a short tail expressly "against a corrupt
        latest". So a torn newest snapshot falls back to the next-older one, and
        only when none are readable do we return None instead of raising -- a
        single bad snapshot must not make `machine status` fail.
        """
        if not self.snapshots_dir.is_dir():
            return None
        seqs = sorted(
            (
                int(entry.stem)
                for entry in self.snapshots_dir.iterdir()
                if entry.suffix == ".json" and entry.stem.isdigit()
            ),
            reverse=True,
        )
        for seq in seqs:
            path = self.snapshots_dir / f"{seq}.json"
            try:
                return Snapshot.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValidationError, OSError):
                continue
        return None

    def take_signal(self) -> tuple[bool, Any]:
        """Consume a pending operator poke, if any.

        Returns ``(present, payload)``: ``present`` is True when a signal file was
        consumed; ``payload`` is the JSON the poke carried (``None`` for a bare
        poke, an empty file, or an unparseable one -- a hand-touched signal is a
        valid bare wake).

        Claims the signal by renaming it to a private consume path first: `poke`
        renames a fresh signal into place from another process, so a
        read-then-unlink would destroy a poke that landed in between.
        """
        consume = self.signal_path.with_suffix(".consuming")
        try:
            self.signal_path.rename(consume)
        except FileNotFoundError:
            return False, None
        try:
            raw = consume.read_text(encoding="utf-8")
        except OSError:
            raw = ""
        consume.unlink(missing_ok=True)
        if not raw.strip():
            return True, None
        try:
            return True, json.loads(raw)
        except json.JSONDecodeError:
            return True, None

    def poke(self, payload: Any = None) -> None:
        """Drop a signal file so a blocked or armed `wait` wakes (§6 signal-poke).

        The optional *payload* travels to the waking `wait` as its `signal`
        payload (journaled, replay-safe) for the next tool to read.

        Atomic (temp + fsync + rename) like every other journal write: the
        engine's ``take_signal`` polls from another process, and a plain write
        exposes an empty/partial file it would consume as a bare poke,
        dropping the payload.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write(self.signal_path, json.dumps(payload))

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
        atomic_write(self.wait_path, pending.model_dump_json(indent=2) + "\n")

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
    atomic_write(root / "machine.asm.toml", text)


def read_source(root: Path) -> str:
    path = root / "machine.asm.toml"
    if not path.is_file():
        raise JournalError(f"no persisted machine source at {path}")
    return path.read_text(encoding="utf-8")
