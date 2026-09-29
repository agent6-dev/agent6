# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The append-only journal, the snapshots and the single-writer lock of one machine instance.

The journal is the source of truth: each validated fact is appended, then the reduced
blackboard replaces the current one, and replaying the events reproduces the path. Events
re-enter through pydantic (`extra="forbid", frozen=True`). Snapshots serve inspection and
status only.

Layout under the per-repo state dir, `machines/<id>/`::

    machine.asm.toml     # the source the run started from, for replay
    journal.jsonl        # append-only, fsync'd, one event per line
    snapshots/<n>.json   # blackboard and current state, written atomically
    machine.lock         # the flock single-writer guard
    signal               # an operator poke, consumed by a `wait` state
    wait.json            # the persisted next wake, for --exit-on-wait
"""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import pathlib
import shutil
from collections.abc import Generator
from typing import Annotated, Any, Literal

import pydantic
import pydantic_core

from agent6 import paths, portable
from agent6.machine import spec

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

_MODEL_CONFIG = pydantic.ConfigDict(extra="forbid", frozen=True)


class JournalError(spec.MachineError):
    """On-disk journal state (the journal, a pending wait, the source, the lock) is unusable.

    A `MachineError`, so every surface degrades on a broken journal as on a broken machine
    file.
    """

    def __init__(self, message: str) -> None:
        super().__init__([message])


def _now_iso() -> str:
    """Return the current UTC instant as an ISO-8601 timestamp."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds")


class ToolFact(pydantic.BaseModel):
    """The observation one tool state produced.

    Attributes:
        kind: The discriminator.
        exit_code: The command's exit code.
        stdout: The captured stdout.
        timed_out: The command hit its timeout.
        stderr: The captured stderr, for debugging; routing never reads it. Defaulted, so a
            journal line without it still parses.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["tool"] = "tool"
    exit_code: int
    stdout: str
    timed_out: bool
    stderr: str = ""


class WaitFact(pydantic.BaseModel):
    """The observation one wait state produced.

    Attributes:
        kind: The discriminator.
        wake_epoch: The instant the wait was armed for; None for a timerless wait.
        woke_by: A clock tick or an operator poke.
        payload: The poke's payload, so a replay re-reads the identical input; None for a
            bare poke or a tick.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["wait"] = "wait"
    wake_epoch: float | None = None
    woke_by: Literal["tick", "signal"]
    payload: Any = None


class BranchFact(pydantic.BaseModel):
    """The observation one branch state produced: the index of the clause that fired."""

    model_config = _MODEL_CONFIG

    kind: Literal["branch"] = "branch"
    clause_index: int = pydantic.Field(ge=0)


class AgentFact(pydantic.BaseModel):
    """The observation one agent state produced.

    Attributes:
        kind: The discriminator.
        outcome: The label the state routed on.
        reason: The loop's stop reason.
        payload: The validated `finish_session` payload, or None.
        usd: The slice's spend.
        usd_partial: `usd` is a known under-estimate (an unpriced model); status renders it
            with the `~` marker.
        input_tokens: The slice's input tokens.
        output_tokens: The slice's output tokens.
    """

    model_config = _MODEL_CONFIG

    kind: Literal["agent"] = "agent"
    outcome: Literal["ok", "failed", "budget_exhausted", "timeout"]
    reason: str
    payload: dict[str, Any] | None = None
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = pydantic.Field(default=0, ge=0)
    output_tokens: int = pydantic.Field(default=0, ge=0)


Fact = Annotated[ToolFact | WaitFact | BranchFact | AgentFact, pydantic.Field(discriminator="kind")]


class MachineBegin(pydantic.BaseModel):
    """The journal's first event: which machine, at which version, started the instance."""

    model_config = _MODEL_CONFIG

    type: Literal["machine.begin"] = "machine.begin"
    ts: str
    machine: str
    version: int


class StepEvent(pydantic.BaseModel):
    """One transition: the state, the fact it produced, the label and the destination.

    Attributes:
        type: The discriminator.
        ts: When the step was journaled.
        seq: The transition's index, contiguous from 0.
        state: The state that ran.
        label: The outcome label.
        goto: The destination state.
        fact: The observation.
    """

    model_config = _MODEL_CONFIG

    type: Literal["step"] = "step"
    ts: str
    seq: int = pydantic.Field(ge=0)
    state: str
    label: str
    goto: str
    fact: Fact


class MachineNotify(pydantic.BaseModel):
    """A state's `notify` message, journaled on entry.

    Presentation only: it adds no edge and never moves the reducer.
    """

    model_config = _MODEL_CONFIG

    type: Literal["machine.notify"] = "machine.notify"
    ts: str
    state: str
    message: str
    level: Literal["info", "warn", "error"] = "info"


class MachineEnd(pydantic.BaseModel):
    """The journal's terminal event.

    Attributes:
        type: The discriminator.
        ts: When the machine ended.
        status: The terminal's status, or `failed` for a cap or a runtime error.
        reason: Why the machine ended.
        state: The state it ended in.
        transitions: The transitions taken.
        usd: The spend of an agent slice that ended with no StepEvent to book it (a capture
            that could not be reduced).
        usd_partial: That spend is a known under-estimate.
        input_tokens: That slice's input tokens.
        output_tokens: That slice's output tokens.
    """

    model_config = _MODEL_CONFIG

    type: Literal["machine.end"] = "machine.end"
    ts: str
    status: Literal["ok", "failed"]
    reason: str
    state: str
    transitions: int = pydantic.Field(ge=0)
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class AttemptSpend(pydantic.BaseModel):
    """The metered spend of a state attempt a supervisor death orphaned.

    The resuming supervisor journals it from the per-state log before re-running the state,
    so the budget keeps the billed slice. Bookkeeping only: it never moves the reducer.
    """

    model_config = _MODEL_CONFIG

    type: Literal["attempt.spend"] = "attempt.spend"
    ts: str
    seq: int = pydantic.Field(ge=0)
    state: str
    usd: float = 0.0
    usd_partial: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


JournalEvent = Annotated[
    MachineBegin | StepEvent | MachineNotify | MachineEnd | AttemptSpend,
    pydantic.Field(discriminator="type"),
]

_EVENT_ADAPTER: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(JournalEvent)


class Snapshot(pydantic.BaseModel):
    """The blackboard and position after a transition, for inspection and status."""

    model_config = _MODEL_CONFIG

    seq: int = pydantic.Field(ge=0)
    state: str
    blackboard: dict[str, Any]


class PendingWait(pydantic.BaseModel):
    """A wait that is armed but has not fired.

    The instant is computed once and persisted, so a resume or a scheduler tick compares
    against the same instant; the record goes once the wait fires.

    Attributes:
        state: The wait state.
        wake_epoch: The instant; None for a timerless wait, which fires only on a poke.
        seq: The transition this visit of the state belongs to, telling it from an earlier
            visit's uncleared record; 0 parses a record written before the field existed.
    """

    model_config = _MODEL_CONFIG

    state: str
    wake_epoch: float | None = None
    seq: int = pydantic.Field(default=0, ge=0)

    @property
    def wake_at(self) -> str:
        """The wake instant as an ISO-8601 UTC timestamp, or "" for a timerless wait."""
        if self.wake_epoch is None:
            return ""
        return datetime.datetime.fromtimestamp(self.wake_epoch, tz=datetime.UTC).isoformat()


def scrub_lone_surrogates(value: Any) -> Any:
    """Return a parsed JSON value with every lone surrogate replaced.

    Applied where they enter (a tool's stdout, a poke payload), so the blackboard never
    holds one; the next agent request's `model_dump_json` would raise on it.

    Args:
        value: The parsed value.

    Returns:
        The value, scrubbed when it held one.
    """
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        clean = json.dumps(value, ensure_ascii=False).encode("utf-8", "replace").decode("utf-8")
        return json.loads(clean)
    return value


def dump_json(model: pydantic.BaseModel, *, indent: int | None = None) -> str:
    """Serialize one journal or snapshot record, replacing lone surrogates.

    `model_dump_json` raises on a lone surrogate; the fallback writes valid UTF-8 so the
    audit trail is always written.

    Args:
        model: The record.
        indent: The JSON indent, or None for compact.

    Returns:
        The JSON text.
    """
    try:
        return model.model_dump_json(indent=indent)
    except pydantic_core.PydanticSerializationError:
        raw = json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            default=str,
        )
        return raw.encode("utf-8", "replace").decode("utf-8")


# How far back a torn-tail heal looks for the last newline before reading the whole file.
_TAIL_WINDOW = 1 << 20


class MachineJournal:
    """The append-only event log plus the snapshots of one machine instance.

    Attributes:
        snapshot_keep: How many recent snapshots to keep (`[machine].snapshot_keep`); 0 keeps
            all.
        root: The instance directory.
        journal_path: The journal file.
        snapshots_dir: The snapshots directory.
        source_path: The recorded machine source.
        signal_path: The operator poke file.
        wait_path: The pending wait record.
    """

    def __init__(self, root: pathlib.Path, *, snapshot_keep: int = 5) -> None:
        self.snapshot_keep = snapshot_keep
        self.root = root
        self.journal_path = root / "journal.jsonl"
        self.snapshots_dir = root / "snapshots"
        self.source_path = root / "machine.asm.toml"
        self.signal_path = root / "signal"
        self.wait_path = root / "wait.json"

    def ensure_dirs(self) -> None:
        """Create the instance directories."""
        paths.mkdir_for_real_user(self.snapshots_dir)

    def exists(self) -> bool:
        """Return whether the instance has a journal."""
        return self.journal_path.is_file()

    def begin(self, *, machine: str, version: int) -> None:
        """Append the `machine.begin` event."""
        self.append(MachineBegin(ts=_now_iso(), machine=machine, version=version))

    def append(self, event: pydantic.BaseModel) -> None:
        """Append one event as a JSON line, fsync'd.

        A torn previous append (a file not ending in a newline) is truncated off first, so
        this event lands on its own line.
        """
        self._heal_torn_tail()
        line = dump_json(event)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _heal_torn_tail(self) -> None:
        """Truncate a torn final line off the journal, in place."""
        if not self.journal_path.is_file():
            return
        with self.journal_path.open("rb") as fh:
            if fh.seek(0, os.SEEK_END) == 0:
                return
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) == b"\n":
                return
        # `truncate` leaves the file at the old length or the new one; a rewrite would open a
        # window where a kill or a reader sees an empty journal.
        with self.journal_path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            window = min(size, _TAIL_WINDOW)
            fh.seek(size - window)
            tail = fh.read(window)
        cut = tail.rfind(b"\n")
        if cut < 0:
            cut = self.journal_path.read_bytes().rfind(b"\n")
            if cut < 0:
                os.truncate(self.journal_path, 0)  # one torn line, nothing committed
                return
            os.truncate(self.journal_path, cut + 1)
            return
        os.truncate(self.journal_path, size - window + cut + 1)

    def read(self) -> list[Any]:
        """Parse and validate every journal line in order.

        Returns:
            The events; empty when there is no journal.

        Raises:
            JournalError: A line does not validate.
        """
        if not self.journal_path.is_file():
            return []
        raw_lines = self.journal_path.read_bytes().split(b"\n")
        # Bytes, not splitlines(): a crash can tear a multibyte sequence, and splitlines() breaks
        # on U+2028, U+2029 and U+0085, which `model_dump_json` writes literally inside strings.
        if raw_lines and raw_lines[-1] != b"":
            raw_lines.pop()
        events: list[Any] = []
        for lineno, raw in enumerate(raw_lines, start=1):
            if not raw.strip():
                continue
            try:
                events.append(_EVENT_ADAPTER.validate_json(raw))
            except pydantic.ValidationError as exc:
                raise JournalError(
                    f"corrupt journal line {lineno} in {self.journal_path}: {exc}"
                ) from exc
        return events

    def end_event(self) -> MachineEnd | None:
        """Return the terminal event, reading only the journal's tail.

        Returns:
            The end, or None while the instance can still take a verb.

        Raises:
            JournalError: The tail does not validate.
        """
        if not self.journal_path.is_file():
            return None
        with self.journal_path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            start = max(0, size - _TAIL_WINDOW)
            fh.seek(start)
            lines = fh.read().split(b"\n")
        if start > 0:
            lines.pop(0)  # the window opens mid-line
        if lines and lines[-1] != b"":
            lines.pop()  # a torn final line was never committed
        whole = [raw for raw in lines if raw.strip()]
        if not whole:
            events = self.read()
            end = events[-1] if events else None
            return end if isinstance(end, MachineEnd) else None
        try:
            event = _EVENT_ADAPTER.validate_json(whole[-1])
        except pydantic.ValidationError as exc:
            raise JournalError(f"corrupt journal tail in {self.journal_path}: {exc}") from exc
        return event if isinstance(event, MachineEnd) else None

    def write_snapshot(self, snapshot: Snapshot) -> None:
        """Write a snapshot atomically, keeping only the newest `snapshot_keep`.

        Recovery and replay fold the journal; the retained tail is a fallback for a corrupt
        latest.
        """
        paths.mkdir_for_real_user(self.snapshots_dir)
        dest = self.snapshots_dir / f"{snapshot.seq}.json"
        portable.atomic_write(dest, dump_json(snapshot, indent=2) + "\n")
        if self.snapshot_keep <= 0:
            return
        with contextlib.suppress(OSError):
            for entry in self.snapshots_dir.iterdir():
                if (
                    entry.suffix == ".json"
                    and entry.stem.isdigit()
                    and int(entry.stem) <= snapshot.seq - self.snapshot_keep
                ):
                    with contextlib.suppress(OSError):
                        entry.unlink()

    def latest_snapshot(self) -> Snapshot | None:
        """Return the newest readable snapshot, or None when none is.

        A torn newest snapshot falls back to the next older one, so one bad snapshot never
        fails `machine status`.
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
                return Snapshot.model_validate_json(path.read_bytes())
            except (pydantic.ValidationError, OSError):
                continue
        return None

    def take_signal(self) -> tuple[bool, Any]:
        """Claim a pending operator poke, if any.

        The signal is renamed to a claim file, so a poke landing between a read and an unlink
        is never lost; the claim outlives this call until `ack_signal`, so a death before the
        ack re-delivers the poke (at least once, which a wake tolerates).

        Returns:
            Whether a poke was present, and the JSON it carried (None for a bare, empty or
            unparseable poke, each a valid wake).
        """
        consume = self.signal_path.with_suffix(".consuming")
        if not consume.exists():
            try:
                self.signal_path.rename(consume)
            except FileNotFoundError:
                return False, None
        try:
            raw = consume.read_text(encoding="utf-8")
        except OSError:
            raw = ""
        if not raw.strip():
            return True, None
        try:
            return True, scrub_lone_surrogates(json.loads(raw))
        except json.JSONDecodeError:
            return True, None

    def ack_signal(self) -> None:
        """Discard the claimed poke once its wake's StepEvent is durable."""
        self.signal_path.with_suffix(".consuming").unlink(missing_ok=True)

    def read_pending_poke(self) -> tuple[bool, Any]:
        """Read a poke the machine has still to act on, without consuming it.

        Returns:
            Whether one is present (the signal file, or an unacked claim), and its payload.
        """
        for path in (self.signal_path.with_suffix(".consuming"), self.signal_path):
            try:
                raw = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            if not raw.strip():
                return True, None
            try:
                return True, scrub_lone_surrogates(json.loads(raw))
            except json.JSONDecodeError:
                return True, None
        return False, None

    def poke(self, payload: Any = None) -> None:
        """Drop a signal file so a blocked or armed wait wakes.

        Atomic, since `take_signal` polls from another process and would consume a partial
        file as a bare poke.

        Args:
            payload: Travels to the waking wait as its `signal` payload, journaled.
        """
        paths.mkdir_for_real_user(self.root)
        portable.atomic_write(self.signal_path, json.dumps(payload))

    def read_pending_wait(self) -> PendingWait | None:
        """Read the armed wait, if any.

        Returns:
            The record, or None when no wait is armed.

        Raises:
            JournalError: The record is unreadable; the message names the remedy, since firing
                early or skipping the wait would both be worse than refusing.
        """
        if not self.wait_path.is_file():
            return None
        try:
            return PendingWait.model_validate_json(self.wait_path.read_bytes())
        except (pydantic.ValidationError, OSError) as exc:
            raise JournalError(
                f"corrupt pending wait {self.wait_path}: {exc}\n"
                f"  delete it to re-arm the wait from the machine's own state:"
                f" rm {self.wait_path}"
            ) from exc

    def write_pending_wait(self, pending: PendingWait) -> None:
        """Persist the armed wait atomically."""
        paths.mkdir_for_real_user(self.root)
        portable.atomic_write(self.wait_path, dump_json(pending, indent=2) + "\n")

    def clear_pending_wait(self) -> None:
        """Drop the armed wait's record."""
        self.wait_path.unlink(missing_ok=True)


@contextlib.contextmanager
def machine_lock(root: pathlib.Path) -> Generator[None]:
    """Hold the single-writer lock of one machine instance.

    Args:
        root: The instance directory.

    Yields:
        Nothing; the lock is held for the block.

    Raises:
        JournalError: Another runner holds the lock.
    """
    paths.mkdir_for_real_user(root)
    lock_path = root / "machine.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            portable.lock_exclusive(fd, blocking=False)
        except OSError as exc:
            raise JournalError(f"machine is already running (lock held): {lock_path}") from exc
        try:
            yield
        finally:
            portable.unlock(fd)
    finally:
        os.close(fd)


def write_source(root: pathlib.Path, text: str) -> None:
    """Persist the machine source the run started from, for replay."""
    paths.mkdir_for_real_user(root)
    portable.atomic_write(root / "machine.asm.toml", text)


def read_source(root: pathlib.Path) -> str:
    """Read the persisted machine source.

    Args:
        root: The instance directory.

    Returns:
        The source text.

    Raises:
        JournalError: No source is persisted, or it cannot be read.
    """
    path = root / "machine.asm.toml"
    if not path.is_file():
        raise JournalError(f"no persisted machine source at {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise JournalError(f"cannot read persisted machine source at {path}: {exc}") from exc


def write_stop_request(root: pathlib.Path) -> None:
    """Ask the live machine to park at its next transition boundary.

    A marker, not a kill: the state in flight finishes and journals its fact, then the
    engine returns `stopped` with no end, so the instance stays resumable.
    """
    paths.mkdir_for_real_user(root)
    (root / "stop").touch()


def stop_requested(root: pathlib.Path) -> bool:
    """Return whether a stop marker is present."""
    return (root / "stop").is_file()


def clear_stop_request(root: pathlib.Path) -> None:
    """Remove the stop marker."""
    with contextlib.suppress(FileNotFoundError):
        (root / "stop").unlink()


def write_bundle(root: pathlib.Path, machine_path: pathlib.Path) -> None:
    """Persist the bundle the instance starts from: the source plus its `scripts/` tree.

    Replay evidence, and the baseline `bundle_drift` holds every continuation to.
    """
    write_source(root, machine_path.read_text(encoding="utf-8"))
    dst = root / "scripts"
    shutil.rmtree(dst, ignore_errors=True)
    scripts = machine_path.parent / "scripts"
    if scripts.is_dir():
        shutil.copytree(scripts, dst)


def bundle_drift(root: pathlib.Path, machine_path: pathlib.Path) -> str | None:
    """Return the first difference between the working bundle and the recorded one.

    A live instance runs the logic it recorded; an edit takes effect on a new instance.

    Args:
        root: The instance directory.
        machine_path: The working machine file.

    Returns:
        One line naming the difference, or None when the bundles match byte for byte.
    """
    recorded_asm = root / "machine.asm.toml"
    if not recorded_asm.is_file():
        return f"no recorded machine source at {recorded_asm}"
    if recorded_asm.read_bytes() != machine_path.read_bytes():
        return f"{machine_path.name} differs from the recorded {recorded_asm}"
    working = _tree_files(machine_path.parent / "scripts")
    recorded = _tree_files(root / "scripts")
    for rel in sorted(recorded.keys() - working.keys()):
        return f"scripts/{rel} was removed after the instance began"
    for rel in sorted(working.keys() - recorded.keys()):
        return f"scripts/{rel} was added after the instance began"
    for rel in sorted(working):
        if working[rel] != recorded[rel]:
            return f"scripts/{rel} differs from the instance's recorded copy"
    return None


def _tree_files(base: pathlib.Path) -> dict[str, bytes]:
    """Return every file under the directory by relative path, or {} when it is not one."""
    if not base.is_dir():
        return {}
    return {
        p.relative_to(base).as_posix(): p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }
