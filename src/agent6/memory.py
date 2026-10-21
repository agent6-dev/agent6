# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Keep the per-repo agent memory under `<state_dir>/memory/`.

One fact per markdown file, with an index rendered from their first non-empty lines.
The index is injected into every run's system prompt; the model reads and edits the
files with the ordinary tools through a narrow path grant. Model-authored
context, never instructions or secrets; sharing a memory across repos is the
operator copying it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import pathlib
import re
import time
from collections.abc import Collection, Generator, Mapping, Sequence
from typing import Any, Literal, cast

from agent6 import errors, paths, portable

MEMORY_DIR_NAME = "memory"
# Past the cap the operator rulings inject as their newest tail behind a clip marker.
DECISIONS_INJECT_CAP = 4_096
# Past the cap the index is clipped with a pointer, so a runaway index cannot flood every prompt.
INDEX_INJECT_CAP = 4_096
# Beside a lane's store: the sha256 of every file `seed_store` copied in, by name.
SEED_NAME = "memory-seed.json"
# Beside the store, outside the model's write grant: who wrote each fact and which runs read it.
USE_NAME = "memory-use.json"

_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class MemoryStoreError(errors.OperatorError):
    """A memory-store operation failed: a bad name, an unreadable store."""


def memory_dir(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the repo's memory directory."""
    return state_dir / MEMORY_DIR_NAME


def seed_digests(state_dir: pathlib.Path) -> dict[str, str]:
    """Return the seed manifest's name to sha256 map.

    Args:
        state_dir: The lane's state dir.

    Returns:
        Each seeded fact's sha256 as copied, by name. A missing, unreadable or misshapen
        manifest reads as empty, so every file then reads as the lane's own.
    """
    try:
        raw = json.loads(seed_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def seed_path(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the seed manifest's path."""
    return state_dir / SEED_NAME


def use_path(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the use record's path."""
    return state_dir / USE_NAME


@dataclasses.dataclass(frozen=True, slots=True)
class Touch:
    """One session's touch of a fact.

    Attributes:
        session: The session's id.
        at: When, UTC to the minute.
    """

    session: str
    at: str


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryUse:
    """One fact's provenance and use.

    Attributes:
        created: The write that created it; None when the record never saw it made.
        writes: Every recorded write, in order.
        reads: How many times it was read.
        last_read: The last read.
    """

    created: Touch | None = None
    writes: tuple[Touch, ...] = ()
    reads: int = 0
    last_read: Touch | None = None

    @property
    def updated(self) -> Touch | None:
        """The last write, or None."""
        return self.writes[-1] if self.writes else None

    @property
    def writers(self) -> tuple[str, ...]:
        """Every distinct writer, first to last."""
        return tuple(dict.fromkeys(t.session for t in self.writes))


def read_use(state_dir: pathlib.Path) -> dict[str, MemoryUse]:
    """Return the use record by fact name.

    Args:
        state_dir: The repo's state dir.

    Returns:
        Each fact's provenance and use, by name. A missing, unreadable or misshapen file
        reads as empty and a misshapen entry is dropped: the record is a surface, never a
        gate.
    """
    try:
        raw: Any = json.loads(use_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, MemoryUse] = {}
    for name, entry in cast("dict[Any, Any]", raw).items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        use = _use_from_entry(cast("dict[Any, Any]", entry))
        if use is not None:
            out[name] = use
    return out


def _touch(value: Any) -> Touch | None:
    """Return a `{"session", "at"}` object as a Touch, or None when it is not one."""
    if not isinstance(value, dict):
        return None
    session, at = value.get("session"), value.get("at")  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(session, str) or not session or not isinstance(at, str) or not at:
        return None
    return Touch(session, at)


def _use_from_entry(entry: dict[Any, Any]) -> MemoryUse | None:
    """Return one record entry as a MemoryUse, or None when misshapen."""
    reads, raw_writes = entry.get("reads", 0), entry.get("writes")
    if type(reads) is not int or not isinstance(raw_writes, list):
        return None
    touches = [_touch(w) for w in cast("list[Any]", raw_writes)]
    writes = tuple(t for t in touches if t is not None)
    raw_created, raw_last = entry.get("created"), entry.get("last_read")
    created, last_read = _touch(raw_created), _touch(raw_last)
    if len(writes) < len(touches) or (raw_created is not None and created is None):
        return None
    if raw_last is not None and last_read is None:
        return None
    return MemoryUse(created=created, writes=writes, reads=reads, last_read=last_read)


def record_use(
    state_dir: pathlib.Path,
    *,
    session: str,
    wrote: Sequence[str],
    read: Mapping[str, int],
    created: Collection[str] = (),
    deleted: Collection[str] = (),
    when: float | None = None,
) -> None:
    """Fold one session's memory writes and reads into the use record.

    Nothing to record leaves the file alone.

    Args:
        state_dir: The repo's state dir.
        session: The session's id.
        wrote: The facts written; each write appends a touch.
        read: Read counts by fact; they accumulate with the last reader.
        created: The facts the session created; the first creation stands.
        deleted: The facts deleted; each leaves the record first.
        when: The touch time; None is now.
    """
    if not wrote and not read and not deleted:
        return
    with _locked_memory(state_dir):
        _record_use_unlocked(
            state_dir,
            session=session,
            wrote=wrote,
            read=read,
            created=created,
            deleted=deleted,
            when=when,
        )


def _record_use_unlocked(
    state_dir: pathlib.Path,
    *,
    session: str,
    wrote: Sequence[str],
    read: Mapping[str, int],
    created: Collection[str] = (),
    deleted: Collection[str] = (),
    when: float | None = None,
) -> None:
    """Fold a session's writes and reads into the record, with the store lock held."""
    touch = Touch(session, time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(when)))
    use = read_use(state_dir)
    for name in deleted:
        use.pop(name, None)
    for name in wrote:
        prior = use.get(name, MemoryUse())
        made = touch if prior.created is None and name in created else prior.created
        use[name] = dataclasses.replace(prior, created=made, writes=(*prior.writes, touch))
    for name, count in read.items():
        if name in deleted and name not in wrote:
            continue  # a read of a life that ended this execution brings no entry back
        prior = use.get(name, MemoryUse())
        use[name] = dataclasses.replace(prior, reads=prior.reads + count, last_read=touch)
    _write_use(state_dir, use)


def merge_use(
    src_state_dir: pathlib.Path, dst_state_dir: pathlib.Path, *, written: Collection[str]
) -> tuple[int, int]:
    """Carry a lane's use record into the origin's at import, before the lane's state dir goes.

    Args:
        src_state_dir: The lane's state dir.
        dst_state_dir: The origin's state dir.
        written: The facts `merge_memory` carried or updated; their writes travel.

    Returns:
        How many entries had writes carried, and how many had reads folded.
    """
    theirs_all = read_use(src_state_dir)
    if not theirs_all:
        return 0, 0
    with _locked_memory(dst_state_dir):
        ours_all = read_use(dst_state_dir)
        held = {p.stem for p in memory_dir(dst_state_dir).glob("*.md")}
        carried = folded = 0
        for name, theirs in theirs_all.items():
            ours = ours_all.get(name, MemoryUse())
            changed = False
            if name in written and theirs.writes:
                fresh = tuple(t for t in theirs.writes if t not in ours.writes)
                made = theirs.created if ours.created is None else ours.created
                ours = dataclasses.replace(ours, created=made, writes=(*ours.writes, *fresh))
                carried += 1
                changed = True
            if theirs.reads and (name in written or name in held):
                mine, its = ours.last_read, theirs.last_read
                later = its if mine is None or (its is not None and its.at >= mine.at) else mine
                ours = dataclasses.replace(ours, reads=ours.reads + theirs.reads, last_read=later)
                folded += 1
                changed = True
            if changed:
                ours_all[name] = ours
        if carried or folded:
            _write_use(dst_state_dir, ours_all)
    return carried, folded


def _write_use(state_dir: pathlib.Path, use: Mapping[str, MemoryUse]) -> None:
    """Write the use record, sorted by name."""
    body = {
        name: {
            "created": None if entry.created is None else dataclasses.asdict(entry.created),
            "writes": [dataclasses.asdict(t) for t in entry.writes],
            "reads": entry.reads,
            "last_read": None if entry.last_read is None else dataclasses.asdict(entry.last_read),
        }
        for name, entry in sorted(use.items())
    }
    portable.atomic_write(use_path(state_dir), (json.dumps(body, indent=1) + "\n").encode("utf-8"))


def _drop_use(state_dir: pathlib.Path, name: str) -> None:
    """Remove one fact from the use record."""
    use = read_use(state_dir)
    if name in use:
        del use[name]
        _write_use(state_dir, use)


def decisions_path(state_dir: pathlib.Path) -> pathlib.Path:
    """Return the decisions file's path: beside the store, outside the model's write grant."""
    return state_dir / "decisions.md"


def migrate_legacy_decisions(state_dir: pathlib.Path) -> str:
    """Rename a store's old `memory/DECISIONS.md` to `decisions.md` while none exists.

    The store is model-writable, so a `DECISIONS.md` beside an existing `decisions.md` is
    named, never merged.

    Args:
        state_dir: The repo's state dir.

    Returns:
        One line saying what moved or what was ignored, or "" with no such file.
    """
    legacy = memory_dir(state_dir) / "DECISIONS.md"
    if not legacy.is_file():
        return ""
    path = decisions_path(state_dir)
    with _locked_memory(state_dir):
        if not legacy.is_file():
            return ""
        if path.exists():
            return f"ignored a stray {legacy}: rulings live at {path}; delete it to silence this"
        legacy.rename(path)
    return f"operator rulings moved from {legacy} to {path}"


def ensure_decisions_file(state_dir: pathlib.Path) -> None:
    """Create an empty `decisions.md` when none exists; the harness does so at every run start.

    Args:
        state_dir: The repo's state dir.
    """
    path = decisions_path(state_dir)
    if not path.exists():
        paths.mkdir_for_real_user(state_dir)
        portable.atomic_write(path, b"")


@contextlib.contextmanager
def _locked_memory(state_dir: pathlib.Path) -> Generator[None]:
    """Serialize the harness's mutations of one repo's store.

    The model's own edits under its write grant take no lock.

    Args:
        state_dir: The repo's state dir.

    Yields:
        With the store locked.
    """
    paths.mkdir_for_real_user(memory_dir(state_dir))
    with portable.locked_file(memory_dir(state_dir)):
        yield


def record_decision(
    state_dir: pathlib.Path, *, question: str, answer: str, session: str, when: float | None = None
) -> str:
    """Record one operator ruling.

    Args:
        state_dir: The repo's state dir.
        question: The question as asked.
        answer: The answer, verbatim.
        session: The session's id.
        when: The ruling's time; None is now.

    Returns:
        The persisted entry; an identical question and answer already on disk returns
        that entry.
    """
    stamp = time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(when))
    q = question.strip().replace("\n", "\n  ")
    a = answer.strip().replace("\n", "\n  ")
    entry = f"- {stamp} [{session}] Q: {q}\n  A: {a}\n"
    path = decisions_path(state_dir)
    with _locked_memory(state_dir):
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            existing = b""
        ruling = _ruling(entry.rstrip("\n"))
        for known in _entries(existing.decode("utf-8", "replace").strip()):
            if _ruling(known) == ruling:
                return known + "\n"
        portable.atomic_write(path, existing + entry.encode("utf-8"))
    return entry


def _entries(text: str) -> list[str]:
    """Return a decisions file's `- ` entries, each with its continuation lines."""
    entries: list[str] = []
    for line in text.splitlines():
        if line.startswith("- ") or not entries:
            entries.append(line)
        else:
            entries[-1] += "\n" + line
    return entries


def _ruling(entry: str) -> str:
    """Return an entry's question and answer without its stamp and session tag."""
    return entry.split("] ", 1)[-1]


def merge_decisions(src_state_dir: pathlib.Path, dst_state_dir: pathlib.Path) -> tuple[int, int]:
    """Append a lane's rulings to the origin's decisions file.

    A ruling the destination already holds under any tag is skipped, as is a repeat
    within the source.

    Args:
        src_state_dir: The lane's state dir.
        dst_state_dir: The origin's state dir.

    Returns:
        How many entries were appended and how many skipped.
    """
    try:
        text = decisions_path(src_state_dir).read_text(encoding="utf-8")
    except OSError:
        return 0, 0
    path = decisions_path(dst_state_dir)
    with _locked_memory(dst_state_dir):
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""
        known = {_ruling(e) for e in _entries(existing.strip())}
        entries = _entries(text.strip())
        fresh: list[str] = []
        for entry in entries:
            if (ruling := _ruling(entry)) not in known:
                known.add(ruling)
                fresh.append(entry)
        if fresh:
            separator = "\n" if existing and not existing.endswith("\n") else ""
            portable.atomic_write(path, existing + separator + "\n".join(fresh) + "\n")
    return len(fresh), len(entries) - len(fresh)


def _sha256(path: pathlib.Path) -> str:
    """Return a file's sha256 hex digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryMerge:
    """What a lane's memory left in the origin's store at import, by name.

    Attributes:
        carried: New fact files landed.
        updated: Names whose lane edit replaced a copy unchanged since seeding.
        deleted: Names whose lane deletion removed such a copy.
        held: Names changed on both sides, or already taken; the lane's version is kept aside.
    """

    carried: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    held: tuple[str, ...] = ()


def merge_memory(
    src_state_dir: pathlib.Path, dst_state_dir: pathlib.Path, *, held_dir: pathlib.Path
) -> MemoryMerge:
    """Carry a lane's memory into the origin's store the way its branch comes back.

    A copy unchanged since seeding is nothing. A change over a copy the origin has not
    touched lands: an edit replaces the file, a deletion removes it. A change on
    both sides is held. A new name lands, or is held when the origin holds that
    name with other content.

    Args:
        src_state_dir: The lane's state dir.
        dst_state_dir: The origin's state dir.
        held_dir: Where a held lane version is kept.

    Returns:
        What landed, by name.
    """
    src = memory_dir(src_state_dir)
    if not src.is_dir():
        return MemoryMerge()
    dst = memory_dir(dst_state_dir)
    paths.mkdir_for_real_user(dst)
    seeds = seed_digests(src_state_dir)
    lane = {p.stem: p for p in src.glob("*.md") if _is_fact_file(p)}
    landed: dict[str, list[str]] = {"carried": [], "updated": [], "deleted": [], "held": []}
    for name in sorted(seeds.keys() | lane.keys()):
        if _NAME_RE.fullmatch(name) is None:
            continue  # not a memory name (`_check_name`): names no path in either store
        path = lane.get(name)
        origin = dst / f"{name}.md"
        seed = seeds.get(name)
        theirs = _sha256(path) if path is not None else None
        with _locked_memory(dst_state_dir):
            ours = _sha256(origin) if origin.is_file() else None
            fate = _fate(seed, theirs, ours)
            if fate == "skip":
                continue
            if fate == "deleted":
                origin.unlink()
                _drop_use(dst_state_dir, name)
            elif fate == "held":
                if path is not None:
                    paths.mkdir_for_real_user(held_dir)
                    portable.atomic_write(held_dir / path.name, path.read_bytes())
            elif path is not None:  # carried or updated: the lane has the file
                portable.atomic_write(origin, path.read_bytes())
        landed[fate].append(name)
    return MemoryMerge(
        carried=tuple(landed["carried"]),
        updated=tuple(landed["updated"]),
        deleted=tuple(landed["deleted"]),
        held=tuple(landed["held"]),
    )


def _fate(
    seed: str | None, theirs: str | None, ours: str | None
) -> Literal["skip", "carried", "updated", "deleted", "held"]:
    """Return one name's fate at import.

    Args:
        seed: The seeded copy's digest; None when the name is new in the lane.
        theirs: The lane's file digest; None when absent.
        ours: The origin's file digest; None when absent.

    Returns:
        The fate.
    """
    if seed is None:  # new in the lane
        if theirs is None or theirs == ours:
            return "skip"
        return "held" if ours is not None else "carried"
    if theirs in (seed, ours):
        return "skip"  # untouched in the lane, or the same content on both sides
    if ours != seed:
        return "held"  # changed on both sides
    return "deleted" if theirs is None else "updated"


def seed_store(src_state_dir: pathlib.Path, dst_state_dir: pathlib.Path) -> int:
    """Copy the repo's fact files into a fresh state dir, recording each copy's digest.

    A lane's clone has an empty store; copies, never links, so a lane cannot write
    the origin's store mid-run. Anything already there stays, and a store entry that
    is not a fact file is not copied. A name seeded earlier is left alone on a retry,
    a fact the lane has since deleted included. The origin's rulings are copied beside the
    store, when the lane has none; an origin with no store leaves the lane storeless.

    Args:
        src_state_dir: The origin's state dir.
        dst_state_dir: The lane's state dir.

    Returns:
        How many fact files were copied; the rulings copy is not one.
    """
    rulings = decisions_path(src_state_dir)
    target_rulings = decisions_path(dst_state_dir)
    if rulings.is_file() and not target_rulings.exists():
        paths.mkdir_for_real_user(dst_state_dir)
        portable.atomic_write(target_rulings, rulings.read_bytes())
    src = memory_dir(src_state_dir)
    if not src.is_dir():
        return 0
    copied = 0
    digests: dict[str, str] = {}
    dst = memory_dir(dst_state_dir)
    with _locked_memory(dst_state_dir):
        recover_unrecorded_copy = not seed_path(dst_state_dir).exists()
        earlier = seed_digests(dst_state_dir)
        for path in sorted(src.iterdir()):
            if not _is_fact_file(path) or path.stem in earlier:
                continue
            target = dst / path.name
            if target.exists():
                if recover_unrecorded_copy and _sha256(path) == _sha256(target):
                    digests[path.stem] = _sha256(target)
                continue
            portable.atomic_write(target, path.read_bytes())
            copied += 1
            digests[path.stem] = _sha256(target)
        portable.atomic_write(
            seed_path(dst_state_dir), json.dumps(earlier | digests, indent=1) + "\n"
        )
    return copied


def decisions_text(state_dir: pathlib.Path) -> str:
    """Return the decisions text for injection: whole under the cap, else its newest tail."""
    try:
        text = decisions_path(state_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) <= DECISIONS_INJECT_CAP:
        return text
    marker = "... (earlier rulings clipped; the operator sees them all: `agent6 memory decisions`)"
    room = DECISIONS_INJECT_CAP - len(marker) - 1
    kept: list[str] = []
    used = 0
    entries = _entries(text)
    for entry in reversed(entries):
        cost = len(entry) + bool(kept)
        if used + cost > room:
            break
        kept.append(entry)
        used += cost
    if not kept and entries:
        kept.append(entries[-1][-room:])  # the newest ruling alone exceeds the cap
    return marker + ("\n" + "\n".join(reversed(kept)) if kept else "")


def clipped_index(index: str) -> str:
    """Return the index as a prompt carries it: whole under the cap, else its head and a marker."""
    body = index.strip()
    if len(body) <= INDEX_INJECT_CAP:
        return body
    marker = "... (index clipped; the rest are <name>.md files in the memory dir)"
    head = body[: INDEX_INJECT_CAP - len(marker) - 1]
    head = head.rsplit("\n", 1)[0] if "\n" in head else head
    return f"{head}\n{marker}" if head else marker


def _is_fact_file(path: pathlib.Path) -> bool:
    """Return whether a store entry is a fact: `<name>.md` under the name rule, a regular file.

    A symlink is not one: the store is model-writable, and a fact feeds every prompt.

    Args:
        path: The store entry.

    Returns:
        True for a fact.
    """
    return (
        path.suffix == ".md"
        and is_memory_name(path.stem)
        and not path.is_symlink()
        and path.is_file()
    )


def _store_entries(state_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the store's entries, sorted; a store that cannot be listed reads as empty."""
    try:
        return sorted(memory_dir(state_dir).iterdir())
    except OSError:
        return []


def index_entries(state_dir: pathlib.Path) -> tuple[tuple[str, str], ...]:
    """Return each fact's name and first non-empty line, sorted by name.

    A fact that cannot be read is skipped and an empty one reads `(empty)`: memory is
    context, never a reason a run fails. A byte that is not UTF-8 is replaced.

    Args:
        state_dir: The repo's state dir.

    Returns:
        The `(name, first line)` pairs; empty with no store.
    """
    entries: list[tuple[str, str]] = []
    for path in _store_entries(state_dir):
        if not _is_fact_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        first = next((line.strip() for line in text.splitlines() if line.strip()), "(empty)")
        entries.append((path.stem, first.lstrip("#").lstrip(" ")))
    return tuple(entries)


def ignored_names(state_dir: pathlib.Path) -> tuple[str, ...]:
    """Return the names of the store's entries that are not fact files, sorted."""
    return tuple(p.name for p in _store_entries(state_dir) if not _is_fact_file(p))


def index_text(state_dir: pathlib.Path) -> str:
    """Render the index for injection.

    Args:
        state_dir: The repo's state dir.

    Returns:
        One `- <name>: <first line>` per fact, sorted by name; "" with no facts.
    """
    return "\n".join(f"- {name}: {first}" for name, first in index_entries(state_dir))


def is_memory_name(name: str) -> bool:
    """Return whether the name is one the store accepts: lowercase letters, digits and dashes."""
    return _NAME_RE.fullmatch(name) is not None


def _check_name(name: str) -> str:
    """Return the name when the store accepts it.

    Args:
        name: The fact's name.

    Returns:
        The name, unchanged.

    Raises:
        MemoryStoreError: The name is not one the store accepts.
    """
    if _NAME_RE.fullmatch(name) is None:
        raise MemoryStoreError(
            f"bad memory name {name!r}: lowercase letters, digits, and dashes only"
        )
    return name


def add(state_dir: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Write a fact file for `agent6 memory add`.

    Args:
        state_dir: The repo's state dir.
        name: The fact's name.
        body: The fact's text; its first line is what the index shows.

    Returns:
        The fact file's path.

    Raises:
        MemoryStoreError: The body is empty or the name is taken.
    """
    body = body.strip()
    if not body:
        raise MemoryStoreError("memory body must be non-empty")
    d = memory_dir(state_dir)
    path = d / f"{_check_name(name)}.md"
    with _locked_memory(state_dir):
        if path.exists():
            raise MemoryStoreError(f"memory {name!r} exists; edit {path} or pick another name")
        portable.atomic_write(path, body + "\n")
        _record_use_unlocked(state_dir, session="operator", wrote=(name,), created=(name,), read={})
    return path


def remove(state_dir: pathlib.Path, name: str) -> None:
    """Delete a fact file for `agent6 memory rm`.

    Args:
        state_dir: The repo's state dir.
        name: The fact's name.

    Raises:
        MemoryStoreError: The name has no file.
    """
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    with _locked_memory(state_dir):
        if not path.is_file():
            raise MemoryStoreError(f"no memory named {name!r}")
        path.unlink()
        _drop_use(state_dir, name)


def show(state_dir: pathlib.Path, name: str) -> str:
    """Return a fact's text.

    Args:
        state_dir: The repo's state dir.
        name: The fact's name.

    Returns:
        The fact file's whole text.

    Raises:
        MemoryStoreError: The name is bad or the file cannot be read.
    """
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryStoreError(f"no memory named {name!r}") from exc
