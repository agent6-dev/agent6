# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Keep the per-repo agent memory under `<state_dir>/memory/`.

One fact per markdown file plus a `MEMORY.md` index, one line per entry. The
index is injected into every run's system prompt; the model reads and edits the
files with the ordinary tools through a narrow path grant. Model-authored
context, never instructions or secrets; sharing a memory across repos is the
operator copying it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Collection, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from agent6.errors import OperatorError
from agent6.paths import mkdir_for_real_user
from agent6.portable import atomic_write, locked_file

MEMORY_DIR_NAME = "memory"
INDEX_NAME = "MEMORY.md"
# Operator rulings, harness-written and append-only; the model reads it and never writes it.
DECISIONS_NAME = "DECISIONS.md"
DECISIONS_INJECT_CAP = 4_096
# Past the cap the index is clipped with a pointer, so a runaway index cannot flood every prompt.
INDEX_INJECT_CAP = 4_096
# Beside a lane's store: the sha256 of every file `seed_store` copied in, by name.
SEED_NAME = "memory-seed.json"
# Beside the store, outside the model's write grant: who wrote each fact and which runs read it.
USE_NAME = "memory-use.json"

_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class MemoryStoreError(OperatorError):
    """A memory-store operation failed: a bad name, an unreadable store."""


def memory_dir(state_dir: Path) -> Path:
    """Return the repo's memory directory."""
    return state_dir / MEMORY_DIR_NAME


def index_path(state_dir: Path) -> Path:
    """Return the index file's path."""
    return memory_dir(state_dir) / INDEX_NAME


def seed_digests(state_dir: Path) -> dict[str, str]:
    """Return the seed manifest's name to sha256 map.

    A missing, unreadable or misshapen manifest reads as empty, so every file then
    reads as the lane's own.
    """
    try:
        raw = json.loads(seed_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def seed_path(state_dir: Path) -> Path:
    """Return the seed manifest's path."""
    return state_dir / SEED_NAME


def use_path(state_dir: Path) -> Path:
    """Return the use record's path."""
    return state_dir / USE_NAME


@dataclass(frozen=True, slots=True)
class Touch:
    """One session's touch of a fact.

    Attributes:
        session: The session's id.
        at: When, UTC to the minute.
    """

    session: str
    at: str


@dataclass(frozen=True, slots=True)
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


def read_use(state_dir: Path) -> dict[str, MemoryUse]:
    """Return the use record by fact name.

    A missing, unreadable or misshapen file reads as empty and a misshapen entry is
    dropped: the record is a surface, never a gate.
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
    state_dir: Path,
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
    state_dir: Path,
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
        use[name] = replace(prior, created=made, writes=(*prior.writes, touch))
    for name, count in read.items():
        if name in deleted and name not in wrote:
            continue  # a read of a life that ended this execution brings no entry back
        prior = use.get(name, MemoryUse())
        use[name] = replace(prior, reads=prior.reads + count, last_read=touch)
    _write_use(state_dir, use)


def merge_use(
    src_state_dir: Path, dst_state_dir: Path, *, written: Collection[str]
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
                ours = replace(ours, created=made, writes=(*ours.writes, *fresh))
                carried += 1
                changed = True
            if theirs.reads and (name in written or name in held):
                mine, its = ours.last_read, theirs.last_read
                later = its if mine is None or (its is not None and its.at >= mine.at) else mine
                ours = replace(ours, reads=ours.reads + theirs.reads, last_read=later)
                folded += 1
                changed = True
            if changed:
                ours_all[name] = ours
        if carried or folded:
            _write_use(dst_state_dir, ours_all)
    return carried, folded


def _write_use(state_dir: Path, use: Mapping[str, MemoryUse]) -> None:
    """Write the use record, sorted by name."""
    body = {
        name: {
            "created": None if entry.created is None else asdict(entry.created),
            "writes": [asdict(t) for t in entry.writes],
            "reads": entry.reads,
            "last_read": None if entry.last_read is None else asdict(entry.last_read),
        }
        for name, entry in sorted(use.items())
    }
    atomic_write(use_path(state_dir), (json.dumps(body, indent=1) + "\n").encode("utf-8"))


def _drop_use(state_dir: Path, name: str) -> None:
    """Remove one fact from the use record."""
    use = read_use(state_dir)
    if name in use:
        del use[name]
        _write_use(state_dir, use)


def decisions_path(state_dir: Path) -> Path:
    """Return the decisions file's path."""
    return memory_dir(state_dir) / DECISIONS_NAME


@contextmanager
def _locked_memory(state_dir: Path) -> Generator[None]:
    """Serialize the harness's mutations of one repo's store.

    The model's own edits under its write grant take no lock.

    Yields:
        With the store locked.
    """
    mkdir_for_real_user(memory_dir(state_dir))
    with locked_file(memory_dir(state_dir)):
        yield


def record_decision(
    state_dir: Path, *, question: str, answer: str, session: str, when: float | None = None
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
        atomic_write(path, existing + entry.encode("utf-8"))
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


def merge_decisions(src_state_dir: Path, dst_state_dir: Path) -> tuple[int, int]:
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
            atomic_write(path, existing + separator + "\n".join(fresh) + "\n")
    return len(fresh), len(entries) - len(fresh)


def _sha256(path: Path) -> str:
    """Return a file's sha256 hex digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryMerge:
    """What a lane's memory left in the origin's store at import, by name.

    Attributes:
        carried: New names, landed with their index lines.
        updated: Names whose lane edit replaced a copy unchanged since seeding.
        deleted: Names whose lane deletion removed such a copy.
        held: Names changed on both sides, or already taken; the lane's version is kept aside.
    """

    carried: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    held: tuple[str, ...] = ()


def merge_memory(src_state_dir: Path, dst_state_dir: Path, *, held_dir: Path) -> MemoryMerge:
    """Carry a lane's memory into the origin's store the way its branch comes back.

    A copy unchanged since seeding is nothing. A change over a copy the origin has not
    touched lands: an edit replaces the file and its index line, a deletion removes
    both. A change on both sides is held. A new name lands with its index line, or is
    held when the origin holds that name with other content; a file the lane's own
    index does not list stays where it is. A hook edited alone does not travel.

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
    mkdir_for_real_user(dst)
    seeds = seed_digests(src_state_dir)
    src_index = index_text(src_state_dir).splitlines()
    lane = {p.stem: p for p in src.glob("*.md") if p.name not in (INDEX_NAME, DECISIONS_NAME)}
    landed: dict[str, list[str]] = {"carried": [], "updated": [], "deleted": [], "held": []}
    for name in sorted(seeds.keys() | lane.keys()):
        if _NAME_RE.fullmatch(name) is None:
            continue  # not a memory name (`_check_name`): names no path in either store
        path = lane.get(name)
        origin = dst / f"{name}.md"
        seed = seeds.get(name)
        theirs = _sha256(path) if path is not None else None
        hook = _index_hook(src_index, name)
        with _locked_memory(dst_state_dir):
            indexed = _index_has(dst_state_dir, name)
            ours = _sha256(origin) if origin.is_file() else None
            fate = _fate(seed, theirs, ours, hook=hook, indexed=indexed)
            if fate == "skip":
                continue
            if fate == "deleted":
                _drop_index_line(dst_state_dir, name)
                origin.unlink()
            elif fate == "held":
                if path is not None:
                    mkdir_for_real_user(held_dir)
                    atomic_write(held_dir / path.name, path.read_bytes())
            elif path is not None:  # carried or updated: the lane has the file
                if fate == "updated" and hook is not None:
                    _replace_index_line(dst_state_dir, name, hook)
                atomic_write(origin, path.read_bytes())
                if fate == "carried" and hook is not None:
                    _append_index_line(dst_state_dir, name, hook)
        landed[fate].append(name)
    return MemoryMerge(
        carried=tuple(landed["carried"]),
        updated=tuple(landed["updated"]),
        deleted=tuple(landed["deleted"]),
        held=tuple(landed["held"]),
    )


def _fate(
    seed: str | None, theirs: str | None, ours: str | None, *, hook: str | None, indexed: bool
) -> Literal["skip", "carried", "updated", "deleted", "held"]:
    """Return one name's fate at import.

    Args:
        seed: The seeded copy's digest; None when the name is new in the lane.
        theirs: The lane's file digest; None when absent.
        ours: The origin's file digest; None when absent.
        hook: The lane's index line hook; None when its index does not list the name.
        indexed: Whether the origin's index names it.

    Returns:
        The fate.
    """
    if seed is None:  # new in the lane
        if hook is None or theirs is None:
            return "skip"  # unindexed there: invisible there, and stays so
        if theirs == ours:
            return "skip" if indexed else "carried"
        return "held" if ours is not None or indexed else "carried"
    if theirs in (seed, ours):
        return "skip"  # untouched in the lane, or the same content on both sides
    if ours != seed:
        return "held"  # changed on both sides
    return "deleted" if theirs is None else "updated"


def seed_store(src_state_dir: Path, dst_state_dir: Path) -> int:
    """Copy the repo's memory into a fresh state dir, recording each copied fact's digest.

    A lane's clone has an empty store; copies, never links, so a lane cannot write
    the origin's store mid-run. Anything already there stays.

    Args:
        src_state_dir: The origin's state dir.
        dst_state_dir: The lane's state dir.

    Returns:
        How many files were copied.
    """
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
            target = dst / path.name
            if not path.is_file():
                continue
            fact = path.suffix == ".md" and path.name not in (INDEX_NAME, DECISIONS_NAME)
            if target.exists():
                if (
                    recover_unrecorded_copy
                    and fact
                    and path.stem not in earlier
                    and _sha256(path) == _sha256(target)
                ):
                    digests[path.stem] = _sha256(target)
                continue
            atomic_write(target, path.read_bytes())
            copied += 1
            if fact:
                digests[path.stem] = _sha256(target)
        atomic_write(seed_path(dst_state_dir), json.dumps(earlier | digests, indent=1) + "\n")
    return copied


def decisions_text(state_dir: Path) -> str:
    """Return the decisions text for injection: whole under the cap, else its newest tail."""
    try:
        text = decisions_path(state_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if len(text) <= DECISIONS_INJECT_CAP:
        return text
    marker = f"... (earlier rulings clipped; {DECISIONS_NAME} holds all)"
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
    marker = "... (index clipped; read MEMORY.md for the rest)"
    head = body[: INDEX_INJECT_CAP - len(marker) - 1]
    head = head.rsplit("\n", 1)[0] if "\n" in head else head
    return f"{head}\n{marker}" if head else marker


def index_text(state_dir: Path) -> str:
    """Return the index body for injection; "" when absent or unreadable.

    A byte that is not UTF-8 is replaced: read strictly, one would empty the index for
    every run, and the next `memory add` would rebuild the file from that empty read.
    """
    try:
        return index_path(state_dir).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def is_memory_name(name: str) -> bool:
    """Return whether the name is one the store accepts: lowercase letters, digits and dashes."""
    return _NAME_RE.fullmatch(name) is not None


def _check_name(name: str) -> str:
    """Return the name when the store accepts it.

    Raises:
        MemoryStoreError: The name is not one the store accepts.
    """
    if _NAME_RE.fullmatch(name) is None:
        raise MemoryStoreError(
            f"bad memory name {name!r}: lowercase letters, digits, and dashes only"
        )
    return name


def _index_has(state_dir: Path, name: str) -> bool:
    """Return whether the index has a `- name:` line."""
    pattern = re.compile(rf"^\s*[-*]\s*{re.escape(name)}\s*:")
    return any(pattern.match(ln) for ln in index_text(state_dir).splitlines())


# `- name: summary` (what `memory add` writes) or a link line `- [title](name.md)`.
_INDEX_LINE_RE = re.compile(
    r"^\s*[-*]\s*(?:\[[^\]]*\]\(([a-z0-9][a-z0-9-]{0,63})\.md\)|([a-z0-9][a-z0-9-]{0,63})\s*:)"
)


def index_name(line: str) -> str | None:
    """Return the fact an index line names, or None for a line that is not an entry."""
    match = _INDEX_LINE_RE.match(line)
    return None if match is None else (match.group(1) or match.group(2))


def unindexed_names(state_dir: Path) -> tuple[str, ...]:
    """Return the fact files the index does not list, sorted; only the index reaches a prompt."""
    named = {n for n in (index_name(ln) for ln in index_text(state_dir).splitlines()) if n}
    try:
        files = sorted(p.stem for p in memory_dir(state_dir).glob("*.md"))
    except OSError:
        return ()
    return tuple(n for n in files if n not in named and is_memory_name(n))


def _index_hook(index_lines: list[str], name: str) -> str | None:
    """Return the hook text the index line for the name carries, or None when there is none."""
    pattern = re.compile(rf"^\s*[-*]\s*{re.escape(name)}\s*:")
    line = next((ln for ln in index_lines if pattern.match(ln)), None)
    return None if line is None else line.split(":", 1)[1].strip()


def _index_pattern(name: str) -> re.Pattern[bytes]:
    """Return the byte pattern matching the index line for the name."""
    return re.compile(rb"^\s*[-*]\s*" + re.escape(name.encode("utf-8")) + rb"\s*:")


def _drop_index_line(state_dir: Path, name: str) -> None:
    """Remove the index line naming the fact.

    Over bytes: a rewrite through the replacing reader would turn every byte that is
    not UTF-8 into U+FFFD in lines the operator wrote.
    """
    idx = index_path(state_dir)
    pattern = _index_pattern(name)
    try:
        lines = idx.read_bytes().split(b"\n")
    except OSError:
        return
    atomic_write(idx, b"\n".join(ln for ln in lines if not pattern.match(ln)))


def _replace_index_line(state_dir: Path, name: str, hook: str) -> None:
    """Rewrite the index line naming the fact in place, over bytes, appending one when none."""
    idx = index_path(state_dir)
    pattern = _index_pattern(name)
    try:
        lines = idx.read_bytes().split(b"\n")
    except OSError:
        lines = []
    if not any(pattern.match(ln) for ln in lines):
        _append_index_line(state_dir, name, hook)
        return
    new = f"- {name}: {hook}".encode()
    atomic_write(idx, b"\n".join(new if pattern.match(ln) else ln for ln in lines))


def _append_index_line(state_dir: Path, name: str, hook: str) -> None:
    """Atomically append one line to the index."""
    idx = index_path(state_dir)
    try:
        existing = idx.read_bytes()
    except FileNotFoundError:
        existing = b""
    separator = b"\n" if existing and not existing.endswith(b"\n") else b""
    atomic_write(idx, existing + separator + f"- {name}: {hook}\n".encode())


def add(state_dir: Path, name: str, body: str) -> Path:
    """Write a fact file and append its index line, for `agent6 memory add`.

    The file is written first: an unindexed file is invisible and harmless, while an
    index line without its file is a prompt that lies.

    Args:
        state_dir: The repo's state dir.
        name: The fact's name.
        body: The fact's text; its first line is the index hook.

    Returns:
        The fact file's path.

    Raises:
        MemoryStoreError: The body is empty, the name is taken, or the file existed
            unindexed and was re-indexed from its own first line instead.
    """
    body = body.strip()
    if not body:
        raise MemoryStoreError("memory body must be non-empty")
    d = memory_dir(state_dir)
    path = d / f"{_check_name(name)}.md"
    with _locked_memory(state_dir):
        if path.exists():
            if _index_has(state_dir, name):
                raise MemoryStoreError(f"memory {name!r} exists; edit {path} or pick another name")
            first = (path.read_text(encoding="utf-8").strip().splitlines() or [""])[0]
            _append_index_line(state_dir, name, first[:120])
            raise MemoryStoreError(
                f"memory {name!r} existed but was missing from the index; re-indexed it."
                f" The body passed here was not saved; edit {path} to change it."
            )
        atomic_write(path, body + "\n")
        _append_index_line(state_dir, name, body.splitlines()[0][:120])
        _record_use_unlocked(state_dir, session="operator", wrote=(name,), created=(name,), read={})
    return path


def remove(state_dir: Path, name: str) -> None:
    """Delete a fact file and its index line, for `agent6 memory rm`.

    The index line goes first: a file with no line is invisible and a retry can still
    delete it. Either remnant alone is removable.

    Raises:
        MemoryStoreError: The name has neither a file nor an index line.
    """
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    with _locked_memory(state_dir):
        had_line = _index_has(state_dir, name)
        if not path.is_file() and not had_line:
            raise MemoryStoreError(f"no memory named {name!r}")
        if had_line:
            _drop_index_line(state_dir, name)
        if path.is_file():
            path.unlink()
        _drop_use(state_dir, name)


def show(state_dir: Path, name: str) -> str:
    """Return a fact's text.

    Raises:
        MemoryStoreError: The name is bad or the file cannot be read.
    """
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryStoreError(f"no memory named {name!r}") from exc
