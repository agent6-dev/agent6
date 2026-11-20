# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read this project's other sessions.

A run, a plan and an ask are all sessions, and their journals sit side by side under the
project's state dir. Read-only, and confined to the state dir by construction: a session is
named by id, resolved against the buckets on disk, so no path from the model reaches the
filesystem. Entry reads open the journal, not provider transcripts or credentials.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import os
import pathlib
from collections.abc import Generator, Iterator
from typing import Any

import pydantic

from agent6 import kinds, path_safety, terminal
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest, shells
from agent6.tools import results, schema

# Who said what, from which field; settled events only (deltas repeat them), and every steer.
_SPEAKER = {
    "model.result": ("assistant", "text"),
    "session.start": ("user", "user_task"),
    "steer.injected": ("user", "text"),
    "steer.parent": ("parent", "text"),
}

_PREVIEW_CHARS = 200
_MAX_ROW_BYTES = 1 << 20
_OVERSIZE_NOTE = "(scan incomplete: one or more journal records over 1 MiB omitted)"


@dataclasses.dataclass(frozen=True, slots=True)
class Roster:
    """The sessions `read_session` lists, newest first.

    The roster is capped at `ROSTER_MAX`: uncapped, 2000 sessions render about 70k tokens.

    Attributes:
        briefs: The sessions shown.
        more: Whether older sessions were left out; `query` reaches them.
        incomplete: A skipped record could have matched the query.
    """

    briefs: tuple[SessionBrief, ...]
    more: bool
    incomplete: bool = False

    def lines(self) -> tuple[str, ...]:
        """Return one line per brief, plus a note when the roster was cut."""
        shown = tuple(b.line() for b in self.briefs)
        if self.more:
            shown += (f"(only the {len(shown)} newest are shown; narrow with `query`)",)
        return (*shown, _OVERSIZE_NOTE) if self.incomplete else shown


@dataclasses.dataclass(frozen=True, slots=True)
class SessionBrief:
    """One session as the roster shows it.

    Attributes:
        id: The session id.
        mode: The session's mode.
        task: The first 120 characters of the task.
        started: The start timestamp.
        bucket: The bucket the session lives in; re-resolving it per brief makes a query O(N^2).
    """

    id: str
    mode: str
    task: str
    started: str
    bucket: str

    def line(self) -> str:
        """Return the brief's roster line."""
        when = f" · {self.started[:16]}" if self.started else ""
        return f"[{self.id}] {self.mode}{when}: {self.task}"


@contextlib.contextmanager
def _state_fd(
    state_dir: pathlib.Path, path: pathlib.Path, *, directory: bool = False
) -> Generator[int]:
    """Open a record or bucket below the project state root without following links.

    Args:
        state_dir: The project's state root.
        path: The record or bucket below it.
        directory: Whether to open a directory instead of a regular file.

    Yields:
        The descriptor, valid until the context exits.
    """
    fd = path_safety.open_contained(
        path_safety.contain(state_dir, path.relative_to(state_dir)),
        os.O_RDONLY | (os.O_DIRECTORY if directory else 0),
    )
    try:
        yield fd
    finally:
        os.close(fd)


def _directories(state_dir: pathlib.Path) -> Iterator[tuple[sessions_layout.SessionLayout, float]]:
    """Yield session directories and their modification times without following links."""
    for bucket in sessions_layout.SESSION_BUCKETS:
        root = sessions_layout.bucket_dir(state_dir, bucket)
        try:
            with _state_fd(state_dir, root, directory=True) as fd, os.scandir(fd) as entries:
                dirs = [
                    (e.name, e.stat(follow_symlinks=False).st_mtime)
                    for e in entries
                    if e.is_dir(follow_symlinks=False)
                ]
        except (OSError, path_safety.ContainmentError):
            continue
        for sid, mtime in dirs:
            yield sessions_layout.SessionLayout(state_dir, sid, subdir=bucket), mtime


def resolve_reader_id(
    state_dir: pathlib.Path, session_id: str
) -> sessions_layout.SessionLayout | None:
    """Resolve an exact id or an unambiguous prefix inside the reader's project.

    Args:
        state_dir: The project state directory.
        session_id: The id or prefix supplied to the tool.

    Returns:
        The one matching session, or None.
    """
    matches = [
        layout for layout, _ in _directories(state_dir) if layout.session_id.startswith(session_id)
    ]
    matches = [layout for layout in matches if layout.session_id == session_id] or matches
    return matches[0] if len(matches) == 1 else None


def session_briefs(state_dir: pathlib.Path) -> list[SessionBrief]:
    """Return every session in this project, newest first."""
    found: list[tuple[float, SessionBrief]] = []
    for layout, mtime in _directories(state_dir):
        try:
            with (
                _state_fd(state_dir, layout.manifest_path) as fd,
                os.fdopen(fd, "rb", closefd=False) as record,
            ):
                m = manifest.SessionManifest.model_validate_json(record.read())
        except (OSError, path_safety.ContainmentError, pydantic.ValidationError):
            continue
        if m.version > manifest.MANIFEST_VERSION:
            continue
        found.append(
            (
                mtime,
                SessionBrief(
                    id=layout.session_id,
                    mode=m.mode or "?",
                    task=" ".join(m.user_task.split())[:120],
                    started=m.start_ts,
                    bucket=layout.subdir,
                ),
            )
        )
    return [brief for _mtime, brief in sorted(found, key=lambda pair: -pair[0])]


def _journal(layout: sessions_layout.SessionLayout) -> Iterator[dict[str, Any] | None]:
    """Yield journal objects in append order, or None for a record too large to inspect."""
    with (
        _state_fd(layout.state_dir, layout.logs_path) as fd,
        os.fdopen(fd, "rb", closefd=False) as rows,
    ):
        while row := rows.readline(_MAX_ROW_BYTES + 1):
            if len(row) > _MAX_ROW_BYTES:
                # Skip the whole row. Its prefix is not a record the reader can project.
                while row and not row.endswith(b"\n"):
                    row = rows.readline(_MAX_ROW_BYTES)
                yield None
                continue
            try:
                event = json.loads(row.decode("utf-8", errors="replace"))
            except (ValueError, RecursionError):
                continue  # a malformed record or a torn line in a live journal
            if isinstance(event, dict):
                yield event


def _speech(event: dict[str, Any], mode: str) -> tuple[str, str] | None:
    """Return settled speech, excluding side calls and streaming deltas."""
    etype = str(event.get("type", ""))
    if said := _SPEAKER.get(etype):
        speaker, field = said
        if etype == "model.result" and kinds.is_side_actor(str(event.get("actor", "")), mode=mode):
            return None
        body = event.get(field)
        if isinstance(body, str) and body.strip():
            return speaker, body.strip()
    return None


def conversation(layout: sessions_layout.SessionLayout, *, max_chars: int) -> str:
    """Return a session's conversation as plain text, oldest first.

    Truncation keeps the tail: the conclusion is what a later session wants, and the head is
    the task the roster already carries.

    Args:
        layout: The session's layout on disk.
        max_chars: The cap on the returned text, header included.

    Returns:
        The conversation, or a note when the journal is unreadable or empty.
    """
    lines: list[str] = []
    mode = kinds.bucket_mode(layout.subdir)
    incomplete = False
    try:
        for event in _journal(layout):
            if event is None:
                incomplete = True
                continue
            if said := _speech(event, mode):
                lines.append(f"{said[0]}: {said[1]}")
            elif tool := _tool_text(event):
                lines.append(f"[tool] {tool}")
                job = event.get("background_id")
                if event.get("type") == "tool.result" and isinstance(job, str):
                    record = shells.result_record(layout.session_dir / shells.SHELLS_DIR, job) or {}
                    if type(code := record.get("returncode")) is int:
                        lines.append(f"[shell] {job}: exit {code}")
    except (OSError, path_safety.ContainmentError) as exc:
        return f"(no readable journal: {exc})"
    if incomplete:
        lines.append(_OVERSIZE_NOTE)
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        # The header counts against the cap.
        header = "... {cut} earlier characters elided ...\n\n"
        kept = max(max_chars - len(header.format(cut=len(text))), 0)
        text = header.format(cut=len(text) - kept) + text[-kept:] if kept else ""
    return text or "(this session recorded no conversation)"


def _value_text(value: Any) -> str:
    """Return one bounded line of a recorded argument or summary."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except RecursionError:
        return "(value too deeply nested)"
    return terminal.line_preview(text.replace("\n", r"\n"), limit=_PREVIEW_CHARS)


def _tool_text(event: dict[str, Any]) -> str | None:
    """Return a journaled tool call, configured command or result for both readers."""
    etype = event.get("type")
    if etype == "metric.start" or (etype == "verify.start" and type(event.get("call_id")) is int):
        name = "run_metric_command" if etype == "metric.start" else "run_verify_command"
        return f"{name} argv={_value_text(event['command'])}" if "command" in event else name
    name = event.get("name")
    if not isinstance(name, str):
        return None
    label = _value_text(name)
    if etype == "tool.call":
        args = event.get("args_preview")
        if isinstance(args, dict):
            return " ".join(
                [label, *(f"{_value_text(k)}={_value_text(v)}" for k, v in args.items())]
            )
        return label
    if etype == "tool.result":
        return f"{label} result: {_value_text(event.get('summary', '(no recorded summary)'))}"
    return None


def _entry(event: dict[str, Any], mode: str) -> tuple[str, str] | None:
    """Return the kind and visible text of one numbered journal entry."""
    if said := _speech(event, mode):
        speaker, body = said
        kind = "operator" if speaker == "user" else "model"
        if event.get("type") == "session.start":
            source = event.get("task_source")
            kind = source if source in ("operator", "model", "panel") else "task"
        elif speaker == "parent":
            body = f"parent: {body}"
        return kind, body
    # The lookup is journaled before it runs. Its own traffic is never a lookup target.
    if event.get("name") != "read_session" and (tool := _tool_text(event)):
        return "tool", tool
    return None


def read_entries(
    layout: sessions_layout.SessionLayout, args: schema.ReadSessionInput
) -> results.SessionEntriesResult:
    """Read a bounded page, numbering the entries before filtering.

    Args:
        layout: The session resolved from the project's roster of ids.
        args: The entry range, filters and character budget.

    Returns:
        The page, plus the first omitted match's number when there is another page.
    """
    matches: list[tuple[int, str]] = []
    used = 0
    incomplete = False
    next_start = None
    number = 0
    mode = kinds.bucket_mode(layout.subdir)
    needle = args.query.casefold()
    try:
        for event in _journal(layout):
            if event is None:
                incomplete = True
                continue
            if not (entry := _entry(event, mode)):
                continue
            number += 1
            kind, body = entry
            if number < args.start or (args.kinds and kind not in args.kinds):
                continue
            try:
                when = datetime.datetime.fromisoformat(str(event.get("ts", ""))).strftime("%H:%M")
            except ValueError:
                when = "--:--"
            line = f"#{number} {when} {kind}: {body}"
            if needle and needle not in line.casefold():
                continue
            cost = len(line) + bool(matches)
            if len(matches) == args.limit or (matches and used + cost > args.max_chars):
                next_start = number
                break
            matches.append((number, line))
            used += cost
    except (OSError, path_safety.ContainmentError) as exc:
        return results.SessionEntriesResult(f"(no readable journal: {exc})"[: args.max_chars])
    if not matches:
        return results.SessionEntriesResult(
            _OVERSIZE_NOTE if incomplete else "(no matching entries)"
        )
    prefix = _OVERSIZE_NOTE + "\n" if incomplete else ""
    # A late notice may displace matches. Put whole entries on the next page when possible.
    while len(matches) > 1 and used + len(prefix) > args.max_chars:
        next_start, last = matches.pop()
        used -= len(last) + 1
    text = prefix + "\n".join(line for _, line in matches)
    if len(text) > args.max_chars:
        text = text[: args.max_chars - 1] + "…"
    return results.SessionEntriesResult(text, next_start)


def roster(state_dir: pathlib.Path, query: str) -> Roster:
    """Return the sessions to show, newest first: every one, or those matching a query.

    Args:
        state_dir: The project's state dir.
        query: A substring matched against the task or anything said in the session.

    Returns:
        The roster, cut at `ROSTER_MAX`.
    """
    briefs = session_briefs(state_dir)
    if not query:
        return Roster(
            briefs=tuple(briefs[: schema.ROSTER_MAX]), more=len(briefs) > schema.ROSTER_MAX
        )
    needle = query.casefold()
    hits: list[SessionBrief] = []
    incomplete = False
    for brief in briefs:
        if len(hits) > schema.ROSTER_MAX:
            break  # one past the cap: enough to know there are more
        layout = sessions_layout.SessionLayout(state_dir, brief.id, subdir=brief.bucket)
        found = needle in brief.task.casefold() or _journal_contains(layout, needle)
        if found:
            hits.append(brief)
        elif found is None:
            incomplete = True
    return Roster(tuple(hits[: schema.ROSTER_MAX]), len(hits) > schema.ROSTER_MAX, incomplete)


def _journal_contains(layout: sessions_layout.SessionLayout, needle: str) -> bool | None:
    """Return whether a visible entry matches, or None if a skipped record could have matched."""
    mode = kinds.bucket_mode(layout.subdir)
    miss: bool | None = False
    try:
        for event in _journal(layout):
            if event is None:
                miss = None
            elif (entry := _entry(event, mode)) and needle in entry[1].casefold():
                return True
    except (OSError, path_safety.ContainmentError):
        return False
    return miss
