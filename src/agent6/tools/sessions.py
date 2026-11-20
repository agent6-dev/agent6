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
import re
import shlex
import stat
from collections.abc import Generator, Iterator
from typing import Any
from urllib import parse

import pydantic

from agent6 import kinds, path_safety, terminal
from agent6.sessions import journal, manifest, shells
from agent6.sessions import layout as sessions_layout
from agent6.tools import results, schema

# Who said what, from which field; settled events only (deltas repeat them), and every steer.
_SPEAKER = {
    "model.result": ("assistant", "text"),
    "session.start": ("user", "user_task"),
    "steer.injected": ("user", "text"),
    "steer.parent": ("parent", "text"),
}

# The input model checks shape; this allowlist, not the input model, grants disclosure.
_PUBLIC_ARGS: dict[str, tuple[type[pydantic.BaseModel], set[str] | dict[str, Any]]] = {
    cls.TOOL_NAME: (cls, fields)
    for cls, fields in (
        (schema.Agent6DocsInput, {"name"}),
        (schema.ApplyEditInput, {"path": True, "preview": True, "edits": {"__all__": {"kind"}}}),
        (schema.ApplyPatchInput, {"path", "preview"}),
        (schema.AskUserInput, {"questions": {"__all__": {"question", "options"}}, "wait": True}),
        (schema.DagAddTaskInput, {"parent_id", "after", "depends_on", "relevant_paths"}),
        (schema.DagListTasksInput, {"status"}),
        (schema.DagUpdateTaskInput, {"id", "status", "depends_on"}),
        (schema.FetchInput, {"url"}),
        (schema.FindDefinitionInput, {"symbol"}),
        (schema.FindReferencesInput, {"symbol"}),
        (schema.ListDirInput, {"path"}),
        (schema.ListFindingsInput, {"all"}),
        (schema.MergeChildInput, {"id"}),
        (schema.OutlineInput, {"path"}),
        (schema.ParallelInput, {"spec"}),
        (schema.ReadBackgroundInput, {"id", "tail_lines", "wait_s"}),
        (schema.ReadFileInput, {"path", "start_line", "limit"}),
        (schema.RunCommandInput, {"argv", "background"}),
        (schema.SteerChildInput, {"id"}),
        (schema.StopBackgroundInput, {"id"}),
        (schema.UseSkillInput, {"name", "file"}),
    )
}

# Some counts exist only in the summary. Match the whole outcome, never an error or tail.
_COUNT_OUTCOMES = {
    "fetch": r"[1-5][0-9]{2} · [0-9]+ bytes",
    "find_definition": r"[0-9]+ definitions(?: \(truncated\))?",
    "find_references": r"[0-9]+ references(?: \(truncated\))?",
    "list_dir": r"[0-9]+ entries(?:, [0-9]+ hidden)?(?: \(truncated\))?",
    "list_tasks": r"[0-9]+ tasks",
    "outline": r"[0-9]+ symbols(?: \(truncated\))?",
    "read_file": r"[0-9]+ bytes(?: \(truncated\))?",
}
_PREVIEW_CHARS = 200
_MAX_ROW_BYTES = 1 << 20


@dataclasses.dataclass(frozen=True, slots=True)
class Roster:
    """The sessions `read_session` lists, newest first.

    The roster is capped at `ROSTER_MAX`: uncapped, 2000 sessions render about 70k tokens.

    Attributes:
        briefs: The sessions shown.
        more: Whether older sessions were left out; `query` reaches them.
    """

    briefs: tuple[SessionBrief, ...]
    more: bool

    def lines(self) -> tuple[str, ...]:
        """Return one line per brief, plus a note when the roster was cut."""
        shown = tuple(b.line() for b in self.briefs)
        if not self.more:
            return shown
        return (*shown, f"(only the {len(shown)} newest are shown; narrow with `query`)")


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
    """Pin each component below the project state root without following links.

    Args:
        state_dir: The project's state root.
        path: The record or bucket below it.
        directory: Whether to open a directory instead of a regular file.

    Yields:
        The descriptor, valid until the context exits.

    Raises:
        NotRegularFileError: The record is not a regular file.
    """
    parts = path_safety.contain(state_dir, path.relative_to(state_dir)).rel_path.parts
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(state_dir, dir_flags)
    try:
        for i, part in enumerate(parts):
            flags = (
                dir_flags
                if directory or i < len(parts) - 1
                else os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
            )
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        if not directory and not stat.S_ISREG(os.fstat(fd).st_mode):
            raise path_safety.NotRegularFileError(f"Not a regular file: {path.name}")
        yield fd
    finally:
        os.close(fd)


def session_briefs(state_dir: pathlib.Path) -> list[SessionBrief]:
    """Return every session in this project, newest first."""
    found: list[tuple[float, SessionBrief]] = []
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
            try:
                with (
                    _state_fd(state_dir, root / sid / manifest.MANIFEST_NAME) as fd,
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
                        id=sid,
                        mode=m.mode or "?",
                        task=" ".join(m.user_task.split())[:120],
                        started=m.start_ts,
                        bucket=bucket,
                    ),
                )
            )
    return [brief for _mtime, brief in sorted(found, key=lambda pair: -pair[0])]


def _journal(layout: sessions_layout.SessionLayout) -> Iterator[dict[str, Any]]:
    """Yield journal objects in append order without following a link out of state."""
    with (
        _state_fd(layout.state_dir, layout.logs_path) as fd,
        os.fdopen(fd, "rb", closefd=False) as rows,
    ):
        while row := rows.readline(_MAX_ROW_BYTES + 1):
            if len(row) > _MAX_ROW_BYTES:
                # Skip the whole row. Its prefix is not a record the reader can project.
                while row and not row.endswith(b"\n"):
                    row = rows.readline(_MAX_ROW_BYTES)
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
        if etype == "model.result" and kinds.is_side_role(str(event.get("role", "")), mode=mode):
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
    commands: dict[tuple[str, str, int | None], int] = {}
    mode = kinds.bucket_mode(layout.subdir)
    try:
        for event in _journal(layout):
            etype = str(event.get("type", ""))
            if etype in journal.SESSION_START_EVENTS:
                commands.clear()
            if said := _speech(event, mode):
                lines.append(f"{said[0]}: {said[1]}")
            elif etype in ("tool.call", "tool.result", "verify.start", "metric.start"):
                _tool_line(event, lines, commands, layout.session_dir)
    except (OSError, path_safety.ContainmentError) as exc:
        return f"(no readable journal: {exc})"
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        # The header counts against the cap.
        header = "... {cut} earlier characters elided ...\n\n"
        kept = max(max_chars - len(header.format(cut=len(text))), 0)
        text = header.format(cut=len(text) - kept) + text[-kept:] if kept else ""
    return text or "(this session recorded no conversation)"


def _tool_line(
    event: dict[str, Any],
    lines: list[str],
    commands: dict[tuple[str, str, int | None], int],
    session_dir: pathlib.Path,
) -> None:
    """Add the call, then settle a command on that same line when its result arrives.

    Args:
        event: The tool call or result from the journal.
        lines: The conversation so far.
        commands: The unsettled command line positions in this execution.
        session_dir: The session whose shell records hold observed exits.
    """
    etype = event["type"]
    name = {"verify.start": "run_verify_command", "metric.start": "run_metric_command"}.get(
        etype, str(event.get("name", ""))
    )
    call_id = event.get("call_id")
    key = (str(event.get("execution_id") or ""), name, call_id if type(call_id) is int else None)
    if etype in ("verify.start", "metric.start"):
        if etype == "verify.start":
            # A harness gate has no tool call to label.
            index = commands.get(key) if type(call_id) is int else None
        else:
            # A metric start has no call id. Do not guess if more than one call is open.
            pending = [i for (_, tool, _), i in commands.items() if tool == name]
            index = pending[0] if len(pending) == 1 else None
        if index is not None:
            lines[index] = _command_line(name, event.get("command"))
    elif etype == "tool.call":
        line = f"[tool] {name}"
        if name in ("run_command", "run_verify_command", "run_metric_command"):
            commands[key] = len(lines)
            args = event.get("args_preview", event.get("args"))
            argv = args.get("argv") if isinstance(args, dict) else None
            line = _command_line(name, argv)
        lines.append(line)
    elif (index := commands.pop(key, None)) is not None:
        if type(code := event.get("exit_code")) is int:
            lines[index] += f" (exit {code})"
        elif isinstance(job := event.get("background_id"), str) and job:
            record = shells.result_record(session_dir / shells.SHELLS_DIR, job) or {}
            label = terminal.line_preview(job)
            if type(code := record.get("returncode")) is int:
                lines[index] += f" ({label}, exit {code})"
            else:
                lines[index] += f" (handed off as {label})"
        elif event.get("ok") is False:
            lines[index] += " (refused)"


def _command_line(name: str, argv: object) -> str:
    """Return a tool label with one bounded line of its recorded command."""
    label = f"[tool] {name}"
    if isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv):
        return f"{label} {terminal.line_preview(shlex.join(argv))}"
    return label


def _plain_values(value: object) -> bool:
    """Return whether projected text fields have no object or array prefixes."""
    if isinstance(value, list):
        return all(_plain_values(item) for item in value)
    if isinstance(value, dict):
        return all(_plain_values(item) for item in value.values())
    return not isinstance(value, str) or not value.lstrip().startswith(("[", "{"))


def _arg_preview(key: str, value: Any) -> str | None:
    """Return a projected field, or None when its shape is ambiguous."""
    if key == "argv" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
        return None
    if not _plain_values(value):
        return None
    if key == "argv":
        text = shlex.join(value)
    elif key == "url":
        # A clipped authority can end before the @. Never echo an unparseable prefix.
        if (
            not isinstance(value, str)
            or "…" in value
            or "\\" in value
            or any(c.isspace() or not c.isprintable() for c in value)
        ):
            return None
        try:
            parts = parse.urlsplit(value)
            host, port = parts.hostname, parts.port
        except ValueError:
            return None
        if not host or parts.scheme not in ("http", "https"):
            return None
        authority = f"[{host}]" if ":" in host else host
        if port is not None:
            authority += f":{port}"
        text = parse.urlunsplit((parts.scheme, authority, parts.path, "", ""))
    else:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return terminal.line_preview(text, limit=_PREVIEW_CHARS)


def _call_arguments(name: str, args: object) -> Iterator[str]:
    """Yield only recorded fields that pass the tool's disclosure projection."""
    projection = _PUBLIC_ARGS.get(name)
    if projection is None or not isinstance(args, dict):
        return
    cls, fields = projection
    try:
        declared = {k: v for k, v in args.items() if k in cls.model_fields}
        call = cls.model_validate(declared, extra="ignore")
    except pydantic.ValidationError:
        return
    for key, value in call.model_dump(mode="json", include=fields, exclude_unset=True).items():
        if key in args and (shown := _arg_preview(key, value)) is not None:
            yield f"{key}={shown}"


def _outcome(name: str, event: dict[str, Any]) -> str:
    """Return the recorded disposition or a known built-in quantity."""
    ok = event.get("ok")
    if ok is False:
        return "refused"
    if (
        name in ("run_command", "run_verify_command", "run_metric_command")
        and type(code := event.get("exit_code")) is int
    ):
        return f"exit {code}"
    summary = event.get("summary")
    if (
        ok is True
        and (pattern := _COUNT_OUTCOMES.get(name)) is not None
        and isinstance(summary, str)
        and len(summary) <= _PREVIEW_CHARS
        and re.fullmatch(pattern, summary)
    ):
        return summary
    if name == "read_file" and type(count := event.get("lines_total")) is int:
        return f"{count} lines"
    return "ok" if ok is True else "unknown"


def _entry(event: dict[str, Any], mode: str) -> tuple[str, str] | None:
    """Return the kind and visible text of one numbered journal entry."""
    if said := _speech(event, mode):
        speaker, body = said
        if speaker == "parent":
            body = f"parent: {body}"
        return "operator" if speaker == "user" else "model", body
    etype = event.get("type")
    if etype == "metric.start" or (etype == "verify.start" and type(event.get("call_id")) is int):
        name = "run_metric_command" if etype == "metric.start" else "run_verify_command"
        argv = _arg_preview("argv", event.get("command"))
        return "tool", f"{name} argv={argv}" if argv is not None else name
    name = event.get("name")
    # The lookup is journaled before it runs. Its own traffic is never a lookup target.
    if not isinstance(name, str) or name == "read_session":
        return None
    label = terminal.line_preview(name, limit=_PREVIEW_CHARS)
    if etype == "tool.call":
        args = event.get("args_preview", event.get("args"))
        return "tool", " ".join((label, *_call_arguments(name, args)))
    if etype == "tool.result":
        shown = terminal.line_preview(_outcome(name, event), limit=_PREVIEW_CHARS)
        return "tool", f"{label} result: {shown}"
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
    lines: list[str] = []
    remaining = args.max_chars
    next_start = None
    number = 0
    mode = kinds.bucket_mode(layout.subdir)
    needle = args.query.casefold()
    try:
        for event in _journal(layout):
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
            if len(lines) == args.limit or (lines and len(line) > remaining):
                next_start = number
                break
            if len(line) > remaining:
                line = line[: remaining - 1] + "…"
            lines.append(line)
            remaining -= len(line) + 1
    except (OSError, path_safety.ContainmentError) as exc:
        return results.SessionEntriesResult(f"(no readable journal: {exc})"[: args.max_chars])
    return results.SessionEntriesResult("\n".join(lines) or "(no matching entries)", next_start)


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
    for brief in briefs:
        if len(hits) > schema.ROSTER_MAX:
            break  # one past the cap: enough to know there are more
        layout = sessions_layout.SessionLayout(state_dir, brief.id, brief.bucket)
        if needle in brief.task.casefold() or _journal_contains(layout, needle):
            hits.append(brief)
    return Roster(briefs=tuple(hits[: schema.ROSTER_MAX]), more=len(hits) > schema.ROSTER_MAX)


def _journal_contains(layout: sessions_layout.SessionLayout, needle: str) -> bool:
    """Return whether a visible entry matches. A roster hit discloses the match."""
    mode = kinds.bucket_mode(layout.subdir)
    try:
        return any(
            needle in entry[1].casefold()
            for event in _journal(layout)
            if (entry := _entry(event, mode))
        )
    except (OSError, path_safety.ContainmentError):
        return False
