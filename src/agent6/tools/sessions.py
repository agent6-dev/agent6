# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read this project's other sessions.

A run, a plan and an ask are all sessions, and their journals sit side by side under the
project's state dir. Read-only, and confined to the state dir by construction: a session is
named by id, resolved against the buckets on disk, so no path from the model reaches the
filesystem. Entry reads open the journal, not provider transcripts or credentials.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import shlex
from collections.abc import Iterator
from typing import Any

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

# The journal also records rejected calls. Only declared arguments belong in another session.
_ARG_FIELDS = {
    cls.TOOL_NAME: cls.model_fields.keys()
    for cls in (
        *schema.ALL_TOOLS,
        *schema.LOOP_EXTRA_TOOLS,
        *schema.PLAN_EXTRA_TOOLS,
        *schema.ASK_EXTRA_TOOLS,
    )
}


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


def session_briefs(state_dir: pathlib.Path) -> list[SessionBrief]:
    """Return every session in this project, newest first."""
    found: list[tuple[float, SessionBrief]] = []
    for bucket in sessions_layout.SESSION_BUCKETS:
        root = sessions_layout.bucket_dir(state_dir, bucket)
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            try:
                m = manifest.read_manifest(d)
            except manifest.ManifestError:
                continue
            found.append(
                (
                    d.stat().st_mtime,
                    SessionBrief(
                        id=d.name,
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
    path = layout.logs_path
    location = path_safety.SafePath(layout.state_dir, path.relative_to(layout.state_dir), path)
    with os.fdopen(path_safety.open_contained(location, os.O_RDONLY), errors="replace") as rows:
        for row in rows:
            try:
                event = json.loads(row)
            except ValueError:
                continue  # a torn line in a live journal
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


def _declared_value(key: str, value: Any) -> Any:
    """Return only the declared fields of nested arguments."""
    if isinstance(value, list):
        return [_declared_value(key, item) for item in value]
    if isinstance(value, dict):
        nested = {"edits": schema.EditPair, "questions": schema.UserQuestion}.get(key)
        if nested is None:
            return "(object)"
        return {k: _declared_value(k, v) for k, v in value.items() if k in nested.model_fields}
    return value


def _arg_preview(key: str, value: Any) -> str:
    """Return one line of a declared argument, without opaque object fields."""
    if key == "argv" and isinstance(value, list) and all(isinstance(v, str) for v in value):
        text = shlex.join(value)
    elif key == "result":
        # A machine's structured result has its own schema, not a tool argument schema.
        return "(object)"
    else:
        plain = _declared_value(key, value)
        text = plain if isinstance(plain, str) else json.dumps(plain, ensure_ascii=False)
    return terminal.line_preview(text, limit=200)


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
        return "tool", f"{name} argv={_arg_preview('argv', event.get('command'))}"
    name = event.get("name")
    if not isinstance(name, str):
        return None
    if etype == "tool.call":
        fields = _ARG_FIELDS.get(name, ())
        args = event.get("args_preview", event.get("args"))
        pairs = (
            [f"{k}={_arg_preview(k, v)}" for k, v in args.items() if k in fields]
            if isinstance(args, dict)
            else []
        )
        return "tool", " ".join((name, *pairs))
    if etype == "tool.result":
        summary = event.get("summary")
        if not isinstance(summary, str):
            summary = f"exit {event['exit_code']}" if type(event.get("exit_code")) is int else ""
        return "tool", f"{name} result: {terminal.line_preview(summary, limit=200)}".rstrip()
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
    needle = query.lower()
    hits: list[SessionBrief] = []
    for brief in briefs:
        if len(hits) > schema.ROSTER_MAX:
            break  # one past the cap: enough to know there are more
        if needle in brief.task.lower() or _file_contains(
            sessions_layout.bucket_dir(state_dir, brief.bucket)
            / brief.id
            / sessions_layout.LOGS_NAME,
            needle,
        ):
            hits.append(brief)
    return Roster(briefs=tuple(hits[: schema.ROSTER_MAX]), more=len(hits) > schema.ROSTER_MAX)


def _file_contains(path: pathlib.Path, needle: str) -> bool:
    """Return whether the file contains the needle, reading in chunks.

    A journal reaches megabytes; reading whole ones costs about 1 GB per query over a few
    hundred sessions.

    Args:
        path: A session's journal.
        needle: The lowercase text to find; the file's case is ignored.

    Returns:
        Whether the needle occurs; False when the file cannot be read.
    """
    overlap = len(needle)
    try:
        with path.open("r", errors="replace") as fh:
            tail = ""
            while chunk := fh.read(1 << 16):
                if needle in (tail + chunk).lower():
                    return True
                tail = chunk[-overlap:] if overlap else ""
    except OSError:
        return False
    return False
