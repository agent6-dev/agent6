# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read this project's other sessions.

A run, a plan and an ask are all sessions, and their journals sit side by side under the
project's state dir. Read-only, and confined to the state dir by construction: a session is
named by id, resolved against the buckets on disk, so no path from the model reaches the
filesystem. The journals hold conversations, not credentials.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
from typing import Any

from agent6 import kinds, terminal
from agent6.sessions import journal, manifest
from agent6.sessions import layout as sessions_layout
from agent6.tools import schema

# Who said what, from which field; settled events only (deltas repeat them), and every steer.
_SPEAKER = {
    "model.result": ("assistant", "text"),
    "session.start": ("user", "user_task"),
    "steer.injected": ("user", "text"),
    "steer.parent": ("parent", "text"),
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
    try:
        raw = layout.logs_path.read_text(errors="replace")
    except OSError as exc:
        return f"(no readable journal: {exc})"
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn last line on a live session
        if not isinstance(event, dict):
            continue
        etype = str(event.get("type", ""))
        if etype in journal.SESSION_START_EVENTS:
            commands.clear()
        said = _SPEAKER.get(etype)
        if said is None:
            if etype in ("tool.call", "tool.result", "verify.start", "metric.start"):
                _tool_line(event, lines, commands)
            continue
        if etype == "model.result" and kinds.is_side_role(
            str(event.get("role", "")), mode=kinds.bucket_mode(layout.subdir)
        ):
            continue  # a side call's answer, not this session's own
        speaker, field = said
        body = str(event.get(field, "")).strip()
        if body:
            lines.append(f"{speaker}: {body}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        # The header counts against the cap.
        header = "... {cut} earlier characters elided ...\n\n"
        kept = max(max_chars - len(header.format(cut=len(text))), 0)
        text = header.format(cut=len(text) - kept) + text[-kept:] if kept else ""
    return text or "(this session recorded no conversation)"


def _tool_line(
    event: dict[str, Any], lines: list[str], commands: dict[tuple[str, str, int | None], int]
) -> None:
    """Add the call, then settle a command on that same line when its result arrives.

    Args:
        event: The tool call or result from the journal.
        lines: The conversation so far.
        commands: The unsettled command line positions in this execution.
    """
    etype = event["type"]
    name = str(event.get("name", ""))
    if etype == "verify.start":
        name = "run_verify_command"
    elif etype == "metric.start":
        name = "run_metric_command"
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
        elif event.get("background_id"):
            lines[index] += f" (running as {terminal.line_preview(str(event['background_id']))})"
        elif event.get("ok") is False:
            lines[index] += " (refused)"


def _command_line(name: str, argv: object) -> str:
    """Return a tool label with one bounded line of its recorded command."""
    label = f"[tool] {name}"
    if isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv):
        return f"{label} {terminal.line_preview(shlex.join(argv))}"
    return label


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
