# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read this project's other sessions.

A run, a plan and an ask are all sessions, and their journals sit side by side under the
project's state dir. Read-only, and confined to the state dir by construction: a session is
named by id, resolved against the buckets on disk, so no path from the model reaches the
filesystem. The journals hold conversations, not credentials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent6.kinds import is_side_role
from agent6.sessions.layout import LOGS_NAME, SESSION_BUCKETS, SessionLayout, bucket_dir
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.schema import ROSTER_MAX

# Who said what, from which field; settled events only (deltas repeat them), and every steer.
_SPEAKER = {
    "role.result": ("assistant", "text"),
    "session.start": ("user", "user_task"),
    "loop.steer.injected": ("user", "text"),
}


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


def session_briefs(state_dir: Path) -> list[SessionBrief]:
    """Return every session in this project, newest first."""
    found: list[tuple[float, SessionBrief]] = []
    for bucket in SESSION_BUCKETS:
        root = bucket_dir(state_dir, bucket)
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            try:
                m = read_manifest(d)
            except ManifestError:
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


def conversation(layout: SessionLayout, *, max_chars: int) -> str:
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
    journal = layout.logs_path
    try:
        raw = journal.read_text(errors="replace")
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
        said = _SPEAKER.get(etype)
        if said is None:
            if etype == "tool.call":
                lines.append(f"[tool] {event.get('name', '')}")
            continue
        if etype == "role.result" and is_side_role(str(event.get("role", ""))):
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


def roster(state_dir: Path, query: str) -> Roster:
    """Return the sessions to show, newest first: every one, or those matching a query.

    Args:
        state_dir: The project's state dir.
        query: A substring matched against the task or anything said in the session.

    Returns:
        The roster, cut at `ROSTER_MAX`.
    """
    briefs = session_briefs(state_dir)
    if not query:
        return Roster(briefs=tuple(briefs[:ROSTER_MAX]), more=len(briefs) > ROSTER_MAX)
    needle = query.lower()
    hits: list[SessionBrief] = []
    for brief in briefs:
        if len(hits) > ROSTER_MAX:
            break  # one past the cap: enough to know there are more
        if needle in brief.task.lower() or _file_contains(
            bucket_dir(state_dir, brief.bucket) / brief.id / LOGS_NAME, needle
        ):
            hits.append(brief)
    return Roster(briefs=tuple(hits[:ROSTER_MAX]), more=len(hits) > ROSTER_MAX)


def _file_contains(path: Path, needle: str) -> bool:
    """Return whether the file contains the needle, reading in chunks.

    A journal reaches megabytes; reading whole ones costs about 1 GB per query over a few
    hundred sessions.
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
