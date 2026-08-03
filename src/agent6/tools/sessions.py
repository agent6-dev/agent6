# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read this project's other sessions.

A run, a plan and an ask are all sessions, and their journals sit side by side
under the project's state dir. Without this the model can only see its own: to
use what an earlier session worked out, an operator had to copy it by hand.

Read-only, and confined to the state dir by construction -- a session is named
by id, resolved against the buckets on disk, so no path from the model reaches
the filesystem. The journals hold conversations, not credentials: secrets are
never written to a transcript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agent6.runs.layout import RUN_BUCKETS, RunLayout, session_layout
from agent6.runs.manifest import ManifestError, read_manifest

# What a reader needs from another session: who said what. Deltas are the same
# prose arriving in pieces, so only the settled events are folded.
_SPEAKER = {"role.result": "assistant", "run.start": "user"}


@dataclass(frozen=True, slots=True)
class SessionBrief:
    """One session as the roster shows it."""

    id: str
    mode: str
    task: str
    started: str

    def line(self) -> str:
        when = f" · {self.started[:16]}" if self.started else ""
        return f"[{self.id}] {self.mode}{when}: {self.task}"


def session_briefs(state_dir: Path) -> list[SessionBrief]:
    """Every session in this project, newest first."""
    found: list[tuple[float, SessionBrief]] = []
    for bucket in RUN_BUCKETS:
        root = state_dir / bucket
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
                    ),
                )
            )
    return [brief for _mtime, brief in sorted(found, key=lambda pair: -pair[0])]


def conversation(layout: RunLayout, *, max_chars: int) -> str:
    """*layout*'s conversation as plain text, oldest first, tail-truncated.

    Truncation keeps the TAIL: a session's conclusion is what a later one
    usually wants, and the head is the task the roster already carries.
    """
    lines: list[str] = []
    journal = layout.run_dir / "events.jsonl"
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
        speaker = _SPEAKER.get(etype)
        if speaker == "user":
            body = str(event.get("user_task", ""))
        elif speaker == "assistant":
            body = str(event.get("text", ""))
        elif etype == "tool.call":
            lines.append(f"[tool] {event.get('name', '')}")
            continue
        else:
            continue
        body = body.strip()
        if body:
            lines.append(f"{speaker}: {body}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        cut = len(text) - max_chars
        text = f"... {cut} earlier characters elided ...\n\n{text[-max_chars:]}"
    return text or "(this session recorded no conversation)"


def matching_sessions(state_dir: Path, query: str) -> list[SessionBrief]:
    """The sessions whose task or journal contains *query*, newest first."""
    needle = query.lower()
    hits: list[SessionBrief] = []
    for brief in session_briefs(state_dir):
        if needle in brief.task.lower():
            hits.append(brief)
            continue
        layout = session_layout(state_dir, brief.id)
        if layout is None:
            continue
        try:
            raw = (layout.run_dir / "events.jsonl").read_text(errors="replace")
        except OSError:
            continue
        if needle in raw.lower():
            hits.append(brief)
    return hits
