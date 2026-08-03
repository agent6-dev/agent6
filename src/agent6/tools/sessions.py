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

from agent6.runs.layout import LOGS_NAME, RUN_BUCKETS, RunLayout
from agent6.runs.manifest import ManifestError, read_manifest

# What a reader needs from another session: who said what, and from which
# field. Deltas are the same prose arriving in pieces, so only the settled
# events are folded. The operator speaks twice: the task, and every steer --
# without the steers the transcript reads as if the session went that way on
# its own.
_SPEAKER = {
    "role.result": ("assistant", "text"),
    "run.start": ("user", "user_task"),
    "loop.steer.injected": ("user", "text"),
}


# A roster is context the model pays for on every call, so it is capped. The
# newest sessions are the ones a reader wants; `query` is how you reach an older
# one. 2000 sessions rendered ~70k tokens before this.
ROSTER_MAX = 40


@dataclass(frozen=True, slots=True)
class Roster:
    """What `read_session` lists, and whether it is the whole story."""

    briefs: tuple[SessionBrief, ...]
    more: bool

    def lines(self) -> tuple[str, ...]:
        shown = tuple(b.line() for b in self.briefs)
        if not self.more:
            return shown
        return (*shown, f"(only the {len(shown)} newest are shown; narrow with `query`)")


@dataclass(frozen=True, slots=True)
class SessionBrief:
    """One session as the roster shows it."""

    id: str
    mode: str
    task: str
    started: str
    # Which bucket it lives in. Carried rather than re-resolved: looking it up
    # per brief re-scanned every bucket, which made a query O(N^2) -- 44s at
    # 2000 sessions, with the loop blocked the whole time.
    bucket: str

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
                        bucket=bucket,
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
        speaker, field = said
        body = str(event.get(field, "")).strip()
        if body:
            lines.append(f"{speaker}: {body}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        # The header counts against the cap: added on top of a max_chars slice,
        # the result was longer than the caller asked for.
        header = "... {cut} earlier characters elided ...\n\n"
        kept = max(max_chars - len(header.format(cut=len(text))), 0)
        text = header.format(cut=len(text) - kept) + text[-kept:] if kept else ""
    return text or "(this session recorded no conversation)"


def roster(state_dir: Path, query: str) -> Roster:
    """The sessions to show, newest first: every one, or those matching *query*.

    A query matches the task or anything said in the session.
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
            state_dir / brief.bucket / brief.id / LOGS_NAME, needle
        ):
            hits.append(brief)
    return Roster(briefs=tuple(hits[:ROSTER_MAX]), more=len(hits) > ROSTER_MAX)


def _file_contains(path: Path, needle: str) -> bool:
    """Whether *path* contains *needle*, read in chunks.

    A real journal reaches megabytes (every streamed delta is persisted), and
    reading whole ones into memory to answer a yes/no was ~1 GB per call across
    a couple of hundred sessions.
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
