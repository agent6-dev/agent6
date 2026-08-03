# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One durable, rewritable scratchpad per repo: ``<state_dir>/notes.md``.

The complement to ``agent6.memory``, not a replacement. Memories are
append-only by design so the audit trail survives an invalidation, which makes
them right for observations and wrong for anything that has to stay readable:
after twenty sessions they are a log. Notes are ONE document the agent
restructures -- strike through a resolved item, merge two, delete a section --
which is the only way a working document survives its own growth.

Whole-file replace, not patch: free restructuring is the point, and the agent
read the file moments before writing it.

Outside the workspace, beside the memories it complements: never in a diff,
never committed by accident, never mounted into the jail.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from agent6.portable import atomic_write, lock_exclusive, unlock

# What the agent may keep. The cap is the point, not a safety margin: an
# uncapped notes file silently eats the context window it is injected into, and
# refusing at the boundary is what forces the agent to curate.
NOTES_MAX_CHARS = 16_000


class NotesError(Exception):
    """A notes read/write was refused."""


def notes_path(state_dir: Path) -> Path:
    return state_dir / "notes.md"


@contextmanager
def _lock_notes(state_dir: Path) -> Generator[None]:
    """Serialize the read-modify-write, like the memory store's.

    The file is per-repo and every write REPLACES it, so two live sessions
    without this are a lost update: the second write erases the first's
    restructuring wholesale rather than merging it.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_dir / ".notes.lock", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        lock_exclusive(fd, blocking=True)
        yield
    finally:
        with contextlib.suppress(OSError):
            unlock(fd)
        os.close(fd)


def read_notes(state_dir: Path) -> str:
    """The current notes, or "" when there are none.

    A missing file is the ordinary first-session case. An unreadable one (hand
    mangled, wrong encoding) is the operator's business and must not take the
    session down, so it reads as empty -- the agent writes fresh notes over it.
    """
    try:
        return notes_path(state_dir).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def write_notes(state_dir: Path, content: str) -> int:
    """Replace the notes with *content*. Returns the character count written.

    Refuses past ``NOTES_MAX_CHARS`` rather than truncating: silently dropping
    the tail of the agent's own notes loses exactly the newest thinking, and
    the refusal is what tells it to prune.
    """
    if len(content) > NOTES_MAX_CHARS:
        raise NotesError(
            f"notes are {len(content)} chars, over the {NOTES_MAX_CHARS} cap:"
            " prune them (drop what is resolved, merge what repeats) and write again"
        )
    with _lock_notes(state_dir):
        atomic_write(notes_path(state_dir), content)
    return len(content)
