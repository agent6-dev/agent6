# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory add/list/show/rm/decisions` commands.

Store refusals (a bad name, an unreadable store) raise `MemoryStoreError`, an
`OperatorError` the cli_main boundary presents; no per-command arms.
"""

from __future__ import annotations

import pathlib
import sys

from agent6 import memory, paths
from agent6.ui.cli import _steer


def _state() -> pathlib.Path:
    """Return the cwd repo's state dir, with an old `memory/DECISIONS.md` moved out of the store.

    The move, or a stray file ignored, is reported on stderr: stdout is the command's output.
    """
    state = paths.state_dir(pathlib.Path.cwd())
    if note := memory.migrate_legacy_decisions(state):
        print(note, file=sys.stderr)
    return state


def _cmd_memory_add(name: str, body: str) -> int:
    """Write a memory and print its path.

    Returns:
        The exit code, 0.
    """
    path = memory.add(_state(), name, body)
    print(f"wrote {path}")
    return 0


def _cmd_memory_list() -> int:
    """Print the rendered index, each entry's use record, then the store files that are not facts.

    A run's prompt gets the index clipped; this prints all of it.

    Returns:
        The exit code, 0.
    """
    state = _state()
    entries = memory.index_entries(state)
    if not entries:
        print(f"(no memories; files live under {memory.memory_dir(state)})")
    use = memory.read_use(state)
    for name, first in entries:
        print(f"- {name}: {first}")
        print(f"    {format_use(use.get(name, memory.MemoryUse()))}")
    if ignored := memory.ignored_names(state):
        print(f"ignored (not a fact file): {', '.join(ignored)}")
    return 0


def format_use(use: memory.MemoryUse) -> str:
    """Return a memory's use record as one line.

    The parts are `written <date> by <session>`, `edited <date> by <session>` and
    `read once|N times, last <date> by <session>` or `never read`; a part the record does
    not hold is left out.
    """
    parts: list[str] = []
    created, updated = use.created, use.updated
    if created is not None:
        parts.append(f"written {_when_by(created)}")
    if updated is not None and updated != created:
        parts.append(f"edited {_when_by(updated)}")
    if use.reads:
        times = "once" if use.reads == 1 else f"{use.reads} times"
        last = "" if use.last_read is None else f", last {_when_by(use.last_read)}"
        parts.append(f"read {times}{last}")
    else:
        parts.append("never read")
    return ", ".join(parts)


def _when_by(touch: memory.Touch) -> str:
    """Return a touch as `<date> by <session>`."""
    return f"{touch.at[:10]} by {touch.session}"


def _cmd_memory_show(name: str) -> int:
    """Print a memory's body.

    Returns:
        The exit code, 0.
    """
    print(memory.show(_state(), name), end="")
    return 0


def _cmd_memory_decisions() -> int:
    """Print the harness's recorded rulings.

    Returns:
        The exit code, 0.
    """
    path = memory.decisions_path(_state())
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    if not text.strip():
        print(f"(no rulings recorded; the harness writes them to {path})")
        return 0
    print(text, end="")
    return 0


def _cmd_memory_rm(name: str, *, yes: bool = False) -> int:
    """Delete a memory file, asking first on a terminal.

    Args:
        name: The memory.
        yes: Delete without asking.

    Returns:
        The exit code, 0; a declined question deletes nothing.
    """
    state = _state()
    if not yes:
        try:
            first = memory.show(state, name).strip().splitlines()[:1]
        except memory.MemoryStoreError:
            # `remove` reports the missing file.
            first = None
        if first is not None and not _steer.delete_confirmed(
            f"memory rm will delete {name}\n  {first[0][:100] if first else '(an empty file)'}",
            "Delete it?",
        ):
            return 0
    memory.remove(state, name)
    print(f"removed {name}")
    return 0
