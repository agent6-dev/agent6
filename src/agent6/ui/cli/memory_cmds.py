# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory add/list/show/rm/decisions` commands.

Store refusals (a bad name, an unreadable store) raise MemoryStoreError, an
OperatorError the cli_main boundary presents; no per-command arms.
"""

from __future__ import annotations

from pathlib import Path

from agent6.memory import (
    MemoryUse,
    Touch,
    add,
    decisions_path,
    index_name,
    index_text,
    memory_dir,
    read_use,
    remove,
    show,
    unindexed_names,
)
from agent6.paths import state_dir


def _cmd_memory_add(name: str, body: str) -> int:
    path = add(state_dir(Path.cwd()), name, body)
    print(f"wrote {path}")
    return 0


def _cmd_memory_list() -> int:
    """The whole index (a run's prompt gets it clipped), each entry followed
    by its use record (who wrote it, who read it) when the harness has one,
    then the files the index does not list."""
    state = state_dir(Path.cwd())
    text = index_text(state)
    orphans = unindexed_names(state)
    if not text:
        print(f"(no memories; files live under {memory_dir(state)})")
    use = read_use(state)
    for line in text.splitlines():
        print(line)
        name = index_name(line)
        if name is not None:
            print(f"    {format_use(use.get(name, MemoryUse()))}")
    if orphans:
        print(f"not in the index (no run sees them; `memory rm` deletes): {', '.join(orphans)}")
    return 0


def format_use(use: MemoryUse) -> str:
    """One line: `written <date> by <session>, edited <date> by <session>,
    read once|N times, last <date> by <session>` or `never read`; a part the
    record does not hold is left out (a fact it never saw created has no
    `written`)."""
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


def _when_by(touch: Touch) -> str:
    return f"{touch.at[:10]} by {touch.session}"


def _cmd_memory_show(name: str) -> int:
    print(show(state_dir(Path.cwd()), name), end="")
    return 0


def _cmd_memory_decisions() -> int:
    state = state_dir(Path.cwd())
    path = decisions_path(state)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        print(f"(no rulings recorded; the harness writes them to {path})")
        return 0
    print(text, end="")
    return 0


def _cmd_memory_rm(name: str) -> int:
    remove(state_dir(Path.cwd()), name)
    print(f"removed {name}")
    return 0
