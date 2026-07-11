# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory add/list/invalidate` commands."""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryStoreError,
)
from agent6.memory import (
    add as memory_add,
)
from agent6.memory import (
    invalidate as memory_invalidate,
)
from agent6.memory import (
    list_entries as memory_list,
)
from agent6.ui.cli._common import _state_dir


def _sgr(text: str, code: str) -> str:
    """Wrap in an ANSI style, tty only, so piped output stays plain."""
    return f"\x1b[{code}m{text}\x1b[0m" if sys.stdout.isatty() else text


def _cmd_memory_add(scope: MemoryScope, body: str) -> int:
    try:
        entry = memory_add(_state_dir(Path.cwd()), scope, body)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{entry.scope} {entry.id} created at {entry.created_at}")
    return 0


def _cmd_memory_list(scope: MemoryScope | None, *, include_invalidated: bool) -> int:
    try:
        entries = memory_list(_state_dir(Path.cwd()), scope)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    shown = [e for e in entries if include_invalidated or e.is_active]
    if not shown:
        if entries:
            print("no active memories. Pass --include-invalidated to see invalidated ones.")
        else:
            print('no memories yet. Add one with `agent6 memory add <scope> "<text>"`.')
        return 0
    # Group by scope so the category prints once, and lead with the body: the
    # opaque id and timestamp recede (dim) below the content they belong to.
    groups: dict[str, list[MemoryEntry]] = {}
    for e in shown:
        groups.setdefault(e.scope, []).append(e)
    for i, (scope_name, items) in enumerate(groups.items()):
        print("" if i == 0 else "\n", end="")
        print(_sgr(scope_name, "1"))
        for e in items:
            active = e.is_active
            for line in e.body.splitlines():
                print(f"  {line}" if active else _sgr(f"  {line}", "2"))
            meta = f"{e.id}  ·  {e.created_at}"
            if not active:
                meta = f"[invalidated] {meta}  ·  {e.invalidation_reason or 'no reason'}"
            print(_sgr(f"  {meta}", "2"))
    return 0


def _cmd_memory_invalidate(memory_id: str, reason: str) -> int:
    try:
        entry = memory_invalidate(_state_dir(Path.cwd()), memory_id, reason)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"invalidated {entry.scope} {entry.id} at {entry.invalidated_at}")
    return 0
