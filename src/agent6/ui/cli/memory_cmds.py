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
from agent6.memory import (
    set_pinned as memory_set_pinned,
)
from agent6.ui.cli._common import _state_dir, sgr
from agent6.workflows import MEMORIES_MAX_CHARS, MEMORY_ENTRY_MAX_CHARS


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
        # The block trims across ALL scopes, so the over-cap warning must too --
        # a --scope listing would otherwise miss a global overflow.
        everything = entries if scope is None else memory_list(_state_dir(Path.cwd()), None)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    shown = [e for e in entries if include_invalidated or e.is_active]
    pinned_cost = sum(
        min(len(e.body), MEMORY_ENTRY_MAX_CHARS) + 48
        for e in everything
        if e.pinned and e.is_active
    )
    if pinned_cost > MEMORIES_MAX_CHARS:
        print(
            "[agent6] pinned memories exceed the memory block cap"
            f" ({pinned_cost:,} > {MEMORIES_MAX_CHARS:,} chars);"
            " oldest pinned will be elided from the <memories> block"
        )
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
        print(sgr(scope_name, "1"))
        for e in items:
            active = e.is_active
            for line in e.body.splitlines():
                print(f"  {line}" if active else sgr(f"  {line}", "2"))
            meta = f"{e.id}  ·  {e.created_at}"
            if e.pinned:
                meta = f"[pinned] {meta}"
            if not active:
                meta = f"[invalidated] {meta}  ·  {e.invalidation_reason or 'no reason'}"
            print(sgr(f"  {meta}", "2"))
    return 0


def _cmd_memory_invalidate(memory_id: str, reason: str) -> int:
    try:
        entry = memory_invalidate(_state_dir(Path.cwd()), memory_id, reason)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"invalidated {entry.scope} {entry.id} at {entry.invalidated_at}")
    return 0


def _cmd_memory_pin(memory_id: str) -> int:
    try:
        entry = memory_set_pinned(_state_dir(Path.cwd()), memory_id, True)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"pinned {entry.scope} {entry.id} (exempt from the <memories> block trim)")
    return 0


def _cmd_memory_unpin(memory_id: str) -> int:
    try:
        entry = memory_set_pinned(_state_dir(Path.cwd()), memory_id, False)
    except MemoryStoreError as exc:
        print(f"MEMORY ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"unpinned {entry.scope} {entry.id}")
    return 0
