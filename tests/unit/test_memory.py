# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the persistent agent memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.memory import (
    MemoryStoreError,
    add,
    invalidate,
    list_entries,
)


def test_add_persists_and_roundtrips(tmp_path: Path) -> None:
    e1 = add(tmp_path, "facts", "the verify command takes 3 seconds")
    e2 = add(tmp_path, "facts", "the worker pool is 4")
    entries = list_entries(tmp_path, "facts")
    ids = [e.id for e in entries]
    assert ids == [e1.id, e2.id]
    assert entries[0].body == "the verify command takes 3 seconds"
    assert entries[0].is_active
    assert entries[1].body == "the worker pool is 4"


def test_add_rejects_empty_body(tmp_path: Path) -> None:
    with pytest.raises(MemoryStoreError):
        add(tmp_path, "facts", "   \n")


def test_list_across_scopes(tmp_path: Path) -> None:
    a = add(tmp_path, "facts", "fact a")
    b = add(tmp_path, "decisions", "decision b")
    c = add(tmp_path, "preferences", "pref c")
    all_ids = {e.id for e in list_entries(tmp_path)}
    assert all_ids == {a.id, b.id, c.id}
    only_facts = [e.id for e in list_entries(tmp_path, "facts")]
    assert only_facts == [a.id]


def test_invalidate_preserves_body(tmp_path: Path) -> None:
    e = add(tmp_path, "decisions", "use ruff for everything")
    inv = invalidate(tmp_path, e.id, "switched to dprint")
    assert not inv.is_active
    assert inv.invalidation_reason == "switched to dprint"
    # Body is preserved.
    again = list_entries(tmp_path, "decisions")
    assert len(again) == 1
    assert again[0].body == "use ruff for everything"
    assert again[0].invalidated_at == inv.invalidated_at


def test_invalidate_twice_fails(tmp_path: Path) -> None:
    e = add(tmp_path, "facts", "x")
    invalidate(tmp_path, e.id, "wrong")
    with pytest.raises(MemoryStoreError):
        invalidate(tmp_path, e.id, "still wrong")


def test_invalidate_unknown_id_fails(tmp_path: Path) -> None:
    add(tmp_path, "facts", "x")
    with pytest.raises(MemoryStoreError):
        invalidate(tmp_path, "0" * 26, "missing")


def test_multiline_body_roundtrips(tmp_path: Path) -> None:
    body = "first line\nsecond line\n\nthird with blank above"
    e = add(tmp_path, "preferences", body)
    again = list_entries(tmp_path, "preferences")
    assert len(again) == 1
    assert again[0].id == e.id
    assert again[0].body == body


def test_invalidate_reason_with_newline_stays_single_line(tmp_path: Path) -> None:
    """A multi-line reason is a single-line `key: value` meta field; a newline
    leaked its tail into the entry body on the next parse (truncating the reason,
    corrupting the body). It must be flattened to one line."""
    e = add(tmp_path, "facts", "stable body")
    invalidate(tmp_path, e.id, "first sentence.\nsecond detail with the tail.")
    again = list_entries(tmp_path, "facts")
    assert len(again) == 1  # no phantom split entry
    assert again[0].body == "stable body"  # body untouched
    assert "\n" not in again[0].invalidation_reason
    assert "second detail with the tail." in again[0].invalidation_reason


def test_add_body_shaped_like_delimiter_rejected(tmp_path: Path) -> None:
    """A body line shaped like the entry delimiter (`### <ULID>`) split the entry
    on the next parse — truncating it and forging a phantom entry under the
    embedded id. It must be refused loudly, not silently corrupt the store."""
    with pytest.raises(MemoryStoreError, match="delimiter"):
        add(tmp_path, "facts", "text\n### 01ARZ3NDEKTSV4RRFFQ69G5FAV\nmore")


def test_add_normal_markdown_heading_still_allowed(tmp_path: Path) -> None:
    """A normal `### Heading` is not a ULID, so it must not be refused."""
    e = add(tmp_path, "facts", "### My Section\nbody content")
    again = list_entries(tmp_path, "facts")
    assert len(again) == 1
    assert again[0].id == e.id
    assert again[0].body == "### My Section\nbody content"


def test_concurrent_writers_cannot_lose_or_resurrect_entries(tmp_path: Path) -> None:
    """Both mutators rewrite the WHOLE shared scope file; unlocked, the loser
    of two concurrent read-modify-writes was silently discarded (a dropped
    add, or an invalidation resurrected by a racing add). The mutators now
    hold the memories flock; two threaded writers must both land."""
    import threading

    from agent6 import memory as mem

    state = tmp_path / "state"
    first = mem.add(state, "facts", "entry one")
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _add(body: str) -> None:
        try:
            barrier.wait(timeout=5)
            mem.add(state, "facts", body)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_add, args=("entry two",))
    t2 = threading.Thread(target=_add, args=("entry three",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert errors == []
    bodies = {e.body for e in mem.list_entries(state, "facts")}
    assert {"entry one", "entry two", "entry three"} <= bodies  # nothing lost
    mem.invalidate(state, first.id, "stale")
    active = {e.body for e in mem.list_entries(state, "facts") if not e.invalidated_at}
    assert "entry one" not in active
