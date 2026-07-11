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
