# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The per-repo memory store: one fact per file plus the MEMORY.md index."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.memory import MemoryStoreError, add, index_text, memory_dir, remove, show


def test_add_writes_file_and_index_line(tmp_path: Path) -> None:
    path = add(tmp_path, "build-quirk", "The build needs FOO=1.\nDetails here.")
    assert path == memory_dir(tmp_path) / "build-quirk.md"
    assert path.read_text() == "The build needs FOO=1.\nDetails here.\n"
    assert index_text(tmp_path) == "- build-quirk: The build needs FOO=1."


def test_add_refuses_duplicate_and_bad_names(tmp_path: Path) -> None:
    add(tmp_path, "one", "fact")
    with pytest.raises(MemoryStoreError, match="exists"):
        add(tmp_path, "one", "other")
    for bad in ("Has-Caps", "sl/ash", "..", "-lead", "a" * 65):
        with pytest.raises(MemoryStoreError, match="bad memory name"):
            add(tmp_path, bad, "x")
    with pytest.raises(MemoryStoreError, match="non-empty"):
        add(tmp_path, "empty", "   ")


def test_remove_deletes_file_and_index_line(tmp_path: Path) -> None:
    add(tmp_path, "keep", "kept fact")
    add(tmp_path, "drop", "dropped fact")
    remove(tmp_path, "drop")
    assert not (memory_dir(tmp_path) / "drop.md").exists()
    assert index_text(tmp_path) == "- keep: kept fact"
    with pytest.raises(MemoryStoreError, match="no memory named"):
        remove(tmp_path, "drop")


def test_show_reads_one_entry(tmp_path: Path) -> None:
    add(tmp_path, "fact", "body text")
    assert show(tmp_path, "fact") == "body text\n"
    with pytest.raises(MemoryStoreError, match="no memory named"):
        show(tmp_path, "absent")


def test_index_text_degrades_to_empty(tmp_path: Path) -> None:
    """Memory is context: an absent or unreadable index is "" for injection,
    never an error that kills every run in the repo."""
    assert index_text(tmp_path) == ""
    d = memory_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "MEMORY.md").write_bytes(b"\xff\xfe broken")
    assert index_text(tmp_path) == ""
