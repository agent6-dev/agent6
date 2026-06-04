# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `_expand_task_file_refs` (CLI @path inlining)."""

from __future__ import annotations

from pathlib import Path

from agent6.cli import (
    _TASK_FILE_REF_MAX_BYTES,  # pyright: ignore[reportPrivateUsage]
    _expand_task_file_refs,  # pyright: ignore[reportPrivateUsage]
)


def test_inlines_existing_file(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello world\n", encoding="utf-8")
    out = _expand_task_file_refs("read @notes.md please", tmp_path)
    assert '<file path="notes.md">' in out
    assert "hello world" in out
    assert "</file>" in out
    # The surrounding sentence words are still there.
    assert out.startswith("read ")
    assert out.rstrip().endswith("please")


def test_leaves_missing_path_untouched(tmp_path: Path) -> None:
    out = _expand_task_file_refs("see @does/not/exist for context", tmp_path)
    assert out == "see @does/not/exist for context"


def test_does_not_touch_email_addresses(tmp_path: Path) -> None:
    out = _expand_task_file_refs("ping user@example.com about it", tmp_path)
    assert out == "ping user@example.com about it"


def test_rejects_escape_via_parent(tmp_path: Path) -> None:
    # Create a file OUTSIDE root; the @ref should fail the relative_to check.
    outside = tmp_path.parent / "agent6_test_escape.txt"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        sub = tmp_path / "work"
        sub.mkdir()
        out = _expand_task_file_refs("look at @../agent6_test_escape.txt", sub)
        assert "secret" not in out
        assert "@../agent6_test_escape.txt" in out
    finally:
        outside.unlink(missing_ok=True)


def test_ignores_directories(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    out = _expand_task_file_refs("see @subdir for context", tmp_path)
    assert out == "see @subdir for context"


def test_truncates_large_files(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (_TASK_FILE_REF_MAX_BYTES + 5000))
    out = _expand_task_file_refs("@big.txt", tmp_path)
    assert "truncated" in out
    assert "5000 bytes omitted" in out


def test_handles_binary_with_replacement(tmp_path: Path) -> None:
    binf = tmp_path / "bin.dat"
    binf.write_bytes(b"\xff\xfe\x00garbage")
    out = _expand_task_file_refs("@bin.dat", tmp_path)
    assert '<file path="bin.dat">' in out
    # Replacement char appears, no crash.
    assert "\ufffd" in out or "garbage" in out


def test_multiple_refs_in_one_string(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "b.txt").write_text("BBB", encoding="utf-8")
    out = _expand_task_file_refs("compare @a.txt with @b.txt", tmp_path)
    assert "AAA" in out
    assert "BBB" in out


def test_empty_task_is_passthrough(tmp_path: Path) -> None:
    assert _expand_task_file_refs("", tmp_path) == ""


def test_no_refs_is_passthrough(tmp_path: Path) -> None:
    assert _expand_task_file_refs("just a task", tmp_path) == "just a task"
