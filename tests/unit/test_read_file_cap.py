# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""read_file bounds what it pulls into memory.

read_contained loaded the whole file before slicing, so start_line/limit did
not bound the read: a multi-gigabyte file (a checked-in blob, a log, a file a
command produced) OOM-crashed the unsandboxed agent. read_file now reads at
most MAX_READ_CHARS and flags `truncated`; a normal file is untouched.
"""

from __future__ import annotations

from pathlib import Path

from agent6.config import Config
from agent6.tools._fs_tools import MAX_READ_CHARS
from agent6.tools.dispatch import ToolDispatcher


def _read(root: Path, **args: object) -> dict[str, object]:
    d = ToolDispatcher(root=root, config=Config(), isolation="none")
    try:
        return d.dispatch("read_file", {"path": "f.txt", **args}).to_wire()
    finally:
        d.close()


def test_a_file_over_the_cap_is_truncated_not_read_whole(tmp_path: Path) -> None:
    # One char past the cap is enough to prove the read is bounded, without
    # allocating a genuinely huge file in the test.
    (tmp_path / "f.txt").write_text("a" * (MAX_READ_CHARS + 5000), encoding="utf-8")
    out = _read(tmp_path)
    content = out["content"]
    assert out["truncated"] is True
    assert isinstance(content, str) and len(content) == MAX_READ_CHARS, "read not bounded"


def test_a_ranged_read_of_a_huge_file_is_also_bounded(tmp_path: Path) -> None:
    # Even with start_line/limit, the read must not load the whole file.
    (tmp_path / "f.txt").write_text(("x\n") * (MAX_READ_CHARS), encoding="utf-8")  # ~2*cap bytes
    out = _read(tmp_path, start_line=1, limit=3)
    assert out["truncated"] is True
    assert out["lines_returned"] == 3  # the window is served from the capped prefix


def test_a_normal_file_is_untouched(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello\nworld\n", encoding="utf-8")
    out = _read(tmp_path)
    assert out["content"] == "hello\nworld\n"
    assert "truncated" not in out  # the flag is omitted on a full read


def test_a_file_exactly_at_the_cap_is_not_flagged(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a" * MAX_READ_CHARS, encoding="utf-8")
    out = _read(tmp_path)
    content = out["content"]
    assert "truncated" not in out
    assert isinstance(content, str) and len(content) == MAX_READ_CHARS
