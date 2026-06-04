# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.tools.patch_apply — parser + applier."""

from __future__ import annotations

import pytest

from agent6.tools.patch_apply import (
    PatchError,
    apply_patch_text,
    parse_patch,
)


def test_parse_simple_single_hunk() -> None:
    patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    p = parse_patch(patch)
    assert p.target_path == "foo.py"
    assert p.is_create is False
    assert len(p.hunks) == 1


def test_apply_replace_one_line() -> None:
    original = "a\nb\nc\n"
    patch = "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    path, new = apply_patch_text(patch, original)
    assert path == "foo.py"
    assert new == "a\nB\nc\n"


def test_apply_multi_hunk_offset_tracking() -> None:
    original = "1\n2\n3\n4\n5\n6\n7\n8\n9\n"
    # First hunk inserts a line near the top; second hunk replaces near the
    # bottom. The second hunk's `old_start` is in original-file coordinates,
    # so the applier must track the cumulative offset (+1) from the first.
    patch = (
        "--- a/n.txt\n"
        "+++ b/n.txt\n"
        "@@ -1,3 +1,4 @@\n"
        " 1\n"
        "+1.5\n"
        " 2\n"
        " 3\n"
        "@@ -7,3 +8,3 @@\n"
        " 7\n"
        "-8\n"
        "+EIGHT\n"
        " 9\n"
    )
    _, new = apply_patch_text(patch, original)
    assert new == "1\n1.5\n2\n3\n4\n5\n6\n7\nEIGHT\n9\n"


def test_apply_pure_insertion_hunk() -> None:
    # `@@ -3,0 +4,2 @@` style: zero lines in original, anchored at line 3.
    original = "a\nb\nc\n"
    patch = "--- a/f.txt\n+++ b/f.txt\n@@ -3,1 +3,3 @@\n c\n+x\n+y\n"
    _, new = apply_patch_text(patch, original)
    assert new == "a\nb\nc\nx\ny\n"


def test_apply_pure_deletion_hunk() -> None:
    original = "a\nb\nc\n"
    patch = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,2 @@\n a\n-b\n c\n"
    _, new = apply_patch_text(patch, original)
    assert new == "a\nc\n"


def test_create_via_dev_null() -> None:
    patch = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
    path, new = apply_patch_text(patch, None)
    assert path == "new.py"
    assert new == "x = 1\ny = 2\n"


def test_create_when_file_exists_errors() -> None:
    patch = "--- /dev/null\n+++ b/exists.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
    with pytest.raises(PatchError, match="already exists"):
        apply_patch_text(patch, "old contents\n")


def test_context_mismatch_errors_with_helpful_message() -> None:
    original = "a\nDIFFERENT\nc\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    with pytest.raises(PatchError, match="Context mismatch"):
        apply_patch_text(patch, original)


def test_missing_file_errors() -> None:
    patch = "--- a/missing.py\n+++ b/missing.py\n@@ -1,1 +1,1 @@\n-a\n+b\n"
    with pytest.raises(PatchError, match="does not exist"):
        apply_patch_text(patch, None)


def test_delete_via_plus_dev_null_rejected() -> None:
    patch = "--- a/f.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-a\n"
    with pytest.raises(PatchError, match="deletion"):
        apply_patch_text(patch, "a\n")


def test_multi_file_patch_rejected() -> None:
    patch = (
        "--- a/one.py\n"
        "+++ b/one.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+A\n"
        "--- a/two.py\n"
        "+++ b/two.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-b\n"
        "+B\n"
    )
    with pytest.raises(PatchError, match="Multi-file"):
        apply_patch_text(patch, "a\n")


def test_hunk_header_count_mismatch_rejected() -> None:
    # Header says 3 old lines but body only supplies 2.
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n"
    with pytest.raises(PatchError, match="declares 3"):
        apply_patch_text(patch, "a\nb\n")


def test_skips_git_diff_preamble() -> None:
    # `git diff` emits `diff --git ...` and `index ...` lines before `---`.
    patch = (
        "diff --git a/f.py b/f.py\n"
        "index abc..def 100644\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+A\n"
    )
    _, new = apply_patch_text(patch, "a\n")
    assert new == "A\n"


def test_no_newline_at_eof_on_old_side() -> None:
    # Original lacks a trailing newline; replacement adds one.
    original = "a\nb"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+B\n"
    _, new = apply_patch_text(patch, original)
    assert new == "a\nB\n"


def test_no_newline_at_eof_on_new_side() -> None:
    original = "a\nb\n"
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n\\ No newline at end of file\n"
    _, new = apply_patch_text(patch, original)
    assert new == "a\nB"


def test_omitted_count_means_one() -> None:
    # `@@ -1 +1 @@` is shorthand for `@@ -1,1 +1,1 @@`.
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+A\n"
    _, new = apply_patch_text(patch, "a\n")
    assert new == "A\n"


def test_empty_patch_rejected() -> None:
    with pytest.raises(PatchError, match="Empty"):
        apply_patch_text("", None)


def test_bare_path_header_no_a_b_prefix() -> None:
    # Accept patches without the conventional `a/`/`b/` prefix.
    patch = "--- f.py\n+++ f.py\n@@ -1 +1 @@\n-a\n+A\n"
    path, new = apply_patch_text(patch, "a\n")
    assert path == "f.py"
    assert new == "A\n"
