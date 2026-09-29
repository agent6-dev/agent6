# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Diagnostics for apply_edit and apply_patch.

The dry-run preview, and on a failed match the closest on-disk region so the model retries
without re-reading.
"""

from __future__ import annotations

import difflib

from agent6.tools import results


def preview_result(
    path: str,
    old_text: str | None,
    new_text: str,
    *,
    bytes_before: int,
    bytes_after: int,
    applied: list[str] | None = None,
    deleting: bool = False,
    healed: tuple[str, ...] = (),
) -> results.PreviewResult:
    """Build the dry-run response for an edit tool called with `preview=true`.

    Nothing is written. The diff is capped at 8000 characters so a preview of a large rewrite
    does not dump the whole file into the conversation.

    Args:
        path: The workspace-relative path.
        old_text: The file's text before the edit, or None for a new file.
        new_text: The file's text after the edit.
        bytes_before: The size on disk before, measured by the caller so it matches an apply.
        bytes_after: The size on disk after.
        applied: The edits that would apply, when the caller tracks them.
        deleting: Whether the edit deletes the file.
        healed: The edits that only matched after an indent shift.

    Returns:
        The unified diff, its hunk count and the byte counts.
    """
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    label_a = "/dev/null" if old_text is None else f"a/{path}"
    label_b = "/dev/null" if deleting else f"b/{path}"
    diff_iter = difflib.unified_diff(old_lines, new_lines, fromfile=label_a, tofile=label_b, n=3)
    diff = "".join(diff_iter)
    if not diff and (old_text is None or deleting):
        diff = f"--- {label_a}\n+++ {label_b}\n"
    hunks = sum(1 for line in diff.splitlines() if line.startswith("@@ "))
    truncated = False
    max_diff_chars = 8000
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + f"\n... <truncated {len(diff) - max_diff_chars} chars>\n"
        truncated = True
    return results.PreviewResult(
        path=path,
        diff=diff or "(no changes)",
        hunks=hunks,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        truncated=truncated,
        would_apply=None if applied is None else tuple(applied),
        healed=healed,
    )


# Above this the closest-match scan (quadratic) is skipped and the diagnostic gives file shape.
CLOSEST_MATCH_MAX_LINES = 6000


def _leading_ws(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _reindent(lines: list[str], old_base: str, new_base: str) -> list[str] | None:
    """Replace each non-blank line's leading `old_base` with `new_base`.

    Indentation beyond the base is kept and blank lines pass through.

    Args:
        lines: The lines to shift.
        old_base: The indent each non-blank line must start with.
        new_base: The indent that replaces it.

    Returns:
        The shifted lines, or None when a non-blank line does not start with `old_base`.
    """
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append(ln)
        elif ln.startswith(old_base):
            out.append(new_base + ln[len(old_base) :])
        else:
            return None
    return out


def indent_tolerant_replacement(file_text: str, old_string: str, new_string: str) -> str | None:
    """Apply an edit whose `old_string` matches exactly one region up to a uniform indent shift.

    Correct lines at the wrong indent depth is the dominant weak-model mistake. The shift
    derived from the first content line must reproduce the matched region byte for byte before
    it is applied to `new_string`, so a wrong region cannot be hit.

    Args:
        file_text: The file's text.
        old_string: The text the edit expected, at the wrong indent.
        new_string: The replacement, at the same wrong indent.

    Returns:
        The edited text, or None when the shift is not provably safe (no match, several
        matches, or a non-uniform shift) so the caller keeps the exact-match error.
    """
    old_lines = old_string.split("\n")
    file_lines = file_text.split("\n")
    n = len(old_lines)
    if n == 0 or n > len(file_lines):
        return None
    old_stripped = [ln.strip() for ln in old_lines]
    if not any(old_stripped):  # all-blank old_string: nothing to anchor safely
        return None
    starts = [
        i
        for i in range(len(file_lines) - n + 1)
        if [ln.strip() for ln in file_lines[i : i + n]] == old_stripped
    ]
    if len(starts) != 1:  # no match, or ambiguous -> never guess
        return None
    start = starts[0]
    region = file_lines[start : start + n]
    old_base = next(_leading_ws(o) for o in old_lines if o.strip())
    new_base = next(_leading_ws(r) for r in region if r.strip())
    if _reindent(old_lines, old_base, new_base) != region:
        return None  # the shift is not uniform across the region -> unsafe
    new_region = _reindent(new_string.split("\n"), old_base, new_base)
    if new_region is None:
        return None  # new_string can't take the same shift cleanly -> fall back
    return "\n".join(file_lines[:start] + new_region + file_lines[start + n :])


def closest_on_disk_region(file_text: str, old_string: str) -> tuple[int, str, float] | None:
    """Find the file region most similar to a not-found `old_string`.

    A failed edit hands the model the exact on-disk text to retry with, instead of telling it
    to re-read the file.

    Args:
        file_text: The file's text.
        old_string: The text the edit expected.

    Returns:
        The 1-based start line, the region's text and the similarity ratio of the best window
        with `old_string`'s line count, or None when the file is empty or oversized.
    """
    file_lines = file_text.splitlines()
    if not file_lines or len(file_lines) > CLOSEST_MATCH_MAX_LINES:
        return None
    old_lines = old_string.splitlines() or [old_string]
    n = max(1, min(len(old_lines), len(file_lines)))
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(old_string)
    best_ratio = -1.0
    best_idx = 0
    for i in range(0, len(file_lines) - n + 1):
        window = "\n".join(file_lines[i : i + n])
        matcher.set_seq1(window)
        # quick_ratio is a cheap upper bound; skip windows that cannot win.
        if matcher.quick_ratio() <= best_ratio:
            continue
        r = matcher.ratio()
        if r > best_ratio:
            best_ratio = r
            best_idx = i
    region = "\n".join(file_lines[best_idx : best_idx + n])
    return best_idx + 1, region, best_ratio


def edit_mismatch_error(path: str, edit_index: int, file_text: str, old_string: str) -> str:
    """Build the not-found error for `apply_edit`.

    Args:
        path: The workspace-relative path.
        edit_index: The failed edit's 1-based index.
        file_text: The file's text.
        old_string: The text the edit expected.

    Returns:
        The error, carrying the closest on-disk region to retry with, or the file's shape when
        no region is at least half similar.
    """
    region_info = closest_on_disk_region(file_text, old_string)
    if region_info is not None and region_info[2] >= 0.5:
        start_line, region, ratio = region_info
        end_line = start_line + len(region.splitlines()) - 1
        diff = "\n".join(
            difflib.unified_diff(
                old_string.splitlines(),
                region.splitlines(),
                fromfile="your_old_string",
                tofile=f"on_disk_lines_{start_line}-{end_line}",
                lineterm="",
                n=1,
            )
        )
        whitespace_only = [ln.strip() for ln in old_string.splitlines()] == [
            ln.strip() for ln in region.splitlines()
        ]
        why = (
            "It matches your old_string except for whitespace/indentation."
            if whitespace_only
            else f"It is the closest region on disk ({ratio:.0%} similar)."
        )
        return (
            f"old_string not found in {path} (edit #{edit_index}). {why} Retry"
            f" apply_edit using the EXACT on-disk text below as old_string; do"
            f" NOT call read_file first; this IS the current content of lines"
            f" {start_line}-{end_line}.\n"
            f"<<<ON_DISK (copy this verbatim, without these <<< >>> markers)\n"
            f"{region}\n"
            f">>>ON_DISK\n"
            f"difference (- your old_string, + on disk):\n{diff}"
        )
    # File shape only: no body to copy, so the model cannot take a wrong anchor.
    lines = file_text.splitlines()
    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:]) if len(lines) > 10 else ""
    size = len(file_text.encode("utf-8"))
    snippet = f"file size: {size} bytes, {len(lines)} lines\nfirst 5 lines:\n{head}"
    if tail:
        snippet += f"\n...\nlast 5 lines:\n{tail}"
    return (
        f"old_string not found in {path} (edit #{edit_index}). Your old_string"
        f" does not match the file content byte-for-byte and no close region"
        f" exists, so it likely targets the wrong file or a stale expectation."
        f" Re-read with read_file, then retry with a shorter, uniquely-anchored"
        f" old_string. File shape (orientation only, do NOT use as old_string"
        f" verbatim):\n{snippet}"
    )
