# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Diagnostics for apply_edit / apply_patch: dry-run preview, and on a failed
match the closest on-disk region so the model can retry without re-reading.
"""

from __future__ import annotations

import difflib
from typing import Any


def preview_result(
    path: str,
    old_text: str | None,
    new_text: str,
    *,
    applied: list[str] | None = None,
) -> dict[str, Any]:
    """Build the dry-run response for ``apply_edit``/``apply_patch`` with
    ``preview=true``. Returns the unified diff (old vs new) and a hunk
    count, but does NOT write anything to disk.

    Lets the agent sanity-check a complex multi-edit call
    before committing to it. Diff is bounded so a preview of a 100k-line
    rewrite doesn't dump the whole file back into the conversation.
    """
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    label_a = "/dev/null" if old_text is None else f"a/{path}"
    label_b = f"b/{path}"
    diff_iter = difflib.unified_diff(old_lines, new_lines, fromfile=label_a, tofile=label_b, n=3)
    diff = "".join(diff_iter)
    hunks = sum(1 for line in diff.splitlines() if line.startswith("@@ "))
    truncated = False
    _MAX_DIFF_CHARS = 8000
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + f"\n... <truncated {len(diff) - _MAX_DIFF_CHARS} chars>\n"
        truncated = True
    result: dict[str, Any] = {
        "preview": True,
        "path": path,
        "diff": diff or "(no changes)",
        "hunks": hunks,
        "bytes_before": len(old_text or ""),
        "bytes_after": len(new_text),
        "truncated": truncated,
    }
    if applied is not None:
        result["would_apply"] = applied
    return result


# Cap the closest-match scan so a failed edit on a very large file does not turn
# into a quadratic diff. Above this the diagnostic falls back to file shape.
CLOSEST_MATCH_MAX_LINES = 6000


def closest_on_disk_region(file_text: str, old_string: str) -> tuple[int, str, float] | None:
    """Find the file region most similar to a not-found ``old_string``.

    Returns ``(1-based start line, region text, similarity ratio)`` for the best
    contiguous window with the same line count as ``old_string``, or None when
    the scan is skipped (empty or oversized file). This lets a failed
    ``apply_edit`` hand the model the EXACT on-disk text to retry with, instead
    of telling it to re-read the whole file (the dominant small-model time sink).
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
    """Build the not-found error for ``apply_edit``. Prefers a copy-paste-able
    closest on-disk region so the model retries directly; falls back to file
    shape only when no region is similar enough to be useful."""
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
            f" apply_edit using the EXACT on-disk text below as old_string — do"
            f" NOT call read_file first; this IS the current content of lines"
            f" {start_line}-{end_line}.\n"
            f"<<<ON_DISK (copy this verbatim, without these <<< >>> markers)\n"
            f"{region}\n"
            f">>>ON_DISK\n"
            f"difference (- your old_string, + on disk):\n{diff}"
        )
    # No region similar enough: orient with file shape only (no body to copy,
    # so the model cannot plagiarise a wrong anchor).
    lines = file_text.splitlines()
    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:]) if len(lines) > 10 else ""
    snippet = f"file size: {len(file_text)} bytes, {len(lines)} lines\nfirst 5 lines:\n{head}"
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
