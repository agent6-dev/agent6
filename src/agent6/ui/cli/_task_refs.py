# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Inline `@path` file references in the task string `agent6 run` was given."""

from __future__ import annotations

import re
from pathlib import Path

_TASK_FILE_REF_RE = re.compile(r"(?<![\w@/])@([A-Za-z0-9_./\-]+)")


TASK_FILE_REF_MAX_BYTES = 64 * 1024  # cap per file - bigger reads need an explicit tool call.


def expand_task_file_refs(task: str, root: Path) -> str:
    """Inline `@path` references in `task` that resolve to files under `root`.

    Behaviour:
      - The match must start at a word boundary that excludes `@` and `/`
        (so `user@example.com` and `//@noqa` are not touched).
      - The path must resolve (via ``Path.resolve``) to a regular file
        whose resolved path is inside ``root``. Symlinks that escape are
        rejected the same way the sandbox would reject them.
      - File contents are truncated to ``TASK_FILE_REF_MAX_BYTES`` and
        decoded as UTF-8 with replacement; binary files therefore appear
        as garbled text rather than crashing the run.
      - Unresolved references are left as-is. We never raise.
    """
    root_resolved = root.resolve()

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(1)
        try:
            candidate = (root / rel).resolve()
        except (OSError, RuntimeError):
            return match.group(0)
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            return match.group(0)
        if not candidate.is_file():
            return match.group(0)
        try:
            raw = candidate.read_bytes()
        except OSError:
            return match.group(0)
        truncated = raw[:TASK_FILE_REF_MAX_BYTES]
        text = truncated.decode("utf-8", errors="replace")
        suffix = ""
        if len(raw) > TASK_FILE_REF_MAX_BYTES:
            suffix = (
                f"\n... (truncated, {len(raw) - TASK_FILE_REF_MAX_BYTES} bytes omitted; "
                "use read_file with an explicit range for the rest)"
            )
        return f'\n<file path="{rel}">\n{text}{suffix}\n</file>\n'

    return _TASK_FILE_REF_RE.sub(_replace, task)
