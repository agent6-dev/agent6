# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Inline `@path` file references in the task string `agent6 run` was given."""

from __future__ import annotations

import os
import pathlib
import re
from collections.abc import Iterable

from agent6 import path_safety

_TASK_FILE_REF_RE = re.compile(r"(?<![\w@/])@([A-Za-z0-9_./\-]+)")


TASK_FILE_REF_MAX_BYTES = 64 * 1024  # per file; a bigger read needs an explicit tool call


def expand_task_file_refs(task: str, root: pathlib.Path, hide: Iterable[str] = ()) -> str:
    """Inline the `@path` references in the task that name readable workspace files.

    A match starts at a word boundary that excludes `@` and `/`, so `user@example.com`
    is untouched. A path resolves inside the host-side read boundary
    (`path_safety.host_workspace`): a regular file inside the root, outside `.git` and the
    hidden set, read through a descriptor walked from the root. Any other reference is left as
    typed. At most `TASK_FILE_REF_MAX_BYTES` are read, and decoded lossily. Never raises.

    Args:
        task: The words the operator typed; text a model or a repository file wrote is never
            passed, since a reference in it would read a file the operator did not name.
        root: The workspace root.
        hide: `[sandbox].hide_paths`; agent6's private dirs are hidden without it.

    Returns:
        The task with a `<file>` block after each resolved reference, which stays as typed.
    """
    workspace = path_safety.host_workspace(root, hide)

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(1)
        try:
            fd = path_safety.open_contained(workspace.resolve_read(rel), os.O_RDONLY)
            with os.fdopen(fd, "rb") as handle:
                raw = handle.read(TASK_FILE_REF_MAX_BYTES)
                omitted = os.fstat(fd).st_size - len(raw)
        except (OSError, path_safety.ContainmentError):
            return match.group(0)
        text = raw.decode("utf-8", errors="replace")
        suffix = ""
        if omitted > 0:
            suffix = (
                f"\n... (truncated, {omitted} bytes omitted; "
                "use read_file with an explicit range for the rest)"
            )
        return f'{match.group(0)}\n<file path="{rel}">\n{text}{suffix}\n</file>\n'

    return _TASK_FILE_REF_RE.sub(_replace, task)
