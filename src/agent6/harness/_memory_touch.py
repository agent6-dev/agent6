# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Which facts of the memory store a tool call touched, and how.

The loop's bookkeeping for the use record (`memory.record_use`) reads this
after every dispatched call.
"""

from __future__ import annotations

import pathlib
from typing import Any

from agent6 import memory
from agent6.tools import patch_apply, results


def memory_store_facts(
    state_dir: pathlib.Path | None, name: str, result: results.ToolResult, tool_input: Any
) -> dict[str, str] | None:
    """Return the memory-store facts a tool call addressed, each with what it did.

    Judged on the model's input paths: the store sits outside the workspace
    root, so only an absolute path reaches it. `apply_patch` names its files in
    the headers; every one must be under the store, since a patch over the store
    and the workspace together is workspace work. A fact is a file the store's
    name rule accepts; the index, the rulings and any other file are not.

    Args:
        state_dir: The run's state directory, or None when no store is wired.
        name: The tool's name.
        result: The tool's result.
        tool_input: The model's input to the tool.

    Returns:
        Fact name to `read`, `create`, `edit` or `delete`; {} when the call was
        about the store but no fact; None when it was not wholly about the store.
    """
    if state_dir is None or not isinstance(tool_input, dict):
        return None
    try:
        sections = (
            patch_apply.split_patch_files(str(tool_input.get("patch", "")))
            if name == "apply_patch"
            else []
        )
        if tool_input.get("path"):
            paths = [str(tool_input["path"])]
            ops = [
                ("create" if result.created else "edit")
                if isinstance(result, results.EditResult)
                else patch_apply.patch_op(sections[0])
                if sections
                else "read"
            ]
        else:
            paths = [patch_apply.patch_target_path(section) for section in sections]
            ops = [patch_apply.patch_op(section) for section in sections]
    except patch_apply.PatchError:
        return None
    if not paths or not all(p.startswith("/") for p in paths):
        return None
    # Both sides resolved: a symlinked state home never sits under the store's raw path.
    store = memory.memory_dir(state_dir).resolve()
    resolved = [pathlib.Path(p).resolve() for p in paths]
    if not all(p.is_relative_to(store) for p in resolved):
        return None
    return {
        p.stem: op
        for p, op in zip(resolved, ops, strict=True)
        if p.parent == store and p.suffix == ".md" and memory.is_memory_name(p.stem)
    }
