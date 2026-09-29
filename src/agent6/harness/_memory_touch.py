# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Which facts of the memory store a tool call touched, and how: the loop's
per-execution bookkeeping for the use record (`memory.record_use`) reads this
after every dispatched call."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.memory import is_memory_name, memory_dir
from agent6.tools.patch_apply import PatchError, patch_op, patch_target_path, split_patch_files
from agent6.tools.results import EditResult, ToolResult


def memory_store_facts(
    state_dir: Path | None, name: str, result: ToolResult, tool_input: Any
) -> dict[str, str] | None:
    """The facts a tool call addressed in the memory store, each with what
    the call did to it (`read`; `create`, `edit` or `delete` from the
    edit result's `created` or the patch's headers); None when the call was not
    (wholly) about the store. Judged on the model's input paths: the
    store sits outside the workspace root, so only an absolute path
    reaches it (a result's path is store-relative, and matching on it
    never fires). `apply_patch` normally carries no `path` and names its
    files in the headers; every one must be under the store (a patch over
    the store and the workspace together is workspace work). A fact is a
    file the store's name rule accepts; the index, the rulings and any
    other file are not, so a call about them alone answers {}."""
    if state_dir is None or not isinstance(tool_input, dict):
        return None
    try:
        sections = (
            split_patch_files(str(tool_input.get("patch", ""))) if name == "apply_patch" else []
        )
        if tool_input.get("path"):
            paths = [str(tool_input["path"])]
            ops = [
                ("create" if result.created else "edit")
                if isinstance(result, EditResult)
                else patch_op(sections[0])
                if sections
                else "read"
            ]
        else:
            paths = [patch_target_path(section) for section in sections]
            ops = [patch_op(section) for section in sections]
    except PatchError:
        return None
    if not paths or not all(p.startswith("/") for p in paths):
        return None
    # Both sides resolved: the model is told the store's unresolved path
    # (a symlinked state home), and a resolved path never sits under it.
    store = memory_dir(state_dir).resolve()
    resolved = [Path(p).resolve() for p in paths]
    if not all(p.is_relative_to(store) for p in resolved):
        return None
    return {
        p.stem: op
        for p, op in zip(resolved, ops, strict=True)
        if p.parent == store and p.suffix == ".md" and is_memory_name(p.stem)
    }
