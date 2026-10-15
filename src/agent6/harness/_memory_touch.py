# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Which facts of the memory store a tool call touched, and how.

The loop's bookkeeping for the use record (`memory.record_use`) reads this
after every dispatched call.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from agent6 import memory
from agent6.tools import patch_apply, results


@dataclasses.dataclass(frozen=True, slots=True)
class MemoryTouch:
    """What one tool call did inside the memory store.

    Attributes:
        facts: Fact name to `read`, `create`, `edit` or `delete`.
        store_paths: Every path under the store, spelled store-relative as the results spell it.
        whole_store: Whether every path the call named is under the store.
    """

    facts: dict[str, str]
    store_paths: tuple[str, ...]
    whole_store: bool


def memory_store_touch(
    state_dir: pathlib.Path | None, name: str, result: results.ToolResult, tool_input: Any
) -> MemoryTouch | None:
    """Return what a tool call did inside the memory store.

    Judged on the model's input paths: the store sits outside the workspace
    root, so only an absolute path reaches it. `apply_patch` names its files in
    the headers, and one patch may address the store and the workspace both.
    A fact is a file the store's name rule accepts; other files are not.

    Args:
        state_dir: The run's state directory, or None when no store is wired.
        name: The tool's name.
        result: The tool's result.
        tool_input: The model's input to the tool.

    Returns:
        The touch; None when no path is under the store.
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
    # Both sides resolved: a symlinked state home never sits under the store's raw path.
    store = memory.memory_dir(state_dir).resolve()
    addressed = [
        (pathlib.Path(p).resolve(), op)
        for p, op in zip(paths, ops, strict=True)
        if pathlib.Path(p).is_absolute()
    ]
    inside = [(p, op) for p, op in addressed if p.is_relative_to(store)]
    if not inside:
        return None
    return MemoryTouch(
        facts={
            p.stem: op
            for p, op in inside
            if p.parent == store and p.suffix == ".md" and memory.is_memory_name(p.stem)
        },
        store_paths=tuple(str(p.relative_to(store)) for p, _ in inside),
        whole_store=len(inside) == len(paths),
    )
