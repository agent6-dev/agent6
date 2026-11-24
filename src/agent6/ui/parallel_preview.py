# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Describe an unsent parallel directive without starting work."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib

from agent6 import directive, terminal
from agent6.config import layer, validation
from agent6.sessions import manifest


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelPreview:
    """The lines to display and whether the draft has a complete spec and task."""

    text: str
    ready: bool


def describe(
    text: str,
    cwd: pathlib.Path,
    config_paths: tuple[pathlib.Path, ...] = (),
    *,
    preset: str = "",
    new_work: bool = False,
    session_dir: pathlib.Path | None = None,
) -> ParallelPreview | None:
    """Describe each task and its attempts under the selected child limit.

    Args:
        text: The unsent composer text.
        cwd: The repository.
        config_paths: The explicit config files in load order.
        preset: The selected preset.
        new_work: Each task starts its own parent rather than sharing a child limit.
        session_dir: An existing run replays its flag-selected preset.

    Returns:
        The requested work or the parse error. Ordinary text has no preview.
    """
    try:
        tasks = directive.parse_directive(text)
        if tasks is None:
            return None
        if session_dir is not None and not preset:
            with contextlib.suppress(manifest.ManifestError):
                preset = manifest.read_manifest(session_dir).harness.replay_preset
        limit = layer.load_effective(
            cwd, config_paths, preset=preset
        ).config.harness.parallel.max_children
        counts = [len(directive.parse_spec(t.spec, limit=limit)) for t in tasks]
    except (directive.DirectiveError, validation.ConfigError) as exc:
        return ParallelPreview(terminal.scrub_terminal_controls(str(exc)), False)
    total = sum(counts)
    if not new_work and total > limit:
        return ParallelPreview(
            f"{total} children exceed [harness.parallel].max_children = {limit}", False
        )
    children = "child" if total == 1 else "children"
    noun = "task" if len(tasks) == 1 else "tasks"
    lines = [f"Parallel request: {total} {children} for {len(tasks)} {noun}"]
    for index, (task, count) in enumerate(zip(tasks, counts, strict=True), 1):
        attempts = "attempt" if count == 1 else "attempts"
        lines.append(f"{index}. {count} {attempts} ({task.spec or 'default'}): {task.task}")
    return ParallelPreview(terminal.scrub_terminal_controls("\n".join(lines)), True)
