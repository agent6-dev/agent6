# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fresh and resumed legs construct the SAME Workflow.

The two call sites drifted: resume silently dropped state_dir (so the memory
dir path, the memory index, and the memory-write verify exclusion all vanished
on resumed legs), the interactive REPL hook, and the prompt-revision wiring.
The kwarg sets are pinned mechanically so an input added to one lifecycle
fails here unless it is a deliberate leg-local seed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import agent6.app.resume
import agent6.app.run

# Seeds a fresh leg plants and a resumed leg restores elsewhere: pins come
# back from the snapshot (the loop's carryover), the standing-goal node lives
# in the restored graph. Anything else fresh-only is drift.
LEG_LOCAL_SEEDS = {"initial_pins", "standing_goal"}


def _workflow_kwargs(module: ModuleType) -> set[str]:
    assert module.__file__ is not None
    src = Path(module.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Workflow"
        ):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no Workflow(...) call found in {module.__file__}")


def test_resume_constructs_the_same_workflow_as_run() -> None:
    fresh = _workflow_kwargs(agent6.app.run)
    resumed = _workflow_kwargs(agent6.app.resume)
    assert fresh - resumed == LEG_LOCAL_SEEDS, "fresh-only kwargs beyond the deliberate seeds"
    assert resumed - fresh == set(), "resume-only kwargs have no fresh twin"
