# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fresh and resumed legs run the SAME leg body.

The two lifecycles once each constructed the Workflow and drifted (resume
silently dropped state_dir, the interactive REPL hook, and the prompt-revision
wiring). Now neither constructs one: both hand `LegInputs` to `_leg.run_leg`,
the one place the Workflow is built, so an input added to one lifecycle cannot
be missing from the other. Pinned structurally: a `Workflow(...)` call in
either lifecycle module is the drift returning.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import agent6.app._leg
import agent6.app.resume
import agent6.app.run


def _calls(module: ModuleType, name: str) -> int:
    assert module.__file__ is not None
    src = Path(module.__file__).read_text(encoding="utf-8")
    return sum(
        1
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )


def test_neither_lifecycle_builds_its_own_workflow() -> None:
    assert _calls(agent6.app.run, "Workflow") == 0
    assert _calls(agent6.app.resume, "Workflow") == 0
    assert _calls(agent6.app._leg, "Workflow") == 1  # pyright: ignore[reportPrivateUsage]


def test_both_lifecycles_run_the_one_leg_body() -> None:
    assert _calls(agent6.app.run, "run_leg") == 1
    assert _calls(agent6.app.resume, "run_leg") == 1


def test_both_lifecycles_detach_under_the_invocations_flags() -> None:
    """A `/detach` spawns a background `resume`; each lifecycle hands it this
    invocation's overrides as flags (`override_flags`), or the detached leg
    runs under the config's defaults: an `--auto-approve` run detached and
    waited on its first approval with nobody attached."""
    assert _calls(agent6.app.run, "override_flags") == 1
    assert _calls(agent6.app.resume, "override_flags") == 1
