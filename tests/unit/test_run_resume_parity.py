# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fresh and resumed executions run the SAME execution body.

The two lifecycles once each constructed the Harness and drifted (resume
silently dropped state_dir, the interactive REPL hook, and the prompt-revision
wiring). Now neither constructs one: both hand `ExecutionInputs` to `_execution.run_execution`,
the one place the Harness is built, so an input added to one lifecycle cannot
be missing from the other. Pinned structurally: a `Harness(...)` call in
either lifecycle module is the drift returning.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import agent6.app._execution
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
    assert _calls(agent6.app.run, "Harness") == 0
    assert _calls(agent6.app.resume, "Harness") == 0
    assert _calls(agent6.app._execution, "Harness") == 1  # pyright: ignore[reportPrivateUsage]


def test_both_lifecycles_run_the_one_execution_body() -> None:
    assert _calls(agent6.app.run, "run_execution") == 1
    assert _calls(agent6.app.resume, "run_execution") == 1


def test_both_lifecycles_detach_under_the_invocations_flags() -> None:
    """A `/detach` spawns a background `resume`; each lifecycle hands it this
    invocation's overrides as flags (`override_flags`), or the detached execution
    runs under the config's defaults: an `--auto-approve` run detached and
    waited on its first approval with nobody attached."""
    assert _calls(agent6.app.run, "override_flags") == 1
    assert _calls(agent6.app.resume, "override_flags") == 1


def test_both_lifecycles_hand_a_detach_to_the_one_helper() -> None:
    assert _calls(agent6.app.run, "detach_to_background") == 1
    assert _calls(agent6.app.resume, "detach_to_background") == 1


def test_every_budget_override_survives_a_detach() -> None:
    """A detached execution re-reads config for anything the flags do not carry, so a
    dropped flag silently restores the config default: `--max-percent` was
    missing, and its default is -1 (unlimited)."""
    import argparse

    from agent6.app._setup import BudgetOverrides

    args = argparse.Namespace(max_usd=2.0, max_tokens_fallback=1000, max_percent=5.0)
    overrides = BudgetOverrides.from_args(args)
    argv = overrides.argv()
    for field in ("max_usd", "max_tokens_fallback", "max_percent"):
        assert getattr(overrides, field) is not None
        assert f"--{field.replace('_', '-')}" in argv, field


def test_the_model_flag_survives_a_detach() -> None:
    """A `--model` route rides the detached execution's argv like the budget and
    sandbox flags, else the detached execution re-reads the config's model."""
    from agent6.app._setup import override_flags
    from agent6.kinds import ModelRoute

    assert override_flags(None, None, ModelRoute("p", "m")) == ["--model", "p/m"]
    assert override_flags(None, None, None) == []
