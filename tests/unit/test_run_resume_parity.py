# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fresh and resumed executions share one execution body.

Both hand `ExecutionInputs` to `execution.run_execution`, the one place the Harness is built.
A `Harness(...)` call in either lifecycle module would let their wiring drift apart.
"""

from __future__ import annotations

import ast
import pathlib
import types

import agent6.app.execution
import agent6.app.resume
import agent6.app.run


def _calls(module: types.ModuleType, name: str) -> int:
    assert module.__file__ is not None
    src = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    return sum(
        1
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


def test_neither_lifecycle_builds_its_own_workflow() -> None:
    assert _calls(agent6.app.run, "Harness") == 0
    assert _calls(agent6.app.resume, "Harness") == 0
    assert _calls(agent6.app.execution, "Harness") == 1  # pyright: ignore[reportPrivateUsage]


def test_both_lifecycles_run_the_one_execution_body() -> None:
    assert _calls(agent6.app.run, "run_execution") == 1
    assert _calls(agent6.app.resume, "run_execution") == 1


def test_both_lifecycles_detach_under_the_invocations_flags() -> None:
    """Both lifecycles detach under the invocation's flags.

    A `/detach` spawns a background `resume`; each lifecycle hands it this invocation's overrides as
    flags (`override_flags`), or the detached execution runs under the config's defaults: an
    `--auto-approve` run detached would wait on its first approval with nobody attached.
    """
    assert _calls(agent6.app.run, "override_flags") == 1
    assert _calls(agent6.app.resume, "override_flags") == 1


def test_both_lifecycles_hand_a_detach_to_the_one_helper() -> None:
    assert _calls(agent6.app.run, "detach_to_background") == 1
    assert _calls(agent6.app.resume, "detach_to_background") == 1


def test_every_budget_override_survives_a_detach() -> None:
    """Every budget override survives a detach.

    A detached execution re-reads config for anything the flags do not carry, so a dropped flag
    silently restores the config default; `--max-plan-points` defaults to -1 (unlimited).
    """
    import argparse

    from agent6.app import session_setup

    args = argparse.Namespace(max_usd=2.0, max_tokens_fallback=1000, max_plan_points=5.0)
    overrides = session_setup.BudgetOverrides.from_args(args)
    argv = overrides.argv()
    for field in ("max_usd", "max_tokens_fallback", "max_plan_points"):
        assert getattr(overrides, field) is not None
        assert f"--{field.replace('_', '-')}" in argv, field


def test_the_model_flag_survives_a_detach() -> None:
    """The model flag survives a detach.

    A `--model` route rides the detached execution's argv like the budget and sandbox flags, or the
    detached execution re-reads the config's model.
    """
    from agent6 import kinds
    from agent6.app import session_setup

    assert session_setup.override_flags(None, None, kinds.ModelRoute("p", "m")) == [
        "--model",
        "p/m",
    ]
    assert session_setup.override_flags(None, None, None) == []


def test_both_lifecycles_share_the_preparation_steps() -> None:
    """A preparation step wired into one lifecycle cannot be missing from the other."""
    for step in (
        "hold_workspace_and_select_isolation",
        "take_single_writer",
        "headless_check",
        "operator_channels",
        "workspace_notices",
        "undo_forker",
        "settle_gate",
    ):
        assert _calls(agent6.app.run, step) >= 1, step
        assert _calls(agent6.app.resume, step) == 1, step
