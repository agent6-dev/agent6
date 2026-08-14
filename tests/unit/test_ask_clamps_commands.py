# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ask` never runs a command unwatched."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

from agent6.config import Config


def _cfg(run_commands: str) -> Config:
    return Config.model_validate({"sandbox": {"run_commands": run_commands}})


def test_auto_approval_becomes_a_prompt_in_ask() -> None:
    """An ask is a question with the operator sitting there, often in a
    directory that is not a repo. `run_commands = "yes"` there means the answer
    to "give me a command to convert these files" could run it first."""
    assert _cfg("yes").with_run_commands_clamped().sandbox.run_commands == "ask"


@pytest.mark.parametrize("setting", ["ask", "no"])
def test_the_clamp_only_ever_tightens(setting: str) -> None:
    """`no` must stay refused: a run may narrow a boundary the operator set,
    never widen one. `ask` is already the clamped value."""
    assert _cfg(setting).with_run_commands_clamped().sandbox.run_commands == setting


def test_the_clamp_leaves_the_rest_of_the_config_alone() -> None:
    cfg = Config.model_validate(
        {"sandbox": {"run_commands": "yes", "protect_git": True}, "preset": "quick"}
    )
    clamped = cfg.with_run_commands_clamped()
    assert clamped.sandbox.protect_git is True
    assert clamped.preset == "quick"
    assert cfg.sandbox.run_commands == "yes"  # the operator's config is untouched


def test_the_ask_lifecycle_clamps_before_anything_reads_the_knob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The clamp has to land before the session is built, or the tool gate, the
    status line and the detach prompt each answer differently."""
    from agent6.app import run as run_mod
    from agent6.app.preflight import SessionRefused

    seen: list[str] = []

    def capture(cfg: Config, **_kw: object) -> str:
        seen.append(cfg.sandbox.run_commands)
        raise SessionRefused(2)

    monkeypatch.setattr(run_mod, "select_isolation", capture)
    monkeypatch.chdir(tmp_path)
    modes: tuple[tuple[Literal["run", "plan", "ask"], str], ...] = (
        ("ask", "ask"),
        ("run", "yes"),
    )
    for mode, expected in modes:
        seen.clear()
        run_mod.run_task(_cfg("yes"), "q", frontend=MagicMock(), mode=mode)
        assert seen == [expected], f"{mode} saw {seen}"


def test_no_commands_pins_the_knob_shut() -> None:
    """The symmetric flag to --auto-approve: one knob, two per-invocation pins.
    A btw uses it, but an operator asking a quick question in a strange repo
    has the same reason to."""
    for start in ("yes", "ask", "no"):
        cfg = Config.model_validate({"sandbox": {"run_commands": start}})
        assert cfg.with_sandbox_overrides(no_commands=True).sandbox.run_commands == "no"


def test_tightening_needs_no_permission_but_widening_does() -> None:
    """--auto-approve must never resurrect a withheld "no" (a flag cannot grant
    what the standing policy denied); --no-commands always may, because
    tightening is always allowed."""
    withheld = Config.model_validate({"sandbox": {"run_commands": "no"}})
    assert withheld.with_sandbox_overrides(auto_approve=True).sandbox.run_commands == "no"
    asked = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    assert asked.with_sandbox_overrides(auto_approve=True).sandbox.run_commands == "yes"


def test_an_explicit_auto_approve_survives_the_ask_clamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The clamp exists to stop an ask inheriting a STANDING `run_commands =
    "yes"` while nobody watches. An operator typing --auto-approve on this
    invocation is the opposite: the most specific layer, and unreachable by the
    LLM.

    Clamping it made the flag inert, and every headless `ask --auto-approve`
    was then refused by the approval preflight -- whose message recommends
    --auto-approve, the flag that had just been undone.
    """
    from agent6.app import run as run_mod
    from agent6.app._setup import SandboxOverrides
    from agent6.app.preflight import SessionRefused

    seen: list[str] = []

    def capture(cfg: Config, **_kw: object) -> str:
        seen.append(cfg.sandbox.run_commands)
        raise SessionRefused(2)

    monkeypatch.setattr(run_mod, "select_isolation", capture)
    monkeypatch.chdir(tmp_path)
    run_mod.run_task(
        _cfg("ask"),
        "q",
        frontend=MagicMock(),
        mode="ask",
        sandbox_overrides=SandboxOverrides(auto_approve=True),
    )
    assert seen == ["yes"], f"the operator's own flag was undone: {seen}"
