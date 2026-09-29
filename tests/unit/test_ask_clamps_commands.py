# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 ask` never runs a command unwatched."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

from agent6.config import Config


def _cfg(run_commands: str) -> Config:
    return Config.model_validate({"sandbox": {"run_commands": run_commands}})


def test_auto_approval_becomes_a_prompt_in_ask() -> None:
    """Auto approval becomes a prompt in ask.

    An ask is a question with the operator sitting there; `run_commands = "yes"` could run the
    command it was asked to write.
    """
    assert _cfg("yes").with_run_commands_clamped().sandbox.run_commands == "ask"


@pytest.mark.parametrize("setting", ["ask", "no"])
def test_the_clamp_only_ever_tightens(setting: str) -> None:
    """`no` stays refused: a run may narrow a boundary the operator set, never widen one."""
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
    """The ask lifecycle clamps before the session is built, so every reader of the knob agrees."""
    from agent6.app import run as run_mod
    from agent6.app.preflight import SessionRefusedError

    seen: list[str] = []

    def capture(cfg: Config, **_kw: object) -> str:
        seen.append(cfg.sandbox.run_commands)
        raise SessionRefusedError(2)

    monkeypatch.setattr(run_mod, "select_isolation", capture)
    monkeypatch.chdir(tmp_path)
    modes: tuple[tuple[Literal["run", "plan", "ask"], str], ...] = (
        ("ask", "ask"),
        ("run", "yes"),
    )
    for mode, expected in modes:
        seen.clear()
        run_mod.run_task(_cfg("yes"), "q", started_at=time.time(), frontend=MagicMock(), mode=mode)
        assert seen == [expected], f"{mode} saw {seen}"


def test_no_commands_pins_the_knob_shut() -> None:
    """--no-commands is the symmetric flag to --auto-approve: one knob, two per-invocation pins."""
    for start in ("yes", "ask", "no"):
        cfg = Config.model_validate({"sandbox": {"run_commands": start}})
        assert cfg.with_sandbox_overrides(no_commands=True).sandbox.run_commands == "no"


def test_tightening_needs_no_permission_but_widening_does() -> None:
    """Tightening needs no permission but widening does.

    --auto-approve never resurrects a withheld "no"; --no-commands always may tighten.
    """
    withheld = Config.model_validate({"sandbox": {"run_commands": "no"}})
    assert withheld.with_sandbox_overrides(auto_approve=True).sandbox.run_commands == "no"
    asked = Config.model_validate({"sandbox": {"run_commands": "ask"}})
    assert asked.with_sandbox_overrides(auto_approve=True).sandbox.run_commands == "yes"


def test_an_explicit_auto_approve_survives_the_ask_clamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit --auto-approve survives the ask clamp.

    The clamp stops an ask inheriting a standing `run_commands = "yes"`; the flag typed on this
    invocation is the most specific layer and unreachable by the LLM.
    """
    from agent6.app import run as run_mod
    from agent6.app._setup import SandboxOverrides
    from agent6.app.preflight import SessionRefusedError

    seen: list[str] = []

    def capture(cfg: Config, **_kw: object) -> str:
        seen.append(cfg.sandbox.run_commands)
        raise SessionRefusedError(2)

    monkeypatch.setattr(run_mod, "select_isolation", capture)
    monkeypatch.chdir(tmp_path)
    run_mod.run_task(
        _cfg("ask"),
        "q",
        started_at=time.time(),
        frontend=MagicMock(),
        mode="ask",
        sandbox_overrides=SandboxOverrides(auto_approve=True),
    )
    assert seen == ["yes"], f"the operator's own flag was undone: {seen}"
