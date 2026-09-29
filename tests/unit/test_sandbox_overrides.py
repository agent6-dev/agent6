# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI wiring of the per-invocation sandbox/approval opt-outs.

Unit tests already cover the pieces (detect.resolve_isolation env setter,
Config.with_sandbox_overrides, the confirm gate). These cover the wiring the
CLI does on top: parsing the flags into _SandboxOverrides and applying them,
and the env mechanism a `machine run` relies on to reach its agent subprocesses.
"""

from __future__ import annotations

import argparse

import pytest

from agent6.app import _setup
from agent6.config import Config
from agent6.sandbox import detect


def _args(**kw: bool) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_from_args_reads_both_flags() -> None:
    o = _setup.SandboxOverrides.from_args(
        _args(dangerously_disable_sandbox=True, auto_approve=True)
    )
    assert o.disable_sandbox is True
    assert o.auto_approve is True


def test_from_args_defaults_false_when_flags_absent() -> None:
    # Commands that do not offer the flags (or an older namespace) get False.
    o = _setup.SandboxOverrides.from_args(_args())
    assert o.disable_sandbox is False
    assert o.auto_approve is False


def test_apply_flag_path_forces_none_and_auto_approve() -> None:
    cfg = Config()
    assert cfg.sandbox.isolation == "auto"
    out = _setup.SandboxOverrides(disable_sandbox=True, auto_approve=True).apply(cfg)
    assert out.sandbox.isolation == "none"
    assert out.sandbox.run_commands == "yes"


def test_apply_noop_when_no_flags() -> None:
    cfg = Config()
    assert _setup.SandboxOverrides().apply(cfg) is cfg


def _linux_env() -> detect.Environment:
    return detect.Environment(
        in_container=False,
        container_signals=(),
        kernel=detect.KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def test_machine_env_mechanism_forces_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # `machine run --dangerously-disable-sandbox` sets this env var; the machine
    # supervisor's resolve_isolation (the same function) must then resolve to none
    # regardless of the machine's configured isolation, and it passes that to each
    # agent subprocess in the request.
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    assert detect.resolve_isolation("strict", _linux_env()) == "none"
    assert detect.resolve_isolation("auto", _linux_env()) == "none"


def test_overrides_render_as_the_flags_that_set_them() -> None:
    """Overrides render as the flags that set them.

    A continuation this invocation spawns (a detached resume) carries the overrides as flags; an
    execution that drops them runs under the config's defaults (an `--auto-approve` run detaches and
    waits on its first approval; a `--max-usd` cap falls back to the config's).
    """
    sandbox = _setup.SandboxOverrides(disable_sandbox=True, auto_approve=True)
    budget = _setup.BudgetOverrides(max_usd=0.25, max_tokens_fallback=1000)
    assert sandbox.argv() == ["--dangerously-disable-sandbox", "--auto-approve"]
    assert budget.argv() == ["--max-usd", "0.25", "--max-tokens-fallback", "1000"]
    assert _setup.override_flags(budget, sandbox, None) == [*budget.argv(), *sandbox.argv()]
    assert _setup.override_flags(None, None, None) == []
    assert _setup.SandboxOverrides().argv() == [] and _setup.BudgetOverrides().argv() == []
