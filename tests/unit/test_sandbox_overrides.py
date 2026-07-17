# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""CLI wiring of the per-invocation sandbox/approval opt-outs.

Unit tests already cover the pieces (detect.select_profile env setter,
Config.with_sandbox_overrides, the confirm gate). These cover the wiring the
CLI does on top: parsing the flags into _SandboxOverrides and applying them,
and the env mechanism a `machine run` relies on to reach its agent subprocesses.
"""

from __future__ import annotations

import argparse

import pytest

from agent6.app._setup import SandboxOverrides as _SandboxOverrides
from agent6.config import Config
from agent6.sandbox.detect import Environment, KernelInfo, select_profile


def _args(**kw: bool) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_from_args_reads_both_flags() -> None:
    o = _SandboxOverrides.from_args(_args(dangerously_disable_sandbox=True, auto_approve=True))
    assert o.disable_sandbox is True
    assert o.auto_approve is True


def test_from_args_defaults_false_when_flags_absent() -> None:
    # Commands that do not offer the flags (or an older namespace) get False.
    o = _SandboxOverrides.from_args(_args())
    assert o.disable_sandbox is False
    assert o.auto_approve is False


def test_apply_flag_path_forces_none_and_auto_approve() -> None:
    cfg = Config()
    assert cfg.sandbox.profile == "auto"
    out = _SandboxOverrides(disable_sandbox=True, auto_approve=True).apply(cfg)
    assert out.sandbox.profile == "none"
    assert out.sandbox.run_commands == "yes"


def test_apply_noop_when_no_flags() -> None:
    cfg = Config()
    assert _SandboxOverrides().apply(cfg) is cfg


def _linux_env() -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        sandbox_available=True,
    )


def test_machine_env_mechanism_forces_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # `machine run --dangerously-disable-sandbox` sets this env var; the machine
    # supervisor's select_profile (the same function) must then resolve to none
    # regardless of the machine's configured profile, and it passes that to each
    # agent subprocess in the request.
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    assert select_profile("strict", _linux_env()) == "none"
    assert select_profile("auto", _linux_env()) == "none"
