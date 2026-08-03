# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`warn_sandbox_gaps`: the run-entry warning when the resolved isolation
confines less than its name promises (`none`, or strict without Landlock)."""

from __future__ import annotations

import pytest

from agent6.app.confine import check_network_support, warn_sandbox_gaps
from agent6.config import Config, SandboxConfig
from agent6.sandbox.detect import Environment, KernelInfo


def _env(landlock_abi: int) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        landlock_abi=landlock_abi,
        sandbox_available=True,
    )


def _cfg(tool_network: str = "auto") -> Config:
    return Config(
        sandbox=SandboxConfig(tool_network=tool_network)  # type: ignore[arg-type]
    )


def test_none_warns_unsandboxed(capsys: pytest.CaptureFixture[str]) -> None:
    warn_sandbox_gaps("none", _env(4), _cfg())
    assert "UNSANDBOXED" in capsys.readouterr().err


def test_strict_without_landlock_warns(capsys: pytest.CaptureFixture[str]) -> None:
    """strict on a Landlock-less kernel (ABI 0) silently lost a documented
    layer: the launcher's best-effort ruleset enforces nothing and no surface
    said so, breaking the "no silent downgrade, always loudly" contract."""
    warn_sandbox_gaps("strict", _env(0), _cfg())
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "Landlock" in err


def test_strict_with_landlock_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    warn_sandbox_gaps("strict", _env(2), _cfg())
    assert capsys.readouterr().err == ""


def test_hardened_auto_warns_tool_network_degrade(capsys: pytest.CaptureFixture[str]) -> None:
    """tool_network='auto' (the secure default) can't be offline on hardened
    (no netns), so it degrades to sharing the host network -- and must SAY so,
    never silently."""
    warn_sandbox_gaps("hardened", _env(4), _cfg("auto"))
    err = capsys.readouterr().err
    assert "WARNING" in err and "tool_network" in err and "network namespace" in err


def test_hardened_allow_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    # An operator who set tool_network='allow' asked for the tool to have the
    # network, so no degrade warning; the isolation itself is fine.
    warn_sandbox_gaps("hardened", _env(4), _cfg("allow"))
    assert capsys.readouterr().err == ""


def test_explicit_block_refuses_on_hardened() -> None:
    """tool_network='block' is an ENFORCE setting: it needs a netns only strict
    provides, so on hardened we refuse (name what's unsupported + the fix)
    rather than run silently under-confined. 'auto' degrades instead."""
    err = check_network_support(_cfg("block"), "hardened")
    assert err is not None
    assert "tool_network = 'block'" in err and "auto" in err and "strict" in err
    # auto is NOT refused (it degrades with a warning) -> None.
    assert check_network_support(_cfg("auto"), "hardened") is None
    # On strict, block is enforceable -> no refusal.
    assert check_network_support(_cfg("block"), "strict") is None
