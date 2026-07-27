# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`warn_sandbox_gaps`: the run-entry warning when the resolved profile
confines less than its name promises (`none`, or strict without Landlock)."""

from __future__ import annotations

import pytest

from agent6.app.egress import warn_sandbox_gaps
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


def test_none_warns_unsandboxed(capsys: pytest.CaptureFixture[str]) -> None:
    warn_sandbox_gaps("none", _env(4))
    assert "UNSANDBOXED" in capsys.readouterr().err


def test_strict_without_landlock_warns(capsys: pytest.CaptureFixture[str]) -> None:
    """strict on a Landlock-less kernel (ABI 0) silently lost a documented
    layer: the launcher's best-effort ruleset enforces nothing and no surface
    said so, breaking the "no silent downgrade, always loudly" contract."""
    warn_sandbox_gaps("strict", _env(0))
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "Landlock" in err


def test_strict_with_landlock_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    warn_sandbox_gaps("strict", _env(2))
    assert capsys.readouterr().err == ""


def test_hardened_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    warn_sandbox_gaps("hardened", _env(4))
    assert capsys.readouterr().err == ""
