# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the agent-process Landlock wiring in agent6.cli.

These never call the real ``apply_agent_landlock`` (which is irrevocable and
would confine the test process); the symbol is monkeypatched with a recorder.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent6 import cli
from agent6.detect import Environment, KernelInfo
from agent6.sandbox import LandlockNotSupportedError
from agent6.sandbox.landlock import LandlockReport


def _env(*, major: int, minor: int) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw=f"{major}.{minor}.0", major=major, minor=minor),
        userns_supported=True,
    )


def _cfg() -> Any:
    # Minimal stand-in: one OpenAI-compatible provider on the default port.
    entry = SimpleNamespace(base_url="https://openrouter.ai/api/v1")
    return SimpleNamespace(providers=SimpleNamespace(values=lambda: [entry]))


def _report() -> LandlockReport:
    return LandlockReport(
        abi=4,
        fs_read=(Path("/"),),
        fs_write=(Path("/"),),
        tcp_connect_ports=(443,),
        tcp_supported=True,
    )


def test_agent_landlock_applied_on_hardened(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _rec(**kwargs: Any) -> LandlockReport:
        calls.append(kwargs)
        return _report()

    monkeypatch.setattr(cli, "apply_agent_landlock", _rec)
    err = cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "hardened", _env(major=6, minor=14)
    )
    assert err is None
    assert len(calls) == 1
    # Ports are derived from the configured providers (default 443 here),
    # not blanket-allowed.
    assert calls[0]["tcp_connect_ports"] == (443,)


def test_agent_landlock_skipped_on_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _rec(**kwargs: Any) -> LandlockReport:
        calls.append(kwargs)
        return _report()

    monkeypatch.setattr(cli, "apply_agent_landlock", _rec)
    err = cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "strict", _env(major=6, minor=14)
    )
    assert err is None
    assert calls == []


def test_agent_landlock_skipped_when_kernel_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _rec(**kwargs: Any) -> LandlockReport:
        calls.append(kwargs)
        return _report()

    monkeypatch.setattr(cli, "apply_agent_landlock", _rec)
    err = cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "hardened", _env(major=5, minor=10)
    )
    assert err is None
    assert calls == []


def test_agent_landlock_warns_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**kwargs: Any) -> LandlockReport:
        raise LandlockNotSupportedError("ABI 0")

    monkeypatch.setattr(cli, "apply_agent_landlock", _raise)
    err = cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "hardened", _env(major=6, minor=14)
    )
    # A kernel without Landlock degrades with a warning, not a refusal.
    assert err is None


def test_agent_landlock_refuses_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(**kwargs: Any) -> LandlockReport:
        raise OSError("EPERM")

    monkeypatch.setattr(cli, "apply_agent_landlock", _raise)
    err = cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "hardened", _env(major=6, minor=14)
    )
    # A kernel that supports Landlock but rejects our ruleset is fail-closed:
    # the run is refused rather than proceeding unconfined.
    assert err is not None
    assert "Landlock" in err
