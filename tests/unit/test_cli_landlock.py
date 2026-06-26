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

from agent6.cli import egress as cli  # _maybe_apply_agent_landlock lives here now
from agent6.detect import Environment, KernelInfo
from agent6.sandbox import LandlockNotSupportedError
from agent6.sandbox.landlock import LandlockReport


def _env(*, major: int, minor: int) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw=f"{major}.{minor}.0", major=major, minor=minor),
        userns_supported=True,
        sandbox_available=True,
    )


def _cfg(agent_network: str = "providers") -> Any:
    # Minimal stand-in: one OpenAI-compatible provider on the default port.
    entry = SimpleNamespace(base_url="https://openrouter.ai/api/v1")
    return SimpleNamespace(
        providers=SimpleNamespace(values=lambda: [entry]),
        sandbox=SimpleNamespace(agent_network=agent_network),
    )


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


def test_agent_landlock_read_roots_include_python_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    calls: list[dict[str, Any]] = []

    def _rec(**kwargs: Any) -> LandlockReport:
        calls.append(kwargs)
        return _report()

    monkeypatch.setattr(cli, "apply_agent_landlock", _rec)
    cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "hardened", _env(major=6, minor=14)
    )
    reads = calls[0]["read_paths"]
    # The agent (and the curator subprocess it re-execs) must read its own
    # Python install + source, or running from an unrelated cwd fails.
    assert Path(sys.prefix) in reads
    assert Path(cli.__file__).resolve().parents[2] in reads  # the agent6 source root


def test_agent_landlock_read_roots_include_jail_child_exec_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent read set must be a SUPERSET of the jail child's read+exec roots.

    The jail launcher (hardened) grants the child read+exec on /usr /bin /sbin
    /lib /lib64 /etc /dev by opening each from inside the agent's own Landlock
    domain. If the agent omits one, that open is denied, the child's rule is
    silently skipped, and the child cannot exec ANY binary needing it -- every
    run_command/verify/commit then fails execve EACCES (rc 127) on a no-userns
    host. /dev is the gap on a merged-/usr host (/bin /lib /lib64 /sbin are
    symlinks into /usr there); the rest matter on a split-/usr host.
    """
    calls: list[dict[str, Any]] = []

    def _rec(**kwargs: Any) -> LandlockReport:
        calls.append(kwargs)
        return _report()

    monkeypatch.setattr(cli, "apply_agent_landlock", _rec)
    cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(), "hardened", _env(major=6, minor=14)
    )
    reads = calls[0]["read_paths"]
    assert Path("/usr") in reads
    assert Path("/etc") in reads
    # The dirs that were missing and broke jail-child exec on no-userns hosts.
    for d in ("/bin", "/sbin", "/lib", "/lib64", "/dev"):
        if Path(d).exists():
            assert Path(d) in reads, f"agent read roots must include {d} (jail child exec)"


def test_agent_landlock_open_network_imposes_no_tcp_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _rec(**kwargs: Any) -> LandlockReport:
        calls.append(kwargs)
        return _report()

    monkeypatch.setattr(cli, "apply_agent_landlock", _rec)
    err = cli._maybe_apply_agent_landlock(  # pyright: ignore[reportPrivateUsage]
        _cfg(agent_network="open"), "hardened", _env(major=6, minor=14)
    )
    assert err is None
    # FS Landlock still applies on hardened, but agent_network="open" imposes
    # no TCP-connect restriction.
    assert calls[0]["tcp_connect_ports"] == ()


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
