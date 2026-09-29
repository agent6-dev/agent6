# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.sandbox.detect."""

from __future__ import annotations

import pytest

import agent6.sandbox.detect as detect_mod
from agent6.sandbox.detect import (
    Environment,
    IsolationUnavailableError,
    KernelInfo,
    _parse_kernel,  # pyright: ignore[reportPrivateUsage]
    detect_container_signals,
    resolve_isolation,
)


def test_parse_kernel_basic() -> None:
    k = _parse_kernel("6.7.5-arch1")
    assert (k.major, k.minor) == (6, 7)


def test_parse_kernel_too_old() -> None:
    k = _parse_kernel("5.10.0")
    assert (k.major, k.minor) == (5, 10)


def test_parse_kernel_unknown() -> None:
    k = _parse_kernel("garbage")
    assert (k.major, k.minor) == (0, 0)


def test_detect_container_signals_returns_tuple() -> None:
    # Just make sure it's a tuple of strs and doesn't crash.
    signals = detect_container_signals()
    assert isinstance(signals, tuple)
    for s in signals:
        assert isinstance(s, str)


def test_detect_container_signals_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rootless podman: /run/.containerenv, no /.dockerenv, often no "podman" cgroup token.
    from pathlib import Path

    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        s = str(self)
        if s == "/run/.containerenv":
            return True
        if s == "/.dockerenv":
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    def fake_read_text(self: Path, **k: object) -> str:
        return "0::/user.slice/session.scope"

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.delenv("REMOTE_CONTAINERS", raising=False)
    monkeypatch.delenv("CODESPACES", raising=False)
    signals = detect_container_signals()
    assert "/run/.containerenv" in signals
    assert "/.dockerenv" not in signals


def _env(*, userns: bool, landlock_abi: int = 4) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=userns,
        landlock_abi=landlock_abi,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def test_detected_profile_strict_when_userns_supported() -> None:
    assert _env(userns=True).detected_isolation == "strict"


def test_detected_profile_hardened_when_userns_blocked() -> None:
    assert _env(userns=False, landlock_abi=1).detected_isolation == "hardened"


def test_detected_profile_none_without_userns_or_landlock() -> None:
    # With neither userns nor Landlock there is no confinement, so the truthful answer is `none`.
    assert _env(userns=False, landlock_abi=0).detected_isolation == "none"


def test_select_profile_strict_refuses_silent_downgrade() -> None:
    with pytest.raises(IsolationUnavailableError, match="user namespaces"):
        resolve_isolation("strict", _env(userns=False))


def test_select_profile_strict_passes_when_supported() -> None:
    assert resolve_isolation("strict", _env(userns=True)) == "strict"


def test_select_profile_hardened_ok_at_abi3_plus() -> None:
    """ABI 3 (Linux 6.2) is the floor for explicit hardened, where Landlock confines truncate."""
    assert resolve_isolation("hardened", _env(userns=True)) == "hardened"  # abi 4
    assert resolve_isolation("hardened", _env(userns=False, landlock_abi=3)) == "hardened"


def test_select_profile_hardened_refuses_without_landlock() -> None:
    # An explicit request the kernel cannot back is refused with a remedy, never under-delivered.
    with pytest.raises(IsolationUnavailableError, match="Landlock"):
        resolve_isolation("hardened", _env(userns=False, landlock_abi=0))


@pytest.mark.parametrize("abi", [1, 2])
def test_select_profile_hardened_refuses_below_abi3(abi: int) -> None:
    """An explicit `hardened` on Landlock ABI 1 or 2 refuses, naming the floor and the fix.

    Those ABIs confine path writes but not truncation, so the label would over-promise.
    """
    with pytest.raises(IsolationUnavailableError, match="truncat") as exc:
        resolve_isolation("hardened", _env(userns=False, landlock_abi=abi))
    msg = str(exc.value)
    assert "ABI 3" in msg and "6.2" in msg and "auto" in msg


@pytest.mark.parametrize("abi", [1, 2, 3])
def test_select_profile_auto_stays_hardened_below_abi3(abi: int) -> None:
    """`auto` on ABI 1 or 2 still resolves to hardened; partial truncate confinement is a warning.

    Dropping to none would leave common hosts (Debian 12, kernel 6.1) unconfined.
    """
    env = _env(userns=False, landlock_abi=abi)
    assert env.detected_isolation == "hardened"
    assert resolve_isolation("auto", env) == "hardened"


def _env_c(*, userns: bool, in_container: bool) -> Environment:
    return Environment(
        in_container=in_container,
        container_signals=("docker",) if in_container else (),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=userns,
        landlock_abi=4,
        seccomp_arch_supported=True,
        sandbox_available=True,
    )


def test_select_profile_explicit_none_is_self_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit `isolation = "none"` is the operator's consent; the startup warning is the net.
    monkeypatch.delenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", raising=False)
    assert resolve_isolation("none", _env_c(userns=True, in_container=False)) == "none"
    assert resolve_isolation("none", _env_c(userns=False, in_container=True)) == "none"


def test_select_profile_auto_reaches_none_only_without_any_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `auto` resolves to none only on a host with no confinement mechanism, and loudly.
    monkeypatch.delenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", raising=False)
    assert resolve_isolation("auto", _env_c(userns=True, in_container=False)) == "strict"
    assert resolve_isolation("auto", _env_c(userns=False, in_container=False)) == "hardened"
    assert resolve_isolation("auto", _env(userns=False, landlock_abi=0)) == "none"


def test_env_setter_forces_none_over_any_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # AGENT6_DANGEROUSLY_DISABLE_SANDBOX forces the unsandboxed isolation whatever the config asked.
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    assert resolve_isolation("auto", _env_c(userns=True, in_container=False)) == "none"
    assert resolve_isolation("strict", _env_c(userns=True, in_container=False)) == "none"
    assert resolve_isolation("hardened", _env_c(userns=False, in_container=False)) == "none"


def test_env_setter_forces_none_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    env = Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="", major=0, minor=0),
        userns_supported=False,
        landlock_abi=0,
        seccomp_arch_supported=True,
        sandbox_available=False,
    )
    assert resolve_isolation("strict", env) == "none"


def test_probe_landlock_abi_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A probe error reads as "no Landlock", never as a confinement the kernel may not deliver.
    from agent6.sandbox.landlock import LandlockError

    detect_mod.probe_landlock_abi.cache_clear()

    def _boom() -> int:
        raise LandlockError("probe failed")

    monkeypatch.setattr(detect_mod, "landlock_abi", _boom)
    try:
        assert detect_mod.probe_landlock_abi() == 0
    finally:
        detect_mod.probe_landlock_abi.cache_clear()


def test_sandbox_disabled_by_env_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", raising=False)
    assert detect_mod.sandbox_disabled_by_env() is False
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    assert detect_mod.sandbox_disabled_by_env() is True
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "yes")  # only "1" counts
    assert detect_mod.sandbox_disabled_by_env() is False


def test_select_profile_auto_never_unsandboxes_while_a_mechanism_exists() -> None:
    # auto never resolves to none while the host offers userns or Landlock.
    assert resolve_isolation("auto", _env_c(userns=True, in_container=True)) != "none"
    assert resolve_isolation("auto", _env_c(userns=False, in_container=True)) != "none"
    assert resolve_isolation("hardened", _env(userns=False)) == "hardened"


def test_select_profile_unknown_raises() -> None:
    with pytest.raises(IsolationUnavailableError, match=r"unknown sandbox\.isolation"):
        resolve_isolation("lax", _env(userns=True))


def _no_sandbox_env() -> Environment:
    """An Environment as detected on a non-Linux host (no kernel sandbox)."""
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="unknown", major=0, minor=0),
        userns_supported=False,
        landlock_abi=0,
        seccomp_arch_supported=True,
        sandbox_available=False,
    )


def test_detected_profile_none_without_sandbox() -> None:
    assert _no_sandbox_env().detected_isolation == "none"


def test_select_profile_auto_is_none_without_sandbox() -> None:
    assert resolve_isolation("auto", _no_sandbox_env()) == "none"


def test_select_profile_strict_refused_without_sandbox() -> None:
    with pytest.raises(IsolationUnavailableError, match="Linux kernel sandbox"):
        resolve_isolation("strict", _no_sandbox_env())


def test_select_profile_hardened_refused_without_sandbox() -> None:
    with pytest.raises(IsolationUnavailableError, match="Linux kernel sandbox"):
        resolve_isolation("hardened", _no_sandbox_env())


def test_sandbox_available_matches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent6.sandbox.detect as detect_mod

    monkeypatch.setattr(detect_mod.sys, "platform", "darwin")
    assert detect_mod.sandbox_available() is False
    monkeypatch.setattr(detect_mod.sys, "platform", "linux")
    assert detect_mod.sandbox_available() is True
