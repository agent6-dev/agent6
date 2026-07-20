# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.sandbox.detect."""

from __future__ import annotations

import pytest

import agent6.sandbox.detect as detect_mod
from agent6.sandbox.detect import (
    Environment,
    KernelInfo,
    ProfileUnavailableError,
    _parse_kernel,  # pyright: ignore[reportPrivateUsage]
    detect_container_signals,
    select_profile,
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
    # Rootless podman: /run/.containerenv present, no /.dockerenv, and the cgroup
    # often lacks a "podman" token -- so the file marker is what catches it.
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
        sandbox_available=True,
    )


def test_detected_profile_strict_when_userns_supported() -> None:
    assert _env(userns=True).detected_profile == "strict"


def test_detected_profile_hardened_when_userns_blocked() -> None:
    assert _env(userns=False, landlock_abi=1).detected_profile == "hardened"


def test_detected_profile_none_without_userns_or_landlock() -> None:
    # hardened's ONLY filesystem boundary is Landlock (no mount namespace).
    # A Linux host offering neither userns nor Landlock has no confinement
    # mechanism at all, so the truthful resolution is `none` (callers warn
    # loudly), never a hardened that would silently confine nothing.
    assert _env(userns=False, landlock_abi=0).detected_profile == "none"


def test_select_profile_auto_follows_environment() -> None:
    assert select_profile("auto", _env(userns=True)) == "strict"
    assert select_profile("auto", _env(userns=False)) == "hardened"
    assert select_profile("auto", _env(userns=False, landlock_abi=0)) == "none"


def test_select_profile_strict_refuses_silent_downgrade() -> None:
    with pytest.raises(ProfileUnavailableError, match="user namespaces"):
        select_profile("strict", _env(userns=False))


def test_select_profile_strict_passes_when_supported() -> None:
    assert select_profile("strict", _env(userns=True)) == "strict"


def test_select_profile_hardened_ok_with_landlock() -> None:
    assert select_profile("hardened", _env(userns=True)) == "hardened"
    assert select_profile("hardened", _env(userns=False, landlock_abi=1)) == "hardened"


def test_select_profile_hardened_refuses_without_landlock() -> None:
    # Mirrors the strict/userns refusal: an explicit request the kernel cannot
    # back is refused with a remedy, never silently under-delivered.
    with pytest.raises(ProfileUnavailableError, match="Landlock"):
        select_profile("hardened", _env(userns=False, landlock_abi=0))


def _env_c(*, userns: bool, in_container: bool) -> Environment:
    return Environment(
        in_container=in_container,
        container_signals=("docker",) if in_container else (),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=userns,
        landlock_abi=4,
        sandbox_available=True,
    )


def test_select_profile_explicit_none_is_self_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Explicit `profile = "none"` is the operator's consent by itself (an
    # operator-only, LLM-unreachable config value); it no longer needs a second
    # env-var gate. Allowed on a bare host and in a container; the loud
    # run-startup warning is the safety net.
    monkeypatch.delenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", raising=False)
    assert select_profile("none", _env_c(userns=True, in_container=False)) == "none"
    assert select_profile("none", _env_c(userns=False, in_container=True)) == "none"


def test_select_profile_auto_reaches_none_only_without_any_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `auto` resolves to none only when the host offers NO confinement
    # mechanism (non-Linux, or Linux without userns AND without Landlock);
    # with either mechanism present it resolves to the real profile. The
    # none resolution is loud: callers warn, and auto-approved run_command
    # additionally hits the unconfined confirm gate.
    monkeypatch.delenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", raising=False)
    assert select_profile("auto", _env_c(userns=True, in_container=False)) == "strict"
    assert select_profile("auto", _env_c(userns=False, in_container=False)) == "hardened"
    assert select_profile("auto", _env(userns=False, landlock_abi=0)) == "none"


def test_env_setter_forces_none_over_any_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # AGENT6_DANGEROUSLY_DISABLE_SANDBOX is a per-invocation SETTER: it forces
    # the unsandboxed profile regardless of what config requested.
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    assert select_profile("auto", _env_c(userns=True, in_container=False)) == "none"
    assert select_profile("strict", _env_c(userns=True, in_container=False)) == "none"
    assert select_profile("hardened", _env_c(userns=False, in_container=False)) == "none"


def test_env_setter_forces_none_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")
    env = Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="", major=0, minor=0),
        userns_supported=False,
        landlock_abi=0,
        sandbox_available=False,
    )
    assert select_profile("strict", env) == "none"


def test_probe_landlock_abi_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A probe error must read as "no Landlock" (0): profile resolution then
    # refuses hardened / resolves auto to the loudly-warned none, instead of
    # promising confinement the kernel may not deliver.
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
    # The critical invariant: auto NEVER resolves to none while the host still
    # offers a real confinement mechanism (userns or Landlock).
    assert select_profile("auto", _env_c(userns=True, in_container=True)) != "none"
    assert select_profile("auto", _env_c(userns=False, in_container=True)) != "none"
    assert select_profile("hardened", _env(userns=False)) == "hardened"


def test_select_profile_unknown_raises() -> None:
    with pytest.raises(ProfileUnavailableError, match=r"unknown sandbox\.profile"):
        select_profile("lax", _env(userns=True))


def _no_sandbox_env() -> Environment:
    """An Environment as detected on a non-Linux host (no kernel sandbox)."""
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="unknown", major=0, minor=0),
        userns_supported=False,
        landlock_abi=0,
        sandbox_available=False,
    )


def test_detected_profile_none_without_sandbox() -> None:
    assert _no_sandbox_env().detected_profile == "none"


def test_select_profile_auto_is_none_without_sandbox() -> None:
    assert select_profile("auto", _no_sandbox_env()) == "none"


def test_select_profile_strict_refused_without_sandbox() -> None:
    with pytest.raises(ProfileUnavailableError, match="Linux kernel sandbox"):
        select_profile("strict", _no_sandbox_env())


def test_select_profile_hardened_refused_without_sandbox() -> None:
    with pytest.raises(ProfileUnavailableError, match="Linux kernel sandbox"):
        select_profile("hardened", _no_sandbox_env())


def test_sandbox_available_matches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent6.sandbox.detect as detect_mod

    monkeypatch.setattr(detect_mod.sys, "platform", "darwin")
    assert detect_mod.sandbox_available() is False
    monkeypatch.setattr(detect_mod.sys, "platform", "linux")
    assert detect_mod.sandbox_available() is True
