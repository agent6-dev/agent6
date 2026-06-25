# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.detect."""

from __future__ import annotations

import pytest

import agent6.detect as detect_mod
from agent6.detect import (
    Environment,
    KernelInfo,
    _parse_kernel,  # pyright: ignore[reportPrivateUsage]
    detect_container_signals,
    select_profile,
)


def test_parse_kernel_basic() -> None:
    k = _parse_kernel("6.7.5-arch1")
    assert (k.major, k.minor) == (6, 7)
    assert k.supports_landlock_tcp is True
    assert k.supports_landlock_fs is True


def test_parse_kernel_too_old() -> None:
    k = _parse_kernel("5.10.0")
    assert k.supports_landlock_tcp is False
    assert k.supports_landlock_fs is False


def test_parse_kernel_landlock_fs_only() -> None:
    k = _parse_kernel("6.6.99")
    assert k.supports_landlock_fs is True
    assert k.supports_landlock_tcp is False


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


def _env(*, userns: bool) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=userns,
        sandbox_available=True,
    )


def test_detected_profile_strict_when_userns_supported() -> None:
    assert _env(userns=True).detected_profile == "strict"


def test_detected_profile_hardened_when_userns_blocked() -> None:
    assert _env(userns=False).detected_profile == "hardened"


def test_select_profile_auto_follows_environment() -> None:
    assert select_profile("auto", _env(userns=True)) == "strict"
    assert select_profile("auto", _env(userns=False)) == "hardened"


def test_select_profile_strict_refuses_silent_downgrade() -> None:
    with pytest.raises(RuntimeError, match="user namespaces"):
        select_profile("strict", _env(userns=False))


def test_select_profile_strict_passes_when_supported() -> None:
    assert select_profile("strict", _env(userns=True)) == "strict"


def test_select_profile_hardened_always_ok() -> None:
    assert select_profile("hardened", _env(userns=True)) == "hardened"


def _env_c(*, userns: bool, in_container: bool) -> Environment:
    return Environment(
        in_container=in_container,
        container_signals=("docker",) if in_container else (),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=userns,
        sandbox_available=True,
    )


def test_select_profile_none_allowed_in_container(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit unsandboxed opt-out is allowed inside a real container (a
    # filesystem marker is present); the run-startup warning tells the operator
    # loudly. The gate requires STRONG (filesystem) evidence, so simulate it.
    monkeypatch.delenv("AGENT6_ALLOW_NO_SANDBOX", raising=False)
    monkeypatch.setattr(detect_mod, "_has_strong_container_evidence", lambda: True)
    assert select_profile("none", _env_c(userns=False, in_container=True)) == "none"
    assert select_profile("none", _env_c(userns=True, in_container=True)) == "none"


def test_select_profile_none_refused_on_bare_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # No outer boundary on a bare host -> refuse unless the operator confirms.
    monkeypatch.delenv("AGENT6_ALLOW_NO_SANDBOX", raising=False)
    monkeypatch.setattr(detect_mod, "_has_strong_container_evidence", lambda: False)
    with pytest.raises(RuntimeError, match="UNSANDBOXED"):
        select_profile("none", _env_c(userns=True, in_container=False))


def test_select_profile_none_refused_on_bare_host_with_only_envvar_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FINDING 1: a WEAK env-var signal (REMOTE_CONTAINERS / CODESPACES) makes
    # env.in_container True, but it is forgeable by a stray exported var on a real
    # bare host. The gate must require STRONG filesystem evidence, so an env-var-
    # only "container" is still REFUSED without the operator's confirmation.
    monkeypatch.delenv("AGENT6_ALLOW_NO_SANDBOX", raising=False)
    monkeypatch.setenv("REMOTE_CONTAINERS", "true")
    # Bare host: neither filesystem marker exists -> no strong evidence.
    monkeypatch.setattr(detect_mod, "_has_strong_container_evidence", lambda: False)
    env = Environment(
        in_container=True,  # env-var signal flipped this on (weak)
        container_signals=("REMOTE_CONTAINERS",),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        sandbox_available=True,
    )
    with pytest.raises(RuntimeError, match="UNSANDBOXED"):
        select_profile("none", env)


def test_select_profile_none_allowed_with_filesystem_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FINDING 1: a real container with a filesystem marker (/.dockerenv or
    # /run/.containerenv) is STRONG evidence and still permits profile=none.
    monkeypatch.delenv("AGENT6_ALLOW_NO_SANDBOX", raising=False)
    monkeypatch.setattr(detect_mod, "_has_strong_container_evidence", lambda: True)
    env = Environment(
        in_container=True,
        container_signals=("/.dockerenv",),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=True,
        sandbox_available=True,
    )
    assert select_profile("none", env) == "none"


def test_has_strong_container_evidence_filesystem_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The helper keys ONLY off filesystem markers, never env vars.
    from pathlib import Path

    real_exists = Path.exists

    def only_dockerenv(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return True
        if str(self) == "/run/.containerenv":
            return False
        return real_exists(self)

    def never_exists(self: Path) -> bool:
        return False

    monkeypatch.setenv("REMOTE_CONTAINERS", "true")
    monkeypatch.setattr(Path, "exists", never_exists)
    assert detect_mod._has_strong_container_evidence() is False  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(Path, "exists", only_dockerenv)
    assert detect_mod._has_strong_container_evidence() is True  # pyright: ignore[reportPrivateUsage]


def test_select_profile_none_bare_host_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_ALLOW_NO_SANDBOX", "1")
    assert select_profile("none", _env_c(userns=True, in_container=False)) == "none"


def test_select_profile_auto_never_unsandboxes_on_linux() -> None:
    # The critical invariant: auto NEVER silently resolves to none on Linux.
    assert select_profile("auto", _env_c(userns=True, in_container=True)) != "none"
    assert select_profile("auto", _env_c(userns=False, in_container=True)) != "none"
    assert select_profile("hardened", _env(userns=False)) == "hardened"


def test_select_profile_unknown_raises() -> None:
    with pytest.raises(RuntimeError, match=r"unknown sandbox\.profile"):
        select_profile("lax", _env(userns=True))


def _no_sandbox_env() -> Environment:
    """An Environment as detected on a non-Linux host (no kernel sandbox)."""
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="unknown", major=0, minor=0),
        userns_supported=False,
        sandbox_available=False,
    )


def test_detected_profile_none_without_sandbox() -> None:
    assert _no_sandbox_env().detected_profile == "none"


def test_select_profile_auto_is_none_without_sandbox() -> None:
    assert select_profile("auto", _no_sandbox_env()) == "none"


def test_select_profile_strict_refused_without_sandbox() -> None:
    with pytest.raises(RuntimeError, match="Linux kernel sandbox"):
        select_profile("strict", _no_sandbox_env())


def test_select_profile_hardened_refused_without_sandbox() -> None:
    with pytest.raises(RuntimeError, match="Linux kernel sandbox"):
        select_profile("hardened", _no_sandbox_env())


def test_sandbox_available_matches_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent6.detect as detect_mod

    monkeypatch.setattr(detect_mod.sys, "platform", "darwin")
    assert detect_mod.sandbox_available() is False
    monkeypatch.setattr(detect_mod.sys, "platform", "linux")
    assert detect_mod.sandbox_available() is True
