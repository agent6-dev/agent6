# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.detect."""

from __future__ import annotations

import pytest

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


def _env(*, userns: bool) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="6.14.0", major=6, minor=14),
        userns_supported=userns,
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
    assert select_profile("hardened", _env(userns=False)) == "hardened"


def test_select_profile_unknown_raises() -> None:
    with pytest.raises(RuntimeError, match=r"unknown sandbox\.profile"):
        select_profile("lax", _env(userns=True))
