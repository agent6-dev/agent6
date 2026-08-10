# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`detect_env` asks the jail binary itself whether `strict` actually works.

`detect.probe_userns_supported` runs `/usr/bin/unshare`, which answers a
narrower question and is wrong in both directions: it under-reports where an
AppArmor profile grants the *agent6-jail* binary userns but not unshare, and it
over-reports inside Docker with a relaxed seccomp profile, where unshare
succeeds and AppArmor then denies the jail's mount. So the real binary settles
it either way.
"""

from __future__ import annotations

import pytest

from agent6.app import _setup
from agent6.sandbox.detect import Environment, KernelInfo


def _env(userns: bool, *, sandbox: bool = True) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="7.0.0", major=7, minor=0),
        userns_supported=userns,
        landlock_abi=4 if sandbox else 0,
        sandbox_available=sandbox,
    )


def _fail_probe() -> bool:
    pytest.fail("strict_namespaces_work should not be called here")


def test_detect_env_does_not_probe_where_there_is_no_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-Linux host has no jail to ask."""
    monkeypatch.setattr(_setup, "detect", lambda: _env(False, sandbox=False))
    monkeypatch.setattr(_setup, "strict_namespaces_work", _fail_probe)
    assert _setup.detect_env().detected_isolation == "none"


def test_detect_env_keeps_userns_when_the_jail_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_setup, "detect", lambda: _env(True))
    monkeypatch.setattr(_setup, "strict_namespaces_work", lambda: True)
    assert _setup.detect_env().userns_supported is True


def test_detect_env_drops_to_hardened_when_the_jail_cannot_do_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Docker case, measured: `unshare` succeeds under a relaxed seccomp
    profile and the default AppArmor profile then denies the jail's `mount`, so
    the cheap probe promised `strict` and every command died with a raw
    "namespace setup failed: EACCES". A capability we cannot deliver has to
    resolve DOWN to one we can, not be announced and then fail."""
    monkeypatch.setattr(_setup, "detect", lambda: _env(True))
    monkeypatch.setattr(_setup, "strict_namespaces_work", lambda: False)
    env = _setup.detect_env()
    assert env.userns_supported is False
    assert env.detected_isolation == "hardened"


def test_detect_env_upgrades_to_strict_via_jail_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # The AppArmor-isolation case: unshare blocked, but the jail binary can userns.
    monkeypatch.setattr(_setup, "detect", lambda: _env(False))
    monkeypatch.setattr(_setup, "strict_namespaces_work", lambda: True)
    assert _setup.detect_env().userns_supported is True


def test_detect_env_stays_hardened_when_jail_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_setup, "detect", lambda: _env(False))
    monkeypatch.setattr(_setup, "strict_namespaces_work", lambda: False)
    assert _setup.detect_env().userns_supported is False


def test_detect_env_skips_probe_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_setup, "detect", lambda: _env(False, sandbox=False))
    monkeypatch.setattr(_setup, "strict_namespaces_work", _fail_probe)  # not consulted
    assert _setup.detect_env().userns_supported is False
