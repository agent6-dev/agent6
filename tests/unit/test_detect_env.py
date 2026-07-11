# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`detect_env` re-checks userns via the jail binary on AppArmor-restricted hosts.

`detect.probe_userns_supported` uses `/usr/bin/unshare`, which under-reports when
an AppArmor profile grants the *agent6-jail* binary userns but not unshare. So
`detect_env` confirms with the real jail binary before dropping to `hardened`.
"""

from __future__ import annotations

import pytest

from agent6.detect import Environment, KernelInfo
from agent6.ui.cli import _common


def _env(userns: bool, *, sandbox: bool = True) -> Environment:
    return Environment(
        in_container=False,
        container_signals=(),
        kernel=KernelInfo(raw="7.0.0", major=7, minor=0),
        userns_supported=userns,
        sandbox_available=sandbox,
    )


def _fail_probe() -> bool:
    pytest.fail("strict_namespaces_work should not be called here")


def test_detect_env_keeps_userns_when_cheap_probe_already_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_common, "detect", lambda: _env(True))
    monkeypatch.setattr(_common, "strict_namespaces_work", _fail_probe)  # not consulted
    assert _common.detect_env().userns_supported is True


def test_detect_env_upgrades_to_strict_via_jail_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # The AppArmor-profile case: unshare blocked, but the jail binary can userns.
    monkeypatch.setattr(_common, "detect", lambda: _env(False))
    monkeypatch.setattr(_common, "strict_namespaces_work", lambda: True)
    assert _common.detect_env().userns_supported is True


def test_detect_env_stays_hardened_when_jail_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "detect", lambda: _env(False))
    monkeypatch.setattr(_common, "strict_namespaces_work", lambda: False)
    assert _common.detect_env().userns_supported is False


def test_detect_env_skips_probe_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "detect", lambda: _env(False, sandbox=False))
    monkeypatch.setattr(_common, "strict_namespaces_work", _fail_probe)  # not consulted
    assert _common.detect_env().userns_supported is False
