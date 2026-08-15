# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared jail-test gate: skip where the jail cannot run, FAIL where a
host policy blocks a capable kernel (a userns-restricted dev machine read
green while 35 jail tests silently skipped and 72 failed unguarded)."""

from __future__ import annotations

import pytest

from agent6.sandbox.detect import Environment, KernelInfo
from tests import jail_env


def _env(
    *,
    sandbox: bool = True,
    userns: bool = True,
    container: bool = False,
    landlock: int = 8,
) -> Environment:
    return Environment(
        in_container=container,
        container_signals=(),
        kernel=KernelInfo(raw="6.8.0", major=6, minor=8),
        userns_supported=userns,
        landlock_abi=landlock,
        seccomp_arch_supported=True,
        sandbox_available=sandbox,
    )


def test_healthy_env_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jail_env, "detect", _env)
    try:
        jail_env.require_userns_jail()
    except (pytest.skip.Exception, pytest.fail.Exception) as exc:  # pragma: no cover
        pytest.fail(f"a healthy env must neither skip nor fail: {exc}")


def test_no_kernel_sandbox_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jail_env, "detect", lambda: _env(sandbox=False, userns=False))
    with pytest.raises(pytest.skip.Exception):
        jail_env.require_userns_jail()


def test_container_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    def _reason(_env: object) -> str | None:
        return "container blocks userns"

    monkeypatch.setattr(jail_env, "detect", lambda: _env(userns=False, container=True))
    monkeypatch.setattr(jail_env, "degrade_reason", _reason)
    with pytest.raises(pytest.skip.Exception):
        jail_env.require_userns_jail()


def test_host_policy_block_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _reason(_env: object) -> str | None:
        return "user.max_user_namespaces = 0"

    monkeypatch.setattr(jail_env, "detect", lambda: _env(userns=False))
    monkeypatch.setattr(jail_env, "degrade_reason", _reason)
    monkeypatch.delenv("AGENT6_TEST_SKIP_JAIL", raising=False)
    with pytest.raises(pytest.fail.Exception) as exc:
        jail_env.require_userns_jail()
    assert "host policy" in str(exc.value)
    assert "user.max_user_namespaces" in str(exc.value)


def test_explicit_env_turns_the_failure_into_a_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reason(_env: object) -> str | None:
        return "blocked"

    monkeypatch.setattr(jail_env, "detect", lambda: _env(userns=False))
    monkeypatch.setattr(jail_env, "degrade_reason", _reason)
    monkeypatch.setenv("AGENT6_TEST_SKIP_JAIL", "1")
    with pytest.raises(pytest.skip.Exception):
        jail_env.require_userns_jail()
