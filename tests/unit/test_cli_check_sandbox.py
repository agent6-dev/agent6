# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` runs its probes under the host's *effective* profile.

Pure-logic tests: the jail itself is stubbed out, so these run on any host
(no namespaces required). They pin the behaviour that on a host that can only
run `hardened` (default-seccomp Docker, AppArmor-restricted Ubuntu) the check
PASSES rather than spuriously failing against a `strict` jail the agent would
never use there.
"""

from __future__ import annotations

import pytest

from agent6.cli import check_cmds
from agent6.types import CommandResult, JailPolicy


def _fake_result(argv: tuple[str, ...], rc: int) -> CommandResult:
    return CommandResult(argv=argv, returncode=rc, stdout="", stderr="", duration_s=0.0)


@pytest.fixture
def stub_jail(monkeypatch: pytest.MonkeyPatch) -> list[JailPolicy]:
    """Stub landlock_abi + run_in_jail; record every policy the check builds."""
    seen: list[JailPolicy] = []
    monkeypatch.setattr(check_cmds, "landlock_abi", lambda: 8)

    def fake_run(policy: JailPolicy) -> CommandResult:
        seen.append(policy)
        # getent (network probe) "fails" (blocked); everything else succeeds.
        rc = 2 if policy.argv[0].endswith("getent") else 0
        return _fake_result(policy.argv, rc)

    monkeypatch.setattr(check_cmds, "run_in_jail", fake_run)
    return seen


def _force_profile(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    monkeypatch.setattr(check_cmds, "detect_env", object)  # returns a throwaway env stub
    monkeypatch.setattr(check_cmds, "apparmor_userns_restricted", lambda: False)  # no advisory

    def fake_select(_req: str, _env: object) -> str:
        return profile

    monkeypatch.setattr(check_cmds, "select_profile", fake_select)


def test_check_sandbox_hardened_passes_and_skips_network(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "hardened")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective profile (auto): hardened" in out
    # Network probe is reported n/a, not run, under hardened.
    assert "jail_blocks_network: n/a under hardened" in out
    assert all(p.profile == "hardened" for p in stub_jail)
    assert not any(p.argv[0].endswith("getent") for p in stub_jail)


def test_check_sandbox_strict_runs_network_probe(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "strict")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective profile (auto): strict" in out
    # The network probe actually runs under strict, with profile=strict.
    getent = [p for p in stub_jail if p.argv[0].endswith("getent")]
    assert len(getent) == 1
    assert getent[0].profile == "strict"
    assert getent[0].allow_network is False


def test_check_sandbox_none_skips_probes(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "none")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    # No kernel sandbox -> reported FAIL, and no jail invocations attempted.
    assert rc == 1, out
    assert "effective profile (auto): none" in out
    assert stub_jail == []
