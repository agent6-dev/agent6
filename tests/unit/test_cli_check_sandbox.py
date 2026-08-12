# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` runs its probes under the host's *effective* isolation.

Pure-logic tests: the jail itself is stubbed out, so these run on any host
(no namespaces required). They pin the behaviour that on a host that can only
run `hardened` (default-seccomp Docker, AppArmor-restricted Ubuntu) the check
PASSES rather than spuriously failing against a `strict` jail the agent would
never use there.
"""

from __future__ import annotations

import pytest

from agent6.types import CommandResult, JailPolicy
from agent6.ui.cli import check_cmds


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


def _force_profile(
    monkeypatch: pytest.MonkeyPatch, isolation: str, reason: str | None = None
) -> None:
    monkeypatch.setattr(check_cmds, "detect_env", object)  # returns a throwaway env stub

    def _reason(_env: object) -> str | None:
        return reason

    monkeypatch.setattr(check_cmds, "degrade_reason", _reason)

    def fake_select(_req: str, _env: object) -> str:
        return isolation

    monkeypatch.setattr(check_cmds, "resolve_isolation", fake_select)


def test_check_sandbox_hardened_passes_and_skips_network(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "hardened")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (auto): hardened" in out
    # Network probe is reported n/a, not run, under hardened.
    assert "jail_blocks_network: n/a under hardened" in out
    assert all(p.isolation == "hardened" for p in stub_jail)
    assert not any(p.argv[0].endswith("getent") for p in stub_jail)


def test_check_sandbox_strict_runs_network_probe(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    _force_profile(monkeypatch, "strict")
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (auto): strict" in out
    # The network probe actually runs under strict, with isolation=strict.
    getent = [p for p in stub_jail if p.argv[0].endswith("getent")]
    assert len(getent) == 1
    assert getent[0].isolation == "strict"
    assert getent[0].network == "none"


def test_check_sandbox_none_skips_probes(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.sandbox.jail import ToolMountNotes

    _force_profile(monkeypatch, "none")
    monkeypatch.setattr(
        check_cmds,
        "tool_mount_notes",
        lambda: ToolMountNotes(exposes_home_dir=("~/.local/bin/x -> ~/.local/share/x",)),
    )
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    # No kernel sandbox -> reported FAIL, and no jail invocations attempted.
    assert rc == 1, out
    assert "effective isolation (auto): none" in out
    assert stub_jail == []
    # Nothing is confined under "none": grant language about tool dirs would
    # describe a boundary that does not exist, so the block is absent.
    assert "granted read-only" not in out
    assert "mounted read-only" not in out


def test_check_sandbox_degraded_names_why(
    monkeypatch: pytest.MonkeyPatch, stub_jail: list[JailPolicy], capsys: pytest.CaptureFixture[str]
) -> None:
    """A degraded level never appears without its cause. Reproduced on a
    userns-blocked host (user.max_user_namespaces = 0): the line read
    `effective isolation (auto): hardened` and nothing said why, while
    `check config` did (Eric hit exactly this)."""
    from agent6.sandbox.jail import ToolMountNotes

    why = "unprivileged user namespaces are disabled (user.max_user_namespaces = 0)"
    _force_profile(monkeypatch, "hardened", reason=why)
    monkeypatch.setattr(
        check_cmds,
        "tool_mount_notes",
        lambda: ToolMountNotes(exposes_home_dir=("~/.local/bin/x -> ~/.local/share/x",)),
    )
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"not strict: {why}" in out
    # Under hardened nothing is MOUNTED (no mount namespace): the tool-dir
    # exposure is a Landlock read grant and the words must say so.
    assert "granted read-only (Landlock path rules)" in out
    assert "mounted read-only into the jail" not in out
