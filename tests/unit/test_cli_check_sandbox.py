# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` runs its probes under the host's effective isolation.

The jail is stubbed out, so these run on any host. On a host that can only run `hardened` the check
passes rather than failing against a `strict` jail the agent would never use.
"""

from __future__ import annotations

import pathlib
import types

import pytest

from agent6 import kinds
from agent6.config import Config, SandboxConfig
from agent6.sandbox import detect, jail, landlock
from agent6.ui.cli import check_cmds


def _fake_result(argv: tuple[str, ...], rc: int) -> kinds.CommandResult:
    return kinds.CommandResult(argv=argv, returncode=rc, stdout="", stderr="", duration_s=0.0)


@pytest.fixture
def stub_jail(monkeypatch: pytest.MonkeyPatch) -> list[kinds.JailPolicy]:
    """Stub landlock_abi + run_in_jail; record every policy the check builds."""
    seen: list[kinds.JailPolicy] = []
    monkeypatch.setattr(check_cmds, "landlock_abi", lambda: 8)

    def fake_run(policy: kinds.JailPolicy) -> kinds.CommandResult:
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


def _honour_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the resolver to return exactly what the config asked for."""
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _reason(_env: object) -> str | None:
        return None

    def _resolve(requested: str, _env: object) -> str:
        return requested

    monkeypatch.setattr(check_cmds, "degrade_reason", _reason)
    monkeypatch.setattr(check_cmds, "resolve_isolation", _resolve)


def test_check_sandbox_reports_a_landlock_probe_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A host policy denying the Landlock syscall is a failed probe, not a traceback."""
    _force_profile(monkeypatch, "hardened")

    def _denied() -> int:
        raise landlock.LandlockError(
            "landlock_create_ruleset version probe failed: Operation not permitted"
        )

    def _run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return _fake_result(policy.argv, 0)

    monkeypatch.setattr(check_cmds, "landlock_abi", _denied)
    monkeypatch.setattr(check_cmds, "run_in_jail", _run)

    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]

    assert rc == 1
    assert "[FAIL] landlock_abi: landlock_create_ruleset" in capsys.readouterr().out


def test_check_sandbox_hardened_passes_and_skips_network(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
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
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent6.sandbox import tool_paths

    _force_profile(monkeypatch, "none")
    monkeypatch.setattr(
        check_cmds,
        "tool_mount_notes",
        lambda: tool_paths.ToolMountNotes(exposes_home_dir=("~/.local/bin/x -> ~/.local/share/x",)),
    )
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    # No kernel sandbox -> reported FAIL, and no jail invocations attempted.
    assert rc == 1, out
    assert "effective isolation (auto): none" in out
    assert stub_jail == []
    # Nothing is confined under "none", so grant language about tool dirs is absent.
    assert "granted read-only" not in out
    assert "mounted read-only" not in out


def test_check_sandbox_degraded_names_why(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A degraded level never appears without its cause.

    On a userns-blocked host the line reads `effective isolation (auto): hardened` and why.
    """
    from agent6.sandbox import tool_paths

    why = "unprivileged user namespaces are disabled (user.max_user_namespaces = 0)"
    _force_profile(monkeypatch, "hardened", reason=why)
    monkeypatch.setattr(
        check_cmds,
        "tool_mount_notes",
        lambda: tool_paths.ToolMountNotes(exposes_home_dir=("~/.local/bin/x -> ~/.local/share/x",)),
    )
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"not strict: {why}" in out
    # Under hardened nothing is mounted: the tool-dir exposure is a Landlock read grant.
    assert "granted read-only (Landlock path rules)" in out
    assert "mounted read-only into the jail" not in out
    assert "1 tool on the PATH" in out


def test_check_sandbox_probes_the_isolation_the_config_selects(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The probes exercise the jail a run here would use.

    With `sandbox.isolation = "hardened"` on a strict-capable host, the section agrees with `check
    config`.
    """
    _honour_request(monkeypatch)
    cfg = Config(sandbox=SandboxConfig(isolation="hardened"))
    rc = check_cmds._cmd_check_sandbox(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (hardened): hardened" in out
    assert stub_jail and all(p.isolation == "hardened" for p in stub_jail)


def test_check_sandbox_names_the_degrade_reason_only_for_auto(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The degrade reason prints only for `auto`.

    `degrade_reason` answers why `auto` does not reach `strict`; `check config` and the run's
    `warn_sandbox_gaps` gate the line on `isolation == "auto"`, and `check sandbox` agrees.
    """
    why = "unprivileged user namespaces are disabled (user.max_user_namespaces = 0)"
    _force_profile(monkeypatch, "hardened", reason=why)
    cfg = Config(sandbox=SandboxConfig(isolation="hardened"))
    rc = check_cmds._cmd_check_sandbox(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation (hardened): hardened" in out
    assert "not strict" not in out


def test_check_names_a_jail_binary_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unusable AGENT6_JAIL_BIN is named as the binary's own refusal in each section.

    It is not reported as a host that blocks user namespaces.
    """
    refusal = "agent6-jail at /opt/agent6-jail cannot be executed: Exec format error. Reinstall it"

    def _binary_refusal() -> object:
        raise jail.JailUnavailableError(refusal)

    monkeypatch.setattr(check_cmds, "detect_env", _binary_refusal)
    rc = check_cmds._cmd_check_sandbox(None)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 1, out
    assert f"[FAIL] jail_binary: {refusal}" in out
    assert "user namespaces" not in out
    assert stub_jail == []
    checks = check_cmds._check_config_section(Config())  # pyright: ignore[reportPrivateUsage]
    assert (checks[0].name, checks[0].status, checks[0].detail) == (
        "config.isolation",
        "FAIL",
        refusal,
    )
    assert "user namespaces" not in capsys.readouterr().out


def test_check_sandbox_fails_on_an_isolation_this_host_refuses(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit level the host refuses is a FAIL naming the refusal.

    Probes on another level.
    """
    monkeypatch.setattr(check_cmds, "detect_env", object)

    def _refuse(req: str, _env: object) -> str:
        raise detect.IsolationUnavailableError(
            f"sandbox.isolation = {req!r} requires user namespaces"
        )

    monkeypatch.setattr(check_cmds, "resolve_isolation", _refuse)
    rc = check_cmds._cmd_check_sandbox(  # pyright: ignore[reportPrivateUsage]
        Config(sandbox=SandboxConfig(isolation="strict"))
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "requires user namespaces" in out
    assert stub_jail == []


def test_check_sandbox_names_which_opt_out_left_nothing_to_probe(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`none` from the config is the operator's opt-out.

    The skip line does not blame the platform.
    """
    _honour_request(monkeypatch)
    rc = check_cmds._cmd_check_sandbox(  # pyright: ignore[reportPrivateUsage]
        Config(sandbox=SandboxConfig(isolation="none"))
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "sandbox.isolation = 'none': commands run unconfined" in out
    assert "no kernel sandbox" not in out
    assert stub_jail == []


def test_check_sandbox_fails_when_its_config_cannot_be_loaded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sandbox probe under defaults cannot certify the unusable config a run would load."""
    bad = tmp_path / "bad.toml"
    bad.write_text("not = [valid", encoding="utf-8")

    def _sandbox_passes(_cfg: Config | None) -> int:
        return 0

    monkeypatch.setattr(check_cmds, "_cmd_check_sandbox", _sandbox_passes)

    rc = check_cmds._cmd_check(bad, section="sandbox")  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] config_load:" in out.split("== summary ==", 1)[1]


def test_check_config_runs_the_refusal_ladder_a_run_applies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`check config` fails on an explicit knob the isolation cannot honour, with the run's refusal.

    The case: hardened plus network = session.
    """
    env = types.SimpleNamespace(
        kernel=types.SimpleNamespace(raw="6.8"),
        userns_supported=True,
        sandbox_available=True,
        landlock_abi=5,
    )

    def _no_reason(_env: object) -> str | None:
        return None

    def _as_requested(requested: str, _env: object) -> str:
        return requested

    monkeypatch.setattr(check_cmds, "detect_env", lambda: env)
    monkeypatch.setattr(check_cmds, "degrade_reason", _no_reason)
    monkeypatch.setattr(check_cmds, "resolve_isolation", _as_requested)
    cfg = Config.model_validate({"sandbox": {"isolation": "hardened", "network": "session"}})
    checks = check_cmds._check_config_section(cfg)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    refusal = next(c for c in checks if c.name == "config.refusal")
    assert refusal.status == "FAIL"
    assert "sandbox.network = 'session' requires the strict isolation" in refusal.detail
    assert "[FAIL] a run would refuse" in out
    ok = Config.model_validate({"sandbox": {"isolation": "hardened"}})
    checks = check_cmds._check_config_section(ok)  # pyright: ignore[reportPrivateUsage]
    assert next(c for c in checks if c.name == "config.refusal").status == "PASS"


@pytest.mark.needs_namespaces
def test_check_sandbox_runs_its_probes_unstubbed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's one "is my sandbox working" command runs its probes unstubbed once."""
    if check_cmds.landlock_abi() < 1:
        pytest.skip("no Landlock: the command's own landlock_abi row fails here")
    monkeypatch.chdir(tmp_path)
    rc = check_cmds._cmd_check_sandbox()  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "effective isolation" in out
