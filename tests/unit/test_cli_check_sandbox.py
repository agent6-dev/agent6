# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` reports the boundaries a run here gets, then probes that jail.

The jail is stubbed out, so these run on any host. On a host that can only run `partial` the
check passes rather than failing against a `full` jail the agent would never use.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds, paths
from agent6.app import _setup
from agent6.config import Config, SandboxConfig
from agent6.sandbox import detect, jail, landlock, tool_paths
from agent6.ui.cli import _check_sandbox, check_cmds

_ENV = detect.Environment(
    in_container=False,
    container_signals=(),
    kernel=detect.KernelInfo(raw="6.8", major=6, minor=8),
    userns_supported=True,
    landlock_abi=5,
    seccomp_arch_supported=True,
    sandbox_available=True,
)
_MOUNTS = (
    tool_paths.ToolMount(pathlib.Path("/opt/tools/bin"), "on the jail's PATH"),
    tool_paths.ToolMount(pathlib.Path("/opt/tools/lib/x/bin"), "/opt/tools/bin/x links here"),
)


def _fake_result(argv: tuple[str, ...], rc: int) -> kinds.CommandResult:
    return kinds.CommandResult(argv=argv, returncode=rc, stdout="", stderr="", duration_s=0.0)


@pytest.fixture
def stub_jail(monkeypatch: pytest.MonkeyPatch) -> list[kinds.JailPolicy]:
    """Stub landlock_abi, run_in_jail and the tool scan; record every policy the check builds."""
    seen: list[kinds.JailPolicy] = []
    monkeypatch.setattr(_check_sandbox, "landlock_abi", lambda: 8)

    def fake_run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        seen.append(policy)
        # getent (network probe) "fails" (blocked); everything else succeeds.
        rc = 2 if policy.argv[0].endswith("getent") else 0
        return _fake_result(policy.argv, rc)

    monkeypatch.setattr(_check_sandbox, "run_in_jail", fake_run)
    monkeypatch.setattr(tool_paths, "operator_tool_mounts", lambda: _MOUNTS)
    return seen


def _force_profile(
    monkeypatch: pytest.MonkeyPatch, isolation: str, reason: str | None = None
) -> None:
    monkeypatch.setattr(_setup, "detect_env", lambda: _ENV)

    def _reason(_env: object) -> str | None:
        return reason

    monkeypatch.setattr(detect, "degrade_reason", _reason)

    def fake_select(_req: str, _env: object) -> str:
        return isolation

    monkeypatch.setattr(detect, "resolve_isolation", fake_select)


def _honour_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the resolver to return exactly what the config asked for."""
    _force_profile(monkeypatch, "unused")

    def _resolve(requested: str, _env: object) -> str:
        return requested

    monkeypatch.setattr(detect, "resolve_isolation", _resolve)


def _row(checks: list[kinds.DoctorCheck], name: str) -> kinds.DoctorCheck:
    return next(c for c in checks if c.name == name)


def _commands_block(out: str) -> list[str]:
    """Return the jailed-commands rows of the report."""
    block = out.split("  jailed commands", 1)[1].split("\n\n", 1)[0]
    return [line for line in block.splitlines()[1:] if line.startswith("    ")]


def test_check_sandbox_reports_a_landlock_probe_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A host policy denying the Landlock syscall is a failed probe, not a traceback."""
    _force_profile(monkeypatch, "partial")

    def _denied() -> int:
        raise landlock.LandlockError(
            "landlock_create_ruleset version probe failed: Operation not permitted"
        )

    def _run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return _fake_result(policy.argv, 0)

    monkeypatch.setattr(_check_sandbox, "landlock_abi", _denied)
    monkeypatch.setattr(_check_sandbox, "run_in_jail", _run)

    checks = _check_sandbox.check_sandbox_section(Config())

    assert _row(checks, "sandbox").status == "FAIL"
    assert "[FAIL] landlock_abi: landlock_create_ruleset" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("network", "told"),
    [
        ("host", "this config gives the run's commands the host's network (sandbox.network)"),
        ("auto", "this config gives the run's commands their own network, with no route off"),
    ],
)
def test_check_sandbox_names_the_network_its_probe_ran_on(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
    network: str,
    told: str,
) -> None:
    """The probe read `blocked` under `network = "host"`, where a command reaches the internet.

    The probe's jail has no network whatever the config says, so the line says that and what
    the config gives the run's commands.
    """
    _force_profile(monkeypatch, "full")
    cfg = Config.model_validate({"sandbox": {"network": network}})
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "PASS", out
    assert (
        "jail_blocks_network: a DNS lookup failed (exit 2) in a probe jail with no network" in out
    )
    assert told in out
    assert stub_jail


def test_check_sandbox_partial_passes_and_skips_network(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_profile(monkeypatch, "partial")
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "PASS", out
    assert "isolation: partial  (sandbox.isolation = auto;" in out
    # Network probe is reported n/a, not run, under partial.
    assert "jail_blocks_network: n/a under partial" in out
    assert all(p.isolation == "partial" for p in stub_jail)
    assert not any(p.argv[0].endswith("getent") for p in stub_jail)


def test_check_sandbox_full_runs_network_probe(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_profile(monkeypatch, "full")
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "PASS", out
    assert "isolation: full  (sandbox.isolation = auto; userns yes, Landlock ABI 5" in out
    # The network probe actually runs under full, with isolation=full.
    getent = [p for p in stub_jail if p.argv[0].endswith("getent")]
    assert len(getent) == 1
    assert getent[0].isolation == "full"
    assert getent[0].network == "none"


def test_check_sandbox_none_skips_probes(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host with no jail is a FAIL, and the commands table says what that reaches."""
    _force_profile(monkeypatch, "none")
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "FAIL", out
    assert "isolation: none  (this host has no jail)" in out
    assert "[FAIL] jail: no jail to probe: this host has no jail; commands run unconfined" in out
    assert stub_jail == []
    user = paths.effective_user()
    rows = _commands_block(out)
    assert rows[0] == (
        "    rw  /  (sandbox disabled: the whole filesystem and network, as"
        f" {user.name} (uid {user.uid}), no syscall filter)"
    )
    assert f"    rw  {user.home.resolve()}  (your home: ~/.ssh, ~/.aws and every credential" in out
    assert f"    rw  {paths.global_config_dir()}  (not hidden: secrets.toml is readable)" in out
    # Nothing is confined under "none", so no tool mount or grant rows.
    assert "operator tools" not in out


def test_check_sandbox_degraded_names_why(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A degraded level never appears without its cause.

    On a userns-blocked host the header reads `isolation: partial` and why.
    """
    why = "unprivileged user namespaces are disabled (user.max_user_namespaces = 0)"
    _force_profile(monkeypatch, "partial", reason=why)
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "PASS", out
    assert f"not full: {why}" in out
    # Under partial nothing is mounted: the tool dirs are Landlock read grants, one row each.
    assert "    ro  /opt/tools/bin  (operator tools: on the jail's PATH)" in out
    assert "    ro  /opt/tools/lib/x/bin  (operator tools: /opt/tools/bin/x links here)" in out
    assert "    --  /  (Landlock: denied except the paths below)" in out


def test_every_operator_tool_mount_is_a_row(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report counted the links into home dirs; the bin dirs on PATH were named nowhere."""
    _force_profile(monkeypatch, "full")
    _check_sandbox.check_sandbox_section(Config())
    rows = _commands_block(capsys.readouterr().out)
    for mount in _MOUNTS:
        assert f"    ro  {mount.path}  (operator tools: {mount.why})" in rows


def test_rows_sort_by_path_with_a_root_and_a_home_row(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each table opens at `/`, names the home directory, and lists parents before children."""
    _force_profile(monkeypatch, "full")
    _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    home = paths.effective_user().home.resolve()
    files = out.split("  in-process file tools", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
    assert files[0] == "    --  /  (refused: only the paths below)"
    assert f"    --  {home}  (refused, except the rows under it below)" in files
    rows = _commands_block(out)
    assert rows[0] == "    --  /  (an empty root: only the paths below exist)"
    assert f"    --  {home}  (not mounted: only the rows under it below)" in rows
    for table in (files, rows):
        found = [pathlib.Path(line.split()[1]) for line in table]
        assert found == sorted(found, key=lambda p: p.parts)
        assert len(found) == len(set(found)), "one row per path"


def test_a_grant_of_the_home_directory_replaces_its_row(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A home grant is listed as the grant it is, warned about, and not as refused."""
    _force_profile(monkeypatch, "full")
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path.resolve()
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(tmp_path),)))
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    warn = _row(checks, "sandbox.grants")
    assert warn.status == "WARN"
    assert warn.detail.startswith(f"sandbox.extra_read_paths grants {str(tmp_path)!r}")
    assert f"  [WARN] {warn.detail}" in out
    assert f"    ro  {home}  (sandbox.extra_read_paths)" in out
    assert f"    --  {home}  (" not in out


def test_check_sandbox_probes_the_isolation_the_config_selects(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The probes exercise the jail a run here would use."""
    _honour_request(monkeypatch)
    cfg = Config(sandbox=SandboxConfig(isolation="partial"))
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "PASS", out
    assert "isolation: partial  (sandbox.isolation = partial;" in out
    assert stub_jail and all(p.isolation == "partial" for p in stub_jail)


def test_check_sandbox_names_the_degrade_reason_only_for_auto(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`degrade_reason` answers why `auto` misses `full`; an explicit level is no degrade."""
    why = "unprivileged user namespaces are disabled (user.max_user_namespaces = 0)"
    _force_profile(monkeypatch, "partial", reason=why)
    cfg = Config(sandbox=SandboxConfig(isolation="partial"))
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert "isolation: partial  (sandbox.isolation = partial;" in out
    assert "not full" not in out


def test_check_names_a_jail_binary_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unusable AGENT6_JAIL_BIN is named as the binary's own refusal.

    It is not reported as a host that blocks user namespaces.
    """
    refusal = "agent6-jail at /opt/agent6-jail cannot be executed: Exec format error. Reinstall it"

    def _binary_refusal() -> object:
        raise jail.JailUnavailableError(refusal)

    monkeypatch.setattr(_setup, "detect_env", _binary_refusal)
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert [(c.name, c.status, c.detail) for c in checks] == [
        ("sandbox.isolation", "FAIL", refusal)
    ]
    assert f"[FAIL] isolation selection: {refusal}" in out
    assert "user namespaces" not in out
    assert stub_jail == []


def test_check_sandbox_fails_on_an_isolation_this_host_refuses(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit level the host refuses is a FAIL naming it; nothing probes another level."""
    monkeypatch.setattr(_setup, "detect_env", lambda: _ENV)

    def _refuse(req: str, _env: object) -> str:
        raise detect.IsolationUnavailableError(
            f"sandbox.isolation = {req!r} requires user namespaces"
        )

    monkeypatch.setattr(detect, "resolve_isolation", _refuse)
    cfg = Config(sandbox=SandboxConfig(isolation="full"))
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert _row(checks, "sandbox.isolation").status == "FAIL"
    assert "requires user namespaces" in out
    assert stub_jail == []


def test_check_sandbox_names_which_opt_out_left_nothing_to_probe(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`none` from the config is the operator's opt-out: a WARN naming the setting, no FAIL."""
    _honour_request(monkeypatch)
    cfg = Config(sandbox=SandboxConfig(isolation="none"))
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert _row(checks, "sandbox").status == "WARN", out
    assert "isolation: none  (sandbox.isolation = none)" in out
    assert "[WARN] jail: no jail to probe: sandbox.isolation = none; commands run unconfined" in out
    assert "no jail" in out and "this host has no jail" not in out
    assert stub_jail == []


def test_check_sandbox_fails_when_its_config_cannot_be_loaded(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sandbox probe under defaults cannot certify the unusable config a run would load."""
    bad = tmp_path / "bad.toml"
    bad.write_text("not = [valid", encoding="utf-8")

    rc = check_cmds._cmd_check((bad,), section="sandbox")  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr().out
    assert rc == 1
    assert "== sandbox ==" not in out
    assert "[FAIL] config_load:" in out.split("== summary ==", 1)[1]


def test_check_sandbox_runs_the_refusal_ladder_a_run_applies(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The section fails on an explicit knob the isolation cannot honour, with the run's refusal.

    The case: partial plus network = session.
    """
    _honour_request(monkeypatch)
    cfg = Config.model_validate({"sandbox": {"isolation": "partial", "network": "session"}})
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    refusal = _row(checks, "sandbox.refusal")
    assert refusal.status == "FAIL"
    assert "sandbox.network = 'session' requires the full isolation" in refusal.detail
    assert "[FAIL] a run would refuse" in out
    ok = Config.model_validate({"sandbox": {"isolation": "partial"}})
    checks = _check_sandbox.check_sandbox_section(ok)
    assert _row(checks, "sandbox.refusal").status == "PASS"


@pytest.mark.needs_namespaces
def test_check_sandbox_runs_its_probes_unstubbed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator's one "is my sandbox working" command runs its probes unstubbed once."""
    if _check_sandbox.landlock_abi() < 1:
        pytest.skip("no Landlock: the command's own landlock_abi row fails here")
    monkeypatch.chdir(tmp_path)
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert all(c.status == "PASS" for c in checks), out
    assert "isolation: " in out
