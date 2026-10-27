# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` reports the boundaries a run here gets, then probes that jail.

The jail is stubbed out, so these run on any host. On a host that can only run `partial` the
check passes rather than failing against a `full` jail the agent would never use.
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

import pytest

from agent6 import kinds, paths
from agent6.app import _setup
from agent6.config import Config, SandboxConfig
from agent6.sandbox import detect, jail, landlock, tool_paths
from agent6.ui.cli import _check_render, _check_sandbox, _sandbox_paths, check_cmds

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
    tool_paths.ToolMount(pathlib.Path("/opt/tools/bin"), on_path=True),
    tool_paths.ToolMount(
        pathlib.Path("/opt/tools/lib/x"),
        links=(tool_paths.ToolLink(pathlib.Path("/opt/tools/bin/x")),),
    ),
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


@pytest.fixture
def linked_install_roots(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Discover three sibling installs and one lone install through real tool links."""
    bins = tmp_path / "bin"
    bins.mkdir()
    for parent, name, links in (
        ("venvs", "bakabakabaka", ("baka",)),
        ("venvs", "pyxargs", ("pyxargs", "xargs")),
        ("venvs", "uv", ("uv", "uvx")),
        ("tools", "agent6", ("agent6",)),
    ):
        target = tmp_path / parent / name / "bin" / links[0]
        target.parent.mkdir(parents=True)
        target.touch()
        for link in links:
            (bins / link).symlink_to(target)
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins,))
    _force_profile(monkeypatch, "full")
    monkeypatch.setattr(_check_sandbox, "landlock_abi", lambda: 5)

    def run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return _fake_result(policy.argv, 2 if policy.argv[0].endswith("getent") else 0)

    monkeypatch.setattr(_check_sandbox, "run_in_jail", run)
    return tmp_path


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
    assert "[SKIP] jail_blocks_network: partial has no network namespace" in out
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
    assert "isolation: none  (sandbox.isolation = auto;" in out
    assert "[FAIL] jail: no jail to probe: this host has no jail; commands run unconfined" in out
    assert stub_jail == []
    user = paths.effective_user()
    rows = _commands_block(out)
    assert rows[0] == (
        "    rw        /  ! (sandbox disabled: the whole filesystem and network, as"
        f" {user.name} (uid {user.uid}), no syscall filter)"
    )
    assert (
        f"    rw        {user.home.resolve()}  ! (your home: ~/.ssh, ~/.aws and every credential"
        in out
    )
    assert (
        f"    rw        {paths.global_config_dir()}  ! (not hidden: secrets.toml is readable)"
        in out
    )
    # Nothing is confined under "none", so no tool mount or grant rows.
    assert not any("  tool  " in line for line in rows)


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
    assert f"'full' is unavailable because {why}" in out
    # Under partial nothing is mounted: the tool dirs are Landlock read grants, one row each.
    assert "    ro  tool  /opt/tools/bin  (on the jail's PATH)" in out
    assert "    ro  tool  /opt/tools/lib/x  (/opt/tools/bin/x links into /opt/tools/lib/x)" in out
    assert "    --        /  (denied except the paths below)" in out


def test_every_operator_tool_mount_is_a_row(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report counted the links into home dirs; the bin dirs on PATH were named nowhere."""
    _force_profile(monkeypatch, "full")
    _check_sandbox.check_sandbox_section(Config())
    rows = _commands_block(capsys.readouterr().out)
    assert "    ro  tool  /opt/tools/bin  (on the jail's PATH)" in rows
    assert "    ro  tool  /opt/tools/lib/x  (/opt/tools/bin/x links into /opt/tools/lib/x)" in rows


@pytest.mark.parametrize("isolation", ["full", "partial"])
def test_sibling_install_roots_share_a_row(
    monkeypatch: pytest.MonkeyPatch,
    linked_install_roots: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    isolation: str,
) -> None:
    """Three install roots share one row with a link per root and all extra links counted."""
    _force_profile(monkeypatch, isolation)
    base = linked_install_roots
    _, mounts_before = tool_paths.operator_tool_paths()

    _check_sandbox.check_sandbox_section(Config())

    out = capsys.readouterr().out
    rows = [line for line in _commands_block(out) if f" {base}/venvs/" in line]
    mode = "ro" if isolation == "full" else "rw"
    assert rows == [
        f"    {mode}  tool  {base}/venvs/{{bakabakabaka,pyxargs,uv}}  "
        f"(3 install roots; {base}/bin/baka, pyxargs and uv link into them, 2 more)"
    ]
    files = out.split("  in-process file tools", 1)[1].split("\n\n", 1)[0]
    assert "install roots" not in files and "  tool  " not in files
    _, mounts_after = tool_paths.operator_tool_paths()
    assert mounts_after == mounts_before
    assert base / "venvs" not in mounts_after
    assert all(base / "venvs" / name in mounts_after for name in ("bakabakabaka", "pyxargs", "uv"))


def test_lone_install_root_stays_separate(
    linked_install_roots: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A root without a sibling keeps its path and single-root link annotation."""
    base = linked_install_roots
    _check_sandbox.check_sandbox_section(Config())
    rows = [line for line in _commands_block(capsys.readouterr().out) if f" {base}/tools/" in line]
    assert rows == [
        f"    ro  tool  {base}/tools/agent6  ({base}/bin/agent6 links into {base}/tools/agent6)"
    ]


def test_path_directory_beside_install_roots_stays_separate(
    monkeypatch: pytest.MonkeyPatch,
    linked_install_roots: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sibling PATH directory stays outside the braces even when a tool links into it."""
    base = linked_install_roots
    path_dir = base / "venvs" / "bin"
    path_dir.mkdir()
    target = path_dir / "real"
    target.touch()
    (base / "bin" / "path-tool").symlink_to(target)
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (base / "bin", path_dir))

    _check_sandbox.check_sandbox_section(Config())

    rows = [line for line in _commands_block(capsys.readouterr().out) if f" {base}/venvs/" in line]
    assert len(rows) == 2
    assert (
        f"    ro  tool  {path_dir}  (on the jail's PATH; "
        f"{base}/bin/path-tool links into {path_dir}; "
        f"install root not widened: {base}/venvs holds a PATH directory)"
    ) in rows
    assert (
        f"    ro  tool  {base}/venvs/{{bakabakabaka,pyxargs,uv}}  "
        f"(3 install roots; {base}/bin/baka, pyxargs and uv link into them, 2 more)"
    ) in rows


def test_install_root_groups_keep_distinct_access_modes(
    linked_install_roots: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A writable install root cannot join read-only siblings in the command table."""
    base = linked_install_roots
    writable = base / "venvs" / "uv"
    cfg = Config(sandbox=SandboxConfig(extra_write_paths=(str(writable),)))

    _check_sandbox.check_sandbox_section(cfg)

    out = capsys.readouterr().out
    rows = [line for line in _commands_block(out) if f" {base}/venvs/" in line]
    assert rows == [
        f"    ro  tool  {base}/venvs/{{bakabakabaka,pyxargs}}  "
        f"(2 install roots; {base}/bin/baka and pyxargs link into them, 1 more)",
        f"    rw  tool  {writable}  "
        f"({base}/bin/uv links into {writable}, 1 more; sandbox.extra_write_paths)",
    ]
    files = out.split("  in-process file tools", 1)[1].split("\n\n", 1)[0]
    assert f"    rw        {writable}  (sandbox.extra_write_paths)" in files


def test_link_target_directories_beside_install_roots_stay_separate(
    linked_install_roots: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Links into ordinary directories do not make those directories install roots."""
    base = linked_install_roots
    for name in ("lib", "libexec"):
        directory = base / "venvs" / name
        directory.mkdir()
        target = directory / "real"
        target.touch()
        (base / "bin" / name).symlink_to(target)

    _check_sandbox.check_sandbox_section(Config())

    rows = [line for line in _commands_block(capsys.readouterr().out) if f" {base}/venvs/" in line]
    assert len(rows) == 3
    assert any(f"{base}/venvs/{{bakabakabaka,pyxargs,uv}}" in line for line in rows)
    for name in ("lib", "libexec"):
        assert (
            f"    ro  tool  {base}/venvs/{name}  ({base}/bin/{name} links into {base}/venvs/{name})"
        ) in rows


def test_install_root_groups_keep_other_provenance(
    linked_install_roots: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Folding equal-access roots retains their configured grant annotations."""
    base = linked_install_roots
    root = base / "venvs" / "pyxargs"
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(root),)))

    _check_sandbox.check_sandbox_section(cfg)

    rows = [line for line in _commands_block(capsys.readouterr().out) if f" {base}/venvs/" in line]
    assert rows == [
        f"    ro  tool  {base}/venvs/{{bakabakabaka,pyxargs,uv}}  "
        f"(3 install roots; {base}/bin/baka, pyxargs and uv link into them, 2 more; "
        "pyxargs: sandbox.extra_read_paths)"
    ]


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
    files = [line for line in files if line.startswith("    ")]
    assert files[0] == "    --        /  (refused except the paths below)"
    assert f"    --        {home}  (refused, except the rows under it below)" in files
    rows = _commands_block(out)
    assert rows[0] == "    --        /  (empty except /proc, /dev and the paths below)"
    assert f"    --        {home}  (not mounted, except the rows under it below)" in rows
    for table in (files, rows):
        found = [pathlib.Path(line[14:].split("  ", 1)[0]) for line in table]
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
    assert f"  ! {warn.detail}" in out
    assert f"    ro        {home}  (sandbox.extra_read_paths)" in out
    assert f"    --        {home}  (" not in out


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
    assert "'full' is unavailable" not in out


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
    assert "isolation: none  (sandbox.isolation = none;" in out
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
    assert "sandbox.network = 'session' requires sandbox.isolation = 'full'" in refusal.detail
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


def test_a_root_grant_covers_the_default_rows() -> None:
    """`extra_read_paths = ["/"]` listed `ro /` over `-- / (refused ...)` and a refused home."""
    root = pathlib.Path("/")
    grant = _sandbox_paths.Row("ro", root, "sandbox.extra_read_paths")
    defaults = (
        _sandbox_paths.Row("--", root, "refused except the paths below"),
        _sandbox_paths.Row("--", pathlib.Path("/home/op"), "refused, except the rows under it"),
    )
    assert _sandbox_paths.with_defaults([grant], *defaults) == [grant]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("isolation", ["full", "partial"])
def test_policy_paths_merge_discovery_and_config_without_losing_access(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
    isolation: str,
) -> None:
    """Overlapping grants preserve each scope's access, including hidden children."""
    _force_profile(monkeypatch, isolation)
    root = pathlib.Path.cwd()
    cfg = Config.model_validate(
        {
            "sandbox": {
                "isolation": isolation,
                "extra_read_paths": ["/opt/tools/bin", "/opt/tools/bin", str(root / "vendor")],
                "extra_write_paths": ["/opt/tools/bin"],
                "hide_paths": ["/opt/tools/bin/private"],
                "extra_device_paths": ["/dev/random"],
            }
        }
    )
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    for scope in ("in-process file tools", "jailed commands"):
        block = out.split(f"  {scope}", 1)[1].split("\n\n", 1)[0]
        rows = re.findall(r"^    (?:rw|ro|--|dev)\s+(?:(?:sys|net|tool)\s+)?(/\S*)", block, re.M)
        assert rows and all(n == 1 for n in collections.Counter(rows).values()), block
        vendor = next(line for line in block.splitlines() if f"{root}/vendor  " in line)
        assert vendor.lstrip().startswith("rw  "), vendor
        if scope == "jailed commands" and isolation == "partial":
            assert (
                "    rw        /opt/tools/bin/private  ! "
                "(not hidden: partial cannot mask this grant)" in block
            ), block
        else:
            assert "    --        /opt/tools/bin/private  (hidden)" in block, block
    tool = next(line for line in _commands_block(out) if "/opt/tools/bin  " in line)
    assert tool.startswith("    rw  tool  "), tool
    assert "on the jail's PATH" in tool
    assert "sandbox.extra_read_paths" in tool and "sandbox.extra_write_paths" in tool
    assert "    rw  sys   /dev/random  (sandbox.extra_device_paths)" in out


@pytest.mark.parametrize("isolation", ["full", "partial"])
def test_device_rows_use_each_nodes_policy_access(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
    isolation: str,
) -> None:
    """The five device grants are separate from shared memory, with distinct read/write rights."""
    _force_profile(monkeypatch, isolation)
    _check_sandbox.check_sandbox_section(Config())
    rows = _commands_block(capsys.readouterr().out)
    for name, mode in (
        ("null", "rw"),
        ("zero", "rw"),
        ("urandom", "ro"),
        ("random", "ro"),
        ("full", "rw"),
    ):
        assert f"    {mode}  sys   /dev/{name}" in rows
    shm = next(line for line in rows if " /dev/shm  " in line)
    assert shm.startswith(f"    {'rw' if isolation == 'full' else '--'}        /dev/shm")
    assert "shared memory" in shm and "multiprocessing" not in shm


@pytest.mark.parametrize(
    ("isolation", "protect", "mode", "detail"),
    [
        ("full", "auto", "ro", "configured auto, read-only under full"),
        ("partial", "on", "rw", "configured on, writable: protection requires full isolation"),
        ("full", "off", "rw", "configured off, writable"),
        ("none", "on", "rw", "configured on, writable: protection requires full isolation"),
    ],
)
def test_git_row_names_configuration_and_effective_protection(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
    isolation: str,
    protect: str,
    mode: str,
    detail: str,
) -> None:
    """Writable Git metadata does not imply that protection was disabled in configuration."""
    _force_profile(monkeypatch, isolation)
    cfg = Config.model_validate({"sandbox": {"protect_git": protect}})
    _check_sandbox.check_sandbox_section(cfg)
    rows = _commands_block(capsys.readouterr().out)
    git = pathlib.Path.cwd() / ".git"
    row = next(line for line in rows if f" {git}  " in line)
    assert row.startswith(f"    {mode}        {git}  ")
    assert f"sandbox.protect_git: {detail}" in row


@pytest.mark.parametrize("isolation", ["partial", "none"])
def test_skipped_network_probe_keeps_the_commands_exit_code(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
    isolation: str,
) -> None:
    """A missing network namespace is a SKIP, not a successful network probe or a new failure."""
    _force_profile(monkeypatch, isolation)
    cfg = tmp_path / "check.toml"
    cfg.write_text(f'[sandbox]\nisolation = "{isolation}"\n', encoding="utf-8")
    rc = check_cmds._cmd_check((cfg,), section="sandbox")  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"[SKIP] jail_blocks_network: {isolation} has no network namespace" in out
    for name in ("landlock_abi", "jail_true", "jail_blocks_etc_write"):
        assert name in out
    assert not any(p.argv[0].endswith("getent") for p in stub_jail)


def test_report_colours_and_plain_modes_carry_the_same_words(
    monkeypatch: pytest.MonkeyPatch,
    linked_install_roots: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Colour adds hierarchy without colouring permissions or hiding words in plain output."""
    _force_profile(monkeypatch, "partial", reason="user namespaces blocked: try unshare -U -r true")
    monkeypatch.setenv("COLUMNS", "4000")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    _check_sandbox.check_sandbox_section(Config())
    coloured = capsys.readouterr().out
    assert "\x1b[1mpartial\x1b[0m" in coloured
    assert "\x1b[32m[PASS]\x1b[0m" in coloured
    assert "\x1b[2m[SKIP]\x1b[0m" in coloured
    assert "\x1b[33m!\x1b[0m" in coloured
    assert "    ro  \x1b[2msys " in coloured
    plain = re.sub(r"\x1b\[[0-9;]*m", "", coloured)
    assert (
        f"    rw  tool  {linked_install_roots}/venvs/{{bakabakabaka,pyxargs,uv}}  "
        f"(3 install roots; {linked_install_roots}/bin/baka, pyxargs and uv link into them, 2 more)"
    ) in _commands_block(plain)
    for legend in (
        "sys: system paths, including granted device nodes",
        "net: files supporting host name resolution and TLS verification",
        "tool: operator tools and their runtime dependencies",
    ):
        assert legend in plain
    for tty, no_color in ((True, True), (False, False)):
        monkeypatch.setattr(sys.stdout, "isatty", lambda tty=tty: tty)
        if no_color:
            monkeypatch.setenv("NO_COLOR", "1")
        else:
            monkeypatch.delenv("NO_COLOR", raising=False)
        _check_sandbox.check_sandbox_section(Config())
        assert capsys.readouterr().out == plain


def test_tool_discovery_retains_links_beside_path_reason(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A containing directory keeps link provenance and counts even when it is also on PATH."""
    _force_profile(monkeypatch, "full")
    bins, runtime = tmp_path / "bins", tmp_path / "data" / "uv" / "python"
    bins.mkdir()
    runtime.mkdir(parents=True)
    for target in (bins / "real", runtime / "python"):
        target.touch()
    for name in ("a", "b"):
        (bins / name).symlink_to(bins / "real")
    (bins / "python").symlink_to(runtime / "python")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins,))
    monkeypatch.setattr(_check_sandbox, "landlock_abi", lambda: 5)

    def run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        return _fake_result(policy.argv, 2 if policy.argv[0].endswith("getent") else 0)

    monkeypatch.setattr(_check_sandbox, "run_in_jail", run)
    _check_sandbox.check_sandbox_section(Config())
    rows = _commands_block(capsys.readouterr().out)
    bin_row = next(line for line in rows if f" {bins}  " in line)
    assert "on the jail's PATH" in bin_row
    assert f"{bins}/a links into {bins}, 1 more" in bin_row
    runtime_row = next(line for line in rows if f" {runtime}  " in line)
    assert "uv-managed interpreters" not in runtime_row
    assert f"{bins}/python links into {runtime}" in runtime_row


def test_regular_path_tools_have_no_install_root_row(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A regular PATH tool has no link annotation or row exposing sibling credentials."""
    root = tmp_path / "x"
    bins = root / "bin"
    bins.mkdir(parents=True)
    (bins / "tool").touch()
    (root / "credentials.toml").touch()
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins,))

    rows = _sandbox_paths._tool_rows()

    assert rows == [_sandbox_paths.Row("ro", bins, "on the jail's PATH", "tool")]
    _check_render.rows(rows)
    assert capsys.readouterr().out.strip() == f"ro  tool  {bins}  (on the jail's PATH)"


@pytest.mark.parametrize("path_subdir", [".", "bin", "other/bin"])
def test_linked_tool_keeps_bin_when_the_install_root_holds_path(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    path_subdir: str,
) -> None:
    """A PATH directory at or under the install root prevents widening and names the reason."""
    bins = tmp_path / "pathdir"
    bins.mkdir()
    root = tmp_path / "y"
    target = root / "bin" / "py"
    target.parent.mkdir(parents=True)
    target.touch()
    link = bins / "py"
    link.symlink_to(target)
    path_dir = root / path_subdir
    path_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins, path_dir))

    rows = _sandbox_paths._tool_rows()

    assert {r.path for r in rows} == {bins, path_dir, target.parent}
    bin_row = next(r for r in rows if r.path == target.parent)
    note = "on the jail's PATH; " if path_subdir == "bin" else ""
    note += (
        f"{link} links into {target.parent}; "
        f"install root not widened: {root} holds a PATH directory"
    )
    assert bin_row == _sandbox_paths.Row("ro", target.parent, note, "tool")
    _check_render.rows([bin_row])
    assert capsys.readouterr().out.strip() == f"ro  tool  {target.parent}  ({note})"


@pytest.mark.parametrize("plain_first", [False, True])
def test_linked_tool_row_names_the_directory_link(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    plain_first: bool,
) -> None:
    """Name the directory link whose parent keeps a shim's target visible."""
    bins = tmp_path / "pathdir"
    bins.mkdir()
    pythons = tmp_path / "data" / "python"
    root = pythons / "cpython-3.12.13-linux"
    target = root / "bin" / "py"
    target.parent.mkdir(parents=True)
    target.touch()
    directory = pythons / "cpython-3.12-linux"
    directory.symlink_to(root.name, target_is_directory=True)
    shim = bins / "py"
    shim.symlink_to(directory / "bin" / "py")
    if plain_first:
        plain = pythons / "bin" / "py"
        plain.parent.mkdir()
        plain.touch()
        (bins / "a").symlink_to(plain)
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins,))

    rows = _sandbox_paths._tool_rows()

    assert {r.path for r in rows} == {bins, pythons}
    row = next(r for r in rows if r.path == pythons)
    note = f"{shim} links into {pythons} through {directory}"
    if plain_first:
        note += ", 1 more"
    assert row == _sandbox_paths.Row("ro", pythons, note, "tool")
    _check_render.rows([row])
    assert capsys.readouterr().out.strip() == f"ro  tool  {pythons}  ({note})"


@pytest.mark.parametrize("parent_kind", ["ancestor", "home", "path", "private", "private-parent"])
def test_linked_tool_keeps_resolved_root_when_link_parent_is_unsafe(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    parent_kind: str,
) -> None:
    """Refuse an unsafe directory-link parent and report the resolved-root fallback."""
    home = tmp_path / "home"
    bins = home / ".local" / "bin"
    bins.mkdir(parents=True)
    root = tmp_path / "data" / "python" / "cpython-3.12.13-linux"
    target = root / "bin" / "py"
    target.parent.mkdir(parents=True)
    target.touch()
    private_parent = tmp_path / "private-holder"
    private = private_parent / "agent6"
    private.mkdir(parents=True)
    parent = {
        "ancestor": tmp_path,
        "home": home,
        "path": root.parent,
        "private": private,
        "private-parent": private_parent,
    }[parent_kind]
    directory = parent / "cpython-3.12-linux"
    directory.symlink_to(root, target_is_directory=True)
    shim = bins / "py"
    shim.symlink_to(directory / "bin" / "py")
    path_dirs = [bins]
    if parent_kind == "path":
        path_dir = parent / "other" / "bin"
        path_dir.mkdir(parents=True)
        path_dirs.append(path_dir)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(private_parent))
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: tuple(path_dirs))

    _, mounts = tool_paths.operator_tool_paths()

    assert mounts == tuple(sorted((*path_dirs, root)))
    rows = [r for r in _sandbox_paths._tool_rows() if r.path == root]
    refusal = f"{parent} holds a PATH directory" if parent_kind == "path" else str(parent)
    note = f"{shim} links into {root}; install root not widened: {refusal}"
    assert rows == [_sandbox_paths.Row("ro", root, note, "tool")]
    _check_render.rows(rows)
    assert capsys.readouterr().out.strip() == f"ro  tool  {root}  ({note})"


@pytest.mark.parametrize("target_dir", ["bin", "libexec"])
def test_linked_tools_grant_the_install_root_only_for_bin_targets(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target_dir: str,
) -> None:
    """Linked bin tools grant their install root; other targets grant only their directory."""
    home = tmp_path / "home"
    bins = home / ".local" / "bin"
    bins.mkdir(parents=True)
    root = tmp_path / "package"
    target = root / target_dir / "py"
    target.parent.mkdir(parents=True)
    target.touch()
    for name in ("py", "python3"):
        (bins / name).symlink_to(target)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins,))
    grant = root if target_dir == "bin" else root / "libexec"

    _, mounts = tool_paths.operator_tool_paths()
    assert mounts == tuple(sorted((bins, grant)))
    rows = [r for r in _sandbox_paths._tool_rows() if r.path == grant]
    assert rows == [
        _sandbox_paths.Row("ro", grant, f"{bins}/py links into {grant}, 1 more", "tool")
    ]
    _check_render.rows(rows)
    assert capsys.readouterr().out.strip() == (
        f"ro  tool  {grant}  ({bins}/py links into {grant}, 1 more)"
    )


@pytest.mark.parametrize(
    "root_kind", ["ancestor", "filesystem", "home", "home-link", "private", "private-link"]
)
def test_linked_tool_keeps_bin_when_the_install_root_is_unsafe(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    """A blocked install root keeps the bin grant and names the refusal in its annotation."""
    home = tmp_path / "home"
    bins = home / ".local" / "bin"
    bins.mkdir(parents=True)
    root = {
        "ancestor": tmp_path,
        "filesystem": pathlib.Path("/"),
        "home": home,
        "home-link": home,
        "private": tmp_path / "private-holder",
        "private-link": tmp_path / "private-holder",
    }[root_kind]
    target = root / "bin" / "sh"
    if root_kind != "filesystem":
        target.parent.mkdir(parents=True)
        target.touch()
    if root_kind.startswith("private"):
        config = root / "config"
        config.mkdir()
        if root_kind == "private-link":
            alias = tmp_path / "config-link"
            alias.symlink_to(config, target_is_directory=True)
            config = alias
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    link = bins / "sh"
    link.symlink_to(target)
    if root_kind == "filesystem":
        # Model an unmerged /bin on hosts where /bin resolves to /usr/bin.
        resolve = pathlib.Path.resolve

        def unmerged(path: pathlib.Path, strict: bool = False) -> pathlib.Path:
            return target if path == link else resolve(path, strict=strict)

        monkeypatch.setattr(pathlib.Path, "resolve", unmerged)
    if root_kind == "home-link":
        alias = tmp_path / "home-link"
        alias.symlink_to(home, target_is_directory=True)
        home = alias
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tool_paths, "_tool_bin_dirs", lambda: (bins,))

    _, mounts = tool_paths.operator_tool_paths()
    assert root not in mounts
    if root_kind != "filesystem":
        assert target.parent in mounts
    rows = [r for r in _sandbox_paths._tool_rows() if r.path == target.parent]
    assert rows == [
        _sandbox_paths.Row(
            "ro",
            target.parent,
            f"{link} links into {target.parent}; install root not widened: {root}",
            "tool",
        )
    ]


def test_resource_annotations_follow_explicit_grants(
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit resource grant does not keep the default unavailable annotation."""
    _force_profile(monkeypatch, "partial")
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=("/proc", "/dev/shm")))
    _check_sandbox.check_sandbox_section(cfg)
    rows = _commands_block(capsys.readouterr().out)
    for path in ("/proc", "/dev/shm"):
        row = next(line for line in rows if f" {path}  " in line)
        assert row.startswith(f"    ro        {path}  ")
        assert "unavailable" not in row
        assert "sandbox.extra_read_paths" in row


def test_network_provenance_does_not_override_a_write_grant(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path discovered for network support keeps its configured write access."""
    from agent6.tools import policy

    _force_profile(monkeypatch, "full")
    path = tmp_path / "resolver"
    path.touch()
    monkeypatch.setattr(policy, "host_network_reads", lambda: (path,))
    cfg = Config(sandbox=SandboxConfig(network="host", extra_write_paths=(str(path),)))
    _check_sandbox.check_sandbox_section(cfg)
    rows = [line for line in _commands_block(capsys.readouterr().out) if f" {path}  " in line]
    assert rows == [f"    rw  net   {path}  (sandbox.extra_write_paths)"]


@pytest.mark.parametrize("isolation", ["full", "partial"])
def test_hidden_symlink_row_uses_the_resolved_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
    isolation: str,
) -> None:
    """A hidden symlink's target is masked under full and remains writable under partial."""
    _force_profile(monkeypatch, isolation)
    private = tmp_path / "private"
    private.mkdir()
    link = tmp_path / "link"
    link.symlink_to(private, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    cfg = Config.model_validate({"sandbox": {"isolation": isolation, "hide_paths": [str(link)]}})

    _check_sandbox.check_sandbox_section(cfg)

    rows = _commands_block(capsys.readouterr().out)
    masked = [line for line in rows if f" {private}  " in line or f" {link}  " in line]
    expected = (
        f"    --        {private}  (hidden)"
        if isolation == "full"
        else f"    rw        {private}  ! (not hidden: partial cannot mask this grant)"
    )
    assert masked == [expected]


def test_git_symlink_row_uses_the_protected_metadata_path(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_jail: list[kinds.JailPolicy],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A Git symlink resolves to the same metadata path that the jail protects."""
    _force_profile(monkeypatch, "full")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (tmp_path / ".git").symlink_to(metadata, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    _check_sandbox.check_sandbox_section(Config())
    rows = _commands_block(capsys.readouterr().out)
    assert (
        f"    ro        {metadata}  (sandbox.protect_git: configured auto, read-only under full)"
        in rows
    )


def test_annotations_wrap_without_overflow_or_shortening_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An annotation whose first word cannot fit beside its path starts on the next line."""
    path = pathlib.Path("/" + "x" * 50)
    monkeypatch.setenv("COLUMNS", "70")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    note = "annotations retain every word and the complete path"
    _check_render.rows([_sandbox_paths.Row("ro", path, note)])
    lines = capsys.readouterr().out.splitlines()
    assert all(len(line) <= 70 for line in lines), lines
    assert lines[0] == f"    ro        {path}"
    assert all(line.startswith(" " * 14) for line in lines[1:])
    assert " ".join(line.strip() for line in lines[1:]) == f"({note})"
