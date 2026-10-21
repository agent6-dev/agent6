# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check sandbox` tells the truth about what the launcher enforces.

The report's system-path lists are Python mirrors of constants the Rust launcher owns:
SYSTEM_BINDS, and the `partial` ro_paths and rw_paths bases and device nodes. The `partial`
hide_paths preflight reads them too. A drift from the enforcer would make the report or the
preflight lie, so the mirrors are pinned against the launcher's source text.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from agent6 import kinds, paths
from agent6.app import _setup, confine
from agent6.config import Config, SandboxConfig
from agent6.sandbox import detect, tool_paths
from agent6.tools import policy as tools_policy
from agent6.ui.cli import _check_sandbox

_MAIN_RS = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail" / "src" / "main.rs"
)
_ENV = detect.Environment(
    in_container=False,
    container_signals=(),
    kernel=detect.KernelInfo(raw="6.8", major=6, minor=8),
    userns_supported=True,
    landlock_abi=5,
    seccomp_arch_supported=True,
    sandbox_available=True,
)
_TOOL_DIR = pathlib.Path("/opt/tools/bin")


def test_full_system_binds_mirror_the_launcher() -> None:
    src = _MAIN_RS.read_text(encoding="utf-8")
    m = re.search(r"const SYSTEM_BINDS[^=]*=\s*\[(.*?)\];", src, re.S)
    assert m, "SYSTEM_BINDS not found in main.rs"
    launcher = tuple(re.findall(r'"([^"]+)"', m.group(1)))
    assert launcher == _check_sandbox.FULL_SYSTEM_BINDS


def test_partial_ro_base_mirrors_the_launcher() -> None:
    src = _MAIN_RS.read_text(encoding="utf-8")
    m = re.search(r"let mut ro_paths[^=]*=\s*vec!\[(.*?)\];", src, re.S)
    assert m, "`partial` ro_paths base not found in main.rs"
    launcher = tuple(re.findall(r'PathBuf::from\("([^"]+)"\)', m.group(1)))
    assert launcher == confine.PARTIAL_SYSTEM_RO


def test_partial_rw_base_mirrors_the_launcher() -> None:
    """The preflight's writable host regions are the base `apply_landlock_partial` grants."""
    src = _MAIN_RS.read_text(encoding="utf-8")
    m = re.search(r"let mut rw_paths[^=]*=\s*vec!\[(.*?)\];", src, re.S)
    assert m, "`partial` rw_paths base not found in main.rs"
    launcher = tuple(re.findall(r'PathBuf::from\("([^"]+)"\)', m.group(1)))
    assert launcher == confine.PARTIAL_SYSTEM_RW


def test_partial_device_nodes_mirror_the_launcher() -> None:
    """The preflight's device-node regions are the nodes `apply_landlock_partial` grants.

    The read loop names them all; the write loop for the sink devices names a subset.
    """
    src = _MAIN_RS.read_text(encoding="utf-8")
    body = src[src.index("fn apply_landlock_partial") :].split("\nfn ", 1)[0]
    loops = [
        tuple(f"/dev/{n}" for n in re.findall(r'"([^"]+)"', m))
        for m in re.findall(r"for dev in \[(.*?)\]", body)
    ]
    assert len(loops) == 2, "`partial` device-node rules not found in main.rs"
    assert loops[0] == confine.PARTIAL_DEVICE_NODES
    assert set(loops[1]) <= set(confine.PARTIAL_DEVICE_NODES)


def _force(monkeypatch: pytest.MonkeyPatch, isolation: str, reason: str | None = None) -> None:
    """Pin the host at one level, with passing probes and one operator tool dir."""
    monkeypatch.setattr(_setup, "detect_env", lambda: _ENV)

    def _select(_req: str, _env: object) -> str:
        return isolation

    def _reason(_env: object) -> str | None:
        return reason

    def _run(policy: kinds.JailPolicy) -> kinds.CommandResult:
        rc = 2 if policy.argv[0].endswith("getent") else 0  # the DNS probe reads blocked
        return kinds.CommandResult(
            argv=policy.argv, returncode=rc, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr(detect, "resolve_isolation", _select)
    monkeypatch.setattr(detect, "degrade_reason", _reason)
    monkeypatch.setattr(_check_sandbox, "landlock_abi", lambda: 5)
    monkeypatch.setattr(_check_sandbox, "run_in_jail", _run)
    monkeypatch.setattr(
        tool_paths,
        "operator_tool_mounts",
        lambda: (tool_paths.ToolMount(_TOOL_DIR, "on the jail's PATH"),),
    )


def _verdicts(checks: list[kinds.DoctorCheck]) -> list[tuple[str, str]]:
    """Return the rows that are not PASS."""
    return [(c.name, c.status) for c in checks if c.status != "PASS"]


def test_both_levels_build_their_landlock_rules_from_the_one_set_owner() -> None:
    """Both levels take `landlock_sets`, whose /proc set strips Execute.

    A right the ruleset does not pass to `handle_access` is left unrestricted, so a handled
    set narrower than the full ABI turns a lock off without any rule changing; and
    `AccessFs::from_read` carries Execute, so a read grant on /proc would let a jailed
    command execve files there. The behaviour pins cannot see either: seccomp refuses mknod
    first, and nothing under /proc is executable on a stock kernel.
    """
    src = _MAIN_RS.read_text(encoding="utf-8")
    assert src.count("let handled = AccessFs::from_all(ABI::V3);") == 1
    assert src.count("all: handled & !AccessFs::MakeChar & !AccessFs::MakeBlock,") == 1
    assert src.count("read_noexec: read & !AccessFs::Execute,") == 1
    assert src.count(".handle_access(sets.handled)") == 2, "a level builds its own handled set"
    proc_rules = re.findall(
        r'PathFd::new\("/proc"\)[^;]*?PathBeneath::new\(fd, (sets\.\w+)\)', src, re.S
    )
    assert proc_rules == ["sets.read_noexec"], proc_rules


def test_boundaries_report_covers_every_actor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One glance answers: who is confined, which files, which network."""
    _force(monkeypatch, "full")
    cfg = Config(sandbox=SandboxConfig(hide_paths=("/work/secrets",)))
    checks = _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert _verdicts(checks) == []
    assert "in-process file tools" in out
    assert "jailed commands" in out
    assert f"    rw  {pathlib.Path.cwd()}  (the workspace)" in out
    assert (
        f"    ro  {pathlib.Path.cwd() / '.git'}  (re-bound read-only: sandbox.protect_git = true)"
        in out
    )
    for p in _check_sandbox.FULL_SYSTEM_BINDS:
        assert f"    ro  {p}  (system" in out
    assert "rw  /tmp/agent6-home  (HOME" in out
    assert f"    ro  {_TOOL_DIR}  (operator tools: on the jail's PATH)" in out
    assert "    --  /work/secrets  (masked out of the jail's view)" in out
    assert "  network:   session (sandbox.network = auto)" in out
    assert "  memory:    no cap (sandbox.memory_limit_mb = 0)" in out
    assert "mcp servers: none run" in out
    assert "egress is unbounded" in out
    assert "secrets.toml" in out


def test_boundaries_report_names_the_memory_dir_for_the_file_tools_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report hid the file tools' memory exception under the state-base denial."""
    _force(monkeypatch, "full")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    subdir = repo / "src"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    state_home = tmp_path / "statehome"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    mem = paths.state_dir(repo) / "memory"
    assert mem.is_relative_to(paths.state_base())

    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    files, commands = out.split("  jailed commands", 1)
    assert _verdicts(checks) == []
    assert (f"    rw  {mem}  (repo memory, inside the hidden state dir)") in files
    assert f"    --  {paths.state_base()}  (hidden: tools refuse it)" in files
    assert not any(
        str(mem) in line and "sandbox.extra_write_paths" in line for line in out.splitlines()
    )
    assert f"rw  {mem}" not in commands
    assert not state_home.exists(), "reporting the memory exception must not create the store"


def test_boundaries_report_is_level_aware(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under `partial` the report names Landlock rules and none of `full`'s mounts.

    It also says why `auto` resolved below `full`.
    """
    _force(monkeypatch, "partial", reason="userns blocked (test)")
    checks = _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert _verdicts(checks) == []
    assert "'full' is unavailable because userns blocked (test)" in out
    for p in confine.PARTIAL_SYSTEM_RO:
        assert f"    ro  {p}  (system, by Landlock rule)" in out
    assert f"rw  {paths.jail_cache_home(pathlib.Path.cwd())}  (HOME" in out and "persists" in out
    assert "denied by Landlock" in out
    assert "re-bound" not in out
    assert "private to the command" not in out
    assert "  network:   host (sandbox.network = auto)" in out


def test_boundaries_report_says_what_partial_leaves_writable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under partial the report left out the host's /tmp and said nothing about `.git`.

    A jailed command writes both there: Landlock grants the host's /tmp, and no mount
    namespace exists to re-bind `.git` read-only. The host's /dev/shm is not granted, which
    the report names beside the device nodes that are.
    """
    _force(monkeypatch, "partial")
    _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    git = pathlib.Path.cwd() / ".git"
    assert (
        f"    rw  {git}  (writable: sandbox.protect_git requires sandbox.isolation = 'full')" in out
    )
    assert "    --  /  (Landlock: denied except the paths below)" in out
    assert "    rw  /tmp  (the host's /tmp, shared with every process of yours)" in out
    assert "    --  /proc  (not granted under partial" in out
    assert "    --  /dev/shm  (the host's, not granted" in out


def test_boundaries_report_prints_one_row_for_a_hidden_dev_shm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`hide_paths = ["/dev/shm"]` under partial printed the path twice: not granted, and hidden."""
    _force(monkeypatch, "partial")
    _check_sandbox.check_sandbox_section(Config(sandbox=SandboxConfig(hide_paths=("/dev/shm",))))
    jailed = capsys.readouterr().out.partition("jailed commands (")[2]
    assert jailed.count("  /dev/shm  ") == 1, jailed
    assert "    --  /dev/shm  (denied by Landlock)" in jailed


def test_boundaries_report_lists_what_the_host_network_reads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report lists the resolver and CA store grants bound for the host network.

    Off the host network `full`'s /etc is empty, and the report says so only there.
    """
    reason = "(name resolution and TLS verification on the host network)"
    _force(monkeypatch, "full")
    host = Config(sandbox=SandboxConfig(network="host"))
    _check_sandbox.check_sandbox_section(host)
    out = capsys.readouterr().out
    reads = tools_policy.host_network_reads()
    assert reads
    for path in reads:
        assert f"    ro  {path}  {reason}" in out
    assert "the rest of /etc is empty" not in out
    _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert reason not in out
    assert "    ro  /etc/alternatives  (system; the rest of /etc is empty)" in out
    _force(monkeypatch, "partial")
    _check_sandbox.check_sandbox_section(Config())
    assert reason in capsys.readouterr().out, "partial shares the host network"


def test_boundaries_report_shows_each_grant_as_a_run_binds_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """A symlinked grant was listed by its target alone; a dropped grant reads as not granted."""
    _force(monkeypatch, "full")
    real, link = tmp_path / "tool-1.2", tmp_path / "tool"
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    inside = pathlib.Path.cwd() / "vendored"
    # Pinned to `full`: pytest's tmp_path sits in /tmp, where a link counts under auto.
    cfg = Config(sandbox=SandboxConfig(isolation="full", extra_read_paths=(str(link), str(inside))))
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    bound = f"    ro  {real}  (sandbox.extra_read_paths, configured as {link})"
    dropped = (
        f"    --  {inside}  (sandbox.extra_read_paths: not granted because it sits inside"
        f" {pathlib.Path.cwd()}, which jailed commands write)"
    )
    assert out.count(bound) == 2 and out.count(dropped) == 2, out
    assert f"    ro  {inside}" not in out


def test_boundaries_report_says_when_only_a_link_puts_a_grant_in_the_workspace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """A grant dropped for a link on its way would read "it sits inside" the workspace, falsely."""
    _force(monkeypatch, "full")
    repo, real, grant = tmp_path / "repo", tmp_path / "real", tmp_path / "grant"
    repo.mkdir()
    real.mkdir()
    monkeypatch.chdir(repo)
    (repo / "hop").symlink_to(real, target_is_directory=True)
    grant.symlink_to(repo / "hop", target_is_directory=True)
    cfg = Config(sandbox=SandboxConfig(extra_read_paths=(str(grant),)))
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert (
        f"    --  {grant}  (sandbox.extra_read_paths: not granted because a link on its way"
        f" sits inside {repo}, which jailed commands write)"
    ) in out, out


def test_boundaries_report_names_the_opted_in_persistent_home(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The jail's HOME is a write grant, so the report lists it.

    Under `home = "cache"` `full`'s is the persistent cache dir, named as such.
    """
    _force(monkeypatch, "full")
    _check_sandbox.check_sandbox_section(Config(sandbox=SandboxConfig(home="cache")))
    out = capsys.readouterr().out
    assert (
        f"rw  {paths.jail_cache_home(pathlib.Path.cwd())}  (HOME" in out
        and "sandbox.home = 'cache'" in out
    )
    assert "/tmp/agent6-home" not in out


def test_boundaries_report_names_the_home_under_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`none` has no jail and still a HOME of agent6's own, listed under the root row.

    The HOME grant is listed at every level; so are the private dirs nothing hides there.
    """
    _force(monkeypatch, "none")
    _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert "    rw  /  (sandbox disabled: the whole filesystem and network, as " in out
    home = paths.jail_cache_home(pathlib.Path.cwd())
    assert (
        f"rw  {home}  (HOME, persists across runs: sandbox.isolation = 'none' has no private /tmp)"
    ) in out
    assert f"    rw  {paths.global_config_dir()}  (not hidden: secrets.toml is readable)" in out
    assert (
        f"    rw  {paths.state_base()}  (not hidden: every session's records are writable)" in out
    )
    assert "masked" not in out and "denied by Landlock" not in out


def test_boundaries_report_lists_a_fork_worktrees_git_dir_grant(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inside a fork's worktree, the report lists the repository git dir a jailed command reaches.

    That is the fork manifest's recorded `worktree_git_dir` (under the repository's state
    dir), granted through the execution's own policy builder; a report that read no manifest
    would omit the one path beyond the workspace. In the repository itself no such line
    appears.
    """
    import json
    import subprocess

    from agent6 import git_ops
    from agent6.sessions import layout as sessions_layout

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        check=True,
    )
    worktree = tmp_path / "wt"
    git_ops.add_worktree(repo, worktree, "HEAD")
    git_dir = (repo / ".git").resolve()
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="fork-AAAA11"
    )
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": "fork-AAAA11",
                "mode": "run",
                "worktree": str(worktree),
                "worktree_git_dir": str(git_dir),
            }
        ),
        encoding="utf-8",
    )
    _force(monkeypatch, "full")
    grant = f"ro  {git_dir}  (the repository's .git, which this linked worktree points into)"

    monkeypatch.chdir(worktree)
    checks = _check_sandbox.check_sandbox_section(Config())
    assert _verdicts(checks) == []
    assert grant in capsys.readouterr().out

    monkeypatch.chdir(repo)
    _check_sandbox.check_sandbox_section(Config())
    out = capsys.readouterr().out
    assert grant not in out
    assert f"    ro  {git_dir}  (re-bound read-only: sandbox.protect_git = true)" in out


def test_boundaries_report_says_withheld_rather_than_unapproved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run_commands = "no"` withholds the command tools, and the header says so.

    "approval: sandbox.run_commands = no" would describe a prompting policy for tools the
    model never sees.
    """
    _force(monkeypatch, "full")
    _check_sandbox.check_sandbox_section(Config(sandbox=SandboxConfig(run_commands="no")))
    out = capsys.readouterr().out
    assert "  commands:  withheld from the model (sandbox.run_commands = no)" in out
    assert "approval: sandbox.run_commands" not in out


def test_boundaries_report_names_each_mcp_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config import _sandbox

    _force(monkeypatch, "full")
    cfg = Config(
        mcp=_sandbox.MCPConfig(
            enabled=True,
            servers={
                "docs": _sandbox.MCPServerEntry(url="http://127.0.0.1:9000/mcp"),
                "fs": _sandbox.MCPServerEntry(
                    command=("mcp-fs",),
                    sandbox=_sandbox.MCPSandbox(read_paths=("/data",), network="none"),
                ),
                "off": _sandbox.MCPServerEntry(command=("x",), enabled=False),
                "plain": _sandbox.MCPServerEntry(command=("y",)),
            },
        )
    )
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert "docs: http http://127.0.0.1:9000/mcp" in out
    assert "fs: spawned in the jail" in out and "paths ro+1 rw+0, network none" in out
    assert "off: DISABLED" in out
    # The network the server GETS, never the configured word: `auto` is a
    # namespace of its own under `full`.
    assert "plain: spawned in the jail" in out and "paths ro+0 rw+0, network none" in out


def test_boundaries_report_prints_the_refusal_for_a_network_the_level_cannot_give(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under `partial` the report prints the refusal for a server's `none` or `session` network.

    A run refuses the server before it starts, so its row never names the host network.
    """
    from agent6.config import _sandbox

    _force(monkeypatch, "partial", "no user namespaces")
    cfg = Config(
        mcp=_sandbox.MCPConfig(
            enabled=True,
            servers={
                "quiet": _sandbox.MCPServerEntry(
                    command=("y",), sandbox=_sandbox.MCPSandbox(network="none")
                )
            },
        )
    )
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert "quiet: a run would refuse: MCP server 'quiet' sets sandbox.network = 'none'" in out
    assert "network host" not in out


def test_boundaries_report_gives_an_mcp_server_the_network_the_level_can(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a level with no namespaces, `auto` resolves to the host's network and the report says so.

    Never the knob's own word.
    """
    from agent6.config import _sandbox

    _force(monkeypatch, "partial", "no user namespaces")
    cfg = Config(
        mcp=_sandbox.MCPConfig(
            enabled=True, servers={"plain": _sandbox.MCPServerEntry(command=("y",))}
        )
    )
    _check_sandbox.check_sandbox_section(cfg)
    out = capsys.readouterr().out
    assert "plain: spawned in the jail" in out and "network host" in out
    assert "network auto" not in out
