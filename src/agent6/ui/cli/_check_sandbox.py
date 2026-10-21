# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""List what the file tools, jailed commands and MCP servers can reach, then run the jail probes.

Each table has one row per path, sorted so a parent precedes its children, with a root row and a
home-directory row.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6 import errors, git_ops, kinds, memory
from agent6 import paths as agent6_paths
from agent6.app import _setup, confine, fork_worktrees
from agent6.config import Config
from agent6.sandbox import (
    JailUnavailableError,
    detect,
    landlock,
    landlock_abi,
    run_in_jail,
    tool_paths,
)
from agent6.tools import _path_safety
from agent6.tools import policy as tools_policy

# Mirrors SYSTEM_BINDS in the jail's main.rs; tests/security pins the two against each other.
FULL_SYSTEM_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives")

_TMP = pathlib.Path("/tmp")  # noqa: S108 - the jail's own tmpfs under `full`
_DEV_SHM = pathlib.Path("/dev/shm")  # noqa: S108 - named as not granted
_FILE_TOOLS = (
    "read_file, list_dir, outline, find_definition, find_references, apply_edit, apply_patch"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """One path in a table, with its access and the reason.

    Attributes:
        mode: `rw`, `ro`, `--` (refused or hidden) or `dev` (a device node).
        path: The resolved path.
        note: Why, or which setting grants it.
    """

    mode: str
    path: pathlib.Path
    note: str


def _print_rows(rows: list[Row]) -> None:
    """Print the rows sorted by path components, so a parent precedes its children."""
    for r in sorted(rows, key=lambda r: r.path.parts):
        print(f"    {r.mode}  {r.path}  ({r.note})")


def _with_defaults(rows: list[Row], *defaults: Row) -> list[Row]:
    """Return the rows plus each default whose path no row already names."""
    named = {r.path for r in rows}
    return [*rows, *(d for d in defaults if d.path not in named)]


def grant_rows(grants: tools_policy.Grants) -> list[Row]:
    """Return the rows of the extra path grants the file tools and jailed commands share.

    Args:
        grants: The config's resolved path grants.

    Returns:
        Each row names the resolved path a run binds, with the configured spelling when the two
        differ, and each grant a run leaves out says why.
    """

    def row(mode: str, key: str, path: pathlib.Path, real: pathlib.Path) -> Row:
        spelled = "" if path == real else f", configured as {path}"
        return Row(mode, real, f"sandbox.{key}{spelled}")

    written = {real for _path, real in grants.write}
    out = [
        row("ro", "extra_read_paths", path, real)
        for path, real in grants.read
        if real not in written
    ]
    out += [row("rw", "extra_write_paths", path, real) for path, real in grants.write]
    out += [
        Row(
            "--",
            pathlib.Path(d.path),
            f"{d.key}: not granted because"
            f" {'a link on its way sits' if d.via_link else 'it sits'} inside {d.region},"
            " which jailed commands write",
        )
        for d in grants.dropped
    ]
    return out


def _home_row(
    cfg: Config,
    selected: kinds.IsolationLevel,
    root: pathlib.Path,
    git_grant: pathlib.Path | None,
) -> Row:
    """Return the jail's HOME row: `full`'s tmpfs, or the repository's cache dir and why.

    Args:
        cfg: The config; under `full`, `sandbox.home` picks the tmpfs or the cache dir.
        selected: The resolved isolation level.
        root: The workspace root, whose repository names the cache dir.
        git_grant: The repository git dir a fork's worktree grants, or None.

    Returns:
        The row, read-write either way.
    """
    persistent = tools_policy.persistent_jail_home(cfg, selected, root, git_grant)
    if persistent is None:
        return Row("rw", tools_policy.JAIL_TMP_HOME, "HOME, inside the private /tmp")
    why = (
        "sandbox.home = 'cache'"
        if selected == "full"
        else f"sandbox.isolation = '{selected}' has no private /tmp"
    )
    return Row("rw", persistent, f"HOME, persists across runs: {why}")


def _fork_git_grant(
    cfg: Config, ws: _path_safety.Workspace, selected: kinds.IsolationLevel
) -> pathlib.Path | None:
    """Return the repository git dir a fork's worktree grants a jailed command, or None.

    Resolved through the execution's own policy builder, so the row matches the grant.

    Args:
        cfg: The config, passed to the policy builder.
        ws: The workspace, looked up among the worktrees the session manifests name.
        selected: The resolved isolation level.

    Returns:
        The git dir; None under `none`, outside a git repository, or when no session recorded
        one for this workspace.
    """
    if selected == "none":
        return None
    try:
        repo = git_ops.git_common_dir(ws.root).parent
    except errors.GitError:
        return None
    for worktree, sessions in fork_worktrees.worktree_owners(agent6_paths.state_dir(repo)).items():
        if worktree.resolve() != ws.root:
            continue
        git_dir = next((m.worktree_git_dir for _d, m in sessions if m.worktree_git_dir), None)
        if git_dir is not None:
            tools_policy.jail_policy(ws.root, cfg, selected, ("true",), worktree_git_dir=git_dir)
        return git_dir
    return None


def _tool_rows() -> list[Row]:
    """Return one row per operator tool dir the jail mounts read+exec."""
    return [
        Row("ro", m.path, f"operator tools: {m.why}") for m in tool_paths.operator_tool_mounts()
    ]


def _file_tool_rows(ws: _path_safety.Workspace, grants: tools_policy.Grants) -> list[Row]:
    """Return what the in-process file tools reach, at every isolation level.

    Args:
        ws: The tools' workspace: its root, the memory dir it exempts, the dirs it hides.
        grants: The config's resolved path grants.

    Returns:
        One row per path, with `--` rows for / and the home dir unless a row names them.
    """
    home = agent6_paths.effective_user().home.resolve()
    rows = [Row("rw", ws.root, "the workspace"), *grant_rows(grants)]
    rows += [Row("rw", p, "repo memory, inside the hidden state dir") for p in sorted(ws.exempt)]
    rows += [Row("--", p, "hidden: tools refuse it") for p in sorted(ws.denied)]
    return _with_defaults(
        rows,
        Row("--", pathlib.Path("/"), "refused except the paths below"),
        Row("--", home, "refused, except the rows under it below"),
    )


def _unconfined_rows(cfg: Config, root: pathlib.Path) -> list[Row]:
    """Return what a command reaches with no jail: everything, the private dirs included.

    Args:
        cfg: The config, for the HOME row.
        root: The workspace root, for the HOME row.

    Returns:
        One row per path: /, the home dir, the commands' HOME, and agent6's config and state
        dirs.
    """
    user = agent6_paths.effective_user()
    rows = [
        Row(
            "rw",
            pathlib.Path("/"),
            f"sandbox disabled: the whole filesystem and network, as {user.name}"
            f" (uid {user.uid}), no syscall filter",
        ),
        Row("rw", user.home.resolve(), "your home: ~/.ssh, ~/.aws and every credential in it"),
        _home_row(cfg, "none", root, None),
        Row("rw", agent6_paths.global_config_dir(), "not hidden: secrets.toml is readable"),
        Row("rw", agent6_paths.state_base(), "not hidden: every session's records are writable"),
    ]
    return _with_defaults(rows[:3], *rows[3:])


def _command_rows(
    cfg: Config,
    ws: _path_safety.Workspace,
    grants: tools_policy.Grants,
    selected: kinds.IsolationLevel,
    git_grant: pathlib.Path | None,
) -> list[Row]:
    """Return what a jailed command reaches under `full` or `partial`.

    Args:
        cfg: The config whose `sandbox` settings the rows show.
        ws: The workspace, whose root a command writes.
        grants: The config's resolved path grants.
        selected: The resolved isolation level.
        git_grant: The repository git dir a fork's worktree grants, or None.

    Returns:
        One row per path, with `--` rows for / and the home dir unless a row names them.
    """
    home = agent6_paths.effective_user().home.resolve()
    net = tools_policy.resolve_network(cfg, selected)
    rows = [Row("rw", ws.root, "the workspace")]
    if (ws.root / ".git").exists():
        if selected == "full" and cfg.sandbox.protect_git:
            rows.append(
                Row("ro", ws.root / ".git", "re-bound read-only: sandbox.protect_git = true")
            )
        elif selected != "full":
            rows.append(
                Row(
                    "rw",
                    ws.root / ".git",
                    "writable: sandbox.protect_git requires sandbox.isolation = 'full'",
                )
            )
    if git_grant is not None:
        rows.append(
            Row("ro", git_grant, "the repository's .git, which this linked worktree points into")
        )
    if selected == "full":
        rows.append(
            Row(
                "rw",
                _TMP,
                "a private tmpfs the run's commands share, gone at teardown",
            )
        )
        for p in FULL_SYSTEM_BINDS:
            note = "system"
            if p == "/etc/alternatives" and net != "host":
                note = "system; the rest of /etc is empty"
            rows.append(Row("ro", pathlib.Path(p), note))
        defaults = [
            Row("--", pathlib.Path("/"), "empty except /proc, /dev and the paths below"),
            Row("--", home, "not mounted, except the rows under it below"),
        ]
        hidden = "masked out of the jail's view"
    else:
        rows += [
            Row("rw", pathlib.Path(p), f"the host's {p}, shared with every process of yours")
            for p in confine.PARTIAL_SYSTEM_RW
        ]
        rows += [
            Row("ro", pathlib.Path(p), "system, by Landlock rule")
            for p in confine.PARTIAL_SYSTEM_RO
        ]
        rows.append(
            Row("--", pathlib.Path("/proc"), f"not granted under {selected}: `ps` fails there")
        )
        nodes = confine.PARTIAL_DEVICE_NODES
        defaults = [
            Row("--", pathlib.Path("/"), "Landlock: denied except the paths below"),
            Row("--", home, "Landlock: denied, except the rows under it below"),
            Row(
                "--",
                _DEV_SHM,
                "the host's, not granted: Python multiprocessing cannot use it;"
                f" {', '.join(nodes[:-1])} and {nodes[-1]} are granted",
            ),
        ]
        hidden = "denied by Landlock"
    rows.append(_home_row(cfg, selected, ws.root, git_grant))
    rows += _tool_rows()
    rows += grant_rows(grants)
    if net == "host":
        rows += [
            Row("ro", p, "name resolution and TLS verification on the host network")
            for p in tools_policy.host_network_reads()
        ]
    rows += [
        Row("dev", pathlib.Path(p), "sandbox.extra_device_paths")
        for p in cfg.sandbox.extra_device_paths
    ]
    masked = {*agent6_paths.private_dirs(), *(pathlib.Path(p) for p in cfg.sandbox.hide_paths)}
    rows += [Row("--", p, hidden) for p in sorted(masked)]
    return _with_defaults(rows, *defaults)


def _mcp_lines(cfg: Config, root: pathlib.Path, selected: kinds.IsolationLevel) -> None:
    """Print each MCP server's confinement and network."""
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        cause = "[mcp].enabled = false" if not cfg.mcp.enabled else "no servers configured"
        print(f"  mcp servers: none run ({cause})")
        return
    print(f"  mcp servers ({len(cfg.mcp.servers)} configured, approval per server):")
    for name, srv in sorted(cfg.mcp.servers.items()):
        if not srv.enabled:
            print(f"    {name}: DISABLED")
            continue
        sb = srv.sandbox
        if srv.url:
            where = f"http {srv.url}"
            confinement = "network client only; no filesystem grant"
        elif selected == "none":
            where = "spawned UNCONFINED"
            confinement = "the whole filesystem and network (sandbox.isolation resolved to 'none')"
        elif (refusal := confine.mcp_network_refusal(name, srv, selected)) is not None:
            # A run refuses a network this level cannot give.
            print(f"    {name}: a run would refuse: {refusal}")
            continue
        elif (policy := _setup.mcp_server_policy(cfg, root, selected, srv)) is None:
            where = "spawned UNCONFINED"
            confinement = (
                "the whole filesystem and network (mcp.servers.*.sandbox.unconfined = true)"
            )
        else:
            where = "spawned in the jail"
            ro = len(sb.read_paths) if sb else 0
            rw = len(sb.write_paths) if sb else 0
            confinement = f"paths ro+{ro} rw+{rw}, network {policy.network}"
        print(f"    {name}: {where}  approve={srv.approve}  {confinement}")


def _header(cfg: Config, env: detect.Environment, selected: kinds.IsolationLevel) -> None:
    """Print the resolved isolation and why, then the command, network and memory settings.

    Args:
        cfg: The config.
        env: The detected host.
        selected: The resolved isolation level.
    """
    if selected == "none":
        print(f"  isolation: none  ({_setup.no_jail_cause(cfg, env)})")
    else:
        abi = str(env.landlock_abi) if env.sandbox_available else "n/a (no Linux sandbox)"
        print(
            f"  isolation: {selected}  (sandbox.isolation = {cfg.sandbox.isolation};"
            f" userns {'yes' if env.userns_supported else 'no'}, Landlock ABI {abi},"
            f" kernel {env.kernel.raw})"
        )
        reason = detect.degrade_reason(env)
        if cfg.sandbox.isolation == "auto" and reason is not None:
            print(f"  'full' is unavailable because {reason}")
    gate = {
        "yes": "auto-approved",
        "ask": "prompted per call",
        "no": "withheld from the model",
    }[cfg.sandbox.run_commands]
    print(f"  commands:  {gate} (sandbox.run_commands = {cfg.sandbox.run_commands})")
    net = tools_policy.resolve_network(cfg, selected)
    net_line = {
        "session": "the run's own network, no route off this machine;"
        " reach it with agent6 exec or agent6 forward",
        "host": "this machine's network, unconfined",
        "none": "no network at all",
    }.get(net, net)
    print(f"  network:   {net} (sandbox.network = {cfg.sandbox.network}): {net_line}")
    mem = cfg.sandbox.memory_limit_mb
    print(
        f"  memory:    {mem} MiB per command (sandbox.memory_limit_mb)"
        if mem > 0
        else "  memory:    no cap (sandbox.memory_limit_mb = 0)"
    )


def _probe(
    cfg: Config, isolation: kinds.IsolationLevel, env: detect.Environment
) -> list[kinds.SandboxReport]:
    """Run the jail's live probes under the selected isolation.

    Args:
        cfg: The config, whose network the network probe's detail names.
        isolation: The resolved isolation level; only `full` probes the network.
        env: The detected host, unused.

    Returns:
        One report per probe.
    """
    reports: list[kinds.SandboxReport] = []
    try:
        abi = landlock_abi()
        reports.append(kinds.SandboxReport(name="landlock_abi", ok=abi > 0, detail=f"ABI {abi}"))
    except landlock.LandlockError as exc:
        reports.append(kinds.SandboxReport(name="landlock_abi", ok=False, detail=str(exc)))

    def _jail(*argv: str) -> kinds.CommandResult:
        """Return the result of argv in the selected jail with no network."""
        return run_in_jail(
            kinds.JailPolicy(
                cwd=pathlib.Path.cwd(),
                argv=argv,
                isolation=isolation,
                network="none",
                timeout_s=10.0,
            )
        )

    try:
        res = _jail("/usr/bin/true")
        reports.append(
            kinds.SandboxReport(
                name="jail_true", ok=res.ok, detail=f"/usr/bin/true exited {res.returncode}"
            )
        )
    except JailUnavailableError as exc:
        reports.append(kinds.SandboxReport(name="jail_true", ok=False, detail=str(exc)))

    # Only `full` has a network namespace to probe; `partial` shares this process's network.
    if isolation == "full":
        try:
            res = _jail("/usr/bin/getent", "hosts", "example.com")
            ok = res.returncode != 0
            # The probe's jail has no network whatever the config says.
            net = tools_policy.resolve_network(cfg, isolation)
            gives = (
                "the host's network (sandbox.network)"
                if net == "host"
                else "their own network, with no route off this machine"
            )
            reports.append(
                kinds.SandboxReport(
                    name="jail_blocks_network",
                    ok=ok,
                    detail=(
                        f"a DNS lookup failed (exit {res.returncode}) in a probe jail with no"
                        " network: the jail can block it; this config gives the run's commands"
                        f" {gives}"
                        if ok
                        else "a DNS lookup succeeded in a probe jail with no network: the"
                        " network is reachable"
                    ),
                )
            )
        except JailUnavailableError as exc:
            reports.append(
                kinds.SandboxReport(name="jail_blocks_network", ok=False, detail=str(exc))
            )
    else:
        reports.append(
            kinds.SandboxReport(
                name="jail_blocks_network",
                ok=True,
                detail="n/a because only sandbox.isolation = 'full' has a network namespace",
            )
        )

    try:
        res = _jail("/bin/sh", "-c", "echo x > /etc/agent6-escape || true")
        # /etc is read-only at both levels, so the file must not appear on the host.
        ok = not pathlib.Path("/etc/agent6-escape").exists()
        reports.append(
            kinds.SandboxReport(
                name="jail_blocks_etc_write",
                ok=ok,
                detail=(
                    "/etc/agent6-escape was not created"
                    if ok
                    else "/etc/agent6-escape was created on the host"
                ),
            )
        )
    except JailUnavailableError as exc:
        reports.append(kinds.SandboxReport(name="jail_blocks_etc_write", ok=False, detail=str(exc)))
    return reports


def _probe_checks(
    cfg: Config, env: detect.Environment, selected: kinds.IsolationLevel
) -> list[kinds.DoctorCheck]:
    """Print the probe lines and return the one summary row they fold into.

    No jail means nothing to probe: an operator's `none` is a WARN naming the setting, a
    host that cannot sandbox is a FAIL.

    Args:
        cfg: The config.
        env: The detected host.
        selected: The resolved isolation level.

    Returns:
        The `sandbox` row.
    """
    if selected == "none":
        cause = _setup.no_jail_cause(cfg, env)
        explicit = detect.sandbox_disabled_by_env() or cfg.sandbox.isolation == "none"
        status = "WARN" if explicit else "FAIL"
        detail = f"no jail to probe: {cause}; commands run unconfined"
        print(f"[{status}] jail: {detail}")
        return [kinds.DoctorCheck(name="sandbox", status=status, detail=detail)]
    reports = _probe(cfg, selected, env)
    ok = True
    for r in reports:
        print(f"[{'PASS' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
        ok = ok and r.ok
    detail = "all jail probes passed" if ok else "a jail probe failed"
    return [kinds.DoctorCheck(name="sandbox", status="PASS" if ok else "FAIL", detail=detail)]


def check_sandbox_section(
    cfg: Config, explicit_leaves: frozenset[str] = frozenset()
) -> list[kinds.DoctorCheck]:
    """Print what the file tools, jailed commands and MCP servers can reach, then probe the jail.

    A refusal a run would give is a FAIL row, and a grant of the home directory is a WARN row.

    Args:
        cfg: The config.
        explicit_leaves: The config leaves the operator wrote, which refuse rather than degrade.

    Returns:
        The section's summary rows.
    """
    try:
        env = _setup.detect_env()
        selected = detect.resolve_isolation(cfg.sandbox.isolation, env)
    except (detect.IsolationUnavailableError, JailUnavailableError) as exc:
        print(f"  [FAIL] isolation selection: {exc}")
        return [kinds.DoctorCheck(name="sandbox.isolation", status="FAIL", detail=str(exc))]
    _header(cfg, env, selected)
    out = [
        kinds.DoctorCheck(name="sandbox.isolation", status="PASS", detail=f"selected '{selected}'")
    ]
    root = pathlib.Path.cwd()
    grants = tools_policy.resolve_grants(cfg, root)
    ws = tools_policy.workspace_for(
        cfg, root, memory_dir=memory.memory_dir(agent6_paths.state_dir(root)), grants=grants
    )
    # The same ladder every run applies: an explicit knob this host cannot honour refuses.
    refusal = confine.check_network_support(cfg, selected) or confine.config_refusal(
        cfg, selected, ws.root, explicit_leaves=explicit_leaves
    )
    if refusal is not None:
        detail = f"a run would refuse: {refusal}"
        print(f"  [FAIL] {detail}")
        out.append(kinds.DoctorCheck(name="sandbox.refusal", status="FAIL", detail=detail))
    else:
        out.append(
            kinds.DoctorCheck(
                name="sandbox.refusal",
                status="PASS",
                detail=f"every explicit setting works on '{selected}'",
            )
        )
    for warning in confine.home_grant_warnings(cfg):
        print(f"  [WARN] {warning}")
        out.append(kinds.DoctorCheck(name="sandbox.grants", status="WARN", detail=warning))
    print()
    print(f"  in-process file tools ({_FILE_TOOLS}; no approval):")
    _print_rows(_file_tool_rows(ws, grants))
    print()
    print("  jailed commands (run_command, the verify gate, the metric command):")
    try:
        if selected == "none":
            _print_rows(_unconfined_rows(cfg, ws.root))
        else:
            git_grant = _fork_git_grant(cfg, ws, selected)
            _print_rows(_command_rows(cfg, ws, grants, selected, git_grant))
        print()
        _mcp_lines(cfg, ws.root, selected)
    except JailUnavailableError as exc:
        print(f"  [FAIL] a run would refuse: {exc}")
        return [*out, kinds.DoctorCheck(name="sandbox.refusal", status="FAIL", detail=str(exc))]
    print()
    print(
        "  agent process: its own egress is unbounded (documented in"
        " docs/security.md); the jail bounds commands, not the agent"
    )
    print(
        f"  secrets: {agent6_paths.secrets_path()}  (0600, never passed into a child's environment)"
    )
    print()
    return [*out, *_probe_checks(cfg, env, selected)]
