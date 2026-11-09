# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""List what the file tools, jailed commands and MCP servers can reach, then run the jail probes.

Each table has one row per path, sorted so a parent precedes its children, with a root row and a
home-directory row.
"""

from __future__ import annotations

import pathlib

from agent6 import errors, kinds
from agent6 import paths as agent6_paths
from agent6.app import confine, fork_worktrees, session_runtime, session_setup
from agent6.config import Config
from agent6.context import memory
from agent6.sandbox import (
    JailUnavailableError,
    detect,
    landlock,
    landlock_abi,
    run_in_jail,
)
from agent6.tools import path_safety, target_policy
from agent6.tools import policy as tools_policy
from agent6.ui.cli import check_render, common, sandbox_paths
from agent6.vcs import git_read

_FILE_TOOLS = (
    "read_file, list_dir, outline, find_definition, find_references, apply_edit, apply_patch"
)


def _fork_git_grant(
    cfg: Config, ws: path_safety.Workspace, selected: kinds.IsolationLevel
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
        repo = git_read.git_common_dir(ws.root).parent
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


def _mcp_lines(cfg: Config, root: pathlib.Path, selected: kinds.IsolationLevel) -> None:
    """Print each MCP server's confinement and network."""
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        cause = "[mcp].enabled = false" if not cfg.mcp.enabled else "no servers configured"
        check_render.line(f"  {common.sgr('mcp servers:', '1')} ", f"none run ({cause})")
        return
    check_render.line(
        f"  {common.sgr('mcp servers', '1')} ",
        f"({len(cfg.mcp.servers)} configured, approval per server):",
    )
    for name, srv in sorted(cfg.mcp.servers.items()):
        if not srv.enabled:
            check_render.line(f"    {name}: ", "DISABLED")
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
            check_render.line(f"    {name}: ", f"a run would refuse: {refusal}", warning=True)
            continue
        elif (policy := session_setup.mcp_server_policy(cfg, root, selected, srv)) is None:
            where = "spawned UNCONFINED"
            confinement = (
                "the whole filesystem and network (mcp.servers.*.sandbox.unconfined = true)"
            )
        else:
            where = "spawned in the jail"
            ro = len(sb.extra_read_paths) if sb else 0
            rw = len(sb.extra_write_paths) if sb else 0
            confinement = f"paths ro+{ro} rw+{rw}, network {policy.network}"
        check_render.line(
            f"    {name}: {common.sgr(where, '1')}  ",
            f"approve={srv.approve}  {confinement}",
            warning=where == "spawned UNCONFINED",
        )


def _header(cfg: Config, env: detect.Environment, selected: kinds.IsolationLevel) -> None:
    """Print the resolved isolation and why, then the command, network and memory settings.

    Args:
        cfg: The config.
        env: The detected host.
        selected: The resolved isolation level.
    """
    abi = str(env.landlock_abi) if env.sandbox_available else "n/a (no Linux sandbox)"
    net = target_policy.resolve_network(cfg, selected)
    sockets = ""
    if selected == "partial" or (selected == "full" and net == "host"):
        # The jail shares the host's network namespace, so only the Landlock scope keeps a
        # command off a host daemon's abstract unix socket.
        sockets = (
            " (abstract unix sockets scoped)"
            if env.landlock_abi >= 6
            else " (abstract unix sockets reachable: the scope needs ABI 6)"
        )
    check_render.line(
        f"  isolation: {common.sgr(selected, '1')}  ",
        f"(sandbox.isolation = {cfg.sandbox.isolation};"
        f" userns {'yes' if env.userns_supported else 'no'}, Landlock ABI {abi}{sockets},"
        f" kernel {env.kernel.raw})",
    )
    if selected == "none":
        check_render.line("  ", session_setup.no_jail_cause(cfg, env), warning=True)
    reason = detect.degrade_reason(env)
    if cfg.sandbox.isolation == "auto" and reason is not None:
        check_render.line("  ", f"'full' is unavailable because {reason}", warning=True)
    gate = {
        "allow": "auto-approved",
        "ask": "prompted per call",
        "deny": "withheld from the model",
    }[cfg.sandbox.run_commands]
    check_render.line(
        f"  commands:  {common.sgr(gate, '1')} ",
        f"(sandbox.run_commands = {cfg.sandbox.run_commands})",
    )
    net_line = {
        "session": "the run's own network, no route off this machine;"
        " reach it with agent6 exec or agent6 forward",
        "host": "this machine's network, unconfined",
        "none": "no network at all",
    }.get(net, net)
    check_render.line(
        f"  network:   {common.sgr(net, '1')} ",
        f"(sandbox.network = {cfg.sandbox.network}): {net_line}",
    )
    mem = cfg.sandbox.memory_limit_mib
    effective = f"{mem} MiB per-process data-segment cap" if mem > 0 else "no cap"
    check_render.line(
        f"  memory:    {common.sgr(effective, '1')} ", f"(sandbox.memory_limit_mib = {mem})"
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
        reports.append(
            kinds.SandboxReport(
                name="landlock_abi", status="PASS" if abi > 0 else "FAIL", detail=f"ABI {abi}"
            )
        )
    except landlock.LandlockError as exc:
        reports.append(kinds.SandboxReport(name="landlock_abi", status="FAIL", detail=str(exc)))

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
                name="jail_true",
                status="PASS" if res.ok else "FAIL",
                detail=f"/usr/bin/true exited {res.returncode}",
            )
        )
        # A quiet command's stderr is the launcher's own: what it could not set up (a refused
        # /proc mount, a hostname it could not set), the same words a run prints as degraded.
        notes = " ".join(res.stderr.split())
        reports.append(
            kinds.SandboxReport(
                name="jail_setup",
                status="WARN" if notes else "PASS",
                detail=f"the launcher reported: {notes}"
                if notes
                else "the launcher set up quietly",
            )
        )
    except JailUnavailableError as exc:
        reports.append(kinds.SandboxReport(name="jail_true", status="FAIL", detail=str(exc)))

    try:
        res = _jail("/bin/sh", "-c", "cat /proc/self/status >/dev/null")
        seen = " ".join(res.stderr.split())
        reports.append(
            kinds.SandboxReport(
                name="jail_proc",
                status="PASS" if res.ok else "FAIL",
                detail=(
                    "the jail's /proc answers (cat /proc/self/status)"
                    if res.ok
                    else f"the jail's /proc did not answer: cat /proc/self/status exited"
                    f" {res.returncode}" + (f" ({seen})" if seen else "") + "; ps and every tool"
                    " reading /proc fail the same way in the run's commands"
                ),
            )
        )
    except JailUnavailableError as exc:
        reports.append(kinds.SandboxReport(name="jail_proc", status="FAIL", detail=str(exc)))

    # Only `full` has a network namespace to probe; `partial` shares this process's network.
    if isolation == "full":
        try:
            res = _jail("/usr/bin/getent", "hosts", "example.com")
            ok = res.returncode != 0
            # The probe's jail has no network whatever the config says.
            net = target_policy.resolve_network(cfg, isolation)
            gives = (
                "the host's network (sandbox.network)"
                if net == "host"
                else "their own network, with no route off this machine"
            )
            reports.append(
                kinds.SandboxReport(
                    name="jail_blocks_network",
                    status="PASS" if ok else "FAIL",
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
                kinds.SandboxReport(name="jail_blocks_network", status="FAIL", detail=str(exc))
            )
    else:
        reports.append(
            kinds.SandboxReport(
                name="jail_blocks_network",
                status="SKIP",
                detail=(
                    f"{isolation} has no network namespace; only full can probe network blocking"
                ),
            )
        )

    try:
        res = _jail("/bin/sh", "-c", "echo x > /etc/agent6-escape || true")
        # /etc is read-only at both levels, so the file must not appear on the host.
        ok = not pathlib.Path("/etc/agent6-escape").exists()
        reports.append(
            kinds.SandboxReport(
                name="jail_blocks_etc_write",
                status="PASS" if ok else "FAIL",
                detail=(
                    "/etc/agent6-escape was not created"
                    if ok
                    else "/etc/agent6-escape was created on the host"
                ),
            )
        )
    except JailUnavailableError as exc:
        reports.append(
            kinds.SandboxReport(name="jail_blocks_etc_write", status="FAIL", detail=str(exc))
        )
    return reports


_PROBE_NAMES = (
    "landlock_abi",
    "jail_true",
    "jail_setup",
    "jail_proc",
    "jail_blocks_network",
    "jail_blocks_etc_write",
)


def _probe_checks(
    cfg: Config,
    env: detect.Environment,
    selected: kinds.IsolationLevel,
    reports: list[kinds.SandboxReport],
) -> list[kinds.DoctorCheck]:
    """Print the probe lines and return the one summary row they fold into.

    No jail means nothing to probe: an operator's `none` is a WARN naming the setting, a
    host that cannot sandbox is a FAIL.

    Args:
        cfg: The config.
        env: The detected host.
        selected: The resolved isolation level.
        reports: The probes' results; empty under `none`.

    Returns:
        The `sandbox` row.
    """
    if selected == "none":
        cause = session_setup.no_jail_cause(cfg, env)
        explicit = detect.sandbox_disabled_by_env() or cfg.sandbox.isolation == "none"
        status = "WARN" if explicit else "FAIL"
        detail = f"no jail to probe: {cause}; commands run unconfined"
        for name in _PROBE_NAMES:
            reason = "none has no network namespace" if name == "jail_blocks_network" else cause
            check_render.result("SKIP", name, reason)
        check_render.result(status, "jail", detail)
        return [kinds.DoctorCheck(name="sandbox", status=status, detail=detail)]
    for r in reports:
        check_render.result(r.status, r.name, r.detail)
    statuses = {r.status for r in reports}
    status = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    detail = {
        "PASS": "all jail probes passed",
        "WARN": "the jail runs degraded",
        "FAIL": "a jail probe failed",
    }[status]
    return [kinds.DoctorCheck(name="sandbox", status=status, detail=detail)]


def _container_section(cfg: Config) -> list[kinds.DoctorCheck]:
    """Print the container a run would select, inspected and never started or stopped.

    Args:
        cfg: The config, with `sandbox.isolation = "container"`.

    Returns:
        The section's summary rows.
    """
    check_render.line(
        f"  isolation: {common.sgr('container', '1')}  ",
        "(sandbox.isolation = container; commands run inside the selected container, which the"
        " runtime bounds; agent6 adds no confinement there)",
    )
    out: list[kinds.DoctorCheck] = []
    root = pathlib.Path.cwd()
    try:
        report = session_runtime.describe_container(cfg, root)
    except JailUnavailableError as exc:
        check_render.result("FAIL", "container.discovery", str(exc))
        out.append(kinds.DoctorCheck(name="container.discovery", status="FAIL", detail=str(exc)))
    else:
        check_render.result("PASS", "container.discovery", report.detail)
        out.append(
            kinds.DoctorCheck(name="container.discovery", status="PASS", detail=report.detail)
        )
        for line in report.exposures:
            check_render.line("  ", f"the container exposes a {line}", warning=True)
            out.append(kinds.DoctorCheck(name="container.exposure", status="WARN", detail=line))
    refusal = confine.startup_refusal(cfg, "none", root)
    if refusal is not None:
        check_render.result("FAIL", "a run would refuse", refusal)
        out.append(kinds.DoctorCheck(name="sandbox.refusal", status="FAIL", detail=refusal))
    else:
        detail = "every explicit setting works inside a container"
        out.append(kinds.DoctorCheck(name="sandbox.refusal", status="PASS", detail=detail))
    print()
    check_render.line(f"  {common.sgr('effective controls', '1')} ", "(as a run resolves them):")
    for control in target_policy.container_controls(cfg):
        reason = f"; {control.reason}" if control.reason else ""
        check_render.line(
            "  ", f"{control.leaf}: {control.behaviour} ({control.owner}, {control.probe}){reason}"
        )
    print()
    grants = tools_policy.resolve_grants(cfg, root)
    ws = tools_policy.workspace_for(
        cfg, root, memory_dir=memory.memory_dir(agent6_paths.state_dir(root)), grants=grants
    )
    check_render.line(
        f"  {common.sgr('in-process file tools', '1')} ", f"({_FILE_TOOLS}; no approval):"
    )
    check_render.line("  ", "Filesystem enforcement: in-process path refusal, on the host.")
    check_render.rows(sandbox_paths.file_tool_rows(ws, grants))
    print()
    check_render.line(
        f"  {common.sgr('secrets:', '1')} {agent6_paths.secrets_path()}  ",
        "(0600, never passed into a child's environment, never copied into the container)",
    )
    print()
    return out


def check_sandbox_section(cfg: Config) -> list[kinds.DoctorCheck]:
    """Print what the file tools, jailed commands and MCP servers can reach, then probe the jail.

    A refusal a run would give is a FAIL row, and a grant of the home directory is a WARN row.

    Args:
        cfg: The config.

    Returns:
        The section's summary rows.
    """
    try:
        env = detect.probed_environment()
        selected = detect.resolve_isolation(cfg.sandbox.isolation, env)
    except (detect.IsolationUnavailableError, JailUnavailableError) as exc:
        check_render.result("FAIL", "isolation selection", str(exc))
        return [kinds.DoctorCheck(name="sandbox.isolation", status="FAIL", detail=str(exc))]
    if cfg.sandbox.isolation == "container":
        return _container_section(cfg)
    _header(cfg, env, selected)
    out = [
        kinds.DoctorCheck(name="sandbox.isolation", status="PASS", detail=f"selected '{selected}'")
    ]
    # Probed before the rows, so the /proc row says what the jail here gets, not the policy.
    reports = _probe(cfg, selected, env) if selected != "none" else []
    proc_ok = all(r.status != "FAIL" for r in reports if r.name == "jail_proc")
    root = pathlib.Path.cwd()
    grants = tools_policy.resolve_grants(cfg, root)
    ws = tools_policy.workspace_for(
        cfg, root, memory_dir=memory.memory_dir(agent6_paths.state_dir(root)), grants=grants
    )
    # The same ladder every run applies: an explicit knob this host cannot honour refuses.
    refusal = confine.startup_refusal(cfg, selected, ws.root)
    if refusal is not None:
        detail = f"a run would refuse: {refusal}"
        check_render.result("FAIL", "a run would refuse", refusal)
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
        check_render.line("  ", warning, warning=True)
        out.append(kinds.DoctorCheck(name="sandbox.grants", status="WARN", detail=warning))
    print()
    check_render.line("  ", "sys: system paths, including granted device nodes")
    check_render.line("  ", "net: files supporting host name resolution and TLS verification")
    check_render.line("  ", "tool: operator tools and their runtime dependencies")
    print()
    check_render.line(
        f"  {common.sgr('in-process file tools', '1')} ", f"({_FILE_TOOLS}; no approval):"
    )
    check_render.line(
        "  ", "Filesystem enforcement: in-process path refusal, at every isolation level."
    )
    check_render.rows(sandbox_paths.file_tool_rows(ws, grants))
    print()
    check_render.line(
        f"  {common.sgr('jailed commands', '1')} ",
        "(run_command, the verify gate, the metric command):",
    )
    enforcement = {
        "full": "mount namespace and read-only binds"
        + (", plus Landlock" if env.landlock_abi > 0 else "; Landlock unavailable"),
        "partial": "Landlock; no mount namespace",
        "none": "none; commands run unconfined",
    }[selected]
    check_render.line("  ", f"Filesystem enforcement: {enforcement}.")
    try:
        if selected == "none":
            check_render.rows(sandbox_paths.unconfined_rows(cfg, ws.root))
        else:
            git_grant = _fork_git_grant(cfg, ws, selected)
            check_render.rows(
                sandbox_paths.command_rows(
                    cfg, ws, grants, selected, git_grant, env.landlock_abi, proc_ok=proc_ok
                )
            )
        print()
        _mcp_lines(cfg, ws.root, selected)
    except JailUnavailableError as exc:
        check_render.result("FAIL", "a run would refuse", str(exc))
        return [*out, kinds.DoctorCheck(name="sandbox.refusal", status="FAIL", detail=str(exc))]
    print()
    command_boundary = (
        "commands also run unconfined"
        if selected == "none"
        else "command files are confined; command egress is unbounded"
        if target_policy.resolve_network(cfg, selected) == "host"
        else "command files and egress are confined"
    )
    check_render.line(
        f"  {common.sgr('agent process:', '1')} ",
        f"its own egress is unbounded (documented in docs/security.md); {command_boundary}",
        warning=True,
    )
    check_render.line(
        f"  {common.sgr('secrets:', '1')} {agent6_paths.secrets_path()}  ",
        "(0600, never passed into a child's environment)",
    )
    print()
    return [*out, *_probe_checks(cfg, env, selected, reports)]
