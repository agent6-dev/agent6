# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check`, sandbox + config + MCP + verify pre-flight."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.app._setup import (
    check_provider_keys,
    detect_env,
    mcp_server_policy,
    start_mcp_manager_if_enabled,
    wants_session_network,
)
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import (
    load_effective,
)
from agent6.paths import private_dirs, secrets_path
from agent6.sandbox import (
    JailUnavailableError,
    landlock_abi,
    run_in_jail,
)
from agent6.sandbox.detect import (
    IsolationUnavailableError,
    degrade_reason,
    resolve_isolation,
)
from agent6.sandbox.jail import SessionNetwork, tool_mount_notes
from agent6.tools.policy import Workspace, resolve_network, workspace_for
from agent6.types import CommandResult, IsolationLevel, JailPolicy, SandboxReport

# Mirrors SYSTEM_BINDS in src/agent6/jail/src/main.rs (the strict rootfs's
# read-only host binds); tests/security pins the two against each other.
_STRICT_SYSTEM_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives")
# What hardened's Landlock grants read-only (main.rs ro_paths base set).
_HARDENED_SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev")


def _isolation_means(isolation: IsolationLevel) -> str:
    """One line on what this level bounds, for someone diagnosing a tool.

    States the boundaries, not their consequences for any particular program: a
    reader who knows a command runs with its own filesystem, its own network and
    a filtered syscall set can work out why it behaves differently here, and
    knows which words to search the docs for.
    """
    if isolation == "strict":
        # The network clause is hedged because this section runs before any
        # config is loaded (`agent6 check sandbox` needs none): strict CAN give
        # the run its own network, and `sandbox.network = "host"` declines it.
        # The config section prints what this project resolved to.
        return (
            "commands get their own filesystem view (only granted paths exist),"
            " a private /proc and PID namespace, a filtered syscall set, and --"
            " unless sandbox.network says otherwise -- the run's own network."
            " See docs/security.md."
        )
    if isolation == "hardened":
        return (
            "commands share this host's filesystem, network and /proc, bounded"
            " by Landlock path rules and a filtered syscall set. No namespaces,"
            " so nothing here is private to the run. See docs/security.md."
        )
    return "NOTHING is confined: commands run as you, on this host. See docs/security.md."


def _cmd_check_sandbox() -> int:
    """Run the sandbox boundary self-tests on the host's kernel.

    The probes run under the *effective* isolation this host resolves to
    (`resolve_isolation("auto", ...)`), not a hardcoded one. On a host that
    blocks unprivileged user namespaces (default-seccomp Docker, or Ubuntu
    with `kernel.apparmor_restrict_unprivileged_userns=1`) the effective
    isolation is `hardened`, which is exactly what `agent6 run` would use there;
    testing `strict` instead would report a spurious FAIL for a sandbox the
    agent never uses on this host.
    """
    reports: list[SandboxReport] = []

    # Landlock probe
    abi = landlock_abi()
    reports.append(
        SandboxReport(
            name="landlock_abi",
            ok=abi > 0,
            detail=f"abi={abi}",
        )
    )

    env = detect_env()
    isolation = resolve_isolation("auto", env)
    print(f"  effective isolation (auto): {isolation}")
    reason = degrade_reason(env)
    if reason is not None:
        # A degraded level never appears without its why (same line the run
        # warning and check config print; one owner in detect.degrade_reason).
        print(f"  not strict: {reason}")
    # What that level GIVES, in general terms rather than a catalogue of cases:
    # someone whose tool misbehaves needs to know which boundaries exist here
    # before they can guess why, and these words are what to search the docs for.
    print(f"  {_isolation_means(isolation)}")
    notes = tool_mount_notes()
    # Under "none" nothing is confined, so grant language about tool dirs
    # would describe a boundary that does not exist; the block is jail-only.
    if notes.exposes_home_dir and isolation != "none":
        # Where someone is actually asking. Not a per-run warning: on a normal
        # machine every uv-installed tool in ~/.local/bin points into
        # ~/.local/share, so it is the ordinary state of a dev box.
        how = (
            "mounted read-only into the jail and readable by"
            if isolation == "strict"
            else "granted read-only (Landlock path rules) and readable by"
        )
        print(
            f"  {len(notes.exposes_home_dir)} tool(s) resolve out of their bin dir, so those"
            f" target directories are\n  {how}"
            " jailed commands:"
        )
        for tool in notes.exposes_home_dir:
            print(f"    {tool}")
    if isolation == "none":
        # No kernel sandbox to test (a non-Linux host, or a deliberate `none`
        # opt-out), and running the boundary probes unconfined would let the
        # /etc-write probe actually escape onto the host. Report and stop.
        reports.append(
            SandboxReport(
                name="jail",
                ok=False,
                detail="no kernel sandbox on this platform (effective isolation 'none'); skipped",
            )
        )
        return _print_sandbox_reports(reports)

    cwd = Path.cwd()

    def _jail(*argv: str) -> CommandResult:
        return run_in_jail(
            JailPolicy(cwd=cwd, argv=argv, isolation=isolation, network="none", timeout_s=10.0)
        )

    # Try running `/usr/bin/true` in the jail.
    try:
        res = _jail("/usr/bin/true")
        reports.append(SandboxReport(name="jail_true", ok=res.ok, detail=f"rc={res.returncode}"))
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_true", ok=False, detail=str(exc)))

    # Confirm the child cannot reach the network. Only meaningful under
    # `strict`, the one level with network namespaces: there a child that did
    # not ask for `host` lands in one with no route out. `hardened` has none to
    # give, so a jailed command shares this process's network and there is
    # nothing to probe -- report n/a rather than a misleading pass/fail.
    if isolation == "strict":
        try:
            res = _jail("/usr/bin/getent", "hosts", "example.com")
            ok = res.returncode != 0
            reports.append(
                SandboxReport(
                    name="jail_blocks_network",
                    ok=ok,
                    detail=f"rc={res.returncode} (nonzero = blocked, as expected)",
                )
            )
        except JailUnavailableError as exc:
            reports.append(SandboxReport(name="jail_blocks_network", ok=False, detail=str(exc)))
    else:
        reports.append(
            SandboxReport(
                name="jail_blocks_network",
                ok=True,
                detail=(
                    "n/a under hardened: no per-command network namespace; jailed"
                    " commands share the host network (sandbox.network degrades with"
                    " a warning)"
                ),
            )
        )

    # Confirm child cannot write outside the workspace.
    try:
        res = _jail("/bin/sh", "-c", "echo x > /etc/agent6-escape || true")
        # /etc is read-only (bind-mounted RO under strict, Landlock-denied under
        # hardened), so the file must not appear on the host.
        ok = not Path("/etc/agent6-escape").exists()
        reports.append(
            SandboxReport(
                name="jail_blocks_etc_write",
                ok=ok,
                detail=f"rc={res.returncode}; host /etc/agent6-escape exists: {not ok}",
            )
        )
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_blocks_etc_write", ok=False, detail=str(exc)))

    return _print_sandbox_reports(reports)


def _print_sandbox_reports(reports: list[SandboxReport]) -> int:
    overall_ok = True
    for r in reports:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        overall_ok = overall_ok and r.ok
    return 0 if overall_ok else 1


@dataclass(frozen=True, slots=True)
class _DoctorCheck:
    """One summary row. `status` carries through to the summary line unchanged:
    INFO (advisory, e.g. "run `agent6 connect`") must never render as PASS."""

    name: str
    status: Literal["PASS", "FAIL", "INFO"]
    detail: str


def _cmd_check(config_path: Path | None, *, section: str) -> int:
    """Consolidated pre-flight (sandbox + config + MCP + verify).

    All checks are read-only. The command never spawns the agent loop,
    never makes a network call to the configured providers, and never
    writes to the repo. MCP servers are started just long enough to
    enumerate their tool descriptors and then closed.

    Returns 0 when every selected check passes, 1 otherwise.
    """
    print(f"agent6 check: section={section}")
    print()

    checks: list[_DoctorCheck] = []
    if section in {"all", "sandbox"}:
        print("== sandbox ==")
        rc = _cmd_check_sandbox()
        checks.append(
            _DoctorCheck(
                name="sandbox",
                status="PASS" if rc == 0 else "FAIL",
                detail="all jail probes passed" if rc == 0 else f"check sandbox exit {rc}",
            )
        )
        print()

    try:
        cfg = (
            load_effective(Path.cwd(), config_path).config
            if section in {"all", "mcp", "verify", "config", "boundaries"}
            else None
        )
    except (ConfigError, OSError) as exc:
        cfg = None
        if section in {"all", "mcp", "verify", "config", "boundaries"}:
            print(f"== config ==\n[FAIL] cannot load config: {exc}\n")
            checks.append(_DoctorCheck(name="config_load", status="FAIL", detail=str(exc)))

    if cfg is not None and section in {"all", "config"}:
        print("== config ==")
        checks.extend(_check_config_section(cfg))
        print()

    if cfg is not None and section in {"all", "boundaries"}:
        print("== boundaries ==")
        checks.extend(_check_boundaries_section(cfg))
        print()

    if cfg is not None and section in {"all", "mcp"}:
        print("== mcp ==")
        checks.extend(_doctor_check_mcp(cfg))
        print()

    if cfg is not None and section in {"all", "verify"}:
        print("== verify ==")
        checks.extend(_doctor_check_verify(cfg))
        print()

    print("== summary ==")
    failed = False
    for c in checks:
        print(f"[{c.status}] {c.name}: {c.detail}")
        failed = failed or c.status == "FAIL"
    return 1 if failed else 0


def _check_config_section(cfg: Config) -> list[_DoctorCheck]:
    """Environment detection + isolation selection + static config checks."""
    env = detect_env()
    print(f"  kernel: {env.kernel.raw}")
    print(f"  userns supported: {env.userns_supported}")
    print(f"  sandbox available: {env.sandbox_available}")
    abi_str = str(env.landlock_abi) if env.sandbox_available else "n/a (no Linux sandbox)"
    print(f"  Landlock ABI: {abi_str}")
    print(
        f"  sandbox.isolation = {cfg.sandbox.isolation}"
        f"  network = {cfg.sandbox.network}"
        f"  run_commands = {cfg.sandbox.run_commands}"
    )
    out: list[_DoctorCheck] = []
    try:
        selected = resolve_isolation(cfg.sandbox.isolation, env)
        # The resolved values, not the configured ones: `auto` is the default on
        # both knobs, and what it resolved to on THIS host is the answer someone
        # runs `check` for.
        print(
            f"  -> selected isolation: {selected}"
            f"  commands' network: {resolve_network(cfg, selected)}"
        )
        reason = degrade_reason(env)
        if cfg.sandbox.isolation == "auto" and reason is not None:
            print(f"  -> not strict: {reason}")
        # The tools' file boundary is NOT the selected isolation's: it follows
        # the config values at every level, so print it beside them rather than
        # leaving the operator to infer it from the level.
        ws = workspace_for(cfg, Path.cwd())
        grants = len({*ws.read_roots, *ws.write_roots})
        print(f"  -> tools' files: {ws.root}  (+{grants} granted, -{len(ws.denied)} hidden)")
        out.append(
            _DoctorCheck(name="config.isolation", status="PASS", detail=f"selected {selected}")
        )
    except IsolationUnavailableError as exc:
        print(f"  [FAIL] isolation selection: {exc}")
        out.append(_DoctorCheck(name="config.isolation", status="FAIL", detail=str(exc)))
    out.extend(_doctor_check_config(cfg))
    return out


def _grant_lines(ws: Workspace) -> list[str]:
    """The operator's extra path grants, shared verbatim by both actors."""
    read_only = set(ws.read_roots) - set(ws.write_roots)
    out = [f"    ro  {p}  (sandbox.extra_read_paths)" for p in sorted(read_only)]
    out += [f"    rw  {p}  (sandbox.extra_write_paths)" for p in sorted(ws.write_roots)]
    return out


def _boundaries_commands(cfg: Config, ws: Workspace, selected: IsolationLevel) -> None:
    print(
        "  jailed commands (run_command / verify_command / metric_command;"
        f" approval: sandbox.run_commands = {cfg.sandbox.run_commands}):"
    )
    if selected == "none":
        print("    UNCONFINED: no jail on this host -- commands run as you, everywhere.")
        return
    git_note = (
        "; .git re-bound read-only" if selected == "strict" and cfg.sandbox.protect_git else ""
    )
    print(f"    rw  {ws.root}  (the workspace{git_note})")
    if selected == "strict":
        print("    rw  /tmp  (private to the command, empty each run)")
        print(f"    ro  system: {' '.join(_STRICT_SYSTEM_BINDS)} (+ a minimal /etc)")
    else:
        print(f"    ro  system (Landlock): {' '.join(_HARDENED_SYSTEM_RO)}")
    notes = tool_mount_notes()
    if notes.exposes_home_dir:
        print(
            f"    ro  operator tools: {len(notes.exposes_home_dir)} resolved bin-dir"
            " targets (`agent6 check sandbox` lists each)"
        )
    for line in _grant_lines(ws):
        print(line)
    masked = {*private_dirs(), *(Path(p) for p in cfg.sandbox.hide_paths)}
    verb = "masked out of the jail's view" if selected == "strict" else "denied by Landlock"
    for p in sorted(masked):
        print(f"    --  {p}  ({verb})")
    net = resolve_network(cfg, selected)
    net_line = {
        "session": "the run's own network -- no route off this machine"
        " (reach it: agent6 exec / agent6 forward)",
        "host": "this machine's network, unconfined",
        "none": "no network at all",
    }.get(net, net)
    print(f"    network  {net}: {net_line}")
    mem = cfg.sandbox.memory_limit_mb
    print(
        f"    memory   {mem} MiB per command (sandbox.memory_limit_mb)"
        if mem > 0
        else "    memory   no cap (sandbox.memory_limit_mb = 0)"
    )


def _boundaries_mcp(cfg: Config) -> None:
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
        elif sb is not None and sb.unconfined:
            where = "spawned UNCONFINED"
            confinement = "full host access (mcp.servers.*.sandbox.unconfined = true)"
        else:
            where = "spawned in the jail"
            ro = len(sb.read_paths) if sb else 0
            rw = len(sb.write_paths) if sb else 0
            net = sb.network if sb else "auto"
            confinement = f"paths ro+{ro} rw+{rw}, network {net}"
        print(f"    {name}: {where}  approve={srv.approve}  {confinement}")


def _check_boundaries_section(cfg: Config) -> list[_DoctorCheck]:
    """Every boundary in one place, grouped by ACTOR: who is confined, what
    files it reaches, which network it gets. Resolved values only (what THIS
    host and config give), one line per fact; informational, no probes."""
    env = detect_env()
    try:
        selected = resolve_isolation(cfg.sandbox.isolation, env)
    except IsolationUnavailableError as exc:
        print(f"[FAIL] isolation selection: {exc}")
        return [_DoctorCheck(name="boundaries", status="FAIL", detail=str(exc))]
    print(f"  isolation: {selected}  (sandbox.isolation = {cfg.sandbox.isolation})")
    reason = degrade_reason(env)
    if cfg.sandbox.isolation == "auto" and reason is not None:
        print(f"  not strict: {reason}")

    ws = workspace_for(cfg, Path.cwd())
    print()
    print("  in-process tools (read_file / apply_edit / list_dir / grep -- no approval):")
    print(f"    rw  {ws.root}  (the workspace)")
    for line in _grant_lines(ws):
        print(line)
    for p in sorted(ws.denied):
        print(f"    --  {p}  (hidden: tools refuse it)")

    print()
    _boundaries_commands(cfg, ws, selected)
    print()
    _boundaries_mcp(cfg)
    print()
    print(
        "  agent process: its own egress is NOT bounded (documented in"
        " docs/security.md); the jail bounds commands, not the agent"
    )
    print(
        f"  secrets: {secrets_path()}  (0600; never mounted into any jail,"
        " never passed into a child's env)"
    )
    return [_DoctorCheck(name="boundaries", status="PASS", detail="reported (informational)")]


def _doctor_check_mcp(cfg: Config) -> list[_DoctorCheck]:
    """Start configured MCP servers, enumerate tools, then close them.

    Returns one check per configured server plus a summary check. When
    ``[mcp]`` is disabled or empty, returns a single skip-style PASS so
    the doctor doesn't fail an unconfigured-by-design feature.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        print("(MCP disabled or no servers configured; skipping)")
        return [
            _DoctorCheck(
                name="mcp",
                status="PASS",
                detail="not configured (cfg.mcp.enabled=False or empty servers)",
            )
        ]
    isolation = resolve_isolation("auto", detect_env())
    # A server set to `session` joins the run's network, so `check` has to make
    # one the same way a run does -- otherwise checking such a server reports a
    # failure that only `check` would ever see.
    session_net = SessionNetwork.open() if wants_session_network(cfg, isolation) else None
    manager = start_mcp_manager_if_enabled(cfg, Path.cwd(), isolation, session_net=session_net)
    if manager is None:
        if session_net is not None:
            session_net.close()
        return [_DoctorCheck(name="mcp", status="PASS", detail="no enabled servers")]
    out: list[_DoctorCheck] = []
    try:
        descriptors = manager.descriptors()
        by_server: dict[str, list[str]] = {}
        for d in descriptors:
            by_server.setdefault(d.server_name, []).append(d.tool_name)
        configured = {name for name, srv in cfg.mcp.servers.items() if srv.enabled}
        # What `auto` RESOLVED to on this host, which is the whole reason to run
        # a check: `config show` can only report the word the operator wrote.
        networks = {
            name: "unconfined"
            if (pol := mcp_server_policy(cfg, Path.cwd(), isolation, srv)) is None
            else pol.network
            for name, srv in cfg.mcp.servers.items()
            if srv.enabled
        }
        why_missing = {f.name: f.error for f in manager.failures}
        for name in sorted(configured):
            tools = by_server.get(name, [])
            ok = bool(tools)
            # A server that never started is not one that "exposed no tools":
            # the reason the operator needs is the spawn error, not a symptom.
            # `approve` belongs in a pre-flight for the same reason the network
            # does: it is standing consent for every call this server's tools
            # make, and the operator set it once, possibly a while ago.
            detail = (
                f"{len(tools)} tool(s), network: {networks.get(name, '?')},"
                f" approve: {cfg.mcp.servers[name].approve}"
                if ok
                else why_missing.get(name, "started but exposed no tools")
            )
            print(f"[{'PASS' if ok else 'FAIL'}] mcp.{name}: {detail}")
            out.append(
                _DoctorCheck(name=f"mcp.{name}", status="PASS" if ok else "FAIL", detail=detail)
            )
    finally:
        manager.close()
        if session_net is not None:
            session_net.close()
    return out


def _doctor_check_verify(cfg: Config) -> list[_DoctorCheck]:
    """Verify command sanity: argv non-empty and the head executable resolves.

    Does NOT execute the verify command, that would run an arbitrary
    test suite on every doctor call. Operators can do
    ``./$(verify_command)`` themselves when they want a live run.
    """
    argv = list(cfg.workflow.verify_command)
    if not argv:
        # Optional now: `agent6 run`/`plan` infer one (AGENTS.md -> repo signals
        # -> LLM), falling back to a gateless run. Advisory, not a failure.
        print("[INFO] verify.argv: unset; will be inferred per run (or run gateless)")
        return [_DoctorCheck(name="verify.argv", status="INFO", detail="unset (inferred per run)")]
    head = argv[0]
    resolved = shutil.which(head)
    ok = resolved is not None
    detail = f"resolves to {resolved}" if resolved else f"not found on PATH: {head!r}"
    print(f"[{'PASS' if ok else 'FAIL'}] verify.head: {detail}")
    print(f"       argv = {argv}")
    print(f"       timeout = {cfg.workflow.verify_timeout_s}s")
    return [_DoctorCheck(name="verify.head", status="PASS" if ok else "FAIL", detail=detail)]


def _doctor_check_config(cfg: Config) -> list[_DoctorCheck]:
    """Static config sanity checks: provider keys + worktree git policy."""
    out: list[_DoctorCheck] = []
    if not cfg.providers:
        # Zero providers configured: "all referenced keys resolve" is vacuously
        # true and would signal "ready", but `agent6 run` will reject. Say so.
        detail_env = (
            "no providers configured yet; run `agent6 connect` (required before `agent6 run`)"
        )
        print(f"[INFO] config.provider_keys: {detail_env}")
        out.append(_DoctorCheck(name="config.provider_keys", status="INFO", detail=detail_env))
    else:
        env_err = check_provider_keys(cfg)
        ok_env = env_err is None
        detail_env = "all referenced provider keys resolve" if ok_env else env_err or ""
        print(f"[{'PASS' if ok_env else 'FAIL'}] config.provider_keys: {detail_env}")
        out.append(
            _DoctorCheck(
                name="config.provider_keys",
                status="PASS" if ok_env else "FAIL",
                detail=detail_env,
            )
        )

    detail_git = "push/--force/history rewrites are refused unconditionally (git_ops, no override)"
    print(f"[PASS] config.git_policy: {detail_git}")
    out.append(_DoctorCheck(name="config.git_policy", status="PASS", detail=detail_git))
    return out
