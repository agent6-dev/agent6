# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check`, sandbox + config + MCP + verify pre-flight."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import (
    load_effective,
)
from agent6.detect import ProfileUnavailableError, apparmor_userns_restricted, select_profile
from agent6.sandbox import (
    JailUnavailableError,
    landlock_abi,
    run_in_jail,
)
from agent6.types import CommandResult, JailPolicy, SandboxReport
from agent6.ui.cli._common import (
    _check_provider_keys,
    _start_mcp_manager_if_enabled,
    detect_env,
)


def _cmd_check_sandbox() -> int:
    """Run the sandbox boundary self-tests on the host's kernel.

    The probes run under the *effective* profile this host resolves to
    (`select_profile("auto", ...)`), not a hardcoded one. On a host that
    blocks unprivileged user namespaces (default-seccomp Docker, or Ubuntu
    with `kernel.apparmor_restrict_unprivileged_userns=1`) the effective
    profile is `hardened`, which is exactly what `agent6 run` would use there;
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
            detail=f"abi={abi}; TCP={'yes' if abi >= 4 else 'no (need Linux 6.7)'}",
        )
    )

    profile = select_profile("auto", detect_env())
    print(f"  effective profile (auto): {profile}")
    if profile == "hardened" and apparmor_userns_restricted():
        print(
            "  NOTE: strict is unavailable because unprivileged user namespaces are\n"
            "  blocked by kernel.apparmor_restrict_unprivileged_userns=1 (Ubuntu 24.04+\n"
            "  default). For the stronger strict profile, install the bundled agent6-jail\n"
            "  AppArmor profile (grants userns to just that binary):\n"
            "    agent6 system apparmor install\n"
            "  (or, less surgically, set the sysctl to 0). hardened is still real,\n"
            "  kernel-enforced isolation."
        )
    if profile == "none":
        # No kernel sandbox to test (a non-Linux host, or a deliberate `none`
        # opt-out), and running the boundary probes unconfined would let the
        # /etc-write probe actually escape onto the host. Report and stop.
        reports.append(
            SandboxReport(
                name="jail",
                ok=False,
                detail="no kernel sandbox on this platform (effective profile 'none'); skipped",
            )
        )
        return _print_sandbox_reports(reports)

    cwd = Path.cwd()

    def _jail(*argv: str) -> CommandResult:
        return run_in_jail(
            JailPolicy(cwd=cwd, argv=argv, profile=profile, allow_network=False, timeout_s=10.0)
        )

    # Try running `/usr/bin/true` in the jail.
    try:
        res = _jail("/usr/bin/true")
        reports.append(SandboxReport(name="jail_true", ok=res.ok, detail=f"rc={res.returncode}"))
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_true", ok=False, detail=str(exc)))

    # Confirm child cannot reach the network. This is only a meaningful jail
    # probe under `strict`, where `allow_network=False` puts the child in an
    # empty network namespace. Under `hardened` the jail applies no network
    # rule at all, a jailed command's egress is bounded by the agent-process
    # Landlock applied at run time (SECURITY.md §1, §8 note 2), which this
    # standalone probe does not set up, so testing it here would be testing
    # the wrong thing. Report it as n/a rather than a misleading pass/fail.
    if profile == "strict":
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
                detail="n/a under hardened (egress confined by agent-process Landlock at run time)",
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
            if section in {"all", "mcp", "verify", "config"}
            else None
        )
    except (ConfigError, OSError) as exc:
        cfg = None
        if section in {"all", "mcp", "verify", "config"}:
            print(f"== config ==\n[FAIL] cannot load config: {exc}\n")
            checks.append(_DoctorCheck(name="config_load", status="FAIL", detail=str(exc)))

    if cfg is not None and section in {"all", "config"}:
        print("== config ==")
        checks.extend(_check_config_section(cfg))
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
    """Environment detection + profile selection + static config checks."""
    env = detect_env()
    print(f"  kernel: {env.kernel.raw} (Landlock TCP: {env.kernel.supports_landlock_tcp})")
    print(f"  userns supported: {env.userns_supported}")
    print(f"  sandbox available: {env.sandbox_available}")
    abi_str = str(landlock_abi()) if env.sandbox_available else "n/a (no Linux sandbox)"
    print(f"  Landlock ABI: {abi_str}")
    print(
        f"  sandbox.profile = {cfg.sandbox.profile}"
        f"  agent_network = {cfg.sandbox.agent_network}"
        f"  tool_network = {cfg.sandbox.tool_network}"
        f"  run_commands = {cfg.sandbox.run_commands}"
    )
    out: list[_DoctorCheck] = []
    try:
        selected = select_profile(cfg.sandbox.profile, env)
        print(f"  -> selected profile: {selected}")
        out.append(
            _DoctorCheck(name="config.profile", status="PASS", detail=f"selected {selected}")
        )
    except ProfileUnavailableError as exc:
        print(f"  [FAIL] profile selection: {exc}")
        out.append(_DoctorCheck(name="config.profile", status="FAIL", detail=str(exc)))
    out.extend(_doctor_check_config(cfg))
    return out


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
    manager = _start_mcp_manager_if_enabled(cfg)
    if manager is None:
        return [_DoctorCheck(name="mcp", status="PASS", detail="no enabled servers")]
    out: list[_DoctorCheck] = []
    try:
        descriptors = manager.descriptors()
        by_server: dict[str, list[str]] = {}
        for d in descriptors:
            by_server.setdefault(d.server_name, []).append(d.tool_name)
        configured = {srv.name for srv in cfg.mcp.servers if srv.enabled}
        for name in sorted(configured):
            tools = by_server.get(name, [])
            ok = bool(tools)
            detail = f"{len(tools)} tool(s)" if ok else "started but exposed no tools"
            print(f"[{'PASS' if ok else 'FAIL'}] mcp.{name}: {detail}")
            out.append(
                _DoctorCheck(name=f"mcp.{name}", status="PASS" if ok else "FAIL", detail=detail)
            )
    finally:
        manager.close()
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
        env_err = _check_provider_keys(cfg)
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

    ok_git = cfg.git.allow_push is False
    detail_git = "git.allow_push=False (push is blocked, as required)"
    if not ok_git:
        detail_git = "git.allow_push=True; agent6 never pushes; set it back to false"
    print(f"[{'PASS' if ok_git else 'FAIL'}] config.git_policy: {detail_git}")
    out.append(
        _DoctorCheck(
            name="config.git_policy", status="PASS" if ok_git else "FAIL", detail=detail_git
        )
    )
    return out
