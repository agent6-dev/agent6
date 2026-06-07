# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` — sandbox + config + MCP + verify pre-flight."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from agent6.cli._common import _check_provider_keys, _start_mcp_manager_if_enabled
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config_layer import (
    load_effective,
)
from agent6.detect import detect, select_profile
from agent6.sandbox import (
    JailUnavailableError,
    landlock_abi,
    run_in_jail,
)
from agent6.types import JailPolicy, SandboxReport


def _cmd_check_sandbox() -> int:
    """Run the sandbox boundary self-tests on the host's kernel."""
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

    # Try running `/bin/true` in the jail.
    cwd = Path.cwd()
    try:
        res = run_in_jail(
            JailPolicy(cwd=cwd, argv=("/usr/bin/true",), allow_network=False, timeout_s=10.0)
        )
        reports.append(SandboxReport(name="jail_true", ok=res.ok, detail=f"rc={res.returncode}"))
    except JailUnavailableError as exc:
        reports.append(SandboxReport(name="jail_true", ok=False, detail=str(exc)))

    # Confirm child cannot reach the network (when allow_network=False).
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=cwd,
                argv=("/usr/bin/getent", "hosts", "example.com"),
                allow_network=False,
                timeout_s=10.0,
            )
        )
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

    # Confirm child cannot write outside /workspace.
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=cwd,
                argv=("/bin/sh", "-c", "echo x > /etc/agent6-escape || true"),
                allow_network=False,
                timeout_s=10.0,
            )
        )
        # /etc was bind-mounted RO and Landlock confines writes to /workspace, so the
        # file should not exist on the host.
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

    overall_ok = True
    for r in reports:
        status = "PASS" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.detail}")
        overall_ok = overall_ok and r.ok
    return 0 if overall_ok else 1


@dataclass(frozen=True, slots=True)
class _DoctorCheck:
    name: str
    ok: bool
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
                ok=(rc == 0),
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
            checks.append(_DoctorCheck(name="config_load", ok=False, detail=str(exc)))

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
    overall_ok = True
    for c in checks:
        flag = "PASS" if c.ok else "FAIL"
        print(f"[{flag}] {c.name}: {c.detail}")
        overall_ok = overall_ok and c.ok
    return 0 if overall_ok else 1


def _check_config_section(cfg: Config) -> list[_DoctorCheck]:
    """Environment detection + profile selection + static config checks."""
    env = detect()
    print(f"  kernel: {env.kernel.raw} (Landlock TCP: {env.kernel.supports_landlock_tcp})")
    print(f"  userns supported: {env.userns_supported}")
    print(f"  sandbox available: {env.sandbox_available}")
    abi_str = str(landlock_abi()) if env.sandbox_available else "n/a (no Linux sandbox)"
    print(f"  Landlock ABI: {abi_str}")
    print(
        f"  sandbox.profile = {cfg.sandbox.profile}  network = {cfg.sandbox.network}"
        f"  run_commands = {cfg.sandbox.run_commands}"
    )
    out: list[_DoctorCheck] = []
    try:
        selected = select_profile(cfg.sandbox.profile, env)
        print(f"  -> selected profile: {selected}")
        out.append(_DoctorCheck(name="config.profile", ok=True, detail=f"selected {selected}"))
    except RuntimeError as exc:
        print(f"  [FAIL] profile selection: {exc}")
        out.append(_DoctorCheck(name="config.profile", ok=False, detail=str(exc)))
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
                ok=True,
                detail="not configured (cfg.mcp.enabled=False or empty servers)",
            )
        ]
    manager = _start_mcp_manager_if_enabled(cfg)
    if manager is None:
        return [_DoctorCheck(name="mcp", ok=True, detail="no enabled servers")]
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
            out.append(_DoctorCheck(name=f"mcp.{name}", ok=ok, detail=detail))
    finally:
        manager.close()
    return out


def _doctor_check_verify(cfg: Config) -> list[_DoctorCheck]:
    """Verify command sanity: argv non-empty and the head executable resolves.

    Does NOT execute the verify command — that would run an arbitrary
    test suite on every doctor call. Operators can do
    ``./$(verify_command)`` themselves when they want a live run.
    """
    argv = list(cfg.workflow.verify_command)
    if not argv:
        print("[FAIL] verify.argv: empty")
        return [_DoctorCheck(name="verify.argv", ok=False, detail="empty")]
    head = argv[0]
    resolved = shutil.which(head)
    ok = resolved is not None
    detail = f"resolves to {resolved}" if resolved else f"not found on PATH: {head!r}"
    print(f"[{'PASS' if ok else 'FAIL'}] verify.head: {detail}")
    print(f"       argv = {argv}")
    print(f"       timeout = {cfg.workflow.verify_timeout_s}s")
    return [_DoctorCheck(name="verify.head", ok=ok, detail=detail)]


def _doctor_check_config(cfg: Config) -> list[_DoctorCheck]:
    """Static config sanity checks: provider keys + worktree git policy."""
    out: list[_DoctorCheck] = []
    env_err = _check_provider_keys(cfg)
    ok_env = env_err is None
    detail_env = "all referenced provider keys resolve" if ok_env else env_err or ""
    print(f"[{'PASS' if ok_env else 'FAIL'}] config.provider_keys: {detail_env}")
    out.append(_DoctorCheck(name="config.provider_keys", ok=ok_env, detail=detail_env))

    ok_git = cfg.git.allow_push is False
    detail_git = "git.allow_push=False (push is blocked, as required)"
    if not ok_git:
        detail_git = "git.allow_push=True — agent6 never pushes; set this back to false"
    print(f"[{'PASS' if ok_git else 'FAIL'}] config.git_policy: {detail_git}")
    out.append(_DoctorCheck(name="config.git_policy", ok=ok_git, detail=detail_git))
    return out
