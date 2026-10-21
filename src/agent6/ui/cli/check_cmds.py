# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 check` pre-flight: sandbox, config, MCP and verify."""

from __future__ import annotations

import contextlib
import pathlib
import shlex
import shutil

from agent6 import kinds, verify_infer
from agent6.app import _setup, confine
from agent6.config import (
    Config,
    ConfigError,
    MCPServerEntry,
    layer,
)
from agent6.sandbox import (
    JailUnavailableError,
    detect,
    jail,
    tool_paths,
)
from agent6.tools import mcp_client
from agent6.ui.cli import _check_sandbox


def _cmd_check(config_paths: tuple[pathlib.Path, ...], *, section: str) -> int:
    """Run the selected pre-flight sections and print a summary.

    Nothing writes to the repo: MCP servers start as a run starts them, with the
    workspace read-only, just long enough to list their tools. The one network call
    is the provider's model listing, refreshed for pricing and never fatal.

    Args:
        config_paths: The invocation's `--config`.
        section: The section to run, or "all".

    Returns:
        0 when every selected check passes, 1 otherwise.
    """
    print(f"agent6 check: {'all sections' if section == 'all' else section}")
    print()

    checks: list[kinds.DoctorCheck] = []
    # Every section needs the config: the sandbox probes run under the isolation it selects.
    cfg: Config | None = None
    explicit_leaves: frozenset[str] = frozenset()
    load_error: str | None = None
    try:
        effective = layer.load_effective(pathlib.Path.cwd(), config_paths)
        cfg, explicit_leaves = effective.config, effective.explicit_leaves
    except (ConfigError, OSError) as exc:
        load_error = str(exc)

    if load_error is not None:
        print(f"== config ==\n[FAIL] cannot load config: {load_error}\n")
        checks.append(kinds.DoctorCheck(name="config_load", status="FAIL", detail=load_error))

    if cfg is not None and section in {"all", "sandbox"}:
        print("== sandbox ==")
        checks.extend(_check_sandbox.check_sandbox_section(cfg, explicit_leaves))
        print()

    if cfg is not None and section in {"all", "config"}:
        print("== config ==")
        checks.extend(_doctor_check_config(cfg))
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


def _probe_refusal(
    cfg: Config,
    env: detect.Environment,
    name: str,
    srv: MCPServerEntry,
    isolation: kinds.IsolationLevel,
) -> str | None:
    """Return why the diagnostic leaves the server unstarted, or None when a read-only probe holds.

    A diagnostic never starts a server outside a run's confinement, so a write grant
    of any kind, or no jail at all, means no probe.
    """
    if srv.url:
        return None
    if isolation == "none":
        return f"not probed: no jail ({_setup.no_jail_cause(cfg, env)}); a run starts it unconfined"
    sb = srv.sandbox
    if sb is not None and sb.unconfined:
        return (
            f"not probed: mcp.servers.{name}.sandbox.unconfined = true; a run starts it"
            " as configured, unconfined"
        )
    grants = (
        (f"mcp.servers.{name}.sandbox.write_paths", sb.write_paths if sb else ()),
        ("sandbox.extra_write_paths", cfg.sandbox.extra_write_paths),
        ("sandbox.extra_device_paths", cfg.sandbox.extra_device_paths),
    )
    for leaf, paths in grants:
        if paths:
            return (
                f"not probed: {leaf} grants writes ({', '.join(paths)}); a run starts it"
                " as configured"
            )
    return None


def _doctor_check_mcp(cfg: Config) -> list[kinds.DoctorCheck]:
    """Start each enabled MCP server as a run would, list its tools, then close it.

    A server a run would refuse is a FAIL row; one the diagnostic must not start is a
    WARN row; neither starts. A disabled or empty `[mcp]` is one PASS row.

    Args:
        cfg: The config.

    Returns:
        One row per server.
    """
    if not cfg.mcp.enabled or not cfg.mcp.servers:
        cause = "[mcp].enabled = false" if not cfg.mcp.enabled else "no servers configured"
        print(f"  (no MCP servers to check: {cause})")
        return [kinds.DoctorCheck(name="mcp", status="PASS", detail=f"not configured ({cause})")]
    try:
        env = _setup.detect_env()
        isolation = detect.resolve_isolation(cfg.sandbox.isolation, env)
    except (detect.IsolationUnavailableError, JailUnavailableError) as exc:
        print(f"[FAIL] mcp: {exc}")
        return [kinds.DoctorCheck(name="mcp", status="FAIL", detail=str(exc))]
    root = pathlib.Path.cwd()
    out: list[kinds.DoctorCheck] = []
    probed: dict[str, MCPServerEntry] = {}
    for name, srv in sorted(cfg.mcp.enabled_servers.items()):
        if (refusal := confine.mcp_network_refusal(name, srv, isolation)) is not None:
            detail = f"a run would refuse: {refusal}"
            print(f"  {name}: {detail}")
            out.append(kinds.DoctorCheck(name=f"mcp.{name}", status="FAIL", detail=detail))
        elif (reason := _probe_refusal(cfg, env, name, srv, isolation)) is not None:
            print(f"  {name}: {reason}")
            out.append(kinds.DoctorCheck(name=f"mcp.{name}", status="WARN", detail=reason))
        else:
            probed[name] = srv
    if not probed:
        return out or [kinds.DoctorCheck(name="mcp", status="PASS", detail="no enabled servers")]
    with contextlib.ExitStack() as stack:
        # A server set to `session` joins the run's network, so one is opened for it.
        try:
            session_net = (
                stack.enter_context(contextlib.closing(jail.SessionNetwork.open()))
                if isolation == "full"
                and any(srv.effective_network == "session" for srv in probed.values())
                else None
            )
        except JailUnavailableError as exc:
            print(f"[FAIL] mcp: {exc}")
            return [*out, kinds.DoctorCheck(name="mcp", status="FAIL", detail=str(exc))]
        try:
            specs = [
                _setup.mcp_server_spec(cfg, root, isolation, name, srv, readonly=True)
                for name, srv in probed.items()
            ]
        except JailUnavailableError as exc:
            # The jail's HOME cannot be made.
            print(f"[FAIL] mcp: {exc}")
            return [*out, kinds.DoctorCheck(name="mcp", status="FAIL", detail=str(exc))]
        manager = mcp_client.MCPManager.start(specs, session_net=session_net)
        stack.callback(manager.close)
        by_server: dict[str, list[str]] = {}
        for d in manager.descriptors():
            by_server.setdefault(d.server_name, []).append(d.tool_name)
        why_missing = {f.name: f.error for f in manager.failures}
        for name in sorted(probed):
            tools = by_server.get(name, [])
            ok = bool(tools)
            # A server that never started names its spawn error, not "no tools".
            detail = (
                f"{mcp_client.tool_count(len(tools))}, network: {manager.networks[name]},"
                f" approve: {cfg.mcp.servers[name].approve}"
                if ok
                else why_missing.get(name, "started but exposed no tools")
            )
            print(f"  {name}: {detail}")
            out.append(
                kinds.DoctorCheck(
                    name=f"mcp.{name}", status="PASS" if ok else "FAIL", detail=detail
                )
            )
    return out


def _doctor_check_verify(cfg: Config) -> list[kinds.DoctorCheck]:
    """Check the verify command's shape without running it: argv non-empty, head executable found.

    Args:
        cfg: The config.

    Returns:
        The rows.
    """
    argv = list(cfg.harness.verify_command)
    if not argv:
        # A run infers one from the deterministic tiers or goes gateless; say what this repo infers.
        cwd, hide = pathlib.Path.cwd(), cfg.sandbox.hide_paths
        inferred = verify_infer.infer_verify_command(
            cwd, verify_infer.read_agents_md(cwd, hide), llm_call=None, hide=hide
        )
        passed = verify_infer.passed_over_text(inferred)
        nothing = "nothing here a run can use" if passed else "nothing here to infer from"
        detail = (
            f"unset; a run here infers {shlex.join(inferred.argv)} from {inferred.label}"
            if inferred.argv
            else f"unset; {nothing} (a run asks the reviewer model over the"
            " manifests, else goes gateless)"
        )
        if passed:
            detail += f"; passed over: {passed}"
        print(f"  {detail}")
        return [kinds.DoctorCheck(name="verify.argv", status="INFO", detail=detail)]
    head = argv[0]
    resolved = shutil.which(head, path=tool_paths.jail_search_path())
    ok = resolved is not None
    detail = f"resolves to {resolved}" if resolved else f"not found on the command PATH: {head!r}"
    print(f"  {head}: {detail}")
    print(f"  argv = {argv}")
    print(f"  timeout = {cfg.harness.verify_timeout_s}s")
    return [kinds.DoctorCheck(name="verify.head", status="PASS" if ok else "FAIL", detail=detail)]


def _doctor_check_config(cfg: Config) -> list[kinds.DoctorCheck]:
    """Check the static config: provider keys and the worktree git policy.

    Args:
        cfg: The config.

    Returns:
        The rows.
    """
    out: list[kinds.DoctorCheck] = []
    if not cfg.providers:
        # With no provider, "every key resolves" is vacuously true, but a run would reject.
        detail_env = (
            "no providers configured yet; run `agent6 connect` (required before `agent6 run`)"
        )
        out.append(kinds.DoctorCheck(name="config.provider_keys", status="INFO", detail=detail_env))
    else:
        env_err = _setup.check_provider_keys(cfg)
        ok_env = env_err is None
        detail_env = "all referenced provider keys resolve" if ok_env else env_err or ""
        out.append(
            kinds.DoctorCheck(
                name="config.provider_keys",
                status="PASS" if ok_env else "FAIL",
                detail=detail_env,
            )
        )

    print(f"  provider keys: {detail_env}")
    detail_git = "push, --force and history rewrites are always refused; no setting enables them"
    print(f"  git policy: {detail_git}")
    out.append(kinds.DoctorCheck(name="config.git_policy", status="PASS", detail=detail_git))
    return out
