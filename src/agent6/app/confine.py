# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Warn about, or refuse, what an isolation level does not confine.

The agent process itself is never confined: every boundary is the jail's, and the levels
differ only in which jail features the launcher enables (docs/security.md owns the model).
A partial block reads as a guarantee it cannot keep, so these checks warn or refuse instead.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from agent6 import kinds, paths
from agent6.app import _setup
from agent6.app import reporter as app_reporter
from agent6.config import Config, MCPServerEntry
from agent6.models import registry
from agent6.sandbox import detect, jail, tool_paths
from agent6.tools import policy as tools_policy


def warn_sandbox_gaps(
    isolation: kinds.IsolationLevel,
    env: detect.Environment,
    cfg: Config,
    *,
    root: pathlib.Path,
    worktree_git_dir: pathlib.Path | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Warn once per run about each way the isolation confines less than it promises.

    Every warning here is a degrade: a default the host cannot honor, or an explicit widening
    (root, a persistent HOME). An explicit setting the host cannot honor refuses earlier and
    never reaches here. The warnings are once per run rather than per spawn, since a stderr
    line in every tool result would prompt the model to fight the sandbox.

    Args:
        isolation: The resolved isolation level.
        env: The detected host environment.
        cfg: The resolved config.
        root: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.
        reporter: Where the warnings go.
    """
    if cfg.sandbox.isolation == "auto":
        reason = detect.degrade_reason(env)
        if reason is not None:
            # 'auto' landing under strict is never silent; the same line `check sandbox` prints.
            reporter.warn(f"'auto' selected '{isolation}', not 'strict': {reason}.")
    if isolation == "none":
        origin = (
            "sandbox.isolation = 'none'"
            if cfg.sandbox.isolation == "none"
            else "'auto' found no confinement mechanism on this host"
        )
        reporter.warn(
            f"running UNSANDBOXED ({origin}). "
            "Every command runs as a plain subprocess with no filesystem, network, "
            "or syscall confinement: the LLM's run_command, the verify command, and "
            "any spawned MCP server. sandbox.memory_limit_mb still applies: the "
            "launcher enforces it with confinement off. Only the "
            "surrounding environment, a container for instance, bounds what a command "
            "can reach. Use 'auto', 'strict', or 'hardened' for kernel-enforced "
            "isolation."
        )
    elif isolation == "strict" and env.landlock_abi < 1:
        reporter.warn(
            "'strict' is running WITHOUT its Landlock layer: "
            "this kernel offers no Landlock (needs Linux >= 5.13 with the "
            "Landlock LSM enabled). Namespaces, the pivoted read-only rootfs, "
            "and seccomp still confine commands; the in-jail Landlock "
            "defense-in-depth is absent."
        )
    if isolation == "hardened" and paths.is_root():
        # The root banner names running as root; this names what it costs at this level.
        reporter.warn(
            "running as root under 'hardened': file permissions "
            "no longer narrow what a jailed command reads, so it can read the "
            "root-only files in the granted system set (/etc/shadow, /etc/sudoers, "
            "the host's ssh private keys). 'strict' pivots into a minimal rootfs "
            "where they are absent. Run as your normal user."
        )
    if isolation == "hardened" and cfg.sandbox.protect_git:
        reporter.warn(
            "'hardened' cannot protect .git: the read-only bind "
            "needs a mount namespace, which only 'strict' has. A jailed command can "
            "write .git; the in-process edit tools still refuse. For the same "
            "reason /tmp is the host's shared /tmp. Use 'strict' for a private /tmp "
            "and a protected .git."
        )
    persistent = tools_policy.persistent_jail_home(cfg, isolation)
    if persistent is not None and isolation != "none":
        cause, fix = (
            (
                "sandbox.home = 'cache'",
                "Set sandbox.home = 'tmp' for a HOME that goes with the run.",
            )
            if isolation == "strict"
            else (
                "'hardened' has no private /tmp",
                "Use 'strict' for a HOME that goes with the run.",
            )
        )
        reporter.warn(
            f"{cause}: HOME ({persistent}) persists across runs and is executable, so a "
            "cache poisoned by one run, or a ~/.gitconfig alias, reaches the next jailed "
            f"run (never your own tools). {fix}"
        )
    if isolation == "hardened" and cfg.sandbox.network == "auto":
        reporter.warn(
            "'hardened' has no network namespace, so "
            "sandbox.network = 'auto' cannot give the run its own session "
            "network: jailed commands share this process's host network, which "
            "hardened does not confine. Run on 'strict' for a session "
            "network, or set sandbox.network = 'session' to refuse the run "
            "instead."
        )
    if isolation == "hardened" and env.landlock_abi < 3:
        reporter.warn(
            f"'hardened' on Landlock ABI {env.landlock_abi} (< 3) "
            "does not confine file truncation: a jailed command can truncate "
            "(truncate/ftruncate) files outside its write grants, discarding their "
            "contents. Its other writes stay confined. Full write-confinement needs "
            "Landlock ABI 3 (Linux 6.2): upgrade the kernel, or run on 'strict', "
            "whose mount namespace confines truncation on any ABI."
        )
    for hidden, region, source in unmaskable_exposures(cfg, isolation, root, worktree_git_dir):
        reporter.warn(
            f"jailed commands can read {hidden}: it sits inside"
            f" {region} ({source}), which they are granted, and 'hardened' has no"
            " mount namespace to mask it out. Every command this run starts can"
            " read the provider keys, transcripts, notes, and run history in there."
            " Use 'strict' to keep them masked under the same grant."
        )
    if isolation in ("strict", "hardened"):
        notes = tool_paths.tool_mount_notes()
        for tool in notes.unreachable:
            reporter.warn(
                f"tool {tool} resolves into a dir that is never"
                " mounted into the jail ($HOME itself, or agent6's private dirs),"
                " so it will not run inside sandboxed commands. Move the target"
                " into its own subdirectory."
            )
        # notes.exposes_home_dir is the ordinary state of a dev box; `agent6 check` lists it.


def warn_cleartext_credential_endpoints(
    cfg: Config, *, reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER
) -> None:
    """Warn once per endpoint sending its credential over plaintext http to a non-loopback host.

    An explicit but discouraged config: it runs with the cost named, never a refusal, since an
    internal-network or VPN endpoint is a real case.
    """
    for label in cfg.cleartext_credential_endpoints():
        reporter.warn(
            f"{label} sends its credential over plaintext http"
            " to a non-loopback host: anyone on the network path can read it."
            " Use https where you can."
        )


def check_workspace_outside_private_dirs(root: pathlib.Path) -> str | None:
    """Return the refusal when the workspace and a private dir overlap either way, else None.

    A workspace inside a private dir cannot read its own files; a private dir inside the
    workspace has its transcripts and keys readable by jailed commands and staged into commits.
    """
    resolved = root.resolve()
    for private in paths.private_dirs():
        p = private.resolve()
        if resolved == p or resolved.is_relative_to(p):
            return (
                f"the workspace {str(resolved)!r} is inside agent6's own private"
                f" directory {str(p)!r}, which is hidden from every tool: the run"
                " could not read or write its own files. Work in a directory"
                " outside it."
            )
        if p.is_relative_to(resolved):
            return (
                f"agent6's private directory {str(p)!r} is inside the workspace"
                f" {str(resolved)!r}: its transcripts and keys would be readable"
                " by jailed commands and staged into commits. Keep agent6's state"
                " base outside the workspace."
            )
    return None


def check_protect_git_support(
    cfg: Config, isolation: kinds.IsolationLevel, *, explicitly_set: bool
) -> str | None:
    """Return the refusal when an explicit `protect_git` cannot be honored here, else None.

    The read-only bind of `.git` needs a mount namespace, which only strict has. Landlock has
    no deny rules and its grants are recursive, so carving `.git` out under hardened would cost
    every write at the workspace root. The default degrades with a warning; the in-process
    edit tools refuse writes into `.git` at every level.
    """
    if isolation != "hardened" or not (cfg.sandbox.protect_git and explicitly_set):
        return None
    return (
        "sandbox.protect_git = true requires the strict isolation (a read-only"
        " bind of .git), but this run resolved to 'hardened', where Landlock"
        " could only provide it by refusing every write at the workspace root."
        " Set sandbox.protect_git = false to run here, or use strict."
    )


def check_jail_home(
    cfg: Config, isolation: kinds.IsolationLevel, *, explicitly_set: bool
) -> str | None:
    """Return the refusal when the jail's HOME cannot be what the config says, else None.

    `home = "tmp"` is a private tmpfs, which only strict has: the default degrades to the cache
    dir with a warning, an explicit one refuses. A persistent dir the policy builder could not
    make agent6's own refuses too, without creating anything.
    """
    if isolation != "strict" and explicitly_set and cfg.sandbox.home == "tmp":
        return (
            "sandbox.home = 'tmp' requires the strict isolation (a private /tmp tmpfs),"
            f" but this run resolved to {isolation!r}, where HOME is the persistent"
            f" cache dir {str(paths.jail_cache_home())!r}. Set sandbox.home = 'cache' to"
            " run here, or use strict."
        )
    persistent = tools_policy.persistent_jail_home(cfg, isolation)
    return None if persistent is None else tools_policy.jail_home_refusal(persistent)


def _hardened_grant_regions(
    cfg: Config, root: pathlib.Path, worktree_git_dir: pathlib.Path | None = None
) -> tuple[tuple[pathlib.Path, str], ...]:
    """Return every region the hardened launcher grants a command, labeled by its source.

    Derived from the builders the run uses, so preflight and enforcement cannot drift; the
    fixed sets mirror the launcher's hardened ruleset in jail/src/main.rs.
    """
    policy = tools_policy.jail_policy(
        root, cfg, "hardened", ("true",), worktree_git_dir=worktree_git_dir
    )
    regions: list[tuple[pathlib.Path, str]] = [
        (root, "the workspace"),
        (pathlib.Path("/tmp"), "the host's shared /tmp (hardened has no private tmpfs)"),  # noqa: S108
    ]
    for sysdir in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev"):
        regions.append((pathlib.Path(sysdir), "a system dir every command is granted"))
    for p in policy.extra_ro_paths:
        if pathlib.Path(p) == worktree_git_dir:
            regions.append(
                (pathlib.Path(p), "the repository's .git, which this linked worktree points into")
            )
        else:
            regions.append((pathlib.Path(p), "sandbox.extra_read_paths"))
    home = tools_policy.persistent_jail_home(cfg, "hardened")
    regions += [
        (pathlib.Path(p), "the jail's HOME" if p == home else "sandbox.extra_write_paths")
        for p in policy.extra_rw_paths
    ]
    regions += [(pathlib.Path(p), "an operator tool dir (PATH mount)") for p in policy.tool_paths]
    if cfg.mcp.enabled:
        for name, srv in cfg.mcp.servers.items():
            if not srv.enabled:
                continue
            spol = _setup.mcp_server_policy(cfg, root, "hardened", srv)
            if spol is None:
                continue  # unconfined by explicit opt-out; its own loud path
            regions += [
                (pathlib.Path(p), f"[mcp.servers.{name}] read_paths") for p in spol.extra_ro_paths
            ]
            regions += [
                (pathlib.Path(p), f"[mcp.servers.{name}] write_paths") for p in spol.extra_rw_paths
            ]
            regions.append((pathlib.Path(spol.cwd), f"[mcp.servers.{name}]'s working dir"))
    return tuple(regions)


def unmaskable_exposures(
    cfg: Config,
    isolation: kinds.IsolationLevel,
    root: pathlib.Path,
    worktree_git_dir: pathlib.Path | None = None,
) -> tuple[tuple[pathlib.Path, pathlib.Path, str], ...]:
    """Return the (hidden path, granted region, region source) triples the isolation cannot mask.

    Empty on strict, which masks, and on `none`, which has no jail. Under hardened a hidden
    path overlapping a granted region is exposed in both directions. Paths are resolved
    before containment so a `..` spelling cannot dodge the check.
    """
    if isolation != "hardened":
        return ()
    regions = _hardened_grant_regions(cfg, root, worktree_git_dir)
    out: list[tuple[pathlib.Path, pathlib.Path, str]] = []
    for h in paths.hidden_paths(pathlib.Path(p) for p in cfg.sandbox.hide_paths):
        hr = h.resolve()
        for region, source in regions:
            rr = region.resolve()
            if hr.is_relative_to(rr) or rr.is_relative_to(hr):
                out.append((h, region, source))
                break  # one exposing region per hidden path carries the point
    return tuple(out)


def check_hide_paths_support(
    cfg: Config,
    isolation: kinds.IsolationLevel,
    root: pathlib.Path,
    worktree_git_dir: pathlib.Path | None = None,
) -> str | None:
    """Return the refusal when a `[sandbox].hide_paths` entry cannot be masked here, else None.

    `hide_paths` is only ever explicit, so an entry hardened cannot mask refuses. The
    always-hidden private dirs warn instead: granting a region holding them may be meant.
    """
    if isolation != "hardened":
        return None  # before reading config: every other level masks
    listed = {pathlib.Path(p) for p in cfg.sandbox.hide_paths}
    for hidden, region, source in unmaskable_exposures(cfg, isolation, root, worktree_git_dir):
        if hidden in listed:
            return (
                f"sandbox.hide_paths lists {str(hidden)!r}, which sits inside"
                f" {str(region)!r} ({source}), granted to jailed commands."
                " Masking it needs the mount namespace only 'strict' has:"
                " use strict, drop the entry, or move one of the two."
            )
    return None


def mcp_network_refusal(
    name: str, srv: MCPServerEntry, isolation: kinds.IsolationLevel
) -> str | None:
    """Return the refusal when the server named a network this host cannot give it, else None.

    The `[sandbox].network` rule: `none` and `session` need a network namespace, which only
    strict has, so they refuse on hardened while `auto` degrades with a warning. The run's
    preflight and `agent6 check` both ask here.
    """
    if isolation != "hardened" or not srv.enabled:
        return None
    if srv.effective_network not in ("none", "session"):
        return None
    return (
        f"MCP server {name!r} sets sandbox.network = {srv.effective_network!r},"
        " which needs a network namespace and so the strict isolation; this"
        f" host resolved to {isolation!r}. Use 'auto' to run with a warning,"
        " or 'host' to accept the machine's network."
    )


def check_mcp_network_support(cfg: Config, isolation: kinds.IsolationLevel) -> str | None:
    """Return the first server's network refusal, else None."""
    for name, srv in sorted(cfg.mcp.servers.items()):
        if (refusal := mcp_network_refusal(name, srv, isolation)) is not None:
            return refusal
    return None


def config_refusal(
    cfg: Config,
    isolation: kinds.IsolationLevel,
    workspace: pathlib.Path,
    *,
    explicit_leaves: frozenset[str] = frozenset(),
    worktree_git_dir: pathlib.Path | None = None,
) -> str | None:
    """Return the first refusal for a config this host cannot honor, else None.

    The one list every lifecycle runs, so a check cannot land in one and not another. The
    network checks stay per lifecycle: machines route theirs through `resolve_network_fix`.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.
        workspace: The workspace.
        explicit_leaves: The config leaves the operator wrote, which refuse rather than degrade.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The refusal, or None when every check passes.
    """
    checks: tuple[Callable[[], str | None], ...] = (
        # The HOME check first: `check_hide_paths_support` builds the policy, which creates HOME.
        lambda: check_jail_home(cfg, isolation, explicitly_set="sandbox.home" in explicit_leaves),
        lambda: check_mcp_network_support(cfg, isolation),
        lambda: check_hide_paths_support(cfg, isolation, workspace, worktree_git_dir),
        lambda: check_workspace_outside_private_dirs(workspace),
        lambda: check_protect_git_support(
            cfg, isolation, explicitly_set="sandbox.protect_git" in explicit_leaves
        ),
    )
    for check in checks:
        try:
            err = check()
        except jail.JailUnavailableError as exc:
            return str(exc)
        if err is not None:
            return err
    return None


def check_network_support(cfg: Config, isolation: kinds.IsolationLevel) -> str | None:
    """Return the refusal when the network config needs what the isolation cannot give, else None.

    `only_explicit_states` and `session` both need a network namespace, which only strict has;
    `auto` degrades with a warning instead.
    """
    if isolation != "hardened":
        return None
    sb = cfg.sandbox
    if sb.network == "only_explicit_states":
        return (
            "sandbox.network = 'only_explicit_states' requires the strict"
            " isolation (network namespaces), but this run resolved to"
            " 'hardened'. Use 'auto' or 'host'."
        )
    if sb.network == "session":
        return (
            "sandbox.network = 'session' requires the strict isolation (a"
            " network namespace), but this run resolved to 'hardened', where"
            " a jailed command shares this process's network. Use 'auto' to run"
            " with a warning, or 'host' to accept it."
        )
    return None


def resolved_config_values(cfg: Config) -> dict[str, object]:
    """Return every config leaf whose effective value differs from its raw one.

    The adaptive model settings and the two `auto` sandbox knobs as this host resolves them.
    With no jail binary to probe, the two stay `auto`.
    """
    out = registry.resolved_adaptive_values(cfg)
    if cfg.sandbox.isolation == "auto" or cfg.sandbox.network == "auto":
        try:
            selected = detect.resolve_isolation(cfg.sandbox.isolation, _setup.detect_env())
        except jail.JailBinaryError:
            return out
        if cfg.sandbox.isolation == "auto":
            out["sandbox.isolation"] = selected
        if cfg.sandbox.network == "auto":
            out["sandbox.network"] = tools_policy.resolve_network(cfg, selected)
    return out
