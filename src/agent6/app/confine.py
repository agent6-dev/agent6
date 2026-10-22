# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Warn about, or refuse, what an isolation level does not confine.

The agent process itself is never confined: every boundary is the jail's, and the levels
differ only in which jail features the launcher enables (docs/security.md owns the model).
A half-measure reads as a guarantee it cannot keep, so these checks warn or refuse instead.
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

# `partial`'s fixed Landlock grants in jail/src/main.rs; tests/security pins each against it.
PARTIAL_SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
PARTIAL_SYSTEM_RW = ("/tmp",)  # noqa: S108 - a granted path, never opened
PARTIAL_DEVICE_NODES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/full")


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

    Every warning here is a degrade: a default the host cannot honor, an explicit widening
    (root, a persistent HOME), or a configured grant the run leaves out. An explicit setting
    the host cannot honor refuses earlier and never reaches here. The warnings are once per
    run rather than per spawn, since a stderr line in every tool result would prompt the model
    to fight the sandbox.

    Args:
        isolation: The resolved isolation level.
        env: The detected host environment.
        cfg: The resolved config.
        root: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.
        reporter: Where the warnings go.
    """
    if cfg.sandbox.isolation == "auto" and not detect.sandbox_disabled_by_env():
        reason = detect.degrade_reason(env)
        if reason is not None:
            # `agent6 check sandbox` prints the same reason.
            reporter.warn(
                f"sandbox.isolation = 'auto' resolved to '{isolation}', not 'full',"
                f" because {reason}."
            )
    if isolation == "none":
        # The one owner of the cause; `check sandbox` prints the same.
        reporter.warn(
            f"running UNSANDBOXED ({_setup.no_jail_cause(cfg, env)}). "
            "Every command runs as a plain subprocess with no filesystem, network, "
            "or syscall confinement: the LLM's run_command, the verify command, and "
            "any spawned MCP server. sandbox.memory_limit_mb still applies: the "
            "launcher enforces it with confinement off. Only the "
            "surrounding environment, a container for instance, bounds what a command "
            "can reach. Use sandbox.isolation = 'auto', 'full', or 'partial' for "
            "kernel-enforced isolation."
            + (
                " git.control = 'agent6' is not enforced without the sandbox; "
                "a commit the model runs lands on the checkout's branch."
                if cfg.git.control == "agent6"
                else ""
            )
        )
    elif isolation == "full" and env.landlock_abi < 1:
        reporter.warn(
            "sandbox.isolation = 'full' runs without Landlock, because this kernel has none."
            " Namespaces, the read-only root filesystem and seccomp still confine commands."
            " Landlock needs Linux 5.13+ with the Landlock LSM enabled."
        )
    if isolation == "partial" and paths.is_root():
        # The root banner names running as root; this names what it costs at this level.
        reporter.warn(
            "running as root under sandbox.isolation = 'partial' lets a jailed command read the"
            " root-only files in the system dirs it is granted (/etc/shadow, /etc/sudoers, the"
            " host's ssh private keys). Run as your normal user."
        )
    if isolation == "partial" and cfg.protects_git:
        reporter.warn(
            "sandbox.protect_git = 'auto' resolves to off: bind-remounting .git read-only needs"
            " sandbox.isolation = 'full', but this run resolved to 'partial', so jailed commands"
            " can write .git. Set sandbox.protect_git = 'off' to accept that silently, or"
            " sandbox.isolation = 'full'."
        )
    persistent = tools_policy.persistent_jail_home(cfg, isolation, root, worktree_git_dir)
    if persistent is not None and isolation != "none":
        cause, fix = (
            (
                "sandbox.home = 'cache'",
                "Set sandbox.home = 'tmp' for a HOME that goes with the run.",
            )
            if isolation == "full"
            else (
                "sandbox.isolation = 'partial' has no private /tmp",
                "Set sandbox.isolation = 'full' for a HOME that goes with the run.",
            )
        )
        reporter.warn(
            f"{cause}: HOME ({persistent}) persists across runs and is executable, so a "
            "cache poisoned by one run, or a ~/.gitconfig alias, reaches the next jailed "
            f"run (never your own tools). {fix}"
        )
    if isolation == "partial" and cfg.sandbox.network == "auto":
        reporter.warn(
            "sandbox.network = 'auto' requires sandbox.isolation = 'full' to give the run its own"
            " network, but this run resolved to 'partial', so jailed commands share this host's"
            " network. Set sandbox.network = 'session' to refuse the run instead, or"
            " sandbox.isolation = 'full'."
        )
    if isolation == "partial" and env.landlock_abi < 3:
        reporter.warn(
            f"sandbox.isolation resolved to 'partial' on Landlock ABI {env.landlock_abi}, so a"
            " jailed command can truncate files outside its write grants; its other writes stay"
            " confined. Upgrade the kernel to Linux 6.2+ for Landlock ABI 3, or set"
            " sandbox.isolation = 'full'."
        )
    _warn_grants(cfg, root, reporter)
    for hidden, region, source in unmaskable_exposures(cfg, isolation, root, worktree_git_dir):
        reporter.warn(exposure_warning(hidden, region, source))
    if isolation in ("full", "partial"):
        notes = tool_paths.tool_mount_notes()
        for tool in notes.unreachable:
            reporter.warn(
                f"tool {tool} resolves into a dir that is never"
                " mounted into the jail ($HOME itself, or agent6's private dirs),"
                " so it will not run inside sandboxed commands. Move the target"
                " into its own subdirectory."
            )
        # notes.exposes_home_dir is the ordinary state of a dev box; `agent6 check` lists it.


def exposure_warning(hidden: pathlib.Path, region: pathlib.Path, source: str) -> str:
    """Return the warning for a private dir `partial` cannot mask from a granted region.

    Args:
        hidden: The private dir.
        region: The granted region that overlaps it.
        source: The phrase naming where the grant comes from.

    Returns:
        The warning: what jailed commands can read, which of the two holds the other, and the
        way out.
    """
    kept = "there (provider keys, transcripts, notes, run history)"
    if region.resolve() == hidden.resolve():
        exposed, remedy = f"{hidden} itself ({source})", f"Move it outside {hidden}."
    elif region.resolve().is_relative_to(hidden.resolve()):
        exposed = f"{region}, which sits inside {hidden} ({source})"
        kept, remedy = "under it", f"Move it outside {hidden}."
    else:
        exposed = f"{hidden}, because it sits inside {region} ({source})"
        remedy = (
            "Hiding it under that grant requires sandbox.isolation = 'full', but this run"
            " resolved to 'partial'."
        )
    return (
        f"jailed commands can read {exposed}. Every command this run starts can read what"
        f" agent6 keeps {kept}. {remedy}"
    )


def _warn_grants(cfg: Config, root: pathlib.Path, reporter: app_reporter.Reporter) -> None:
    """Name each grant the run leaves out, and each that is the home directory or above it.

    Args:
        cfg: The resolved config.
        root: The workspace.
        reporter: Where the warnings go.
    """
    for dropped in tools_policy.resolve_grants(cfg, root).dropped:
        reporter.warn(dropped.line())
    for name, srv in cfg.mcp.enabled_servers.items():
        if not srv.jailed:
            continue
        for dropped in _setup.mcp_server_grants(cfg, root, srv, name).dropped:
            if dropped.key.startswith("mcp."):
                reporter.warn(dropped.line())
    for line in home_grant_warnings(cfg):
        reporter.warn(line)


def warn_cleartext_credential_endpoints(
    cfg: Config, *, reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER
) -> None:
    """Warn once per endpoint sending its credential over plaintext http to a non-loopback host.

    An explicit but discouraged config: it runs with the cost named, never a refusal, since an
    internal-network or VPN endpoint is a real case.

    Args:
        cfg: The config whose providers and MCP servers are checked.
        reporter: Where the warnings go.
    """
    for label in cfg.cleartext_credential_endpoints():
        reporter.warn(
            f"{label} sends its credential over plaintext http"
            " to a non-loopback host: anyone on the network path can read it."
            " Use https where you can."
        )


def home_grant_warnings(cfg: Config) -> tuple[str, ...]:
    """Return one warning per path grant that is the operator's home directory or above it.

    Such a grant reaches the credentials other tools keep under the home directory. The grant
    is the operator's to make, so it runs, with the cost named.

    Args:
        cfg: The config whose `sandbox.extra_read_paths` and `extra_write_paths` are checked.

    Returns:
        The warnings; a path in both lists warns once.
    """
    home = paths.effective_user().home.resolve()
    sb = cfg.sandbox
    grants = (
        *(("extra_write_paths", "read and write", p) for p in sb.extra_write_paths),
        # A write grant already reads: the same path granted both ways warns once.
        *(
            ("extra_read_paths", "read", p)
            for p in sb.extra_read_paths
            if p not in sb.extra_write_paths
        ),
    )
    out: list[str] = []
    for key, access, grant in grants:
        real = pathlib.Path(grant).resolve()
        if not home.is_relative_to(real):
            continue
        which = (
            "your home directory" if real == home else f"which holds your home directory ({home})"
        )
        out.append(
            f"sandbox.{key} grants {grant!r}, {which}: the run's commands and file tools can"
            f" {access} what is under it, the credentials other tools keep there included."
            " Under sandbox.isolation = 'full', sandbox.hide_paths hides the paths you list."
        )
    return tuple(out)


def check_workspace_outside_private_dirs(root: pathlib.Path) -> str | None:
    """Return the refusal when the workspace and a private dir overlap either way, else None.

    A workspace inside a private dir cannot read its own files; a private dir inside the
    workspace has its transcripts and keys readable by jailed commands and staged into commits.

    Args:
        root: The workspace.

    Returns:
        The refusal, or None when no private dir overlaps the workspace.
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


def check_workspace_outside_hidden_paths(cfg: Config, root: pathlib.Path) -> str | None:
    """Return the refusal when the workspace is a `[sandbox].hide_paths` entry or under one.

    Such a workspace hides the run's files from its own tools; the launcher's rootfs setup
    fails under the mask, and a write grant on the workspace would bind it back through the
    mask, `.git` included.

    Args:
        cfg: The resolved config.
        root: The workspace.

    Returns:
        The refusal, or None otherwise; an entry inside the workspace is fine.
    """
    resolved = root.resolve()
    for entry in cfg.sandbox.hide_paths:
        if resolved.is_relative_to(pathlib.Path(entry).resolve()):
            return (
                f"the workspace {str(resolved)!r} is under {entry!r}, which"
                " sandbox.hide_paths hides from the run's tools and commands, so the run"
                " could not work on its own files. Drop the entry from sandbox.hide_paths,"
                " or work in a directory outside it."
            )
    return None


def check_protect_git_support(cfg: Config, isolation: kinds.IsolationLevel) -> str | None:
    """Return the refusal when `protect_git = "on"` cannot be honored here, else None.

    `on` requires `full` `sandbox.isolation` to bind-remount `.git` read-only; `auto` degrades
    with a warning instead.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.

    Returns:
        The refusal for `on` off `full`, else None.
    """
    if isolation == "full" or cfg.sandbox.protect_git != "on":
        return None
    return (
        "sandbox.protect_git = 'on' requires sandbox.isolation = 'full' to bind-remount .git"
        f" read-only, but this run resolved to {isolation!r}. Set sandbox.protect_git = 'auto'"
        " to run without it where 'full' is unavailable, or sandbox.isolation = 'full'."
    )


def check_jail_home(
    cfg: Config,
    isolation: kinds.IsolationLevel,
    workspace: pathlib.Path,
    worktree_git_dir: pathlib.Path | None = None,
) -> str | None:
    """Return the refusal when the jail's HOME cannot be what the config says, else None.

    `home = "tmp"` requires `full` `sandbox.isolation` to put HOME on a private /tmp and refuses
    under another level, where `auto` degrades to the repository's cache dir with a warning. A
    persistent dir the policy builder could not make agent6's own refuses too, without creating
    anything.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.
        workspace: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The refusal, or None.
    """
    if isolation != "full" and cfg.sandbox.home == "tmp":
        persistent = paths.jail_cache_home(paths.checkout_root(workspace))
        return (
            "sandbox.home = 'tmp' requires sandbox.isolation = 'full' to put HOME on a private"
            f" /tmp, but this run resolved to {isolation!r}. Set sandbox.home = 'auto' to use"
            f" the persistent HOME {str(persistent)!r} there, or sandbox.isolation = 'full'."
        )
    persistent = tools_policy.persistent_jail_home(cfg, isolation, workspace, worktree_git_dir)
    if persistent is None:
        return None
    return tools_policy.jail_home_refusal(persistent)


def _partial_grant_regions(
    cfg: Config, root: pathlib.Path, worktree_git_dir: pathlib.Path | None = None
) -> tuple[tuple[pathlib.Path, str], ...]:
    """Return every region a jailed command is granted under `partial`, labeled by its source.

    Derived from the builders the run uses, so preflight and enforcement cannot drift. The
    fixed sets mirror the Landlock rules jail/src/main.rs applies under `partial`.

    Args:
        cfg: The resolved config.
        root: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The (region, source) pairs; a source is the phrase a warning or refusal quotes.
    """
    policy = tools_policy.jail_policy(
        root, cfg, "partial", ("true",), worktree_git_dir=worktree_git_dir
    )
    regions: list[tuple[pathlib.Path, str]] = [(root, "the workspace")]
    regions += [(pathlib.Path(p), f"the host's shared {p}") for p in PARTIAL_SYSTEM_RW]
    regions += [
        (pathlib.Path(p), "a system dir every command is granted") for p in PARTIAL_SYSTEM_RO
    ]
    regions += [
        (pathlib.Path(p), "a device node every command is granted") for p in PARTIAL_DEVICE_NODES
    ]
    regions += [(p, "sandbox.extra_device_paths") for p in policy.extra_device_paths]
    network_reads = tools_policy.host_network_reads()
    for p in policy.extra_ro_paths:
        if pathlib.Path(p) == worktree_git_dir:
            regions.append(
                (pathlib.Path(p), "the repository's .git, which this linked worktree points into")
            )
        elif p in network_reads:
            regions.append((p, "what name resolution and TLS verification read"))
        else:
            regions.append((pathlib.Path(p), "sandbox.extra_read_paths"))
    home = tools_policy.persistent_jail_home(cfg, "partial", root, worktree_git_dir)
    regions += [
        (pathlib.Path(p), "the jail's HOME" if p == home else "sandbox.extra_write_paths")
        for p in policy.extra_rw_paths
    ]
    regions += [(pathlib.Path(p), "an operator tool dir (PATH mount)") for p in policy.tool_paths]
    for name, srv in cfg.mcp.enabled_servers.items():
        spol = _setup.mcp_server_policy(cfg, root, "partial", srv)
        if spol is None:
            continue  # not jailed: a `url` or `unconfined` server
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

    Under `partial`, a hidden path is exposed when it holds a granted region or sits inside one.
    Paths are resolved before containment so a `..` spelling cannot dodge the check.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.
        root: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        At most one triple per hidden path; empty outside `partial`.
    """
    if isolation != "partial":
        return ()
    regions = _partial_grant_regions(cfg, root, worktree_git_dir)
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

    `hide_paths` is only ever explicit, so an entry that `partial` cannot mask refuses. The
    always-hidden private dirs warn instead: granting a region holding them may be meant.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.
        root: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The refusal naming the first exposed entry and the region exposing it, else None.
    """
    if isolation != "partial":
        return None  # before reading config: every other level masks
    listed = {pathlib.Path(p) for p in cfg.sandbox.hide_paths}
    for hidden, region, source in unmaskable_exposures(cfg, isolation, root, worktree_git_dir):
        if hidden in listed:
            return (
                f"sandbox.hide_paths lists {str(hidden)!r}, which overlaps {str(region)!r}"
                f" ({source}). Hiding a path that overlaps a grant requires"
                " sandbox.isolation = 'full', but this run resolved to 'partial'. Set"
                " sandbox.isolation = 'full', drop the entry, or move one of the two."
            )
    return None


def mcp_network_refusal(
    name: str, srv: MCPServerEntry, isolation: kinds.IsolationLevel
) -> str | None:
    """Return the refusal when the server named a network this host cannot give it, else None.

    A server's `none` or `session` network requires `full` `sandbox.isolation` and refuses on
    `partial`, while `auto` degrades with a warning. The run's preflight and `agent6 check`
    both ask here, for each server a run starts.

    Args:
        name: The server's name under `[mcp.servers]`.
        srv: The server's config block.
        isolation: The resolved isolation level.

    Returns:
        The refusal for a `none` or `session` network on `partial`, else None.
    """
    if isolation != "partial" or srv.effective_network not in ("none", "session"):
        return None
    return (
        f"MCP server {name!r} sets sandbox.network = {srv.effective_network!r}. Keeping the"
        " server off the host network requires sandbox.isolation = 'full', but this host"
        f" resolved to {isolation!r}. Set its sandbox.network = 'auto' to run with a warning,"
        " or 'host' to accept the host network."
    )


def check_mcp_network_support(cfg: Config, isolation: kinds.IsolationLevel) -> str | None:
    """Return the first network refusal among the servers a run starts, else None.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.

    Returns:
        The refusal, or None.
    """
    for name, srv in sorted(cfg.mcp.enabled_servers.items()):
        if (refusal := mcp_network_refusal(name, srv, isolation)) is not None:
            return refusal
    return None


def config_refusal(
    cfg: Config,
    isolation: kinds.IsolationLevel,
    workspace: pathlib.Path,
    *,
    worktree_git_dir: pathlib.Path | None = None,
) -> str | None:
    """Return the first refusal for a config this host cannot honor, else None.

    The one list every lifecycle runs, so a check cannot land in one and not another. The
    network check stays per lifecycle: a machine runs it once per state.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.
        workspace: The workspace.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The refusal, or None when every check passes.
    """
    checks: tuple[Callable[[], str | None], ...] = (
        # The HOME check first: `check_hide_paths_support` builds the policy, which creates HOME.
        lambda: check_jail_home(cfg, isolation, workspace, worktree_git_dir),
        lambda: check_mcp_network_support(cfg, isolation),
        # Before `check_hide_paths_support`, so a workspace under an entry gets its own refusal.
        lambda: check_workspace_outside_hidden_paths(cfg, workspace),
        lambda: check_hide_paths_support(cfg, isolation, workspace, worktree_git_dir),
        lambda: check_workspace_outside_private_dirs(workspace),
        lambda: check_protect_git_support(cfg, isolation),
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

    `session` requires `full` `sandbox.isolation`, while `auto` degrades with a warning.

    Args:
        cfg: The resolved config.
        isolation: The resolved isolation level.

    Returns:
        The refusal for `sandbox.network = "session"` on `partial`, else None.
    """
    if isolation != "partial":
        return None
    sb = cfg.sandbox
    if sb.network == "session":
        return (
            "sandbox.network = 'session' requires sandbox.isolation = 'full' to give the run its"
            " own network, but this run resolved to 'partial'. Set sandbox.network = 'auto' to"
            " run with a warning, or 'host' to accept this host's network."
        )
    return None


def resolved_config_values(cfg: Config) -> dict[str, object]:
    """Return every config leaf whose effective value differs from its raw one.

    Args:
        cfg: The effective config.

    Returns:
        The values by leaf: the adaptive model settings and the two `auto` sandbox knobs as
        this host resolves them. With no jail binary to probe, the two stay `auto`.
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
