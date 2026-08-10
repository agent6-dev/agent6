# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Warnings and cross-checks for what an isolation level does NOT confine.

The agent process itself is never confined, at any isolation level: every
boundary is the jail's, applied per command, and the levels differ only in
which jail features the launcher enables (docs/security.md owns the model
and the rationale). Nothing bounds the agent's own filesystem or egress; a
partial block on a trusted process reads as a guarantee it cannot keep, so
these checks warn or refuse instead of pretending.
"""

from __future__ import annotations

from pathlib import Path

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config
from agent6.paths import cache_dir, data_dir, global_config_dir, state_base
from agent6.sandbox.detect import Environment
from agent6.sandbox.jail import unreachable_tools
from agent6.types import IsolationLevel


def warn_sandbox_gaps(
    isolation: IsolationLevel,
    env: Environment,
    cfg: Config,
    *,
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print a prominent warning when the isolation confines less than it promises.

    `none` is reached on a host with no confinement mechanism at all
    (non-Linux, or a Linux kernel offering neither userns nor Landlock), or
    when the operator EXPLICITLY sets `isolation = "none"` (the unsandboxed
    opt-out, intended for inside a container). Either way commands run as
    plain subprocesses with no agent6 confinement, so say so loudly.

    `strict` needs only userns; on a kernel without Landlock the jail's
    best-effort ruleset enforces nothing (`restrict_self` returns NotEnforced)
    while namespaces + the pivoted read-only rootfs + seccomp still confine.
    That is a documented layer going missing, so it is loud too -- here, once
    per run, not in the launcher: a per-spawn stderr warning would land in
    every tool result and prompt the model to fight the sandbox.

    `tool_network = "auto"` DEGRADES on a netns-less isolation: with no per-child
    network namespace, a jailed run_command shares the (agent-scoped) host
    network instead of being offline, so say so once per run. Explicit `block`
    never reaches here (check_network_support refused it on hardened).

    `protect_git` degrades the same way: strict-only, because it is a read-only
    bind. An explicitly-set one refuses (check_protect_git_support).
    """
    if isolation == "none":
        reporter.err(
            "[agent6] WARNING: running UNSANDBOXED (sandbox.isolation = 'none'). "
            "Commands -- including the LLM's run_command and verify_command -- "
            "execute as plain subprocesses with NO filesystem, network, or syscall "
            "confinement; the agent is contained only by the surrounding environment "
            "(e.g. the container it runs in). Use 'auto'/'strict'/'hardened' for "
            "kernel-enforced isolation."
        )
    elif isolation == "strict" and env.landlock_abi < 1:
        reporter.err(
            "[agent6] WARNING: 'strict' is running WITHOUT its Landlock layer: "
            "this kernel offers no Landlock (needs Linux >= 5.13 with the "
            "Landlock LSM enabled). Namespaces, the pivoted read-only rootfs, "
            "and seccomp still confine commands; the in-jail Landlock "
            "defense-in-depth is absent."
        )
    if isolation == "hardened" and cfg.sandbox.protect_git:
        reporter.err(
            "[agent6] WARNING: 'hardened' cannot protect .git -- that is a "
            "read-only bind, which needs the mount namespace only 'strict' has. "
            "A jailed command can write .git here; the in-process edit tools "
            "still refuse to. Use 'strict' for the real thing."
        )
    if isolation == "hardened" and cfg.sandbox.tool_network == "auto":
        reporter.err(
            "[agent6] WARNING: 'hardened' has no network namespace, so "
            "sandbox.tool_network = 'auto' cannot make a jailed run_command "
            "offline: it shares this process's host network, which hardened "
            "does not confine. Run on 'strict' for a network-free tool sandbox "
            "and provider-only agent egress, or set sandbox.tool_network = "
            "'block' to refuse rather than run here."
        )
    if isolation in ("strict", "hardened"):
        for tool in unreachable_tools():
            reporter.err(
                f"[agent6] WARNING: tool {tool} resolves into a dir that is never"
                " mounted into the jail ($HOME itself, or agent6's private dirs),"
                " so it will not run inside sandboxed commands. Move the target"
                " into its own subdirectory."
            )


def check_protect_git_support(
    cfg: Config, isolation: IsolationLevel, *, explicitly_set: bool
) -> str | None:
    """A refusal message when `protect_git` was EXPLICITLY asked for and this
    isolation cannot provide it, else None.

    `protect_git` is strict-only. Strict re-binds `.git` read-only, which needs
    a mount namespace. On hardened there is none, so the only tool is Landlock,
    which has no deny rules: protecting `.git` means NOT granting the workspace
    root itself, and a Landlock grant is recursive, so granting the root its
    own create/remove rights would grant them over `.git` too. Carving it out
    therefore cost every top-level write -- `touch newfile`, `mkdir build`,
    `mkfifo` all failed at the workspace root, which is too much to pay.

    The default DEGRADES with a warning (see `warn_sandbox_gaps`); an explicit
    `protect_git = true` refuses, naming what is unsupported and the fix. The
    in-process edit tools still refuse writes into `.git` at every level.
    """
    if isolation != "hardened" or not (cfg.sandbox.protect_git and explicitly_set):
        return None
    return (
        "sandbox.protect_git = true requires the strict isolation (a read-only"
        " bind of .git), but this host supports only 'hardened', where Landlock"
        " could only provide it by refusing every write at the workspace root."
        " Set sandbox.protect_git = false to run here, or use strict."
    )


def check_hide_paths_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal message when hiding cannot be honored on this isolation, else
    None.

    Hiding is a mount-namespace mask, so `hardened` (Landlock has no deny
    rules) cannot honor a hidden path that sits inside a granted region -- the
    workspace or an extra grant. Leaving it readable would be silently
    ineffective security, so refuse instead, naming the pair. Hidden means
    agent6's own config/state/data/cache dirs plus `[sandbox].hide_paths`.

    A hidden path nothing grants is trivially satisfied (Landlock never
    granted it), so ordinary hardened runs are untouched. `none` has no jail
    at all; its blanket unsandboxed warning covers this too.
    """
    if isolation != "hardened":
        return None
    sb = cfg.sandbox
    regions = (
        Path.cwd(),
        *(Path(p) for p in (*sb.extra_read_paths, *sb.extra_write_paths)),
    )
    hidden = (
        *(Path(p) for p in sb.hide_paths),
        global_config_dir(),
        state_base(),
        data_dir(),
        cache_dir(),
    )
    for region in regions:
        for h in hidden:
            if h.is_relative_to(region):
                return (
                    f"{str(h)!r} must stay hidden from jailed commands"
                    " (agent6-private, or sandbox.hide_paths) but sits inside"
                    f" {str(region)!r}, which they can read. Masking it needs the"
                    " mount namespace only 'strict' has: use strict, or move one"
                    " of the two."
                )
    return None


def check_network_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal message if the network config EXPLICITLY enforces something
    this isolation cannot provide, else None.

    Only jailed commands have a network boundary. ``tool_network =
    "only_explicit_states"`` (singling one tool out) and ``"block"`` (no jailed
    network at all) both need a per-command network namespace, which only
    ``strict`` provides. On ``hardened`` we refuse rather than silently
    under-confine, naming what is unsupported and the fix; ``"auto"`` is the
    secure default that DEGRADES with a warning instead. On ``none`` the
    unsandboxed warning already covers it.
    """
    if isolation != "hardened":
        return None
    sb = cfg.sandbox
    if sb.tool_network == "only_explicit_states":
        return (
            "sandbox.tool_network = 'only_explicit_states' requires the strict"
            " isolation (per-command network namespaces), but this host supports"
            " only 'hardened'. Use 'auto' or 'allow'."
        )
    if sb.tool_network == "block":
        return (
            "sandbox.tool_network = 'block' requires the strict isolation (a"
            " per-command network namespace), but this host supports only"
            " 'hardened', where a jailed command shares this process's network."
            " Use 'auto' to run with a warning, or 'allow' to accept it."
        )
    return None
