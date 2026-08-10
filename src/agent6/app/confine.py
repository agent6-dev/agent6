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
from agent6.paths import private_dirs
from agent6.sandbox.detect import Environment
from agent6.sandbox.jail import tool_mount_notes
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

    `tool_network = "auto"` DEGRADES on a netns-less isolation: with no network
    namespace there is no private network to give, so a jailed run_command
    shares the host's, and we say so once per run. Explicit `private` never
    reaches here (check_network_support refused it on hardened).

    `protect_git` degrades the same way: strict-only, because it is a read-only
    bind. An explicitly-set one refuses (check_protect_git_support).
    """
    if isolation == "none":
        reporter.err(
            "[agent6] WARNING: running UNSANDBOXED (sandbox.isolation = 'none'). "
            "Commands -- including the LLM's run_command and verify_command -- "
            "and any spawned MCP server execute as plain subprocesses with NO "
            "filesystem, network, or syscall confinement; the agent is contained "
            "only by the surrounding environment (e.g. the container it runs in). "
            "Use 'auto'/'strict'/'hardened' for kernel-enforced isolation."
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
            "sandbox.tool_network = 'auto' cannot give the run its own private "
            "network: jailed commands share this process's host network, which "
            "hardened does not confine. Run on 'strict' for a private tool "
            "network, or set sandbox.tool_network = 'private' to refuse rather "
            "than run here."
        )
    for hidden, region in unmaskable_exposures(cfg, isolation):
        reporter.err(
            "[agent6] WARNING: jailed commands can READ"
            f" {hidden} -- it sits inside {region}, which they are granted,"
            " and 'hardened' has no mount namespace to mask it out (Landlock"
            " has no deny rules). Provider keys, transcripts, notes and run"
            " history in there are readable by every command this run runs."
            " Use 'strict' to keep them masked under the same grant."
        )
    if isolation in ("strict", "hardened"):
        notes = tool_mount_notes()
        for tool in notes.unreachable:
            reporter.err(
                f"[agent6] WARNING: tool {tool} resolves into a dir that is never"
                " mounted into the jail ($HOME itself, or agent6's private dirs),"
                " so it will not run inside sandboxed commands. Move the target"
                " into its own subdirectory."
            )
        # notes.exposes_home_dir is NOT warned per run: on a normal machine
        # every uv-installed tool in ~/.local/bin points into ~/.local/share,
        # so this fired a dozen times a run and buried the messages that
        # mattered. It is the ordinary state of a dev box, not a surprise --
        # `agent6 check` lists it, where someone is asking.


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


def unmaskable_exposures(cfg: Config, isolation: IsolationLevel) -> tuple[tuple[Path, Path], ...]:
    """`(hidden path, granted region containing it)` pairs this isolation
    cannot mask, hidden-path first. Empty on strict (it masks) and on `none`
    (no jail at all; the blanket unsandboxed warning covers that).

    On `hardened` there is no mount namespace and Landlock has no deny rules,
    so anything inside a granted region -- the workspace, or an extra grant --
    is readable however private it is.
    """
    if isolation != "hardened":
        return ()
    sb = cfg.sandbox
    regions = (
        Path.cwd(),
        *(Path(p) for p in (*sb.extra_read_paths, *sb.extra_write_paths)),
    )
    hidden = (*(Path(p) for p in sb.hide_paths), *private_dirs())
    return tuple((h, region) for region in regions for h in hidden if h.is_relative_to(region))


def check_hide_paths_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal message when an EXPLICIT `[sandbox].hide_paths` entry cannot
    be honored here, else None.

    The same rule the other knobs follow: a default degrades with a warning,
    a value the operator wrote down refuses rather than being silently
    ineffective. `hide_paths` is only ever explicit, so an entry hardened
    cannot mask refuses. The always-hidden private dirs are NOT this: the
    operator granting a region that contains them is a choice they may mean
    (real protection remains -- writes stay confined, seccomp still applies),
    so that is a loud warning instead (`warn_sandbox_gaps`).
    """
    if isolation != "hardened":
        return None  # before reading config: every other level masks
    listed = {Path(p) for p in cfg.sandbox.hide_paths}
    for hidden, region in unmaskable_exposures(cfg, isolation):
        if hidden in listed:
            return (
                f"sandbox.hide_paths lists {str(hidden)!r}, which sits inside"
                f" {str(region)!r} -- a region jailed commands can read. Masking it"
                " needs the mount namespace only 'strict' has. Use strict, drop the"
                " entry, or move one of the two."
            )
    return None


def check_mcp_network_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal when a server EXPLICITLY named a network this host cannot
    give it, else None.

    Per-server, same rule and same vocabulary as `[sandbox].tool_network`, and
    therefore the same guard: a network namespace needs user namespaces, which
    only `strict` has, so `none` and `private` refuse on `hardened` while the
    `auto` default degrades with a warning. Under `none` nothing is confined at all
    and the blanket unsandboxed warning covers it -- the same answer
    `protect_git` and `memory_limit_mb` give, and the same answer the sibling
    knob gives, which this did not until a matrix of the two side by side
    showed them disagreeing.
    """
    if isolation != "hardened":
        return None
    for name, srv in sorted(cfg.mcp.servers.items()):
        if srv.enabled and srv.sandbox is not None and srv.sandbox.network in ("none", "private"):
            return (
                f"MCP server {name!r} sets sandbox.network = {srv.sandbox.network!r},"
                " which needs a network namespace and so the strict isolation; this"
                f" host resolved to {isolation!r}. Use 'auto' to run with a warning,"
                " or 'host' to accept the machine's network."
            )
    return None


def check_network_support(cfg: Config, isolation: IsolationLevel) -> str | None:
    """A refusal message if the network config EXPLICITLY enforces something
    this isolation cannot provide, else None.

    Only jailed commands have a network boundary. ``tool_network =
    "only_explicit_states"`` (singling one tool out) and ``"private"`` (the
    run's own network, with no route off the box) both need a network
    namespace, which only ``strict`` provides. On ``hardened`` we refuse rather than silently
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
            " isolation (network namespaces), but this host supports only"
            " 'hardened'. Use 'auto' or 'host'."
        )
    if sb.tool_network == "private":
        return (
            "sandbox.tool_network = 'private' requires the strict isolation (a"
            " network namespace), but this host supports only 'hardened', where"
            " a jailed command shares this process's network. Use 'auto' to run"
            " with a warning, or 'host' to accept it."
        )
    return None
