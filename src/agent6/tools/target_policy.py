# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve local execution controls without probing or granting transport access."""

from __future__ import annotations

from typing import Literal

from agent6 import kinds
from agent6.config import Config


def jail_home_kind(
    config: Config, isolation: kinds.IsolationLevel
) -> Literal["tmp", "cache", "host"]:
    """Return which HOME a command gets: `auto` is each level's own.

    `tmp` is `full`'s private tmpfs, `cache` is `partial`'s persistent directory, `host` is
    the operator's own HOME where nothing confines the command.

    Args:
        config: The run's config.
        isolation: The resolved isolation level.

    Returns:
        `tmp`, `cache` or `host`.
    """
    word = config.sandbox.home
    if word != "auto":
        return word
    if isolation == "full":
        return "tmp"
    return "cache" if isolation == "partial" else "host"


def resolve_network(
    config: Config, isolation: kinds.IsolationLevel, *, override: kinds.NetworkMode | None = None
) -> kinds.NetworkMode:
    """Return the network a child gets.

    Clamped to what the level can provide: only `full` has namespaces, so everywhere else the
    child shares this process's network and the policy says so (preflight has refused an
    explicit setting it cannot honour and warned about an automatic one).

    Args:
        config: The run's config.
        isolation: The resolved isolation level.
        override: A caller's own answer; an MCP server's reachability is the operator's
            per-server choice.

    Returns:
        "host" outside `full`, else the override when given, else "host" under
        `network = "host"`, else "session".
    """
    if isolation != "full":
        return "host"
    if override is not None:
        return override
    return "host" if config.sandbox.network == "host" else "session"


def effective_controls(
    config: Config, isolation: kinds.IsolationLevel
) -> tuple[kinds.EffectiveControl, ...]:
    """Describe every local sandbox leaf without starting or probing a launcher.

    Resolution is not a successful enforcement probe. Path rows describe the grant policy,
    not the existence of each configured path; startup diagnostics retain skipped grants.

    Args:
        config: The configured sandbox leaves.
        isolation: The resolved local isolation level.

    Returns:
        One immutable row per sandbox leaf.
    """
    sb = config.sandbox
    full, confined = isolation == "full", isolation != "none"
    file_owner: kinds.EnforcementOwner = "agent6" if confined else "host file tools"
    rows: list[kinds.EffectiveControl] = []

    def row(
        name: str,
        effective: kinds.ControlValue,
        behaviour: str,
        owner: kinds.EnforcementOwner = "agent6",
        reason: str = "",
    ) -> None:
        rows.append(
            kinds.EffectiveControl(
                leaf=f"sandbox.{name}",
                configured=getattr(sb, name),
                effective=effective,
                behaviour=behaviour,
                owner=owner,
                reason=reason,
            )
        )

    row(
        "isolation",
        isolation,
        {
            "full": "namespaces, seccomp and supported Landlock",
            "partial": "Landlock and seccomp, no namespaces",
            "none": "no command confinement",
        }[isolation],
        "agent6" if confined else "unsupported",
    )
    row(
        "network",
        resolve_network(config, isolation),
        "run-private network" if full and sb.network != "host" else "host network",
        "agent6" if full else "unsupported",
        "session networking requires full isolation" if not full and sb.network != "host" else "",
    )
    row("home", jail_home_kind(config, isolation), "command HOME selection")
    row(
        "protect_git",
        "on" if full and config.protects_git else "off",
        "read-only command mount and file-tool protection"
        if full and config.protects_git
        else "file-tool protection only"
        if config.protects_git
        else "protection disabled",
        "agent6" if full or not config.protects_git else "host file tools",
        "command-side git protection requires full isolation"
        if not full and config.protects_git
        else "",
    )
    row("run_commands", sb.run_commands, "command permission; session approvals may change it")
    row(
        "fetch_approved_hosts",
        sb.fetch_approved_hosts,
        "approved hosts bypass fetch approval; other hosts require approval"
        if full and sb.network != "host"
        else "fetch withheld on host networking",
    )
    row(
        "memory_limit_mib",
        sb.memory_limit_mib,
        "per-process RLIMIT_DATA in MiB" if sb.memory_limit_mib else "no memory cap",
    )
    for name in ("extra_read_paths", "extra_write_paths"):
        row(
            name,
            getattr(sb, name),
            "resolved grants for commands and host file tools"
            if confined
            else "host file-tool grants only; commands are unconfined",
            file_owner,
        )
    row(
        "extra_device_paths",
        sb.extra_device_paths,
        "device binds and Landlock grants"
        if full
        else "Landlock device grants"
        if confined
        else "no command device restriction",
        "agent6" if confined else "unsupported",
    )
    row(
        "hide_paths",
        sb.hide_paths,
        "command masks and host file-tool denials"
        if full
        else "host file-tool denials; exposed command grants refuse"
        if confined
        else "host file-tool denials only; commands are unconfined",
        file_owner,
    )
    return tuple(sorted(rows, key=lambda control: control.leaf))
