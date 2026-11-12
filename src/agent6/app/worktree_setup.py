# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the environment of a worktree before its first turn.

A `run --worktree` worktree, a fork's worktree, a child's clone and a machine state's clone
hold tracked files only. `[git.worktree].setup_command` runs once in each, through an injected
runner using the worktree's command policy and jail. A session's manifest records that its
worktree is set up. With no command configured, the `## Setup command` block of AGENTS.md
supplies it from the worktree's base commit, never a model-written machine chain tip.
A gate whose first word is a path git does not track cannot run in such a worktree until
a setup command builds it: with none, the start
refuses before anything is created.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shlex
from collections.abc import Sequence
from typing import Protocol

from agent6 import kinds
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.sandbox import tool_paths
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import target_policy, tool_errors, verify_infer
from agent6.vcs.git import read as git_read

# The lines of the command's output a failure's summary carries.
_TAIL_LINES = 20


def _output_tail(res: kinds.CommandResult) -> list[str]:
    """Return the last lines of the command's output, stdout before stderr."""
    lines = [line for part in (res.stdout, res.stderr) for line in part.rstrip().splitlines()]
    return lines[-_TAIL_LINES:]


def infer_if_unset(
    cfg: Config, origin: pathlib.Path, *, reporter: app_reporter.Reporter, at: str | None = None
) -> Config:
    """Return the config with the setup command AGENTS.md states, when none is configured.

    Read once, in the origin, and carried in the config, so the worktree and every child
    run the command the start named.

    Args:
        cfg: The session's config.
        origin: The checkout whose AGENTS.md is read.
        reporter: Receives the line naming what was inferred.
        at: Read this commit's AGENTS.md instead of the checkout's copy.

    Returns:
        The config, its setup command set when the `## Setup command` block gives one.
    """
    if cfg.git.worktree.setup_command:
        return cfg
    agents_md = verify_infer.read_agents_md(origin, cfg.sandbox.hide_paths, at=at)
    argv = verify_infer.setup_from_agents_md(agents_md)
    if argv is None:
        return cfg
    reporter.note(
        f"setup_command not set; inferred from AGENTS.md: {shlex.join(argv)}\n"
        "         (pin it with git.worktree.setup_command in your per-repo config)"
    )
    return cfg.with_setup_command(argv)


def _gate_path(gate: Sequence[str]) -> str:
    """Return the gate's first word as a path into the worktree; "" for a name or a way out."""
    first = gate[0] if gate else ""
    if "/" not in first or pathlib.Path(first).is_absolute():
        return ""
    path = os.path.normpath(first)
    return "" if path.startswith("..") else path


def _no_setup_text(path: str, worktree: str) -> str:
    """Return the refusal naming the gate's path and the two ways to state a setup command.

    Args:
        path: The gate's first word.
        worktree: The worktree, for the message ("a new worktree").

    Returns:
        The refusal: the reason, a bullet per way, and a line on working offline.
    """
    return (
        f"the verify gate runs `{path}`, a path git does not track, so it is absent in"
        f" {worktree}, which holds tracked files only, and no setup command builds it there.\n"
        "  - set one for this repository:  agent6 config set --repo git.worktree.setup_command"
        """ '["uv", "sync"]'\n"""
        "  - or commit it in AGENTS.md: a `## Setup command` heading over a fenced block\n"
        "  it runs in the run's own sandbox: with no route off the machine it has to work"
        " offline (docs/security.md has the recipe)"
    )


def origin_gate(cfg: Config, origin: pathlib.Path) -> tuple[str, ...]:
    """Return the gate the config pins or the origin's own files give, with no model call."""
    if cfg.harness.verify.command or not cfg.harness.verify.infer:
        return cfg.harness.verify.command
    hide = cfg.sandbox.hide_paths
    agents_md = verify_infer.read_agents_md(origin, hide)
    return verify_infer.infer_verify_command(
        origin, agents_md, available=tool_paths.jail_resolves, hide=hide
    ).argv


def gate_refusal(
    cfg: Config, origin: pathlib.Path, gate: Sequence[str], *, worktree: str, at: str | None = None
) -> str | None:
    """Return why a session could never run its gate in a new worktree, or None.

    Args:
        cfg: The session's config.
        origin: The checkout the worktree is cut from.
        gate: The gate the session will be judged by.
        worktree: The worktree, for the message ("a new worktree").
        at: The commit the worktree is cut from; None reads HEAD's AGENTS.md and files.

    Returns:
        The refusal when the gate's first word is a path the commit does not hold and no
        setup command is configured or stated in AGENTS.md; None otherwise,
        and when commands are withheld, which leaves the run gateless.
    """
    path = _gate_path(gate)
    if not path or cfg.sandbox.run_commands == "deny" or cfg.git.worktree.setup_command:
        return None
    agents_md = verify_infer.read_agents_md(origin, cfg.sandbox.hide_paths, at=at)
    if verify_infer.setup_from_agents_md(agents_md) is not None:
        return None
    if path in git_read.tree_paths(origin, at or "HEAD"):
        return None
    return _no_setup_text(gate[0], worktree)


def session_needs_setup(session_dir: pathlib.Path) -> bool:
    """Return whether the session's own worktree or clone still needs setup.

    The manifest records a setup that left the worktree ready, so a resume after one skips
    the command and a fork's new worktree runs it.

    Args:
        session_dir: The session directory.

    Returns:
        Whether setup is still due; False without a worktree or clone of its own, or without
        a readable manifest.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return False
    isolated = manifest.worktree is not None or manifest.parallel is not None
    return isolated and not manifest.worktree_set_up


@dataclasses.dataclass(frozen=True, slots=True)
class Prepared:
    """What preparing a worktree found.

    Attributes:
        problem: What is not ready, the output tail under it; None when the worktree is.
        ran: Whether the setup command ran and left the worktree ready.
        result: The setup outcome, or None when no command ran.
    """

    problem: str | None
    ran: bool
    result: kinds.CommandResult | None = None


class SetupRunner(Protocol):
    """The command policy and execution that worktree preparation needs."""

    @property
    def isolation(self) -> kinds.IsolationLevel:
        """Return the worktree's isolation level."""
        ...

    def command_policy(self) -> str:
        """Return whether commands are allowed, withheld or require approval."""
        ...

    def run_setup(self, argv: tuple[str, ...]) -> kinds.CommandResult:
        """Run setup under the worktree's command policy and jail.

        Args:
            argv: The setup command.

        Returns:
            The outcome, a nonzero exit included.
        """
        ...


def prepare(
    cfg: Config,
    root: pathlib.Path,
    *,
    runner: SetupRunner,
    reporter: app_reporter.Reporter,
) -> Prepared:
    """Run the worktree's setup command, then check that the gate's path exists there.

    Args:
        cfg: The session's config, its gate resolved.
        root: The worktree.
        runner: The worktree's command policy and jailed setup runner.
        reporter: Receives the command line and its output.

    Returns:
        What is not ready, and whether the command ran; withheld commands run nothing and
        leave no gate to check.
    """
    prepared = setup(cfg, runner=runner, reporter=reporter)
    if prepared.problem is not None or runner.command_policy() == "deny":
        return prepared
    argv = cfg.git.worktree.setup_command
    result = prepared.result
    gate = cfg.harness.verify.command
    path = _gate_path(gate)
    if not path or (root / path).exists():
        return prepared
    if argv:
        return Prepared(
            f"the setup command ran ({shlex.join(argv)}), and the verify gate's `{gate[0]}` is"
            f" still missing in {root}",
            ran=False,
            result=result,
        )
    return Prepared(_no_setup_text(gate[0], f"this worktree ({root})"), ran=False)


def setup(cfg: Config, *, runner: SetupRunner, reporter: app_reporter.Reporter) -> Prepared:
    """Run setup without the session-specific gate-path check.

    Args:
        cfg: The config naming the setup command.
        runner: The worktree's command policy and jailed setup runner.
        reporter: Receives the command line, and the last lines of its output when it succeeds.

    Returns:
        The command result and any setup failure.
    """
    argv = cfg.git.worktree.setup_command
    if runner.command_policy() == "deny":
        if argv:
            line = shlex.join(argv)
            reporter.note(f"commands are withheld, and the setup command is one ({line}): not run")
        return Prepared(None, ran=False)
    result: kinds.CommandResult | None = None
    if argv:
        line = shlex.join(argv)
        reporter.note(f"setup command: {line}")
        try:
            result = runner.run_setup(argv)
        except tool_errors.ToolError as exc:
            return Prepared(f"the setup command did not run ({line}): {exc}", ran=False)
        tail = _output_tail(result)
        if result.returncode != 0:
            how = "could not start" if result.exec_failed else f"failed (exit {result.returncode})"
            problem = "\n".join([f"the setup command {how}: {line}", *(f"    {t}" for t in tail)])
            return Prepared(
                problem + _no_route_line(cfg, runner.isolation), ran=False, result=result
            )
        for shown in tail:
            reporter.err(f"         {shown}")
        reporter.note(f"setup command finished in {result.duration_s:.1f}s")
    return Prepared(None, ran=bool(argv), result=result)


def _no_route_line(cfg: Config, isolation: kinds.IsolationLevel) -> str:
    """Return the line a failed setup command ends on where commands have no route out.

    Args:
        cfg: The config holding `sandbox.network`.
        isolation: The worktree's isolation level; under `partial` or `none`, commands share
            the host's network.

    Returns:
        The line, "" when commands share the host's network.
    """
    if target_policy.resolve_network(cfg, isolation) == "host":
        return ""
    return (
        "\n  the command ran with no route off this machine (sandbox.network), so it has to"
        " work offline. See docs/security.md."
    )
