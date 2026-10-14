# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the environment of an isolated workspace before its first turn.

A `run --worktree` worktree, a fork's worktree, a lane's clone and a machine state's clone
hold tracked files only. `[parallel].setup_command` runs once in each, through the session's
own dispatcher, so it gets the jail every command of that run gets. A session's manifest
records that its workspace is set up. With no command configured, the `## Setup command`
block of the origin's AGENTS.md supplies it. A gate whose first word is a path git does not
track cannot run in such a workspace until a setup command builds it: with none, the start
refuses before anything is created.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shlex
from collections.abc import Sequence

from agent6 import git_ops, verify_infer
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import dispatch, results
from agent6.tools import errors as tools_errors

# The lines of the command's output a failure's summary carries.
_TAIL_LINES = 20


def _output_tail(res: results.ExecResult) -> list[str]:
    """Return the last lines of the command's output, stdout before stderr."""
    lines = [line for part in (res.stdout, res.stderr) for line in part.rstrip().splitlines()]
    return lines[-_TAIL_LINES:]


def infer_if_unset(cfg: Config, origin: pathlib.Path, *, reporter: app_reporter.Reporter) -> Config:
    """Return the config with the setup command AGENTS.md states, when none is configured.

    Read once, in the origin, and carried in the config, so the workspace and every lane
    run the command the start named.

    Args:
        cfg: The session's config.
        origin: The checkout whose AGENTS.md is read.
        reporter: Receives the line naming what was inferred.

    Returns:
        The config, its setup command set when the `## Setup command` block gives one.
    """
    if cfg.parallel.setup_command:
        return cfg
    agents_md = verify_infer.read_agents_md(origin, cfg.sandbox.hide_paths)
    argv = verify_infer.setup_from_agents_md(agents_md)
    if argv is None:
        return cfg
    reporter.note(
        f"setup_command not set; inferred from AGENTS.md: {shlex.join(argv)}\n"
        "         (pin it with parallel.setup_command in your per-repo config)"
    )
    return cfg.with_setup_command(argv)


def _gate_path(gate: Sequence[str]) -> str:
    """Return the gate's first word as a path into the workspace; "" for a name or a way out."""
    first = gate[0] if gate else ""
    if "/" not in first or pathlib.Path(first).is_absolute():
        return ""
    path = os.path.normpath(first)
    return "" if path.startswith("..") else path


def _no_setup_text(path: str, workspace: str) -> str:
    """Return the refusal naming the gate's path and the two ways to state a setup command."""
    return (
        f"the verify gate runs `{path}`, a path git does not track, so it is absent in"
        f" {workspace}, which holds tracked files only, and no setup command builds it there.\n"
        "  - set one for this repository:  agent6 config set --repo parallel.setup_command"
        """ '["uv", "sync"]'\n"""
        "  - or state it in AGENTS.md: a `## Setup command` heading over a fenced block"
    )


def origin_gate(cfg: Config, origin: pathlib.Path) -> tuple[str, ...]:
    """Return the gate the config pins or the origin's own files give, with no model call."""
    if cfg.harness.verify_command or not cfg.harness.verify_infer:
        return cfg.harness.verify_command
    hide = cfg.sandbox.hide_paths
    agents_md = verify_infer.read_agents_md(origin, hide)
    return verify_infer.infer_verify_command(origin, agents_md, hide=hide).argv


def gate_refusal(
    cfg: Config, origin: pathlib.Path, gate: Sequence[str], *, workspace: str
) -> str | None:
    """Return why a session could never run its gate in a new workspace, or None.

    Args:
        cfg: The session's config.
        origin: The checkout the workspace is cut from.
        gate: The gate the session will be judged by.
        workspace: The workspace, for the message ("a new worktree").

    Returns:
        The refusal when the gate's first word is a path in the checkout that git does not
        track and no setup command is configured or stated in AGENTS.md; None otherwise,
        and when commands are withheld, which leaves the run gateless.
    """
    path = _gate_path(gate)
    if not path or cfg.sandbox.run_commands == "no" or cfg.parallel.setup_command:
        return None
    agents_md = verify_infer.read_agents_md(origin, cfg.sandbox.hide_paths)
    if verify_infer.setup_from_agents_md(agents_md) is not None:
        return None
    if path in git_ops.tracked_files(origin):
        return None
    return _no_setup_text(gate[0], workspace)


def session_needs_setup(session_dir: pathlib.Path) -> bool:
    """Return whether the session's own worktree or clone has not run its setup command.

    The manifest holds the state, so a resume never runs the command again and a fork's new
    worktree does.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return False
    isolated = manifest.worktree is not None or manifest.parallel is not None
    return isolated and not manifest.workspace_set_up


@dataclasses.dataclass(frozen=True, slots=True)
class Prepared:
    """What preparing a workspace found.

    Attributes:
        problem: What is not ready, the output tail under it; None when the workspace is.
        ran: Whether the setup command ran and left the workspace ready.
    """

    problem: str | None
    ran: bool


def prepare(
    cfg: Config,
    root: pathlib.Path,
    *,
    dispatcher: dispatch.ToolDispatcher,
    reporter: app_reporter.Reporter,
) -> Prepared:
    """Run the workspace's setup command, then check that the gate's path exists there.

    Args:
        cfg: The session's config, its gate resolved.
        root: The workspace.
        dispatcher: The session's dispatcher, whose jail the command runs in.
        reporter: Receives the command line and its output.

    Returns:
        What is not ready, and whether the command ran; withheld commands run nothing and
        leave no gate to check.
    """
    argv = cfg.parallel.setup_command
    if dispatcher.command_policy() == "no":
        if argv:
            line = shlex.join(argv)
            reporter.note(f"commands are withheld, and the setup command is one ({line}): not run")
        return Prepared(None, ran=False)
    if argv and (problem := run(argv, dispatcher=dispatcher, reporter=reporter)) is not None:
        return Prepared(problem, ran=False)
    gate = cfg.harness.verify_command
    path = _gate_path(gate)
    if not path or (root / path).exists():
        return Prepared(None, ran=bool(argv))
    if argv:
        return Prepared(
            f"the setup command ran ({shlex.join(argv)}), and the verify gate's `{gate[0]}` is"
            f" still missing in {root}",
            ran=False,
        )
    return Prepared(_no_setup_text(gate[0], f"this workspace ({root})"), ran=False)


def run(
    argv: tuple[str, ...], *, dispatcher: dispatch.ToolDispatcher, reporter: app_reporter.Reporter
) -> str | None:
    """Run the setup command in the session's jail and say what failed.

    Args:
        argv: The setup command.
        dispatcher: The session's dispatcher, whose jail the command runs in.
        reporter: Receives the command line and its output.

    Returns:
        None when the command exited 0, else the failure, the output tail under it.
    """
    line = shlex.join(argv)
    reporter.note(f"setup command: {line}")
    try:
        res = dispatcher.run_setup(argv)
    except tools_errors.ToolError as exc:
        return f"the setup command did not run ({line}): {exc}"
    tail = _output_tail(res)
    if res.returncode != 0:
        how = "could not start" if res.exec_failed else f"failed (exit {res.returncode})"
        return "\n".join([f"the setup command {how}: {line}", *(f"    {t}" for t in tail)])
    for shown in tail:
        reporter.err(f"         {shown}")
    reporter.note(f"setup command finished in {res.duration_s:.1f}s")
    return None
