# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for the run/resume/fork family: start a run, resume a
paused one from its snapshot, or fork a new run off a prior checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6.ui.cli._common import _add_budget_flags, _add_sandbox_flags, _sub
from agent6.ui.cli.completers import (
    _complete_parallel_models,
    _complete_plan_run_ids,
    _complete_profiles,
    _complete_run_ids,
    _complete_skills,
)


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_p = _sub(sub, "run", help="Run the single-loop agent on a task.")
    run_p.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task description (in quotes). Omit when using --continue.",
    )
    run_p.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    run_profile = run_p.add_argument(
        "--profile",
        default="",
        help="Config profile preset (quick/standard/ultra/paranoid or a custom"
        " [profiles.<name>]). Overrides the top-level `profile` key; your explicit"
        " settings win.",
    )
    run_profile.completer = _complete_profiles  # type: ignore[attr-defined]
    run_p.add_argument(
        "--config",
        type=Path,
        # SUPPRESS (not None): a subparser default would otherwise clobber a
        # top-level `agent6 --config FILE <cmd>` back to None. With SUPPRESS the
        # subparser only sets `config` when --config is given AFTER the
        # subcommand, so both `agent6 --config F run` and `agent6 run --config F`
        # work; the top-level --config supplies the always-present default.
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Explicit config file (layered over global + repo configs).",
    )
    run_p.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help=(
            "Resume the most recent run for this cwd"
            " instead of starting a new one. Mutually exclusive with a"
            " task argument."
        ),
    )
    run_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "REPL mode: after each successful auto-commit, prompt on stdin for"
            " one of /continue (default), /diff, /cost, /undo (git revert HEAD),"
            " /watch, /mcp, /init, /help, /quit. Requires a TTY."
        ),
    )
    run_p.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Open the full-screen TUI on the run (the conversation view; Ctrl+D"
            " toggles the dashboard) instead of the default headless CLI stream."
            " Needs a TTY; mutually exclusive with -i."
            " (Or run `agent6 tui` and start the run from there.)"
        ),
    )
    run_from_plan = run_p.add_argument(
        "--from-plan",
        default="",
        metavar="RUN_ID",
        help=(
            "Use the plan.md from a prior `agent6 plan` run (resolved"
            " under the per-repo run-state dir, exact or unambiguous prefix) as the"
            " task description. Mutually exclusive with a positional task."
        ),
    )
    run_from_plan.completer = _complete_plan_run_ids  # type: ignore[attr-defined]
    run_p.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Plan-first: the agent lays the task out as ordered DAG subtasks"
            " (add_task) before editing, then works them one at a time -- a plan it"
            " builds and follows on its own, no approval step, populating the task"
            " graph. Same as setting [prompt].decompose for this run. Helps on"
            " multi-part tasks and smaller models; a capable model decomposes"
            " implicitly, so measure before leaving it on."
        ),
    )
    run_skill = run_p.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="Prepend an installed skill's instructions to the task (repeatable).",
    )
    run_skill.completer = _complete_skills  # type: ignore[attr-defined]
    run_parallel_flag = run_p.add_argument(
        "--parallel",
        default="",
        metavar="N|m1,m2,...",
        help=(
            "Fan out isolated lanes: an integer N runs N lanes on the worker model,"
            " a comma-separated model list runs one lane per model. Each lane clones"
            " the repo, runs independently, and lands its own branch; results are"
            " auto-compared and ranked (nothing is merged). Capped by"
            " [parallel].max_lanes; combine with --max-usd for a per-lane budget."
        ),
    )
    run_parallel_flag.completer = _complete_parallel_models  # type: ignore[attr-defined]
    _add_budget_flags(run_p)
    _add_sandbox_flags(run_p)


def _add_resume_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    resume_p = _sub(sub, "resume", help="Resume a paused run from its snapshot.")
    resume_run = resume_p.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id under the per-repo run-state dir (omit for the most recent run).",
    )
    resume_run.completer = _complete_run_ids  # type: ignore[attr-defined]
    resume_p.add_argument(
        "--config",
        type=Path,
        # SUPPRESS (not None): a subparser default would otherwise clobber a
        # top-level `agent6 --config FILE <cmd>` back to None. With SUPPRESS the
        # subparser only sets `config` when --config is given AFTER the
        # subcommand, so both `agent6 --config F run` and `agent6 run --config F`
        # work; the top-level --config supplies the always-present default.
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Explicit config file (layered over global + repo configs).",
    )
    resume_p.add_argument(
        "--force-resume",
        action="store_true",
        help="Resume even if the workspace HEAD diverged from the run's last snapshot "
        "(a rebase, reset, or a commit on another line; plain forward movement resumes "
        "without this flag).",
    )
    resume_p.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the headless stream (like `run --tui`).",
    )
    resume_p.add_argument(
        "--steer",
        default="",
        metavar="TEXT",
        help=(
            "Inject TEXT as an operator steering instruction at the resumed"
            " session's first safe boundary (the TUI composer bar's follow-up"
            " uses this)."
        ),
    )
    _add_budget_flags(resume_p)
    _add_sandbox_flags(resume_p)


def _add_fork_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fork_p = _sub(
        sub,
        "fork",
        help=(
            "Clone a run, rolled back to a checkpoint, into a NEW run and continue"
            " it (the source run is never mutated)."
        ),
    )
    fork_src = fork_p.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Source run id or unambiguous prefix to fork from (omit for the most recent run).",
    )
    fork_src.completer = _complete_run_ids  # type: ignore[attr-defined]
    fork_p.add_argument(
        "--at-turn",
        type=int,
        default=None,
        metavar="N",
        dest="at_turn",
        help="Checkpoint turn to fork from (default: the latest checkpoint).",
    )
    fork_p.add_argument(
        "--run-id",
        default="",
        dest="new_run_id",
        help="Explicit id for the new (forked) run (default: generate one).",
    )
    fork_p.add_argument(
        "--no-run",
        action="store_true",
        help="Only create the fork dir; do not continue it (resume it later).",
    )
    fork_p.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Explicit config file (layered over global + repo configs).",
    )
    fork_p.add_argument(
        "--tui",
        action="store_true",
        help="Open the full-screen TUI instead of the headless stream (like `run --tui`).",
    )
    _add_budget_flags(fork_p)
