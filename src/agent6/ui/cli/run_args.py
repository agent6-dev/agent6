# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the `run`, `resume` and `fork` parsers."""

from __future__ import annotations

import argparse

from agent6.config import layer_presets, model
from agent6.ui.cli import common, completers

# `run -i` and `resume -i` share one hook (`app/execution.py`), so they share one help.
_INTERACTIVE_HELP = (
    "Pause for input after each automatic commit; /help lists the prompt's commands."
    " Requires a foreground terminal."
)


def _add_model_flag(parser: argparse.ArgumentParser) -> None:
    """Add `--model` with route completion."""
    arg = parser.add_argument(
        "--model",
        default="",
        metavar="[PROVIDER/]MODEL",
        help=(
            "Use MODEL for this session, over every config file. A bare MODEL keeps the"
            " configured provider; PROVIDER/MODEL names another."
        ),
    )
    arg.completer = completers._complete_model_routes  # type: ignore[attr-defined]


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `run`."""
    run_p = common._sub(sub, "run", help="Work on a coding task in a new session.")
    run_p.add_argument(
        "task",
        nargs="?",
        default="",
        help=(
            "Task for the agent, usually in quotes. On a terminal, omit it to choose whether"
            " to run the newest plan."
        ),
    )
    run_p.add_argument(
        "--session-id", default="", help="Use this id for the new session. Default: generate one."
    )
    run_from = run_p.add_argument(
        "--from",
        dest="seed_from",
        default="",
        metavar="SESSION_ID",
        help=(
            "Start with context from another run, plan, or ask: its task, outcome, diff and"
            " plan. With no TASK, a plan source becomes the task."
        ),
    )
    run_from.completer = completers._complete_session_ids  # type: ignore[attr-defined]
    run_p.add_argument(
        "--pin",
        dest="pins",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Keep TEXT in every model call, even after older context is shortened."
            " Repeatable; the same as /pin."
        ),
    )
    run_skill = run_p.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="Add an installed skill's instructions before the task. Repeatable.",
    )
    run_skill.completer = completers._complete_skills  # type: ignore[attr-defined]
    run_p.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Make the agent split the task into ordered subtasks before editing, then work on"
            " one at a time. Sets harness.prompt.decompose to `on` for this run."
        ),
    )
    run_p.add_argument(
        "--standing",
        default="",
        metavar="GOAL",
        help=(
            "Return to GOAL whenever all other tasks are done or the agent tries to stop."
            " The session ends when you stop it or it reaches its budget or turn limit."
        ),
    )
    run_parallel_flag = run_p.add_argument(
        "--parallel",
        default="",
        metavar="N|SELECTOR,...",
        help=(
            "Run the task in isolated lanes, then join the lane [harness.parallel].judges"
            " rank first onto this run's branch. N runs N lanes on the worker model; a list"
            " runs one lane per SELECTOR (a role name, or [provider/]model)."
        ),
    )
    run_parallel_flag.completer = completers._complete_parallel_models  # type: ignore[attr-defined]
    run_p.add_argument(
        "--worktree",
        action="store_true",
        help=(
            "Work in a new git worktree at HEAD's commit, under [git.worktree].base_dir;"
            " uncommitted changes stay in this checkout. [git.worktree].setup_command builds"
            " what git does not track. Land the work with `agent6 sessions merge ID`."
        ),
    )
    run_profile = run_p.add_argument(
        "--preset",
        default=argparse.SUPPRESS,
        type=model.preset_name,
        help=(
            f"Apply a built-in strategy preset ({'/'.join(layer_presets.BUILTIN_PRESETS)})"
            " or a custom presets.NAME, in place of the configured agent6.preset."
            " Use none to select no preset."
        ),
    )
    run_profile.completer = completers._complete_presets  # type: ignore[attr-defined]
    _add_model_flag(run_p)
    common._add_config_flag(run_p)
    run_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=_INTERACTIVE_HELP,
    )
    run_p.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Show the run in the full-screen terminal interface."
            " Without a terminal the run continues in CLI mode; cannot be used with -i."
        ),
    )
    common._add_budget_flags(run_p)
    common._add_sandbox_flags(run_p)


def _add_resume_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `resume`."""
    resume_p = common._sub(
        sub, "resume", help="Continue a paused or interrupted session from its saved state."
    )
    common._add_session_id(resume_p, completers._complete_resumable_ids)
    resume_p.add_argument(
        "--steer",
        default="",
        metavar="TEXT",
        help=("Give TEXT to the resumed agent before it next starts work."),
    )
    resume_p.add_argument(
        "--merge",
        default="",
        metavar="BRANCH",
        dest="merge_from",
        help=(
            "Merge BRANCH into the run's work before its first turn; a conflict's markers stay"
            " in the files for the run to resolve."
        ),
    )
    resume_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Resume even when the saved commit and the session's current commit have diverged,"
            " or when its agent6/ID branch holds a commit the session did not make (the branch"
            " is then left where it is)."
        ),
    )
    resume_preset = resume_p.add_argument(
        "--preset",
        default=argparse.SUPPRESS,
        type=model.preset_name,
        help=("Use another strategy preset from this resume on. Use none to select no preset."),
    )
    resume_preset.completer = completers._complete_presets  # type: ignore[attr-defined]
    _add_model_flag(resume_p)
    common._add_config_flag(resume_p)
    resume_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=_INTERACTIVE_HELP,
    )
    resume_p.add_argument(
        "--tui",
        action="store_true",
        help="Show the resumed session in the full-screen terminal interface.",
    )
    common._add_budget_flags(resume_p)
    common._add_sandbox_flags(resume_p)


def _add_fork_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `fork`."""
    fork_p = common._sub(
        sub,
        "fork",
        help=(
            "Copy a session at one of its saved turns, then continue the copy without changing"
            " the source. A run copy gets its own git worktree. A plan or ask copy stays in"
            " the current checkout, without edit tools or automatic commits."
        ),
    )
    common._add_session_id(
        fork_p,
        completers._complete_resumable_ids,
        help_text=("Source session id or unambiguous prefix. Default: newest resumable session."),
    )
    fork_p.add_argument(
        "--at-turn",
        type=int,
        default=None,
        metavar="N",
        dest="at_turn",
        help="Use the checkpoint saved for turn N. Default: latest checkpoint.",
    )
    fork_p.add_argument(
        "--session-id",
        default="",
        dest="new_session_id",
        help="Use this id for the new session. Default: generate one.",
    )
    fork_p.add_argument(
        "--steer",
        default="",
        metavar="TEXT",
        help=(
            "Give TEXT to the forked agent before it starts work. Cannot be used with --no-run;"
            " later use `agent6 resume ID --steer TEXT`."
        ),
    )
    fork_p.add_argument(
        "--no-run",
        action="store_true",
        help="Create the copy without continuing it. Start it later with `agent6 resume`.",
    )
    common._add_config_flag(fork_p)
    fork_p.add_argument(
        "--tui",
        action="store_true",
        help="Show the forked session in the full-screen terminal interface.",
    )
    common._add_budget_flags(fork_p)
    # A fork without --no-run continues a run, so it carries the same sandbox overrides.
    common._add_sandbox_flags(fork_p)
