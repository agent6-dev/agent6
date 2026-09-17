# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for the run/resume/fork family: start a run, resume a
paused one from its snapshot, or fork a new run off a prior checkpoint."""

from __future__ import annotations

import argparse

from agent6.config.layer import BUILTIN_PRESETS
from agent6.ui.cli._common import (
    _add_budget_flags,
    _add_config_flag,
    _add_sandbox_flags,
    _add_session_id,
    _sub,
)
from agent6.ui.cli.completers import (
    _complete_model_routes,
    _complete_parallel_models,
    _complete_presets,
    _complete_resumable_ids,
    _complete_session_ids,
    _complete_skills,
)


def _add_model_flag(parser: argparse.ArgumentParser) -> None:
    arg = parser.add_argument(
        "--model",
        default="",
        metavar="[PROVIDER/]MODEL",
        help=(
            "Use MODEL for this session, over every config file. A bare MODEL keeps the"
            " role's configured provider; PROVIDER/MODEL names another. `agent6 model <role>"
            " <provider>` lists a provider's ids. run and ask set the worker; plan sets the"
            " planner. The session records the choice; `resume --model` can change it."
        ),
    )
    arg.completer = _complete_model_routes  # type: ignore[attr-defined]


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_p = _sub(sub, "run", help="Work on a coding task in a new session.")
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
            "Start a new run with context from another run, plan, or ask. Context includes its"
            " task, outcome, diff, key events, and plan text when present. Accepts a session id"
            " or unambiguous prefix. With no TASK, a plan source becomes the task. The source"
            " does not change. Use `agent6 fork` to copy an earlier saved turn instead."
        ),
    )
    run_from.completer = _complete_session_ids  # type: ignore[attr-defined]
    run_p.add_argument(
        "--pin",
        dest="pins",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Add an instruction that stays in every model call, even after older context is"
            " shortened. Repeat for more instructions; this is the same as /pin. A /parallel"
            " lane inherits them."
        ),
    )
    run_skill = run_p.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="Add an installed skill's instructions before the task. Repeat for more skills.",
    )
    run_skill.completer = _complete_skills  # type: ignore[attr-defined]
    run_p.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Make the agent split the task into ordered subtasks before editing, then work on"
            " one at a time. It does not ask you to approve the plan. This sets"
            " prompt.decompose to `on` for this run. It can help smaller models with multi-part"
            " tasks but adds work for models that already plan well."
        ),
    )
    run_p.add_argument(
        "--standing",
        default="",
        metavar="GOAL",
        help=(
            "Keep returning to GOAL whenever all other tasks are done or the agent tries to"
            " stop. New tasks take priority. The session can still end when you stop it or it"
            " reaches its budget or iteration limit. workflow.standing_patience can also end"
            " it after repeated unproductive returns; by default, it never does."
        ),
    )
    run_parallel_flag = run_p.add_argument(
        "--parallel",
        default="",
        metavar="N|[PROVIDER/]MODEL,...",
        help=(
            "Fan out isolated lanes: an integer N runs N lanes on the worker model,"
            " a comma-separated list runs one lane per entry (provider/model, or a"
            " model id on the worker's provider). Each lane clones the repo, runs"
            " independently, and lands its own branch; results are auto-compared"
            " and ranked (nothing is merged). Capped by [parallel].max_lanes;"
            " combine with --max-usd for a per-lane budget."
        ),
    )
    run_parallel_flag.completer = _complete_parallel_models  # type: ignore[attr-defined]
    run_profile = run_p.add_argument(
        "--preset",
        default="",
        help=(
            f"Apply a built-in strategy preset ({'/'.join(BUILTIN_PRESETS)}) or a custom"
            " presets.NAME. This overrides any `preset` selected in a config file and values"
            " from the global and per-repository configs. Values from --config FILE and other"
            " command flags still win."
        ),
    )
    run_profile.completer = _complete_presets  # type: ignore[attr-defined]
    _add_model_flag(run_p)
    _add_config_flag(run_p)
    run_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Pause for input after each automatic commit. The prompt accepts /continue"
            " (default), /diff, /cost, /undo, /watch, /mcp, /init, /help, /quit, and /exit."
            " /undo restores the files from before the last message and continues in a fork."
            " /quit and /exit stop without another prompt. Requires a foreground terminal."
        ),
    )
    run_p.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Show the run in the full-screen terminal interface instead of command-line"
            " output. Ctrl+D switches between the conversation and dashboard. Requires a"
            " terminal and cannot be used with -i. You can also start the run from `agent6 tui`."
        ),
    )
    _add_budget_flags(run_p)
    _add_sandbox_flags(run_p)


def _add_resume_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    resume_p = _sub(
        sub, "resume", help="Continue a paused or interrupted session from its saved state."
    )
    _add_session_id(resume_p, _complete_resumable_ids)
    resume_p.add_argument(
        "--steer",
        default="",
        metavar="TEXT",
        help=(
            "Give TEXT to the resumed agent before it next starts work. The TUI follow-up field"
            " uses this option."
        ),
    )
    resume_p.add_argument(
        "--standing",
        default="",
        metavar="GOAL",
        help=(
            "Give this session the standing goal it did not have: it returns to GOAL whenever"
            " all other tasks are done or the agent tries to stop. A session that already has"
            " one keeps it."
        ),
    )
    resume_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Resume even when the saved commit and the session's current commit have diverged."
            " Use only when its agent6/ID ref was rewritten or replaced; later commits made by"
            " the same session do not require this."
        ),
    )
    resume_preset = resume_p.add_argument(
        "--preset",
        default="",
        help=(
            "Use another strategy preset for this continuation. A preset can change any"
            " setting, so it takes effect only when resuming. The session records it and uses"
            " it on later resumes."
        ),
    )
    resume_preset.completer = _complete_presets  # type: ignore[attr-defined]
    _add_model_flag(resume_p)
    _add_config_flag(resume_p)
    resume_p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Keep the session open for another instruction when the agent stops replying,"
            " instead of ending it. This is the same as `run -i`. Requires a foreground"
            " terminal."
        ),
    )
    resume_p.add_argument(
        "--tui",
        action="store_true",
        help="Show the resumed session in the full-screen terminal interface.",
    )
    _add_budget_flags(resume_p)
    _add_sandbox_flags(resume_p)


def _add_fork_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fork_p = _sub(
        sub,
        "fork",
        help=(
            "Copy a session at one of its saved turns, then continue the copy without changing"
            " the source. A run copy gets its own git worktree. A plan or ask copy stays"
            " read-only in the current checkout."
        ),
    )
    _add_session_id(
        fork_p,
        _complete_resumable_ids,
        help_text=("Source session id or unambiguous prefix. Default: newest resumable session."),
    )
    fork_p.add_argument(
        "--standing",
        default="",
        metavar="GOAL",
        help=(
            "Give the fork the standing goal the source did not have: it returns to GOAL"
            " whenever all other tasks are done or the agent tries to stop. A source that"
            " already has one passes it to the fork, which keeps it."
        ),
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
    _add_config_flag(fork_p)
    fork_p.add_argument(
        "--tui",
        action="store_true",
        help="Show the forked session in the full-screen terminal interface.",
    )
    _add_budget_flags(fork_p)
    # A fork without --no-run continues a run, so it is a paid command like the
    # rest and carries the same approval/sandbox overrides.
    _add_sandbox_flags(fork_p)
