# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for `plan` and `ask`: alternate single-loop modes (planning-
only, Q&A) alongside the main `run`, each with its own default-verb
subcommand tree (see `_inject_default_verb`)."""

from __future__ import annotations

import argparse
import os

from agent6.ui.cli._common import (
    _add_budget_flags,
    _add_config_flag,
    _add_sandbox_flags,
    _add_session_id,
    _sub,
)
from agent6.ui.cli._run_args import _add_model_flag
from agent6.ui.cli.completers import (
    _complete_plan_session_ids,
    _complete_presets,
    _complete_session_ids,
)


def _add_plan_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plan_p = _sub(
        sub,
        "plan",
        help=(
            "Write a plan without editing repository files or making commits; save it as plan.md."
            ' `agent6 plan "TASK"` and `agent6 plan run "TASK"` are the same. A planning'
            " session asks before running commands even when the config approves them"
            " automatically; --auto-approve approves them for the session. Execute the plan"
            " with `agent6 run --from PLAN_ID`. Read or edit it with `agent6 plan show PLAN_ID`"
            " or `agent6 plan edit PLAN_ID`."
        ),
    )
    # `plan <task>` is the bare planning run; `plan show/edit <id>` inspect a
    # prior plan. `run` is the implicit default verb injected by
    # `_inject_default_verb` when the first token isn't a known plan verb, so
    # `plan "fix the bug"` and `plan run "fix the bug"` are the same.
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True, metavar="<subcommand>")
    plan_run = _sub(plan_sub, "run", help="Create a plan for a task.")
    plan_run.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task to plan, usually in quotes. Required.",
    )
    plan_run.add_argument(
        "--session-id", default="", help="Use this id for the new session. Default: generate one."
    )
    plan_profile = plan_run.add_argument(
        "--preset",
        default="",
        help="Apply a strategy preset. `agent6 config presets` lists the choices.",
    )
    plan_profile.completer = _complete_presets  # type: ignore[attr-defined]
    _add_model_flag(plan_run)
    _add_config_flag(plan_run)
    plan_run.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Show the planning session in the full-screen terminal interface instead of"
            " command-line output. Ctrl+D switches between the conversation and dashboard."
            " Requires a terminal. You can also start the plan from `agent6 tui`."
        ),
    )
    _add_budget_flags(plan_run)
    _add_sandbox_flags(plan_run)
    plan_show = _sub(plan_sub, "show", help="Print plan.md from a saved planning session.")
    _add_session_id(
        plan_show,
        _complete_plan_session_ids,
        help_text="Planning session id or unambiguous prefix. Default: newest plan.",
    )
    plan_edit = _sub(
        plan_sub,
        "edit",
        help=(
            "Open plan.md from a saved planning session with $EDITOR."
            f" Resolved command: {os.environ.get('EDITOR', '') or 'vi'}."
        ),
    )
    _add_session_id(
        plan_edit,
        _complete_plan_session_ids,
        help_text="Planning session id or unambiguous prefix. Default: newest plan.",
    )


def _add_ask_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ask_p = _sub(
        sub,
        "ask",
        help=(
            "Ask for a prose answer without allowing file edits or commits; no repository is"
            ' required. `agent6 ask "QUESTION"` and `agent6 ask query "QUESTION"` are the'
            " same. On a foreground terminal, bare `agent6 ask` starts an interactive"
            " conversation. An ask asks before running commands even when the config"
            " approves them automatically; --auto-approve approves them for the session."
            " Saved asks appear in `agent6 sessions list`."
        ),
    )
    # `ask <question>` runs a Q&A. `query` is the implicit default verb injected
    # by `_inject_default_verb` when the first token isn't a known ask verb, so
    # `ask "why ..."` == `ask query "why ..."`.
    ask_sub = ask_p.add_subparsers(dest="ask_command", required=True, metavar="<subcommand>")
    ask_query = _sub(ask_sub, "query", help="Ask a question.")
    ask_query.add_argument(
        "task",
        nargs="?",
        default="",
        help=(
            "Question, usually in quotes. On a foreground terminal, omit it to ask"
            ' interactively. Example: "why does the retry loop double the timeout?"'
        ),
    )
    seed = ask_query.add_mutually_exclusive_group()
    ask_session = seed.add_argument(
        "--from",
        dest="ask_session",
        default="",
        metavar="SESSION_ID",
        help=(
            "Include context from another run, plan, or ask: its task, outcome, diff, key"
            " events, and plan text when present. Accepts a session id or unambiguous prefix."
        ),
    )
    ask_session.completer = _complete_session_ids  # type: ignore[attr-defined]
    seed.add_argument(
        "--from-latest",
        dest="ask_session_latest",
        action="store_true",
        help="Include the same context from the newest run or ask. Planning sessions are excluded.",
    )
    ask_query.add_argument(
        "--file",
        dest="ask_files",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Include PATH's contents in the question. Repeat for more files; this is the same"
            " as putting @PATH in the question."
        ),
    )
    ask_profile = ask_query.add_argument(
        "--preset",
        default="",
        help="Apply a strategy preset. `agent6 config presets` lists the choices.",
    )
    ask_profile.completer = _complete_presets  # type: ignore[attr-defined]
    _add_model_flag(ask_query)
    _add_config_flag(ask_query)
    ask_query.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Keep accepting follow-up questions in this session, with earlier questions and"
            " answers as context. Commands: /cost, /reset, and /quit. Requires a foreground"
            " terminal. A bare `agent6 ask` uses this mode there."
        ),
    )
    _add_budget_flags(ask_query)
    _add_sandbox_flags(ask_query)
