# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parser builders for `plan` and `ask`: alternate single-loop modes (planning-
only, read-only Q&A) alongside the main `run`, each with its own default-verb
subcommand tree (see `_inject_default_verb`)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6.ui.cli._common import _add_budget_flags, _add_sandbox_flags, _sub
from agent6.ui.cli.completers import _complete_plan_run_ids, _complete_profiles, _complete_run_ids


def _add_plan_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plan_p = _sub(
        sub,
        "plan",
        help=(
            "Planning pass: same loop, read-only tools, writes plan.md."
            " Pair with `agent6 run --from-plan <run-id>` to execute."
            " Inspect with `plan show <id>` / `plan edit <id>`."
        ),
    )
    # `plan <task>` is the bare planning run; `plan show/edit <id>` inspect a
    # prior plan. `run` is the implicit default verb injected by
    # `_inject_default_verb` when the first token isn't a known plan verb, so
    # `plan "fix the bug"` and `plan run "fix the bug"` are the same.
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True, metavar="<subcommand>")
    plan_run = _sub(plan_sub, "run", help="Run a planning pass on a task.")
    plan_run.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task to plan (in quotes). Required; `plan show/edit <id>` inspect prior plans.",
    )
    plan_run.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    plan_profile = plan_run.add_argument(
        "--profile", default="", help="Config profile preset (see `agent6 run --profile`)."
    )
    plan_profile.completer = _complete_profiles  # type: ignore[attr-defined]
    plan_run.add_argument(
        "--config",
        type=Path,
        # SUPPRESS (not None): a subparser default would otherwise clobber a
        # top-level `agent6 --config FILE <cmd>` back to None. With SUPPRESS the
        # subparser only sets `config` when --config is given AFTER the
        # subcommand, so both `agent6 --config F plan` and `agent6 plan --config F`
        # work; the top-level --config supplies the always-present default.
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Explicit config file (layered over global + repo configs).",
    )
    _add_budget_flags(plan_run)
    _add_sandbox_flags(plan_run)
    plan_show = _sub(plan_sub, "show", help="Print the plan.md for a prior plan run and exit.")
    plan_show_id = plan_show.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Plan run id (or unambiguous prefix); omit for the most recent plan.",
    )
    plan_show_id.completer = _complete_plan_run_ids  # type: ignore[attr-defined]
    plan_edit = _sub(
        plan_sub,
        "edit",
        help="Open the plan.md for a prior plan run in $EDITOR (default: vi) and exit.",
    )
    plan_edit_id = plan_edit.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Plan run id (or unambiguous prefix); omit for the most recent plan.",
    )
    plan_edit_id.completer = _complete_plan_run_ids  # type: ignore[attr-defined]


def _add_ask_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    ask_p = _sub(
        sub,
        "ask",
        help=(
            "Read-only Q&A: investigate the repo and answer a question in prose"
            " (no edits/commits). Brainstorm, rubber-duck, or ask how to do"
            " something. `ask list` enumerates saved asks."
        ),
    )
    # `ask <question>` runs a Q&A; `ask list` enumerates saved asks. `query` is
    # the implicit default verb injected by `_inject_default_verb` when the first
    # token isn't a known ask verb, so `ask "why ..."` == `ask query "why ..."`.
    ask_sub = ask_p.add_subparsers(dest="ask_command", required=True, metavar="<subcommand>")
    ask_query = _sub(ask_sub, "query", help="Ask a question (the default verb).")
    ask_profile = ask_query.add_argument(
        "--profile", default="", help="Config profile preset (see `agent6 run --profile`)."
    )
    ask_profile.completer = _complete_profiles  # type: ignore[attr-defined]
    ask_query.add_argument(
        "task",
        nargs="?",
        default="",
        help='Question (in quotes), e.g. "why does the broker drop large requests?".',
    )
    ask_query.add_argument(
        "--config",
        type=Path,
        # SUPPRESS so a top-level `agent6 --config F ask` is not clobbered; see
        # the run/plan --config notes above.
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Explicit config file (layered over global + repo configs).",
    )
    ask_run = ask_query.add_argument(
        "--run",
        dest="ask_run",
        default="",
        metavar="RUN_ID",
        help=(
            "Ask about a prior run: seed its task, outcome, diff, and key events"
            " from the run dir (exact id or unambiguous prefix)."
        ),
    )
    ask_run.completer = _complete_run_ids  # type: ignore[attr-defined]
    ask_query.add_argument(
        "--seed-latest",
        dest="ask_seed_latest",
        action="store_true",
        help="Like --run, but seed the most recent run.",
    )
    ask_query.add_argument(
        "--file",
        dest="ask_files",
        action="append",
        default=[],
        metavar="PATH",
        help="Seed a file's contents into the question (repeatable; like an inline @path).",
    )
    ask_query.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "Interactive REPL: keep asking follow-ups in one session (the prior"
            " Q&A is carried as context). /cost, /reset, /quit. Also the default"
            " when no question is given and stdin is a TTY."
        ),
    )
    _add_budget_flags(ask_query)
    _add_sandbox_flags(ask_query)
    _sub(
        ask_sub,
        "list",
        help="List saved asks under the per-repo state dir (asks subdir, newest first) and exit.",
    )
