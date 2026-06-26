# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Assembles the `agent6` argparse parser (subcommands, flags, completers)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6 import __version__
from agent6.cli._common import _add_budget_flags
from agent6.cli.completers import (
    _complete_config_keys,
    _complete_config_values,
    _complete_machine_files,
    _complete_machine_ids,
    _complete_model_provider,
    _complete_models,
    _complete_plan_run_ids,
    _complete_providers,
    _complete_run_ids,
)

# Commands with a default verb: `plan <task>` == `plan run <task>`, and
# `ask <q>` == `ask query <q>`. _inject_default_verb rewrites argv so a bare
# task isn't mistaken for a subcommand name. The explicit forms (`plan run`,
# `ask query`) cover the rare task whose first word is a verb name.
_DEFAULT_VERBS: dict[str, tuple[str, frozenset[str]]] = {
    "plan": ("run", frozenset({"run", "show", "edit"})),
    "ask": ("query", frozenset({"query", "list"})),
}


# Top-level options that may precede the subcommand. `--config` takes a value;
# the rest are flags. _inject_default_verb skips past these to find the command.
_GLOBAL_VALUE_OPTS = frozenset({"--config"})
_GLOBAL_FLAG_OPTS = frozenset({"--allow-root"})


def _command_index(argv: list[str]) -> int | None:
    """Index of the subcommand token, skipping leading global options.

    `["--config", "c.toml", "plan", ...]` -> 2. Returns None if a global help
    or version flag appears first (argparse handles those) or no command is
    found.
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-h", "--help", "--version"):
            return None
        if tok in _GLOBAL_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("--") and "=" in tok and tok.split("=", 1)[0] in _GLOBAL_VALUE_OPTS:
            i += 1
            continue
        if tok in _GLOBAL_FLAG_OPTS:
            i += 1
            continue
        return i
    return None


def _inject_default_verb(argv: list[str]) -> list[str]:
    """Insert the implicit verb for `plan`/`ask` when the next token isn't one.

    `["plan", "fix the bug"]` -> `["plan", "run", "fix the bug"]`;
    `["ask", "why?"]` -> `["ask", "query", "why?"]`. Leading global options
    (`--config FILE`, `--allow-root`) are skipped to find the command. A bare
    `plan`/`ask`, an explicit verb, or `-h`/`--help` is left untouched.
    """
    ci = _command_index(argv)
    if ci is None or argv[ci] not in _DEFAULT_VERBS:
        return argv
    default_verb, verbs = _DEFAULT_VERBS[argv[ci]]
    rest = argv[ci + 1 :]
    # A bare `plan`/`ask` also gets the default verb so the no-task path (offer
    # the most recent plan / start the ask REPL) still runs; only an explicit
    # verb or -h/--help is left alone.
    if rest and (rest[0] in verbs or rest[0] in ("-h", "--help")):
        return argv
    return [*argv[: ci + 1], default_verb, *rest]


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(prog="agent6", description="Sandboxed coding agent.")
    parser.add_argument("--version", action="version", version=f"agent6 {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Explicit config file, layered on top of the global"
            " (~/.config/agent6/config.toml) and the per-repo config"
            " (out of the workspace, under the state dir). Default: use only"
            " those two layers + built-in defaults."
        ),
    )
    parser.add_argument(
        "--allow-root",
        action="store_true",
        help=(
            "Permit running as root (also AGENT6_ALLOW_ROOT=1). Off by default:"
            " running an LLM-driven agent as root is dangerous. Under sudo,"
            " agent6 reads your config/secrets and chowns new files back to you."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the single-loop agent on a task.")
    run_p.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task description (in quotes). Omit when using --continue.",
    )
    run_p.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    run_p.add_argument(
        "--profile",
        default="",
        help="Config profile preset (quick/standard/ultra/paranoid or a custom"
        " [profiles.<name>]). Overrides [workflow].profile; your explicit settings win.",
    )
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
            "REPL mode: after each successful auto-commit, prompt on"
            " stdin for one of /continue (default), /cost, /undo (git"
            " revert HEAD), /help, /quit. Requires a TTY."
        ),
    )
    run_p.add_argument(
        "--no-tui",
        action="store_true",
        help=(
            "Do not auto-spawn the textual dashboard. The TUI is launched"
            " automatically when the `tui` extra is installed and stdout is a"
            " TTY; this opts out (always implied by -i and on non-TTY runs)."
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
    _add_budget_flags(run_p)

    plan_p = sub.add_parser(
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
    plan_sub = plan_p.add_subparsers(dest="plan_command", required=True)
    plan_run = plan_sub.add_parser("run", help="Run a planning pass on a task.")
    plan_run.add_argument(
        "task",
        nargs="?",
        default="",
        help="Task description (in quotes). Omit to execute/offer the most recent plan.",
    )
    plan_run.add_argument("--run-id", default="", help="Explicit run id (default: generate one).")
    plan_run.add_argument(
        "--profile", default="", help="Config profile preset (see `agent6 run --profile`)."
    )
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
    plan_show = plan_sub.add_parser("show", help="Print the plan.md for a prior plan run and exit.")
    plan_show_id = plan_show.add_argument("run_id", help="Plan run id (or unambiguous prefix).")
    plan_show_id.completer = _complete_plan_run_ids  # type: ignore[attr-defined]
    plan_edit = plan_sub.add_parser(
        "edit",
        help="Open the plan.md for a prior plan run in $EDITOR (default: vi) and exit.",
    )
    plan_edit_id = plan_edit.add_argument("run_id", help="Plan run id (or unambiguous prefix).")
    plan_edit_id.completer = _complete_plan_run_ids  # type: ignore[attr-defined]

    ask_p = sub.add_parser(
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
    ask_sub = ask_p.add_subparsers(dest="ask_command", required=True)
    ask_query = ask_sub.add_parser("query", help="Ask a question (the default verb).")
    ask_query.add_argument(
        "--profile", default="", help="Config profile preset (see `agent6 run --profile`)."
    )
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
    ask_sub.add_parser(
        "list",
        help="List saved asks under the per-repo state dir (asks subdir, newest first) and exit.",
    )

    runs_p = sub.add_parser(
        "runs",
        help=(
            "Inspect a specific run: show (liveness/progress), watch (follow"
            " live), diff, transcript, graph. The run id is a positional"
            " everywhere (exact or unambiguous prefix; omit for the most recent)."
        ),
    )
    runs_sub = runs_p.add_subparsers(dest="runs_command", required=True)

    runs_show = runs_sub.add_parser(
        "show",
        help="One-shot liveness + progress of a run, then exit (vs `watch`, which follows).",
    )
    runs_show_id = runs_show.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (omit for the most recent run).",
    )
    runs_show_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_show.add_argument(
        "--json",
        action="store_true",
        help="Emit the status as a single JSON object (for scripts/monitoring).",
    )

    runs_watch = runs_sub.add_parser(
        "watch",
        help="Read-only live view of a run (defaults to the most recent run).",
    )
    runs_watch_id = runs_watch.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (omit for the most recent run).",
    )
    runs_watch_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_watch.add_argument(
        "--plain",
        action="store_true",
        help=(
            "Plain text tail of logs.jsonl (no textual TUI). Useful when"
            " textual is not installed or in headless terminals. Streams each"
            " event as a single line `<elapsed> <type> <key=val ...>` and"
            " follows the file like `tail -f`."
        ),
    )
    runs_watch.add_argument(
        "--since",
        type=int,
        default=0,
        metavar="N",
        help=(
            "With --plain: replay the last N events before starting to follow."
            " 0 (default) starts at end-of-file."
        ),
    )

    runs_diff = runs_sub.add_parser(
        "diff",
        help="Print the git diff produced by a run (manifest.base_sha -> HEAD of run branch).",
    )
    runs_diff_id = runs_diff.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unique prefix). Omit to diff the most recent run.",
    )
    runs_diff_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_diff.add_argument(
        "--stat",
        action="store_true",
        help="Show --stat summary instead of the full patch.",
    )
    runs_diff.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Restrict the diff to these paths.",
    )

    runs_tr = runs_sub.add_parser(
        "transcript",
        help="Render a run's full LLM conversation (the lossless transcripts) as Markdown.",
    )
    runs_tr_id = runs_tr.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unambiguous prefix). Defaults to the most recent run.",
    )
    runs_tr_id.completer = _complete_run_ids  # type: ignore[attr-defined]
    runs_tr.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the raw transcript array (the per-call request/response objects) instead.",
    )
    runs_tr.add_argument(
        "--no-thinking", action="store_true", help="Omit the model's reasoning/thinking blocks."
    )
    runs_tr.add_argument(
        "--tools",
        choices=("both", "calls", "none"),
        default="both",
        help="Show tool calls + results (both), calls only, or neither.",
    )
    runs_tr.add_argument(
        "--seq",
        default="",
        help="Restrict to a round-trip seq window, e.g. 3 or 3-7 (default: all).",
    )

    runs_graph = runs_sub.add_parser(
        "graph",
        help="Render the persisted task graph for a run as a DFS tree.",
    )
    runs_graph_id = runs_graph.add_argument(
        "run_id",
        nargs="?",
        default="",
        help="Run id (or unambiguous prefix). Defaults to the most recent run.",
    )
    runs_graph_id.completer = _complete_run_ids  # type: ignore[attr-defined]

    sub.add_parser(
        "tui",
        help="Open the TUI hub: browse runs and start a new run/plan/ask.",
    )

    prompt_p = sub.add_parser(
        "prompt",
        help="Inspect the assembled system prompt for this repo + config.",
    )
    prompt_sub = prompt_p.add_subparsers(dest="prompt_command", required=True)
    prompt_show = prompt_sub.add_parser(
        "show",
        help=(
            "Print the exact system prompt the worker receives: the static"
            " structural blocks plus the per-repo <repo-priors> block (repo map"
            " + AGENTS.md + recent commits)."
        ),
    )
    prompt_show.add_argument(
        "--mode",
        choices=("run", "plan", "ask", "machine", "agent"),
        default="run",
        help="Which mode's prompt to assemble (default: run).",
    )

    resume_p = sub.add_parser("resume", help="Resume a paused run from its snapshot.")
    resume_run = resume_p.add_argument("run_id", help="Run id under the per-repo run-state dir.")
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
        help="Resume even if snapshot commit is missing or worktree has diverged.",
    )
    resume_p.add_argument(
        "--no-tui",
        action="store_true",
        help="Do not auto-spawn the textual dashboard (see `agent6 run --no-tui`).",
    )
    _add_budget_flags(resume_p)

    fork_p = sub.add_parser(
        "fork",
        help=(
            "Clone a run, rolled back to a checkpoint, into a NEW run and continue"
            " it (the source run is never mutated)."
        ),
    )
    fork_src = fork_p.add_argument(
        "run_id", help="Source run id (or unambiguous prefix) to fork from."
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
        "--no-tui",
        action="store_true",
        help="Do not auto-spawn the textual dashboard (see `agent6 run --no-tui`).",
    )
    _add_budget_flags(fork_p)

    config_p = sub.add_parser(
        "config",
        help="Inspect and materialize the layered config (global + repo + defaults).",
    )
    config_sub = config_p.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser(
        "show",
        help=(
            "Print every effective config value and where it came from"
            " (default / global / repo / flag). `*` marks values that override"
            " the built-in default."
        ),
    )
    config_show.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON instead of a table."
    )
    config_fill = config_sub.add_parser(
        "fill",
        help=(
            "Write the fully-resolved config (every effective value, explicit)"
            " to a file — the global config by default, or the repo config with"
            " --repo. Handy before tightening defaults or for an audit snapshot."
        ),
    )
    config_fill.add_argument(
        "--repo",
        action="store_true",
        help="Write the per-repo config instead of the global config.",
    )
    config_fill.add_argument(
        "--force", action="store_true", help="Overwrite the target file if it already exists."
    )
    config_sub.add_parser(
        "path", help="Print the resolved global + repo config (and secrets) file paths."
    )
    config_get = config_sub.add_parser(
        "get", help="Print a leaf's effective value and which layer set it."
    )
    config_get_key = config_get.add_argument(
        "key", help="Dotted leaf path, e.g. sandbox.agent_network."
    )
    config_get_key.completer = _complete_config_keys  # type: ignore[attr-defined]
    config_get_machine = config_get.add_argument(
        "--machine-file",
        dest="machine_file",
        type=Path,
        default=None,
        metavar="FILE",
        help="View the value with a machine file's [config] overlay applied.",
    )
    config_get_machine.completer = _complete_machine_files  # type: ignore[attr-defined]
    for verb, blurb in (
        ("set", "Set a leaf to a scalar value (global by default)."),
        ("unset", "Remove a leaf, reverting it to the next-lower layer / default."),
        ("add", "Append a value to a list field (e.g. sandbox.allow_urls)."),
        ("remove", "Remove a value from a list field."),
    ):
        p = config_sub.add_parser(verb, help=blurb)
        key_arg = p.add_argument("key", help="Dotted leaf path, e.g. sandbox.agent_network.")
        key_arg.completer = _complete_config_keys  # type: ignore[attr-defined]
        if verb != "unset":
            val_arg = p.add_argument("value", help="Value (TOML-typed; bare text is a string).")
            val_arg.completer = _complete_config_values  # type: ignore[attr-defined]
        p.add_argument(
            "--repo",
            action="store_true",
            help="Write the per-repo config instead of the global config.",
        )
        machine_arg = p.add_argument(
            "--machine-file",
            dest="machine_file",
            type=Path,
            default=None,
            metavar="FILE",
            help="Edit a machine file's [config] overlay (providers.* forbidden).",
        )
        machine_arg.completer = _complete_machine_files  # type: ignore[attr-defined]

    check_p = sub.add_parser(
        "check",
        help=(
            "Pre-flight checks: sandbox + config + provider keys + MCP +"
            " verify_command. Read-only; safe on any clean repo."
        ),
    )
    check_p.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=("all", "sandbox", "config", "mcp", "verify"),
        help=(
            "Limit the report to one section. 'all' (default) runs every check"
            " and prints a single PASS/FAIL summary."
        ),
    )
    check_p.add_argument(
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

    connect_p = sub.add_parser(
        "connect",
        help="Interactively add a provider + API key (stored in the global secrets file).",
    )
    connect_provider = connect_p.add_argument(
        "--provider",
        default="",
        help="Provider name to add/update (e.g. anthropic, openrouter). Prompted if omitted.",
    )
    connect_provider.completer = _complete_providers  # type: ignore[attr-defined]
    connect_p.add_argument(
        "--repo",
        action="store_true",
        help="Write the [providers.*] block to the per-repo config instead of the global config.",
    )

    model_p = sub.add_parser(
        "model",
        help="Show or set which model + thinking level each role uses (planner/worker/reviewer).",
    )
    # choices gives both argparse validation and argcomplete tab-completion for
    # free. default=None (not "") so the omitted case isn't checked against
    # choices, argparse validates choices against a string default otherwise.
    # metavar="role" shows `role` in usage (not the noisy `{planner,...}`); the
    # choices stay listed in the help text. "all" is a pseudo-role (no config
    # field of that name) that sets every role at once, see _cmd_model.
    model_p.add_argument(
        "role",
        nargs="?",
        choices=("planner", "worker", "reviewer", "all"),
        default=None,
        metavar="role",
        help=(
            "Role to set: planner, worker, reviewer, or all (sets every role at"
            " once). Omit to print the current assignments."
        ),
    )
    model_provider = model_p.add_argument(
        "provider",
        nargs="?",
        default="",
        help="Provider name for the role (prompted from connected providers if omitted).",
    )
    # Role-gated (not _complete_providers) so the provider list doesn't bleed
    # into the first positional (role), see _complete_model_provider.
    model_provider.completer = _complete_model_provider  # type: ignore[attr-defined]
    model_model = model_p.add_argument(
        "model",
        nargs="?",
        default="",
        help="Model identifier for the role (prompted from the provider's catalog if omitted).",
    )
    model_model.completer = _complete_models  # type: ignore[attr-defined]
    model_p.add_argument(
        "--thinking",
        choices=("off", "low", "medium", "high"),
        default="",
        help="Reasoning/thinking effort for the role.",
    )
    model_p.add_argument(
        "--repo",
        action="store_true",
        help="Write to the per-repo config instead of the global config.",
    )

    mem_p = sub.add_parser("memory", help="Manage persistent agent memories.")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True)
    mem_add = mem_sub.add_parser("add", help="Append a new memory entry.")
    mem_add.add_argument(
        "scope", choices=("facts", "decisions", "preferences"), help="Memory scope."
    )
    mem_add.add_argument("body", help="Entry body (in quotes).")
    mem_list = mem_sub.add_parser("list", help="List memory entries.")
    mem_list.add_argument(
        "--scope",
        choices=("facts", "decisions", "preferences"),
        default="",
        help="Limit to one scope; omit for all.",
    )
    mem_list.add_argument(
        "--all", action="store_true", help="Include invalidated entries (default: hide)."
    )
    mem_inv = mem_sub.add_parser("invalidate", help="Mark a memory entry as invalidated.")
    mem_inv.add_argument("memory_id", help="26-char ULID of the entry to invalidate.")
    mem_inv.add_argument("reason", help="Why this entry is no longer valid.")

    hist_p = sub.add_parser(
        "history",
        help="Cross-run search over persisted transcripts and run data (per-run views: `runs`).",
    )
    hist_sub = hist_p.add_subparsers(dest="history_command", required=True)
    hist_search = hist_sub.add_parser("search", help="ripgrep-backed search over all runs.")
    hist_search.add_argument("query", help="Pattern (passed to rg --fixed-strings by default).")
    hist_search.add_argument(
        "--regex", action="store_true", help="Interpret query as a regex instead of fixed string."
    )
    hist_search_run = hist_search.add_argument(
        "--run", default="", help="Restrict to a single run id (default: all runs)."
    )
    hist_search_run.completer = _complete_run_ids  # type: ignore[attr-defined]

    init_p = sub.add_parser(
        "init",
        help="Optional setup wizard: per-repo config, verify_command, .gitignore, AGENTS.md.",
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Non-interactive: accept every step's default (never overwrites your files).",
    )
    init_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive prompts and accept the defaults for every step.",
    )
    init_p.add_argument(
        "--profile",
        choices=("py", "rust", "node"),
        default="",
        help=(
            "Ecosystem for the .gitignore build-artifact entries. Auto-detected"
            " from the repo's manifests when omitted (py/rust/node)."
        ),
    )

    review_p = sub.add_parser(
        "review",
        help="Read-only code review of a diff (working tree, branch-vs-base, or arbitrary range).",
    )
    review_p.add_argument(
        "--base",
        default="",
        help="Base ref. Default: review uncommitted changes (working tree vs HEAD).",
    )
    review_p.add_argument(
        "--head",
        default="HEAD",
        help="Head ref (default: HEAD). Only used when --base is set.",
    )
    review_p.add_argument(
        "--paths",
        nargs="*",
        default=(),
        help="Restrict the diff to these paths (forwarded to `git diff -- PATHS`).",
    )
    review_p.add_argument(
        "--model",
        default="",
        help=(
            "Override the reviewer model for this one-shot review "
            "(e.g. claude-sonnet-4-5 for a cheaper read). "
            "Default: reviewer_model from config."
        ),
    )
    review_p.add_argument(
        "--reviewers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run an adversarial REVIEW PANEL of N grounded reviewers instead of one"
            " freeform review. Findings are grounded against the diff (only real,"
            " block-eligible problems gate). 0 (default) = the classic single review."
        ),
    )
    review_p.add_argument(
        "--personas",
        default="",
        help=(
            "Comma-separated adversarial stances for the panel seats, cycled across"
            " --reviewers (e.g. 'security,correctness,tests'). Default: a built-in set."
        ),
    )

    mcp_p = sub.add_parser(
        "mcp",
        help="MCP (Model Context Protocol) integration. See `agent6 mcp serve --help`.",
    )
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_sub.add_parser(
        "serve",
        help=(
            "Run agent6 as an MCP stdio server, exposing run_verify /"
            " run_in_sandbox / apply_patch_in_sandbox / query_dag / list_runs"
            " using the cwd's agent6 config. Speaks line-delimited JSON-RPC"
            " on stdin/stdout; spawn from an MCP-aware client (e.g. VS Code"
            " Copilot's hand-off menu) and configure it with this command."
        ),
    )
    mcp_serve.add_argument(
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

    machine_p = sub.add_parser(
        "machine",
        help="Author-time tooling for agent6 state machines (.asm.toml).",
    )
    machine_sub = machine_p.add_subparsers(dest="machine_command", required=True)
    machine_check = machine_sub.add_parser(
        "check",
        help=(
            "Validate a .asm.toml machine file: parse, type-check, reachability,"
            " bundle paths, and static script lint/types (ruff + ty). No execution."
        ),
    )
    machine_check.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_test = machine_sub.add_parser(
        "test",
        help=(
            "Simulate a machine offline: everything `check` does, plus run the"
            " bundle's scripts/*_test.py mocks in a no-network jail, plus a pure"
            " dry-run (synthesized facts, branch routing against a fixture)."
            " No provider calls, no real network."
        ),
    )
    machine_test.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_test.add_argument(
        "--blackboard",
        type=Path,
        default=None,
        metavar="FIXTURE.toml",
        help="TOML fixture of variable values, overlaid on defaults for branch routing.",
    )
    machine_graph = machine_sub.add_parser(
        "graph",
        help="Emit the machine as a state diagram (mermaid or Graphviz dot).",
    )
    machine_graph.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_graph.add_argument(
        "--format",
        choices=("mermaid", "dot"),
        default="mermaid",
        help="Diagram format (default: mermaid).",
    )
    machine_run = machine_sub.add_parser(
        "run",
        help="Run (or resume) a machine, driving its states to a terminal one.",
    )
    machine_run.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_run.add_argument(
        "--exit-on-wait",
        action="store_true",
        help=(
            "Persist the next wake instant and exit 0 (status 'waiting') at the first"
            " not-ready wait instead of blocking, for an external scheduler to resume."
        ),
    )
    machine_status = machine_sub.add_parser(
        "status",
        help="Report a machine instance's current state, spend, and next wake. Read-only.",
    )
    machine_status_id = machine_status.add_argument(
        "machine_id", help="Machine id (directory under the per-repo state dir, machines subdir)."
    )
    machine_status_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_poke = machine_sub.add_parser(
        "poke",
        help="Signal a waiting machine to wake on its next check (drops a signal file).",
    )
    machine_poke_id = machine_poke.add_argument(
        "machine_id", help="Machine id (directory under the per-repo state dir, machines subdir)."
    )
    machine_poke_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_replay = machine_sub.add_parser(
        "replay",
        help="Deterministically replay a machine's journal offline (no world I/O).",
    )
    machine_replay_id = machine_replay.add_argument(
        "machine_id", help="Machine id (directory under the per-repo state dir, machines subdir)."
    )
    machine_replay_id.completer = _complete_machine_ids  # type: ignore[attr-defined]

    machine_create = machine_sub.add_parser(
        "create",
        help="Draft a .asm.toml machine from a natural-language task (LLM-assisted).",
    )
    machine_create.add_argument("task", help="Natural-language description of the loop to author.")
    machine_create.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Write the draft here (overwriting freely). Default: <machine-name>.asm.toml"
            " in cwd, which is never overwritten."
        ),
    )
    machine_create.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum draft->check->fix attempts before giving up (default: 3).",
    )

    # Shell tab-completion. argcomplete is a hard dependency; the call is a
    # no-op unless the shell sourced its completion script for this binary
    # (see `agent6 --help` and the README for activation instructions).
    return parser
