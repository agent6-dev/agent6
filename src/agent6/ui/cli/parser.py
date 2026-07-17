# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Assembles the `agent6` argparse parser (subcommands, flags, completers)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6 import __version__
from agent6.ui.cli._common import _sub
from agent6.ui.cli._config_args import _add_config_parser, _add_connect_parser, _add_model_parser
from agent6.ui.cli._machine_args import _add_machine_parser
from agent6.ui.cli._plan_args import _add_ask_parser, _add_plan_parser
from agent6.ui.cli._review_args import _add_check_parser, _add_review_parser, _add_system_parser
from agent6.ui.cli._run_args import _add_fork_parser, _add_resume_parser, _add_run_parser
from agent6.ui.cli._runs_args import _add_runs_parser
from agent6.ui.cli._skills_args import _add_skills_parser
from agent6.ui.cli._watch_args import _add_attach_parser, _add_tui_parser, _add_web_parser
from agent6.ui.cli.completers import _complete_run_ids

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
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    _add_run_parser(sub)

    _add_plan_parser(sub)

    _add_ask_parser(sub)

    _add_attach_parser(sub)

    _add_runs_parser(sub)

    _add_tui_parser(sub)

    completions_p = _sub(
        sub,
        "completions",
        help=(
            "Install shell tab-completion for agent6 (detects your shell from"
            " $SHELL; bash/zsh get a guarded source line in their rc, fish and"
            " xonsh a native auto-loaded file). --print emits the script"
            " instead, for `eval` or manual setup."
        ),
    )
    completions_p.add_argument(
        "shell",
        nargs="?",
        # None (not ""): argparse validates a *string* default against choices,
        # and an empty-string choice leaks into completion output as a bogus
        # description-only candidate.
        default=None,
        choices=["bash", "zsh", "fish", "xonsh"],
        metavar="{bash,zsh,fish,xonsh}",
        help="Target shell (default: detect from $SHELL).",
    )
    completions_p.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the completion script to stdout instead of installing it.",
    )

    _add_web_parser(sub)

    prompt_p = _sub(
        sub,
        "prompt",
        help="Inspect the assembled system prompt for this repo + config.",
    )
    prompt_sub = prompt_p.add_subparsers(
        dest="prompt_command", required=True, metavar="<subcommand>"
    )
    prompt_show = _sub(
        prompt_sub,
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

    _add_resume_parser(sub)

    _add_fork_parser(sub)

    _add_config_parser(sub)

    _add_check_parser(sub)

    _add_connect_parser(sub)

    _add_system_parser(sub)

    _add_model_parser(sub)

    mem_p = _sub(sub, "memory", help="Manage persistent agent memories.")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True, metavar="<subcommand>")
    mem_add = _sub(mem_sub, "add", help="Append a new memory entry.")
    mem_add.add_argument(
        "scope", choices=("facts", "decisions", "preferences"), help="Memory scope."
    )
    mem_add.add_argument("body", help="Entry body (in quotes).")
    mem_list = _sub(mem_sub, "list", help="List memory entries.")
    mem_list.add_argument(
        "--scope",
        choices=("facts", "decisions", "preferences"),
        default="",
        help="Limit to one scope; omit for all.",
    )
    mem_list.add_argument(
        "--all", action="store_true", help="Include invalidated entries (default: hide)."
    )
    mem_inv = _sub(mem_sub, "invalidate", help="Mark a memory entry as invalidated.")
    mem_inv.add_argument("memory_id", help="26-char ULID of the entry to invalidate.")
    mem_inv.add_argument("reason", help="Why this entry is no longer valid.")

    _add_skills_parser(sub)

    hist_p = _sub(
        sub,
        "history",
        help="Cross-run search over persisted transcripts and run data (per-run views: `runs`).",
    )
    hist_sub = hist_p.add_subparsers(dest="history_command", required=True, metavar="<subcommand>")
    hist_search = _sub(hist_sub, "search", help="ripgrep-backed search over all runs.")
    hist_search.add_argument("query", help="Pattern (passed to rg --fixed-strings by default).")
    hist_search.add_argument(
        "--regex", action="store_true", help="Interpret query as a regex instead of fixed string."
    )
    hist_search_run = hist_search.add_argument(
        "--run",
        default="",
        metavar="RUN_ID",
        help="Restrict to a single run id (default: all runs).",
    )
    hist_search_run.completer = _complete_run_ids  # type: ignore[attr-defined]

    init_p = _sub(
        sub,
        "init",
        help="Optional setup wizard: per-repo config, verify_command, .gitignore, AGENTS.md.",
    )
    init_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive prompts and accept the defaults for every step"
        " (nothing existing is ever overwritten).",
    )
    init_p.add_argument(
        # Named --ecosystem, not --profile: `run/plan/ask --profile` already mean
        # the config strategy preset (quick/ultra/...), a different concept.
        "--ecosystem",
        dest="profile",
        choices=("py", "rust", "node"),
        default="",
        help=(
            "Ecosystem for the .gitignore build-artifact entries. Auto-detected"
            " from the repo's manifests when omitted (py/rust/node)."
        ),
    )

    _add_review_parser(sub)

    mcp_p = _sub(
        sub,
        "mcp",
        help="MCP (Model Context Protocol) integration. See `agent6 mcp serve --help`.",
    )
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", required=True, metavar="<subcommand>")
    mcp_serve = _sub(
        mcp_sub,
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

    _add_machine_parser(sub)

    # Shell tab-completion. argcomplete is a hard dependency; the call is a
    # no-op unless the shell sourced its completion script for this binary
    # (see `agent6 --help` and the README for activation instructions).
    return parser
