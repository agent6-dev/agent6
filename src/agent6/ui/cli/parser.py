# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Assemble the `agent6` argparse parser: subcommands, flags, completers."""

from __future__ import annotations

import argparse
import pathlib

from agent6 import __version__, paths
from agent6.ui.cli import (
    _common,
    _config_args,
    _machine_args,
    _mcp_args,
    _plan_args,
    _review_args,
    _run_args,
    _sessions_args,
    _skills_args,
    _watch_args,
    completers,
)

# Commands with a default verb (`plan <task>` is `plan run <task>`, a bare `skills` lists); the
# explicit form covers a query whose first word is a verb name. A test pins each verb set.
_DEFAULT_VERBS: dict[str, tuple[str, frozenset[str]]] = {
    "plan": ("run", frozenset({"run", "show", "edit"})),
    "ask": ("query", frozenset({"query"})),
    "history": ("search", frozenset({"search"})),
    "skills": ("list", frozenset({"install", "update", "list", "enable", "disable", "remove"})),
    "memory": ("list", frozenset({"add", "list", "show", "rm", "decisions"})),
    "mcp": ("list", frozenset({"connect", "list", "remove", "serve"})),
    "config": (
        "show",
        frozenset(
            {"show", "fill", "path", "presets", "get", "set", "unset", "add", "remove", "fix"}
        ),
    ),
    "prompt": ("show", frozenset({"show"})),
    "sessions": (
        "list",
        frozenset(
            {
                "commits",
                "compare",
                "diff",
                "dir",
                "graph",
                "list",
                "merge",
                "prune",
                "review",
                "rm",
                "show",
                "transcript",
            }
        ),
    ),
    "machine": (
        "list",
        frozenset(
            {"list", "check", "test", "graph", "run", "status", "poke", "stop", "replay", "create"}
        ),
    ),
}

# Groups whose default verb takes no positional: a bare word after them is a mistyped verb.
_BARE_DEFAULT_GROUPS: frozenset[str] = frozenset(
    {"skills", "memory", "mcp", "prompt", "machine", "sessions"}
)


# Top-level options that may precede the subcommand; `--config` takes a value, the rest are flags.
_GLOBAL_VALUE_OPTS = frozenset({"--config"})
_GLOBAL_FLAG_OPTS = frozenset({"--allow-root"})


def _shell_default_help() -> str:
    """Return the completions `shell` help, naming what detection resolves to.

    Detection walks the process tree (a fish inside bash detects fish); unknown keeps the
    generic wording.
    """
    from agent6.ui.cli import completions_cmd  # noqa: PLC0415  # noqa: PLC0415

    detected = completions_cmd.detect_shell()
    if detected in ("bash", "zsh", "fish", "xonsh"):
        return f"Target shell (default: detected {detected})."
    return "Target shell (default: detect the running shell)."


def _command_index(argv: list[str]) -> int | None:
    """Return the index of the subcommand token, skipping leading global options.

    `["--config", "c.toml", "plan", ...]` gives 2.

    Returns:
        The index, or None when a global help or version flag comes first or no command
        is found.
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
    """Return argv with a group's implicit verb inserted when the next token is not one.

    `["plan", "fix the bug"]` becomes `["plan", "run", "fix the bug"]`. Leading global
    options are skipped to find the command; an explicit verb or `-h`/`--help` is left as is.
    """
    ci = _command_index(argv)
    if ci is None or argv[ci] not in _DEFAULT_VERBS:
        return argv
    default_verb, verbs = _DEFAULT_VERBS[argv[ci]]
    rest = argv[ci + 1 :]
    # A bare `plan` or `ask` gets the verb too, so the no-task path still runs.
    if rest and (rest[0] in verbs or rest[0] in ("-h", "--help")):
        return argv
    if rest and argv[ci] in _BARE_DEFAULT_GROUPS and not rest[0].startswith("-"):
        return argv
    return [*argv[: ci + 1], default_verb, *rest]


def _directories_epilog() -> str:
    """Return where agent6 keeps things, resolved, for the bottom of `--help`.

    Paths only, each a plain env or home lookup, so building the parser stays cheap.
    """
    user = paths.effective_user()
    rows = (
        ("config", paths.global_config_path(user).parent, "config.toml, secrets.toml (0600)"),
        ("state", paths.state_base(user), "per-repo run history, memory, reviews"),
        ("data", paths.data_dir(user), "installed skill packs (skills/)"),
        ("cache", paths.cache_dir(user), "regenerable model lists"),
    )
    width = max(len(str(p)) for _n, p, _w in rows)
    lines = [f"  {name:<6} {path!s:<{width}}  {what}" for name, path, what in rows]
    return "\n".join(
        [
            "directories (XDG):",
            *lines,
            "",
            "`agent6 config path` shows this repo's own state dir and the config files.",
        ]
    )


# The `agent6 --help` groups, in this order: start work, act on a live run, front-ends, records,
# context and tools, setup.
COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("work", ("run", "plan", "ask", "resume", "fork", "review", "machine")),
    ("control", ("attach", "steer", "stop", "answer", "exec", "forward")),
    ("front-ends", ("tui", "web", "acp")),
    ("sessions", ("sessions", "ps", "history")),
    ("context", ("prompt", "skills", "memory")),
    ("tools", ("mcp",)),
    ("setup", ("init", "connect", "model", "config", "check", "completions", "system")),
)


class _GroupedCommandsHelp(argparse.RawDescriptionHelpFormatter):
    """Render the top-level command list by `COMMAND_GROUPS`, each group a titled block."""

    def _format_action(self, action: argparse.Action) -> str:
        """Return the grouped rendering for the subparsers action, argparse's for every other."""
        if not isinstance(action, argparse._SubParsersAction) or action.dest != "command":  # pyright: ignore[reportPrivateUsage]
            return super()._format_action(action)
        entries = {a.dest: a for a in action._choices_actions}  # pyright: ignore[reportPrivateUsage]
        parts: list[str] = []
        for title, names in COMMAND_GROUPS:
            parts.append(f"{'':{self._current_indent}}{title}:\n")
            self._indent()
            for name in names:
                parts.append(super()._format_action(entries[name]))
            self._dedent()
        return "".join(parts)


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    """Return the `agent6` parser."""
    parser = argparse.ArgumentParser(
        prog="agent6",
        description="Sandboxed coding agent.",
        epilog=_directories_epilog(),
        formatter_class=_GroupedCommandsHelp,
    )
    parser.add_argument("--version", action="version", version=f"agent6 {__version__}")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        metavar="FILE",
        help=(
            "A config file layered over the global config (its path is below) and"
            " the per-repo config (kept under the state dir, out of the workspace)."
            " Default: those two and the built-in defaults."
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
    sub = parser.add_subparsers(
        dest="command", required=True, metavar="<command>", title="commands"
    )
    # The commands lead the help page; the options follow them.
    groups = parser._action_groups  # pyright: ignore[reportPrivateUsage]
    groups.insert(0, groups.pop())

    _run_args._add_run_parser(sub)

    _run_args._add_resume_parser(sub)

    _run_args._add_fork_parser(sub)

    _plan_args._add_plan_parser(sub)

    _plan_args._add_ask_parser(sub)

    _review_args._add_review_parser(sub)

    _watch_args._add_attach_parser(sub)
    _watch_args._add_steer_parser(sub)

    _watch_args._add_stop_parser(sub)
    _watch_args._add_answer_parser(sub)
    _watch_args._add_net_parsers(sub)

    _sessions_args._add_sessions_parser(sub)

    ps_p = _common._sub(
        sub,
        "ps",
        help=(
            "The live sessions of every repository on this machine: directory, id,"
            " mode, status, pid, front-end (`sessions` lists one repository's)."
        ),
    )
    ps_p.add_argument(
        "--json",
        action="store_true",
        help="Emit the rows as a JSON array (directory, repo_id, id, mode, status, pid, attached,"
        " coordinator, lanes).",
    )
    ps_p.add_argument(
        "--lanes",
        action="store_true",
        help="List each fan-out's live lanes under its row (folded into a count otherwise;"
        " the JSON form nests them always).",
    )

    hist_p = _common._sub(
        sub,
        "history",
        help=("Search every session's transcripts and records (`sessions` shows one session)."),
    )
    hist_sub = hist_p.add_subparsers(dest="history_command", required=True, metavar="<subcommand>")
    hist_search = _common._sub(hist_sub, "search", help="ripgrep-backed search over all sessions.")
    hist_search.add_argument(
        "query",
        nargs="?",
        default="",
        help="Pattern (passed to rg --fixed-strings by default).",
    )
    hist_search.add_argument(
        "--regex", action="store_true", help="Interpret query as a regex instead of fixed string."
    )
    hist_search_session = hist_search.add_argument(
        "--session",
        default="",
        metavar="SESSION_ID",
        help="Restrict to a single session id (default: all sessions).",
    )
    hist_search_session.completer = completers._complete_session_ids  # type: ignore[attr-defined]

    _watch_args._add_tui_parser(sub)

    _watch_args._add_web_parser(sub)

    _common._sub(
        sub,
        "acp",
        help=(
            "Run agent6 as an ACP (Agent Client Protocol) agent, driven by an"
            " editor. Speaks line-delimited JSON-RPC on stdin/stdout; the"
            " editor spawns this, so nothing else may write to stdout. Config"
            " comes from each session's own directory."
        ),
    )

    init_p = _common._sub(
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
        # `--ecosystem`, since `--preset` is the strategy preset on run, plan and ask.
        "--ecosystem",
        dest="ecosystem",
        choices=("py", "rust", "node"),
        default="",
        help=(
            "Ecosystem for the .gitignore build-artifact entries. Auto-detected"
            " from the repo's manifests when omitted (py/rust/node)."
        ),
    )

    _config_args._add_connect_parser(sub)

    _config_args._add_model_parser(sub)

    _config_args._add_config_parser(sub)

    _review_args._add_check_parser(sub)

    prompt_p = _common._sub(
        sub,
        "prompt",
        help="Inspect the assembled system prompt for this repo + config.",
    )
    prompt_sub = prompt_p.add_subparsers(
        dest="prompt_command", required=True, metavar="<subcommand>"
    )
    prompt_show = _common._sub(
        prompt_sub,
        "show",
        help=(
            "Print everything the model receives on a run's first call here: the"
            " system prompt (static blocks + the per-repo <repo-priors> block), the"
            " tool definitions this config exposes (the API's `tools` field), and"
            " the first user message around the task."
        ),
    )
    prompt_show.add_argument(
        "--mode",
        choices=("run", "plan", "ask", "agent"),
        default="run",
        help="Which mode's exchange to assemble (default: run).",
    )
    prompt_show.add_argument(
        "--json",
        action="store_true",
        help=(
            "One JSON object (mode, system, tools, first_message, mcp_tools_pending)"
            " instead of text."
        ),
    )

    completions_p = _common._sub(
        sub,
        "completions",
        help=(
            "Install shell tab-completion for agent6 (detects the shell you"
            " are running; bash/zsh get a guarded source line in their rc, fish and"
            " xonsh a native auto-loaded file). --print emits the script"
            " instead, for `eval` or manual setup."
        ),
    )
    completions_p.add_argument(
        "shell",
        nargs="?",
        # None, not "": argparse checks a string default against choices; "" leaks into completion.
        default=None,
        choices=["bash", "zsh", "fish", "xonsh"],
        metavar="{bash,zsh,fish,xonsh}",
        help=_shell_default_help(),
    )
    completions_p.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the completion script to stdout instead of installing it.",
    )

    _review_args._add_system_parser(sub)

    _skills_args._add_skills_parser(sub)

    mcp_p = _common._sub(
        sub,
        "mcp",
        help=(
            "MCP (Model Context Protocol): add or remove a server, list them, or serve; a bare"
            " `agent6 mcp` lists them."
        ),
    )
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", required=True, metavar="<subcommand>")
    _mcp_args._add_mcp_server_parsers(mcp_sub)
    mcp_serve = _common._sub(
        mcp_sub,
        "serve",
        help=(
            "Run agent6 as an MCP stdio server over the cwd's agent6 config:"
            " query_dag and list_sessions always, run_in_sandbox only where"
            ' sandbox.run_commands = "yes" (the default "ask" withholds it: nothing'
            " here can answer an approval), and run_verify and apply_patch_in_sandbox"
            " where it also sets a verify command. Speaks line-delimited"
            " JSON-RPC on stdin/stdout; configure an MCP-aware client to spawn"
            " this command."
        ),
    )
    _common._add_config_flag(mcp_serve)

    mem_p = _common._sub(
        sub,
        "memory",
        help=(
            "Manage the repo's agent memory (one fact per file + index); a bare"
            " `agent6 memory` lists it."
        ),
    )
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True, metavar="<subcommand>")
    mem_add = _common._sub(mem_sub, "add", help="Write <name>.md and its index line.")
    mem_add.add_argument("name", help="Memory name (lowercase letters, digits, dashes).")
    mem_add.add_argument("body", help="The fact (in quotes; first line becomes the index hook).")
    _common._sub(mem_sub, "list", help="Print the MEMORY.md index.")
    mem_show = _common._sub(mem_sub, "show", help="Print one memory file.")
    mem_show.add_argument("name", help="Memory name.")
    mem_rm = _common._sub(mem_sub, "rm", help="Delete a memory file and its index line.")
    mem_rm.add_argument("name", help="Memory name.")
    _common._sub(
        mem_sub,
        "decisions",
        help="Print the operator rulings the harness recorded (memory/DECISIONS.md).",
    )

    _machine_args._add_machine_parser(sub)

    return parser
