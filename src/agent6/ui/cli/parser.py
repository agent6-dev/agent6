# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Assembles the `agent6` argparse parser (subcommands, flags, completers)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent6 import __version__
from agent6.ui.cli._common import _add_sandbox_flags, _sub
from agent6.ui.cli._plan_args import _add_ask_parser, _add_plan_parser
from agent6.ui.cli._run_args import _add_fork_parser, _add_resume_parser, _add_run_parser
from agent6.ui.cli._runs_args import _add_runs_parser
from agent6.ui.cli._watch_args import _add_attach_parser, _add_tui_parser, _add_web_parser
from agent6.ui.cli.completers import (
    _complete_config_keys,
    _complete_config_values,
    _complete_machine_files,
    _complete_machine_ids,
    _complete_model_provider,
    _complete_models,
    _complete_providers,
    _complete_run_ids,
    _complete_skills,
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

    config_p = _sub(
        sub,
        "config",
        help="Inspect and materialize the layered config (global + repo + defaults).",
    )
    config_sub = config_p.add_subparsers(
        dest="config_command", required=True, metavar="<subcommand>"
    )
    config_show = _sub(
        config_sub,
        "show",
        help=(
            "Print every effective config value and where it came from"
            " (default / global / repo / flag). `*` marks values that override"
            " the built-in default."
        ),
    )
    config_show.add_argument(
        "key",
        nargs="?",
        default="",
        help="Show just this leaf (or a section prefix, e.g. 'sandbox') UNTRUNCATED.",
    )
    config_show.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON instead of a table."
    )
    config_fill = _sub(
        config_sub,
        "fill",
        help=(
            "Write the fully-resolved config (every effective value, explicit)"
            " to a file: the global config by default, or the repo config with"
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
    _sub(
        config_sub, "path", help="Print the resolved global + repo config (and secrets) file paths."
    )
    config_get = _sub(
        config_sub, "get", help="Print a leaf's effective value and which layer set it."
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
        p = _sub(config_sub, verb, help=blurb)
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

    config_fix = _sub(
        config_sub,
        "fix",
        help=(
            "Drop invalid config entries (unknown keys, stale values) from the global"
            " and repo config, printing each and whether it was global or repo. Repairs"
            " a machine file's [config] overlay instead with --machine-file."
        ),
    )
    config_fix_machine = config_fix.add_argument(
        "--machine-file",
        dest="machine_file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Repair a machine file's [config] overlay instead of the global/repo config.",
    )
    config_fix_machine.completer = _complete_machine_files  # type: ignore[attr-defined]

    check_p = _sub(
        sub,
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

    connect_p = _sub(
        sub,
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
    connect_p.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip the post-save read-only key check (a GET to the provider's /models)."
        " Use for offline/local endpoints (Ollama, llama.cpp) that have no models listing.",
    )

    # `agent6 system <component> <action>`: privileged host/OS setup (uses sudo).
    system_p = _sub(
        sub,
        "system",
        help="Host/OS setup that needs privileges (e.g. the AppArmor profile for the"
        " strict sandbox). Uses sudo.",
    )
    system_sub = system_p.add_subparsers(
        dest="system_command", required=True, metavar="<subcommand>"
    )
    apparmor_p = _sub(
        system_sub,
        "apparmor",
        help="Install/remove the agent6-jail AppArmor profile (Ubuntu 24.04+: lets the"
        " strict sandbox use user namespaces).",
    )
    apparmor_p.add_argument(
        "action",
        choices=("install", "remove", "status"),
        help="install the profile and reload AppArmor, remove it, or report its state.",
    )

    model_p = _sub(
        sub,
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

    skills_p = _sub(sub, "skills", help="Manage operator-installed skills (SKILL.md packs).")
    skills_sub = skills_p.add_subparsers(
        dest="skills_command", required=True, metavar="<subcommand>"
    )
    sk_install = _sub(
        skills_sub,
        "install",
        help="Install skills from a SKILL.md URL, a git repository URL, or a local path.",
    )
    sk_install.add_argument("url", help="Direct SKILL.md URL, git repo URL, or local path.")
    sk_install.add_argument(
        "--force", action="store_true", help="Replace an already-installed skill of the same name."
    )
    sk_update = _sub(skills_sub, "update", help="Re-fetch installed skills from their origins.")
    sk_update_name = sk_update.add_argument(
        "name", nargs="?", default="", help="Skill to update (default: all with an origin)."
    )
    sk_update_name.completer = _complete_skills  # type: ignore[attr-defined]
    _sub(skills_sub, "list", help="List installed skills with state and origin.")
    sk_enable = _sub(skills_sub, "enable", help="Re-enable a skill (or promote it to always-on).")
    sk_enable_name = sk_enable.add_argument("name", help="Skill name.")
    sk_enable_name.completer = _complete_skills  # type: ignore[attr-defined]
    sk_enable.add_argument(
        "--always",
        action="store_true",
        help="Inject the skill's full text into every run's system prompt instead of the index.",
    )
    sk_enable.add_argument(
        "--repo", action="store_true", help="Write to the per-repo config instead of the global."
    )
    sk_disable = _sub(skills_sub, "disable", help="Drop a skill from the index and use_skill.")
    sk_disable_name = sk_disable.add_argument("name", help="Skill name.")
    sk_disable_name.completer = _complete_skills  # type: ignore[attr-defined]
    sk_disable.add_argument(
        "--repo", action="store_true", help="Write to the per-repo config instead of the global."
    )
    sk_remove = _sub(skills_sub, "remove", help="Delete an installed skill from the data dir.")
    sk_remove_name = sk_remove.add_argument("name", help="Skill name.")
    sk_remove_name.completer = _complete_skills  # type: ignore[attr-defined]

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

    review_p = _sub(
        sub,
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

    machine_p = _sub(
        sub,
        "machine",
        help="Author-time tooling for agent6 state machines (.asm.toml).",
    )
    machine_sub = machine_p.add_subparsers(
        dest="machine_command", required=True, metavar="<subcommand>"
    )
    machine_check = _sub(
        machine_sub,
        "check",
        help=(
            "Validate a .asm.toml machine file: parse, type-check, reachability,"
            " bundle paths, and static script lint/types (ruff + ty). No execution."
        ),
    )
    machine_check.add_argument("file", type=Path, help="Path to the .asm.toml machine file.")
    machine_test = _sub(
        machine_sub,
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
    machine_graph = _sub(
        machine_sub,
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
    machine_run = _sub(
        machine_sub,
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
    # Sandbox only; a machine's approvals are config-driven (its states set
    # run_commands), so --auto-approve is not offered here.
    _add_sandbox_flags(machine_run, auto_approve=False)
    machine_status = _sub(
        machine_sub,
        "status",
        help="Report a machine instance's current state, spend, and next wake. Read-only.",
    )
    machine_status_id = machine_status.add_argument(
        "machine_id", help="Machine id (directory under the per-repo state dir, machines subdir)."
    )
    machine_status_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_poke = _sub(
        machine_sub,
        "poke",
        help="Signal a waiting machine to wake on its next check (drops a signal file).",
    )
    machine_poke_id = machine_poke.add_argument(
        "machine_id", help="Machine id (directory under the per-repo state dir, machines subdir)."
    )
    machine_poke_id.completer = _complete_machine_ids  # type: ignore[attr-defined]
    machine_poke_payload = machine_poke.add_mutually_exclusive_group()
    machine_poke_payload.add_argument(
        "--data",
        metavar="JSON",
        help="A JSON value delivered to the waking wait as its poke payload"
        " (readable by the next tool at $AGENT6_MACHINE_DATA_DIR/poke.json).",
    )
    machine_poke_payload.add_argument(
        "--message",
        metavar="TEXT",
        help="Shorthand for --data with a JSON string payload.",
    )
    machine_replay = _sub(
        machine_sub,
        "replay",
        help="Deterministically replay a machine's journal offline (no world I/O).",
    )
    machine_replay_id = machine_replay.add_argument(
        "machine_id", help="Machine id (directory under the per-repo state dir, machines subdir)."
    )
    machine_replay_id.completer = _complete_machine_ids  # type: ignore[attr-defined]

    machine_create = _sub(
        machine_sub,
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
        metavar="N",
        help="Maximum draft->check->fix attempts before giving up (default: 3).",
    )

    # Shell tab-completion. argcomplete is a hard dependency; the call is a
    # no-op unless the shell sourced its completion script for this binary
    # (see `agent6 --help` and the README for activation instructions).
    return parser
