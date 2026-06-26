# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""agent6 command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import argcomplete

from agent6.cli._ask import (
    build_ask_run_digest as _build_ask_run_digest,
)
from agent6.cli._ask import (
    cmd_ask_list as _cmd_ask_list,
)
from agent6.cli._ask import (
    seed_files as _seed_files,
)
from agent6.cli._common import _BudgetOverrides, _enforce_root_policy, _runs_dir
from agent6.cli.check_cmds import _cmd_check
from agent6.cli.config_cmds import (
    _cmd_config_add,
    _cmd_config_fill,
    _cmd_config_get,
    _cmd_config_path,
    _cmd_config_remove,
    _cmd_config_set,
    _cmd_config_show,
    _cmd_config_unset,
)
from agent6.cli.connect import _cmd_connect
from agent6.cli.fork import _cmd_fork
from agent6.cli.machine_cmds import (
    _cmd_machine_check,
    _cmd_machine_create,
    _cmd_machine_graph,
    _cmd_machine_poke,
    _cmd_machine_replay,
    _cmd_machine_run,
    _cmd_machine_status,
    _cmd_machine_test,
)
from agent6.cli.misc_cmds import (
    _cmd_diff,
    _cmd_history_graph,
    _cmd_history_search,
    _cmd_history_transcript,
    _cmd_init,
    _cmd_mcp_serve,
    _cmd_memory_add,
    _cmd_memory_invalidate,
    _cmd_memory_list,
    _cmd_review,
)
from agent6.cli.model import _cmd_model
from agent6.cli.parser import _inject_default_verb, build_parser
from agent6.cli.plan_watch import (
    _cmd_plan_edit,
    _cmd_plan_show,
    _cmd_status,
    _cmd_tui,
    _cmd_watch,
    _most_recent_plan_run_id,
    _most_recent_run_id,
    _resolve_plan_run_id,
)
from agent6.cli.prompt_cmds import _cmd_prompt_show
from agent6.cli.run import (
    _cmd_resume,
    _cmd_run,
)


def _first_markdown_line(text: str, max_len: int = 80) -> str:
    """First non-empty line of a markdown doc (a plan title), `#`/bullet stripped."""
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").lstrip("-*").strip()
        if line:
            return line[:max_len]
    return "(untitled plan)"


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    parser = build_parser()
    argcomplete.autocomplete(parser)
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(_inject_default_verb(raw))
    root_rc = _enforce_root_policy(getattr(args, "allow_root", False))
    if root_rc is not None:
        return root_rc
    if args.command == "run":
        if args.continue_run:
            if args.task:
                print("ERROR: pass either a task OR --continue, not both.", file=sys.stderr)
                return 2
            if args.run_id:
                print(
                    "ERROR: --run-id is incompatible with --continue"
                    " (--continue resolves the most recent run automatically).",
                    file=sys.stderr,
                )
                return 2
            target = _most_recent_run_id(_runs_dir(Path.cwd()))
            if target is None:
                print(
                    "ERROR: --continue: no prior runs for this cwd.",
                    file=sys.stderr,
                )
                return 2
            print(f"[agent6] --continue: resuming {target}", file=sys.stderr)
            return _cmd_resume(
                args.config,
                target,
                force=False,
                no_tui=args.no_tui,
                budget_overrides=_BudgetOverrides.from_args(args),
            )
        if args.from_plan:
            if args.task:
                print(
                    "ERROR: --from-plan is mutually exclusive with a task argument.",
                    file=sys.stderr,
                )
                return 2
            resolved = _resolve_plan_run_id(args.from_plan)
            if resolved is None:
                return 2
            plan_md = (_runs_dir(Path.cwd()) / resolved / "plan.md").read_text(encoding="utf-8")
            task = (
                f"The following plan was prepared by a planning pass at {resolved}."
                f" Execute it.\n\n{plan_md}"
            )
        elif not args.task:
            # No task: fall back to the most recent plan run, the common
            # "I just ran `agent6 plan`, now execute it" flow. At a TTY,
            # confirm before editing; non-interactively, refuse (a bare
            # `run` in a script should not silently start mutating).
            last_plan = _most_recent_plan_run_id(_runs_dir(Path.cwd()))
            if last_plan is None:
                print(
                    "ERROR: 'run' needs a task (or --from-plan <id> / --continue);"
                    " no prior plan found to execute.",
                    file=sys.stderr,
                )
                return 2
            plan_path = _runs_dir(Path.cwd()) / last_plan / "plan.md"
            try:
                plan_md = plan_path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"ERROR: could not read {plan_path}: {exc}", file=sys.stderr)
                return 2
            title = _first_markdown_line(plan_md)
            if not sys.stdin.isatty():
                print(
                    f"ERROR: 'run' needs a task. Most recent plan is {last_plan}"
                    f" ({title}); execute it with: agent6 run --from-plan {last_plan}",
                    file=sys.stderr,
                )
                return 2
            print(f"[agent6] No task given. Most recent plan: {last_plan}  ({title})")
            try:
                ans = input("Execute it now? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans in ("n", "no"):
                print(f"Aborted. Run it later: agent6 run --from-plan {last_plan}")
                return 0
            task = (
                f"The following plan was prepared by a planning pass at {last_plan}."
                f" Execute it.\n\n{plan_md}"
            )
        else:
            task = args.task
        return _cmd_run(
            args.config,
            task,
            run_id=args.run_id,
            interactive=args.interactive,
            no_tui=args.no_tui,
            budget_overrides=_BudgetOverrides.from_args(args),
            profile=getattr(args, "profile", ""),
        )
    if args.command == "plan":
        if args.plan_command == "show":
            return _cmd_plan_show(args.run_id)
        if args.plan_command == "edit":
            return _cmd_plan_edit(args.run_id)
        if not args.task:
            print(
                "ERROR: 'plan' needs a task argument (or `plan show/edit <id>`).",
                file=sys.stderr,
            )
            return 2
        return _cmd_run(
            args.config,
            args.task,
            run_id=args.run_id,
            mode="plan",
            budget_overrides=_BudgetOverrides.from_args(args),
            profile=getattr(args, "profile", ""),
        )
    if args.command == "ask":
        if args.ask_command == "list":
            return _cmd_ask_list()
        # REPL when -i is given, or no question + an interactive stdin.
        repl = args.interactive or (not args.task and sys.stdin.isatty())
        if not args.task and not repl:
            print(
                "ERROR: 'ask' needs a question (in quotes), or -i for the REPL.",
                file=sys.stderr,
            )
            return 2
        question = args.task
        prefix: list[str] = []
        if args.ask_seed_latest or args.ask_run:
            digest = _build_ask_run_digest(Path.cwd(), args.ask_run, latest=args.ask_seed_latest)
            if digest is None:
                return 2
            prefix.append(digest)
        if args.ask_files:
            seeds = _seed_files(Path.cwd(), args.ask_files)
            if seeds:
                prefix.append(seeds)
        if prefix:
            question = "\n\n".join([*prefix, question]) if question else "\n\n".join(prefix)
        return _cmd_run(
            args.config,
            question,
            mode="ask",
            interactive=repl,
            budget_overrides=_BudgetOverrides.from_args(args),
            profile=getattr(args, "profile", ""),
        )
    if args.command == "runs":
        if args.runs_command == "show":
            return _cmd_status(args.run_id, as_json=args.json)
        if args.runs_command == "watch":
            return _cmd_watch(args.run_id, plain=args.plain, since=args.since)
        if args.runs_command == "diff":
            return _cmd_diff(run_id=args.run_id, stat=args.stat, paths=tuple(args.paths))
        if args.runs_command == "transcript":
            return _cmd_history_transcript(
                args.run_id,
                as_json=args.as_json,
                no_thinking=args.no_thinking,
                tools=args.tools,
                seq=args.seq,
            )
        if args.runs_command == "graph":
            return _cmd_history_graph(args.run_id)
    if args.command == "tui":
        return _cmd_tui()
    if args.command == "prompt" and args.prompt_command == "show":
        return _cmd_prompt_show(args.config, mode=args.mode)
    if args.command == "resume":
        return _cmd_resume(
            args.config,
            args.run_id,
            force=args.force_resume,
            no_tui=args.no_tui,
            budget_overrides=_BudgetOverrides.from_args(args),
        )
    if args.command == "fork":
        return _cmd_fork(
            args.config,
            args.run_id,
            at_turn=args.at_turn,
            new_run_id=args.new_run_id,
            no_run=args.no_run,
            no_tui=args.no_tui,
            budget_overrides=_BudgetOverrides.from_args(args),
        )
    if args.command == "config":
        if args.config_command == "show":
            return _cmd_config_show(args.config, as_json=args.as_json)
        if args.config_command == "fill":
            return _cmd_config_fill(args.config, to_repo=args.repo, force=args.force)
        if args.config_command == "path":
            return _cmd_config_path()
        if args.config_command == "get":
            return _cmd_config_get(args.key, machine=args.machine_file)
        if args.config_command == "set":
            return _cmd_config_set(args.key, args.value, repo=args.repo, machine=args.machine_file)
        if args.config_command == "unset":
            return _cmd_config_unset(args.key, repo=args.repo, machine=args.machine_file)
        if args.config_command == "add":
            return _cmd_config_add(args.key, args.value, repo=args.repo, machine=args.machine_file)
        if args.config_command == "remove":
            return _cmd_config_remove(
                args.key, args.value, repo=args.repo, machine=args.machine_file
            )
    if args.command == "check":
        return _cmd_check(args.config, section=args.section)
    if args.command == "connect":
        return _cmd_connect(provider=args.provider, to_repo=args.repo)
    if args.command == "model":
        return _cmd_model(
            args.config,
            role=args.role,
            provider=args.provider,
            model=args.model,
            thinking=args.thinking,
            to_repo=args.repo,
        )
    if args.command == "memory":
        if args.memory_command == "add":
            return _cmd_memory_add(args.scope, args.body)
        if args.memory_command == "list":
            return _cmd_memory_list(args.scope or None, include_invalidated=args.all)
        if args.memory_command == "invalidate":
            return _cmd_memory_invalidate(args.memory_id, args.reason)
    if args.command == "history" and args.history_command == "search":
        return _cmd_history_search(args.query, fixed=not args.regex, run_id=args.run)
    if args.command == "init":
        return _cmd_init(force=args.force, profile=args.profile, assume_yes=args.yes)
    if args.command == "review":
        return _cmd_review(
            args.config,
            base=args.base,
            head=args.head,
            paths=tuple(args.paths),
            model_override=args.model,
            reviewers=args.reviewers,
            personas=args.personas,
        )
    if args.command == "mcp" and args.mcp_command == "serve":
        return _cmd_mcp_serve(args.config)
    if args.command == "machine" and args.machine_command == "check":
        return _cmd_machine_check(args.file)
    if args.command == "machine" and args.machine_command == "test":
        return _cmd_machine_test(args.file, blackboard=args.blackboard)
    if args.command == "machine" and args.machine_command == "graph":
        return _cmd_machine_graph(args.file, fmt=args.format)
    if args.command == "machine" and args.machine_command == "run":
        return _cmd_machine_run(args.file, exit_on_wait=args.exit_on_wait)
    if args.command == "machine" and args.machine_command == "status":
        return _cmd_machine_status(args.machine_id)
    if args.command == "machine" and args.machine_command == "poke":
        return _cmd_machine_poke(args.machine_id)
    if args.command == "machine" and args.machine_command == "replay":
        return _cmd_machine_replay(args.machine_id)
    if args.command == "machine" and args.machine_command == "create":
        return _cmd_machine_create(args.task, output=args.output, max_attempts=args.max_attempts)
    parser.error("unknown command")  # pragma: no cover
    return 2
