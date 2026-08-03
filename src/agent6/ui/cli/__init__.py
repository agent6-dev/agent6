# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""agent6 command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import argcomplete

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
)
from agent6.errors import OperatorError, read_operator_file
from agent6.events import EventWriteError
from agent6.ui.acp import serve_acp
from agent6.ui.cli._ask import (
    build_ask_session_digest,
    cmd_ask_list,
    seed_files,
)
from agent6.ui.cli._common import (
    _enforce_root_policy,
    _plans_dir,
)
from agent6.ui.cli.check_cmds import _cmd_check
from agent6.ui.cli.completions_cmd import cmd_completions
from agent6.ui.cli.config_cmds import (
    _cmd_config_add,
    _cmd_config_fill,
    _cmd_config_fix,
    _cmd_config_get,
    _cmd_config_path,
    _cmd_config_presets,
    _cmd_config_remove,
    _cmd_config_set,
    _cmd_config_show,
    _cmd_config_unset,
)
from agent6.ui.cli.connect import _cmd_connect
from agent6.ui.cli.fork import _cmd_fork
from agent6.ui.cli.history_cmds import (
    _cmd_history_graph,
    _cmd_history_search,
    _cmd_history_transcript,
)
from agent6.ui.cli.init_cmds import _cmd_init
from agent6.ui.cli.machine_cmds import (
    _cmd_machine_check,
    _cmd_machine_create,
    _cmd_machine_graph,
    _cmd_machine_poke,
    _cmd_machine_replay,
    _cmd_machine_run,
    _cmd_machine_status,
    _cmd_machine_test,
)
from agent6.ui.cli.mcp_cmds import _cmd_mcp_serve
from agent6.ui.cli.mcp_connect import cmd_mcp_connect, cmd_mcp_list
from agent6.ui.cli.memory_cmds import (
    _cmd_memory_add,
    _cmd_memory_invalidate,
    _cmd_memory_list,
    _cmd_memory_pin,
    _cmd_memory_unpin,
)
from agent6.ui.cli.model import _cmd_model
from agent6.ui.cli.parser import _command_index, _inject_default_verb, build_parser
from agent6.ui.cli.plan_watch import (
    _cmd_plan_edit,
    _cmd_plan_show,
    _cmd_status,
    _cmd_tui,
    _most_recent_plan_session_id,
    _resolve_plan_session_id,
)
from agent6.ui.cli.prompt_cmds import _cmd_prompt_show
from agent6.ui.cli.resume import _cmd_resume
from agent6.ui.cli.review_cmds import _cmd_review
from agent6.ui.cli.run import _cmd_run
from agent6.ui.cli.sessions_cmds import (
    _cmd_commits,
    _cmd_compare,
    _cmd_diff,
    _cmd_list,
    _cmd_merge,
    _cmd_prune,
    _cmd_sessions_dir,
    _cmd_sessions_rm,
    _cmd_stop,
)
from agent6.ui.cli.skills_cmds import (
    _cmd_skills_disable,
    _cmd_skills_enable,
    _cmd_skills_install,
    _cmd_skills_list,
    _cmd_skills_remove,
    _cmd_skills_update,
)
from agent6.ui.cli.system_cmds import _cmd_system_apparmor
from agent6.ui.cli.watch import _cmd_watch_target
from agent6.ui.cli.web_cmds import _cmd_web


def _first_markdown_line(text: str, max_len: int = 80) -> str:
    """First non-empty line of a markdown doc (a plan title), `#`/bullet stripped."""
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").lstrip("-*").strip()
        if line:
            return line[:max_len]
    return "(untitled plan)"


def _from_plan_task(plan_md: str, session_id: str) -> str:
    """The execution prompt for `run --from-plan`, LEADING with the plan title so
    a listing (the runs table, the DAG root, attach --json) shows the plan, not
    the 'The following plan was prepared...' boilerplate as the run's task."""
    title = _first_markdown_line(plan_md)
    if title.lower().startswith("plan:"):  # the '# Plan: <title>' convention
        title = title[len("plan:") :].strip() or title
    return f"Execute the prepared plan: {title}\n\n(from planning pass {session_id})\n\n{plan_md}"


def cli_main(argv: list[str] | None = None) -> int:
    """Console-script entry point: the boundary that sorts failures by fault.

    An ``OperatorError`` (a bad flag value, an unreadable operator file, an
    invalid config) prints as an ``ERROR:`` refusal at exit 2, no traceback.
    Anything else is a bug in agent6: a one-line ``ERROR: unexpected ...`` plus
    a pointer to a saved traceback, exit 1. Set ``AGENT6_DEBUG=1`` to re-raise
    the full traceback inline (for bug reports). ``main`` itself is left
    unguarded so tests and ``python -m`` see real tracebacks. argparse's
    ``SystemExit`` (bad args / --help) is not an ``Exception`` and passes
    through untouched.
    """
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("\nagent6: interrupted.", file=sys.stderr)
        return 130
    except OperatorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # top-level last resort; re-raised under AGENT6_DEBUG
        if os.environ.get("AGENT6_DEBUG") == "1":
            raise
        print(f"ERROR: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            fd, path = tempfile.mkstemp(prefix="agent6-crash-", suffix=".log")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
            print(f"  full traceback: {path}", file=sys.stderr)
        except OSError:
            pass  # never let crash-reporting itself crash the exit path
        print(
            "  re-run with AGENT6_DEBUG=1 to see it inline; please report this if it persists.",
            file=sys.stderr,
        )
        return 1


def _dispatch_run(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if getattr(args, "parallel", "") and (args.interactive or args.tui):
        print(
            "ERROR: --parallel cannot combine with -i or --tui"
            " (each lane runs headless and detached).",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "parallel", "") and args.session_id:
        print(
            "ERROR: --parallel cannot combine with --session-id (each lane mints its own id).",
            file=sys.stderr,
        )
        return 2
    if args.from_plan:
        if args.task:
            print(
                "ERROR: --from-plan is mutually exclusive with a task argument.",
                file=sys.stderr,
            )
            return 2
        resolved = _resolve_plan_session_id(args.from_plan)
        if resolved is None:
            return 2
        plan_md = read_operator_file(_plans_dir(Path.cwd()) / resolved / "plan.md")
        task = _from_plan_task(plan_md, resolved)
    elif not args.task:
        # No task: fall back to the most recent plan run, the common
        # "I just ran `agent6 plan`, now execute it" flow. At a TTY,
        # confirm before editing; non-interactively, refuse (a bare
        # `run` in a script should not silently start mutating).
        last_plan = _most_recent_plan_session_id(_plans_dir(Path.cwd()))
        if last_plan is None:
            print(
                "ERROR: 'run' needs a task (or --from-plan <id>); no prior plan found to execute.",
                file=sys.stderr,
            )
            return 2
        plan_md = read_operator_file(_plans_dir(Path.cwd()) / last_plan / "plan.md")
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
        task = _from_plan_task(plan_md, last_plan)
    else:
        task = args.task
    return _cmd_run(
        args.config,
        task,
        session_id=args.session_id,
        interactive=args.interactive,
        tui=args.tui,
        decompose=args.decompose,
        seed_from=getattr(args, "seed_from", ""),
        skills=tuple(args.skill),
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
        parallel_spec=getattr(args, "parallel", ""),
        pins=tuple(args.pins),
    )


def _dispatch_plan(args: argparse.Namespace) -> int:
    if args.plan_command == "show":
        return _cmd_plan_show(args.session_id)
    if args.plan_command == "edit":
        return _cmd_plan_edit(args.session_id)
    if not args.task:
        print(
            "ERROR: 'plan' needs a task argument (or `plan show/edit <id>`).",
            file=sys.stderr,
        )
        return 2
    return _cmd_run(
        args.config,
        args.task,
        session_id=args.session_id,
        mode="plan",
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
    )


def _dispatch_ask(args: argparse.Namespace) -> int:
    if args.ask_command == "list":
        return cmd_ask_list()
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
    if args.ask_session_latest or args.ask_session:
        digest = build_ask_session_digest(
            Path.cwd(), args.ask_session, latest=args.ask_session_latest
        )
        if digest is None:
            return 2
        prefix.append(digest)
    if args.ask_files:
        seeds = seed_files(Path.cwd(), args.ask_files)
        if seeds:
            prefix.append(seeds)
    if prefix:
        question = "\n\n".join([*prefix, question]) if question else "\n\n".join(prefix)
    return _cmd_run(
        args.config,
        question,
        mode="ask",
        interactive=repl,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
    )


def _dispatch_attach(args: argparse.Namespace) -> int:
    return _cmd_watch_target(
        args.target, tui=args.tui, json_out=args.json, since=args.since, raw=args.raw
    )


def _dispatch_sessions(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if args.sessions_command in (None, "list"):
        return _cmd_list()
    if args.sessions_command == "show":
        return _cmd_status(args.session_id, as_json=args.json)
    if args.sessions_command == "diff":
        return _cmd_diff(session_id=args.session_id, stat=args.stat, paths=tuple(args.paths))
    if args.sessions_command == "merge":
        return _cmd_merge(
            session_id=args.session_id,
            strategy=args.strategy,
            into=args.into,
            message=args.message,
        )
    if args.sessions_command == "compare":
        return _cmd_compare(session_ids=tuple(args.session_ids))
    if args.sessions_command == "commits":
        return _cmd_commits(session_id=args.session_id)
    if args.sessions_command == "stop":
        return _cmd_stop(session_id=args.session_id)
    if args.sessions_command == "prune":
        return _cmd_prune(delete_squashed=args.delete_squashed)
    if args.sessions_command == "dir":
        return _cmd_sessions_dir()
    if args.sessions_command == "rm":
        return _cmd_sessions_rm(session_id=args.session_id, asks=args.asks)
    if args.sessions_command == "transcript":
        return _cmd_history_transcript(
            args.session_id,
            as_json=args.as_json,
            no_thinking=args.no_thinking,
            tools=args.tools,
            seq=args.seq,
        )
    if args.sessions_command == "graph":
        return _cmd_history_graph(args.session_id)
    raise AssertionError("unreachable")  # pragma: no cover -- runs subparser is required


def _dispatch_tui(args: argparse.Namespace) -> int:
    return _cmd_tui()


def _dispatch_completions(args: argparse.Namespace) -> int:
    return cmd_completions(args.shell, print_only=args.print_only)


def _dispatch_web(args: argparse.Namespace) -> int:
    return _cmd_web(
        args.target,
        config_path=args.config,
        host=args.host,
        port=args.port,
        allow_non_loopback=args.allow_non_loopback,
    )


def _dispatch_prompt(args: argparse.Namespace) -> int:
    if args.prompt_command == "show":
        return _cmd_prompt_show(args.config, mode=args.mode)
    raise AssertionError("unreachable")  # pragma: no cover -- prompt subparser is required


def _dispatch_resume(args: argparse.Namespace) -> int:
    return _cmd_resume(
        args.config,
        args.session_id,
        force=args.force,
        tui=args.tui,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        steer=args.steer,
    )


def _dispatch_fork(args: argparse.Namespace) -> int:
    return _cmd_fork(
        args.config,
        args.session_id,
        at_turn=args.at_turn,
        new_session_id=args.new_session_id,
        no_run=args.no_run,
        tui=args.tui,
        budget_overrides=BudgetOverrides.from_args(args),
        sandbox_overrides=SandboxOverrides.from_args(args),
        steer=args.steer,
    )


def _dispatch_config(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if args.config_command == "show":
        return _cmd_config_show(args.config, as_json=args.as_json, key=args.key)
    if args.config_command == "fill":
        return _cmd_config_fill(args.config, force=args.force)
    if args.config_command == "path":
        return _cmd_config_path()
    if args.config_command == "presets":
        return _cmd_config_presets(args.config)
    if args.config_command == "get":
        return _cmd_config_get(args.config, args.key, machine=args.machine_file)
    if args.config_command == "set":
        return _cmd_config_set(args.key, args.value, repo=args.repo, machine=args.machine_file)
    if args.config_command == "unset":
        return _cmd_config_unset(args.key, repo=args.repo, machine=args.machine_file)
    if args.config_command == "add":
        return _cmd_config_add(args.key, args.value, repo=args.repo, machine=args.machine_file)
    if args.config_command == "remove":
        return _cmd_config_remove(args.key, args.value, repo=args.repo, machine=args.machine_file)
    if args.config_command == "fix":
        return _cmd_config_fix(machine=args.machine_file)
    raise AssertionError("unreachable")  # pragma: no cover -- config subparser is required


def _dispatch_check(args: argparse.Namespace) -> int:
    return _cmd_check(args.config, section=args.section)


def _dispatch_connect(args: argparse.Namespace) -> int:
    return _cmd_connect(provider=args.provider, to_repo=args.repo, verify=args.verify)


def _dispatch_model(args: argparse.Namespace) -> int:
    return _cmd_model(
        args.config,
        role=args.role,
        provider=args.provider,
        model=args.model,
        thinking=args.thinking,
        to_repo=args.repo,
    )


def _dispatch_memory(args: argparse.Namespace) -> int:
    if args.memory_command == "add":
        return _cmd_memory_add(args.scope, args.body)
    if args.memory_command == "list":
        return _cmd_memory_list(args.scope or None, include_invalidated=args.all)
    if args.memory_command == "invalidate":
        return _cmd_memory_invalidate(args.memory_id, args.reason)
    if args.memory_command == "pin":
        return _cmd_memory_pin(args.memory_id)
    if args.memory_command == "unpin":
        return _cmd_memory_unpin(args.memory_id)
    raise AssertionError("unreachable")  # pragma: no cover -- memory subparser is required


def _dispatch_skills(args: argparse.Namespace) -> int:
    if args.skills_command == "install":
        return _cmd_skills_install(args.url, force=args.force)
    if args.skills_command == "update":
        return _cmd_skills_update(args.name)
    if args.skills_command == "list":
        return _cmd_skills_list()
    if args.skills_command == "enable":
        return _cmd_skills_enable(args.name, always=args.always, repo=args.repo)
    if args.skills_command == "disable":
        return _cmd_skills_disable(args.name, repo=args.repo)
    if args.skills_command == "remove":
        return _cmd_skills_remove(args.name)
    raise AssertionError("unreachable")  # pragma: no cover -- skills subparser is required


def _dispatch_history(args: argparse.Namespace) -> int:
    if args.history_command == "search":
        return _cmd_history_search(args.query, fixed=not args.regex, session_id=args.session)
    raise AssertionError("unreachable")  # pragma: no cover -- history subparser is required


def _dispatch_init(args: argparse.Namespace) -> int:
    return _cmd_init(ecosystem=args.ecosystem, assume_yes=args.yes)


def _dispatch_review(args: argparse.Namespace) -> int:
    return _cmd_review(
        args.config,
        base=args.base,
        head=args.head,
        paths=tuple(args.paths),
        model_override=args.model,
        reviewers=args.reviewers,
        personas=args.personas,
    )


def _dispatch_mcp(args: argparse.Namespace) -> int:
    if args.mcp_command == "serve":
        return _cmd_mcp_serve(args.config)
    if args.mcp_command == "connect":
        return cmd_mcp_connect(
            args.name,
            command=args.server_command,
            url=args.url,
            token_env=args.token_env,
            pass_env=args.pass_env,
            to_repo=args.to_repo,
        )
    return cmd_mcp_list()


def _dispatch_machine(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if args.machine_command == "check":
        return _cmd_machine_check(args.file)
    if args.machine_command == "test":
        return _cmd_machine_test(args.file, blackboard=args.blackboard)
    if args.machine_command == "graph":
        return _cmd_machine_graph(args.file, fmt=args.format)
    if args.machine_command == "run":
        return _cmd_machine_run(
            args.file,
            exit_on_wait=args.exit_on_wait,
            disable_sandbox=args.dangerously_disable_sandbox,
            auto_approve=args.auto_approve,
        )
    if args.machine_command == "status":
        return _cmd_machine_status(args.machine_id)
    if args.machine_command == "poke":
        return _cmd_machine_poke(args.machine_id, data=args.data, message=args.message)
    if args.machine_command == "replay":
        return _cmd_machine_replay(args.machine_id)
    if args.machine_command == "create":
        return _cmd_machine_create(args.task, output=args.output, max_attempts=args.max_attempts)
    raise AssertionError("unreachable")  # pragma: no cover -- machine subparser is required


def _dispatch_system(args: argparse.Namespace) -> int:
    if args.system_command == "apparmor":
        return _cmd_system_apparmor(args.action)
    raise AssertionError("unreachable")  # pragma: no cover -- system subparser is required


# command -> per-family dispatcher. Mirrors the `_*_args.py` parser grouping:
# one handler per top-level command, each fanning out over its own subcommands.
_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _dispatch_run,
    "plan": _dispatch_plan,
    "ask": _dispatch_ask,
    "attach": _dispatch_attach,
    "sessions": _dispatch_sessions,
    "tui": _dispatch_tui,
    "completions": _dispatch_completions,
    "web": _dispatch_web,
    "prompt": _dispatch_prompt,
    "resume": _dispatch_resume,
    "fork": _dispatch_fork,
    "config": _dispatch_config,
    "check": _dispatch_check,
    "connect": _dispatch_connect,
    "model": _dispatch_model,
    "memory": _dispatch_memory,
    "skills": _dispatch_skills,
    "history": _dispatch_history,
    "init": _dispatch_init,
    "review": _dispatch_review,
    "machine": _dispatch_machine,
    "mcp": _dispatch_mcp,
    "acp": lambda _args: serve_acp(),
    "system": _dispatch_system,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argcomplete.autocomplete(parser)
    raw = sys.argv[1:] if argv is None else argv
    # Bare `agent6` (no command, no -h/--version): print help rather than the
    # terse argparse "required: <command>" error. The boring, expected thing.
    if _command_index(raw) is None and not any(a in ("-h", "--help", "--version") for a in raw):
        parser.print_help()
        return 0
    args = parser.parse_args(_inject_default_verb(raw))
    # `agent6 system ...` is a privileged host-setup command that legitimately
    # runs as root (it writes /etc and reloads AppArmor); it does not run the
    # LLM, so it is exempt from the "no LLM agent as root" gate.
    if args.command != "system":
        root_rc = _enforce_root_policy(getattr(args, "allow_root", False))
        if root_rc is not None:
            return root_rc
    handler = _DISPATCH.get(args.command)
    if handler is None:  # pragma: no cover -- the top-level subparser is required
        parser.error("unknown command")
    try:
        return handler(args)
    except EventWriteError as exc:
        # A lifecycle stopped because the durable run journal could not be
        # appended; its finally already released locks and egress. One report
        # here beats a per-command arm in every lifecycle.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
