# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""The agent6 command-line interface: the entry point and one dispatcher per command family."""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
import tempfile
import traceback
from collections.abc import Callable

import argcomplete

from agent6 import errors, events, paths
from agent6.ui.cli import _common, _terminal_guard
from agent6.ui.cli import parser as cli_parser


def _first_markdown_line(text: str, max_len: int = 80) -> str:
    """Return the first non-empty line of a markdown doc, heading and bullet marks stripped."""
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").lstrip("-*").strip()
        if line:
            return line[:max_len]
    return "(untitled plan)"


def _plan_title(plan_md: str) -> str:
    """Return a plan's title: its first line, less the `# Plan: <title>` convention."""
    title = _first_markdown_line(plan_md)
    if title.lower().startswith("plan:"):
        title = title[len("plan:") :].strip() or title
    return title


def _from_plan_task(plan_md: str, session_id: str) -> str:
    """Return the task for `run --from <plan>`, the plan's title first so listings show it."""
    title = _plan_title(plan_md)
    return f'Execute the prepared plan: {title}\n\n<plan id="{session_id}">\n{plan_md}\n</plan>'


def _plan_text_for_run(plan_path: pathlib.Path, session_id: str) -> str | None:
    """Return a plan's non-empty markdown, or None after naming the unusable plan."""
    if not plan_path.is_file():
        _common.error(f"plan {session_id!r} has no plan.md")
        return None
    plan_md = errors.read_operator_file(plan_path)
    if not plan_md.strip():
        _common.error(f"plan {session_id!r} has an empty plan.md")
        return None
    return plan_md


def cli_main(argv: list[str] | None = None) -> int:
    """Run the CLI, sorting failures by fault.

    An `OperatorError` prints as a refusal at exit 2, no traceback. Anything else
    is a bug: one line plus a saved traceback, exit 1; `AGENT6_DEBUG=1` re-raises it.
    `main` stays unguarded so tests see real tracebacks.

    Args:
        argv: The arguments; `sys.argv[1:]` when None.

    Returns:
        The exit code.
    """
    with _terminal_guard.guarded_terminal():
        try:
            return main(argv)
        except KeyboardInterrupt:
            print("\nagent6: interrupted.", file=sys.stderr)
            return 130
        except errors.OperatorError as exc:
            _common.error(f"{exc}")
            return 2
        except Exception as exc:  # the last resort; re-raised under AGENT6_DEBUG
            if os.environ.get("AGENT6_DEBUG") == "1":
                raise
            _common.error(f"unexpected {type(exc).__name__}: {exc}")
            try:
                fd, path = tempfile.mkstemp(prefix="agent6-crash-", suffix=".log")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    traceback.print_exc(file=fh)
                print(f"  full traceback: {path}", file=sys.stderr)
            except OSError:
                pass  # crash reporting never crashes the exit path
            print(
                "  re-run with AGENT6_DEBUG=1 to see it inline; if it persists, report it:"
                " https://github.com/agent6-dev/agent6/issues",
                file=sys.stderr,
            )
            return 1


def _dispatch_run(args: argparse.Namespace) -> int:  # noqa: PLR0911, PLR0912
    """Return the exit code of `agent6 run`: the task, a plan to execute, or the REPL."""
    from agent6.app import _setup  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import (  # noqa: PLC0415  # noqa: PLC0415
        _session_prompt,
        plan_watch,
        run,
    )

    if args.interactive and args.tui:
        _common.error("-i cannot combine with --tui (the REPL and the TUI both want the terminal).")
        return 2
    if args.interactive and not _session_prompt.prompting_is_possible():
        # Without a terminal the REPL's first read would end the run mid-task.
        _common.error("-i needs a TTY in the foreground process group; drop -i for a headless run.")
        return 2
    parallel = getattr(args, "parallel", "")
    if parallel and (args.interactive or args.tui):
        _common.error(
            "--parallel cannot combine with -i or --tui (each lane runs headless and detached)."
        )
        return 2
    if parallel and args.session_id:
        _common.error("--parallel cannot combine with --session-id (each lane mints its own id).")
        return 2
    if parallel and args.standing:
        _common.error(
            "--parallel cannot combine with --standing (the lanes take no standing goal)."
        )
        return 2
    seed_from, source_session_id = getattr(args, "seed_from", ""), ""
    if not args.task and seed_from:
        # A plan id alone runs that plan; seeding the same text again would double it.
        from agent6.sessions import id  # noqa: PLC0415  # noqa: PLC0415

        try:
            layout = id.resolve_session(paths.state_dir(pathlib.Path.cwd()), seed_from)
        except id.SessionIdError as exc:
            _common.error(f"{exc}")
            return 2
        if layout.subdir != "plans":
            _common.error(
                "'run' needs a task; --from <id> seeds one, and a plan id alone runs that plan."
            )
            return 2
        plan_md = _plan_text_for_run(layout.session_dir / "plan.md", layout.session_id)
        if plan_md is None:
            return 2
        task, source_session_id = _from_plan_task(plan_md, layout.session_id), layout.session_id
        seed_from = ""
    elif not args.task:
        # No task: the most recent plan, confirmed at a TTY, refused in a script.
        last_plan = plan_watch._most_recent_plan_session_id(_common._plans_dir(pathlib.Path.cwd()))
        if last_plan is None:
            _common.error(
                "'run' needs a task and there is no saved plan. Start with"
                ' `agent6 run "TASK"`, or create one with `agent6 plan "TASK"`.'
            )
            return 2
        plan_md = _plan_text_for_run(
            _common._plans_dir(pathlib.Path.cwd()) / last_plan / "plan.md", last_plan
        )
        if plan_md is None:
            return 2
        title = _plan_title(plan_md)
        if not sys.stdin.isatty():
            _common.error(
                f"'run' needs a task. Most recent plan is {last_plan}"
                f" ({title}); execute it with: agent6 run --from {last_plan}"
            )
            return 2
        print(f"[agent6] No task given. Most recent plan: {last_plan}  ({title})")
        ans = _common.safe_input("Execute it now? [Y/n]: ")
        if ans is None or ans.lower() in ("n", "no"):
            print(f"Aborted. Run it later: agent6 run --from {last_plan}")
            return 0
        task, source_session_id = _from_plan_task(plan_md, last_plan), last_plan
    else:
        task = args.task
    session_id = _minted_session_id(args.session_id, "run")
    rc = run._cmd_run(
        args.config,
        task,
        session_id=session_id,
        interactive=args.interactive,
        tui=args.tui,
        decompose=args.decompose,
        seed_from=seed_from,
        source_session_id=source_session_id,
        skills=tuple(args.skill),
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
        parallel_spec=getattr(args, "parallel", ""),
        standing_goal=getattr(args, "standing", ""),
        pins=tuple(args.pins),
        model=getattr(args, "model", ""),
    )
    # A fan-out ends in its compare summary and the TUI owns its screen; neither prompts.
    if getattr(args, "parallel", "") or args.tui:
        return rc
    return _prompt_for_the_next_input(args, rc, session_id)


def _minted_session_id(explicit: str, mode: str) -> str:
    """Return this invocation's session id, minted before the run when the operator named none.

    The end-of-session prompt then offers the session this invocation created, not
    the repo's newest. Minting reserves nothing on disk.
    """
    from agent6 import kinds  # noqa: PLC0415  # noqa: PLC0415
    from agent6.sessions import id  # noqa: PLC0415  # noqa: PLC0415

    if explicit:
        return explicit
    return id.unused_session_id(paths.state_dir(pathlib.Path.cwd()), kinds.session_bucket(mode))


def _prompt_for_the_next_input(  # noqa: PLR0911
    args: argparse.Namespace, rc: int, session_id: str
) -> int:
    """Ask for the next input instead of ending, when someone is there to type.

    `run` and `plan` end this way; an ask stays a one-shot. Every follow-up runs
    under this invocation's flags.

    Args:
        args: The parsed command line.
        rc: The execution's exit code.
        session_id: This invocation's session.

    Returns:
        The last execution's exit code.
    """
    if rc == 2:
        return rc

    from agent6.app import _setup  # noqa: PLC0415  # noqa: PLC0415
    from agent6.sessions import id  # noqa: PLC0415  # noqa: PLC0415
    from agent6.sessions import manifest as sessions_manifest  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import _session_prompt  # noqa: PLC0415

    if not _session_prompt.prompting_is_possible():
        return rc
    try:
        layout = _common.resolve_or_newest_layout(pathlib.Path.cwd(), session_id)
    except id.SessionIdError:
        # A refused run discarded its husk, so there is no session to continue.
        return rc
    if layout is None or not layout.session_dir.is_dir():
        return rc
    # A parked start never ran; its next step is the resume line already printed.
    with contextlib.suppress(sessions_manifest.ManifestError):
        manifest = sessions_manifest.read_manifest(layout.session_dir)
        if manifest.parked_task or manifest.mode == "ask":
            return rc
    if not _session_prompt.follow_up_on_offer(layout.session_dir):
        return rc
    return _session_prompt.end_of_session_prompt(
        rc=rc,
        session_id=layout.session_id,
        session_dir=layout.session_dir,
        ask=input,
        config_path=args.config,
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        model=getattr(args, "model", ""),
    )


def _dispatch_plan(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 plan` or its show and edit verbs."""
    from agent6.app import _setup  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import (  # noqa: PLC0415  # noqa: PLC0415
        plan_watch,
        run,
    )

    if args.plan_command == "show":
        return plan_watch._cmd_plan_show(args.session_id)
    if args.plan_command == "edit":
        return plan_watch._cmd_plan_edit(args.session_id)
    if not args.task:
        _common.error("'plan' needs a task argument (or `plan show/edit <id>`).")
        return 2
    session_id = _minted_session_id(args.session_id, "plan")
    rc = run._cmd_run(
        args.config,
        args.task,
        session_id=session_id,
        mode="plan",
        tui=args.tui,
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
        model=getattr(args, "model", ""),
    )
    # The TUI owns its screen; it does not prompt.
    if args.tui:
        return rc
    return _prompt_for_the_next_input(args, rc, session_id)


def _dispatch_ask(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 ask`, one-shot or as a REPL."""
    from agent6.app import _setup  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import (  # noqa: PLC0415  # noqa: PLC0415
        _ask,
        _session_prompt,
        run,
    )

    if args.interactive and not _session_prompt.prompting_is_possible():
        _common.error("-i needs a TTY in the foreground process group; drop -i for a one-shot ask.")
        return 2
    # A REPL with -i, or with no question at an interactive foreground stdin.
    repl = args.interactive or (not args.task and _session_prompt.prompting_is_possible())
    if not args.task and not repl:
        _common.error(
            "'ask' needs a question (in quotes), or -i for the REPL on a foreground terminal."
        )
        return 2
    question = args.task
    prefix: list[str] = []
    source_session_id = ""
    if args.ask_session_latest or args.ask_session:
        seed = _ask.build_session_seed(
            pathlib.Path.cwd(), args.ask_session, latest=args.ask_session_latest
        )
        if seed is None:
            return 2
        prefix.append(seed.text)
        source_session_id = seed.source_session_id
    if args.ask_files:
        seeds = _ask.seed_files(pathlib.Path.cwd(), args.ask_files)
        if seeds:
            prefix.append(seeds)
    if prefix:
        question = "\n\n".join([*prefix, question]) if question else "\n\n".join(prefix)
    return run._cmd_run(
        args.config,
        question,
        mode="ask",
        interactive=repl,
        source_session_id=source_session_id,
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        preset=getattr(args, "preset", ""),
        model=getattr(args, "model", ""),
    )


def _dispatch_attach(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 attach`."""
    from agent6.ui.cli import watch  # noqa: PLC0415  # noqa: PLC0415

    return watch._cmd_watch_target(
        args.target,
        tui=args.tui,
        json_out=args.json,
        since=args.since,
        raw=args.raw,
        config_path=args.config,
    )


def _dispatch_steer(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 steer`."""
    from agent6.ui.cli import steer_cmd  # noqa: PLC0415  # noqa: PLC0415

    return steer_cmd._cmd_steer(args.target, args.text, now=args.now)


def _dispatch_stop(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 stop`."""
    from agent6.ui.cli import stop_cmd  # noqa: PLC0415  # noqa: PLC0415

    return stop_cmd._cmd_stop(args.session_id, all_sessions=args.all, after_step=args.after_step)


def _dispatch_answer(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 answer`."""
    from agent6.ui.cli import answer_cmd  # noqa: PLC0415  # noqa: PLC0415

    return answer_cmd._cmd_answer(args.target, tuple(args.answers))


def _dispatch_exec(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 exec [SESSION --] CMD...`, run in the session's network."""
    from agent6.config import (  # noqa: PLC0415
        ConfigError,
        layer,
    )
    from agent6.ui.cli import net_cmds  # noqa: PLC0415  # noqa: PLC0415

    # Only the first `--` separates the optional session; a later one belongs to the command.
    rest: list[str] = list(args.rest)
    target = ""
    if "--" in rest:
        split = rest.index("--")
        before, argv = rest[:split], tuple(rest[split + 1 :])
        if len(before) > 1:
            _common.error(f"at most one session id before `--`, got {' '.join(before)!r}.")
            return 2
        target = before[0] if before else ""
    else:
        argv = tuple(rest)
    if not argv:
        _common.error("give a command (after `--` when naming a session).")
        return 2

    layout = _common.resolve_target(target)
    if layout is None:
        return 2
    try:
        cfg = layer.load_effective(pathlib.Path.cwd(), args.config).config
    except ConfigError as exc:
        _common.error(f"{exc}")
        return 2
    return net_cmds.exec_in_session(layout, cfg, pathlib.Path.cwd(), argv)


def _dispatch_forward(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 forward`, a port of the session's network."""
    from agent6.sessions import ipc  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import net_cmds  # noqa: PLC0415  # noqa: PLC0415

    target, port = args.target, args.port
    if port is None and target.isdigit():
        # A bare number is a port of the newest session; a numeric session id needs both args.
        target, port = "", int(target)

    layout = _common.resolve_target(target)
    if layout is None:
        return 2
    if port is None:
        ports = ipc.listening_ports(layout.session_dir)
        if not ports:
            reason = (
                f"{layout.session_id} is listening on nothing yet."
                if ipc.read_session_netns_pid(layout.session_dir) is not None
                else net_cmds.no_session_network_reason(layout)
            )
            _common.refuse(f"{reason}")
            return 2
        print(f"{layout.session_id} is listening on: {', '.join(str(p) for p in ports)}")
        return 0
    return net_cmds.forward(layout, port, args.local_port)


def _dispatch_sessions(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Return the exit code of a `agent6 sessions` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import (  # noqa: PLC0415
        history_cmds,
        sessions_cmds,
        sessions_compare,
        sessions_merge,
        sessions_show,
    )

    if args.sessions_command == "list":
        return sessions_cmds._cmd_list(as_json=args.list_json, lanes=args.list_lanes)
    if args.sessions_command == "show":
        return sessions_show._cmd_status(args.session_id, as_json=args.json)
    if args.sessions_command == "diff":
        return sessions_cmds._cmd_diff(
            session_id=args.session_id, stat=args.stat, paths=tuple(args.paths)
        )
    if args.sessions_command == "merge":
        return sessions_merge._cmd_merge(
            session_id=args.session_id,
            strategy=args.strategy,
            into=args.into,
            message=args.message,
            config_path=args.config,
        )
    if args.sessions_command == "compare":
        return sessions_compare._cmd_compare(
            session_ids=tuple(args.session_ids),
            config_path=args.config,
            rejudge=args.rejudge,
        )
    if args.sessions_command == "review":
        from agent6.ui.cli import sessions_review  # noqa: PLC0415  # noqa: PLC0415

        return sessions_review._cmd_sessions_review(
            args.config, session_id=args.session_id, model=args.model
        )
    if args.sessions_command == "commits":
        return sessions_cmds._cmd_commits(session_id=args.session_id)
    if args.sessions_command == "prune":
        return sessions_merge._cmd_prune(
            delete_squashed=args.delete_squashed, config_path=args.config
        )
    if args.sessions_command == "dir":
        return sessions_cmds._cmd_sessions_dir(args.session_id)
    if args.sessions_command == "rm":
        return sessions_cmds._cmd_sessions_rm(session_id=args.session_id, asks=args.asks)
    if args.sessions_command == "transcript":
        return history_cmds._cmd_history_transcript(
            args.session_id,
            as_json=args.as_json,
            no_thinking=args.no_thinking,
            tools=args.tools,
            seq=args.seq,
        )
    if args.sessions_command == "graph":
        return history_cmds._cmd_history_graph(args.session_id)
    raise AssertionError("unreachable")  # pragma: no cover -- earlier branches cover every verb


def _dispatch_tui(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 tui`: the hub, or a target's dashboard."""
    from agent6.ui.cli import (  # noqa: PLC0415  # noqa: PLC0415
        plan_watch,
        watch,
    )

    if args.target:
        return watch._cmd_watch_target(
            args.target, tui=True, json_out=False, since=None, raw=False, config_path=args.config
        )
    return plan_watch._cmd_tui(args.config)


def _dispatch_completions(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 completions`."""
    from agent6.ui.cli import completions_cmd  # noqa: PLC0415  # noqa: PLC0415

    return completions_cmd.cmd_completions(args.shell, print_only=args.print_only)


def _dispatch_web(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 web`."""
    from agent6.ui.cli import web_cmds  # noqa: PLC0415  # noqa: PLC0415

    return web_cmds._cmd_web(
        args.target,
        config_path=args.config,
        host=args.host,
        port=args.port,
        allow_non_loopback=args.allow_non_loopback,
    )


def _dispatch_prompt(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 prompt` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import prompt_cmds  # noqa: PLC0415  # noqa: PLC0415

    if args.prompt_command == "show":
        return prompt_cmds._cmd_prompt_show(args.config, mode=args.mode, as_json=args.json)
    raise AssertionError("unreachable")  # pragma: no cover -- prompt subparser is required


def _dispatch_resume(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 resume`, on the named session or the newest resumable one."""
    from agent6.app import (  # noqa: PLC0415  # noqa: PLC0415
        _setup,
        resume,
    )
    from agent6.sessions import id  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import _session_prompt  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import resume as cli_resume  # noqa: PLC0415  # noqa: PLC0415
    from agent6.viewmodel import newest_session_dir  # noqa: PLC0415

    if getattr(args, "interactive", False) and args.tui:
        _common.error("-i cannot combine with --tui (the REPL and the TUI both want the terminal).")
        return 2
    if getattr(args, "interactive", False) and not _session_prompt.prompting_is_possible():
        _common.error(
            "-i needs a TTY in the foreground process group; drop -i for a headless resume."
        )
        return 2
    session_id = args.session_id
    state = paths.state_dir(pathlib.Path.cwd())
    if session_id:
        with contextlib.suppress(id.SessionIdError):
            session_id = id.resolve_session(state, session_id).session_id
    else:
        # Every resumable bucket, so a plan or an ask is found too.
        latest = newest_session_dir(resume.resumable_bucket_dirs(state))
        if latest is None:
            # An empty state dir is not a fault.
            print(
                'nothing to resume yet. Start a session with `agent6 run "<task>"`.',
                file=sys.stderr,
            )
            return 2
        session_id = latest.name
        _common.note(f"resuming most recent session: {session_id}")
    rc = cli_resume._cmd_resume(
        args.config,
        session_id,
        force=args.force,
        tui=args.tui,
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        preset=args.preset,
        steer=args.steer,
        interactive=getattr(args, "interactive", False),
        model=getattr(args, "model", ""),
    )
    # A resumed execution ends the way a fresh one does; the TUI owns its screen.
    return rc if args.tui else _prompt_for_the_next_input(args, rc, session_id)


def _dispatch_fork(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 fork`."""
    from agent6.app import _setup  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import fork  # noqa: PLC0415  # noqa: PLC0415

    return fork._cmd_fork(
        args.config,
        args.session_id,
        at_turn=args.at_turn,
        new_session_id=args.new_session_id,
        no_run=args.no_run,
        tui=args.tui,
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        steer=args.steer,
    )


def _dispatch_config(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Return the exit code of a `agent6 config` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import config_cmds  # noqa: PLC0415

    if args.config_command == "show":
        return config_cmds._cmd_config_show(
            args.config,
            as_json=args.as_json,
            keys=args.keys,
            descriptions=args.descriptions,
            machine=args.machine_file,
        )
    if args.config_command == "fill":
        return config_cmds._cmd_config_fill(force=args.force)
    if args.config_command == "path":
        return config_cmds._cmd_config_path()
    if args.config_command == "presets":
        return config_cmds._cmd_config_presets(args.config)
    if args.config_command == "get":
        return config_cmds._cmd_config_get(args.config, args.key, machine=args.machine_file)
    if args.config_command == "set":
        return config_cmds._cmd_config_set(
            args.key, args.value, repo=args.repo, machine=args.machine_file, config_path=args.config
        )
    if args.config_command == "unset":
        return config_cmds._cmd_config_unset(
            args.key, repo=args.repo, machine=args.machine_file, config_path=args.config
        )
    if args.config_command in ("add", "remove"):
        return config_cmds._config_list_edit(
            args.key,
            args.value,
            repo=args.repo,
            machine=args.machine_file,
            add=args.config_command == "add",
        )
    if args.config_command == "fix":
        return config_cmds._cmd_config_fix(machine=args.machine_file)
    raise AssertionError("unreachable")  # pragma: no cover -- config subparser is required


def _dispatch_check(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 check`."""
    from agent6.ui.cli import check_cmds  # noqa: PLC0415  # noqa: PLC0415

    return check_cmds._cmd_check(args.config, section=args.section)


def _dispatch_connect(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 connect`."""
    from agent6.ui.cli import connect  # noqa: PLC0415  # noqa: PLC0415

    return connect._cmd_connect(
        provider=args.provider, to_repo=args.repo, verify=args.verify, logout=args.logout
    )


def _dispatch_model(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 model`."""
    from agent6.ui.cli import model  # noqa: PLC0415  # noqa: PLC0415

    return model._cmd_model(
        args.config,
        role=args.role,
        route=args.route,
        effort=args.effort,
        to_repo=args.repo,
    )


def _dispatch_memory(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 memory` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import memory_cmds  # noqa: PLC0415

    if args.memory_command == "add":
        return memory_cmds._cmd_memory_add(args.name, args.body)
    if args.memory_command == "list":
        return memory_cmds._cmd_memory_list()
    if args.memory_command == "show":
        return memory_cmds._cmd_memory_show(args.name)
    if args.memory_command == "rm":
        return memory_cmds._cmd_memory_rm(args.name)
    if args.memory_command == "decisions":
        return memory_cmds._cmd_memory_decisions()
    raise AssertionError("unreachable")  # pragma: no cover -- memory subparser is required


def _dispatch_skills(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 skills` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import skills_cmds  # noqa: PLC0415

    if args.skills_command == "install":
        return skills_cmds._cmd_skills_install(args.url, force=args.force, config_path=args.config)
    if args.skills_command == "update":
        return skills_cmds._cmd_skills_update(args.name)
    if args.skills_command == "list":
        return skills_cmds._cmd_skills_list(args.config)
    if args.skills_command == "enable":
        return skills_cmds._cmd_skills_enable(
            args.name, always=args.always, repo=args.repo, config_path=args.config
        )
    if args.skills_command == "disable":
        return skills_cmds._cmd_skills_disable(args.name, repo=args.repo, config_path=args.config)
    if args.skills_command == "remove":
        return skills_cmds._cmd_skills_remove(args.name, args.config)
    raise AssertionError("unreachable")  # pragma: no cover -- skills subparser is required


def _dispatch_ps(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 ps`."""
    from agent6.ui.cli import ps_cmd  # noqa: PLC0415  # noqa: PLC0415

    return ps_cmd.cmd_ps(as_json=args.json, lanes=args.lanes)


def _dispatch_history(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 history` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import history_cmds  # noqa: PLC0415  # noqa: PLC0415

    if args.history_command == "search":
        if not args.query:
            _common.error("'history' needs a query, the pattern to search for.")
            return 2
        return history_cmds._cmd_history_search(
            args.query, fixed=not args.regex, session_id=args.session
        )
    raise AssertionError("unreachable")  # pragma: no cover -- history subparser is required


def _dispatch_init(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 init`."""
    from agent6.ui.cli import init_cmds  # noqa: PLC0415  # noqa: PLC0415

    return init_cmds._cmd_init(
        ecosystem=args.ecosystem, assume_yes=args.yes, config_path=args.config
    )


def _dispatch_review(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 review`."""
    from agent6.ui.cli import review_cmds  # noqa: PLC0415  # noqa: PLC0415

    return review_cmds._cmd_review(
        args.config,
        base=args.base,
        head=args.head,
        paths=tuple(args.paths),
        model=args.model,
        reviewers=args.reviewers,
        personas=args.personas,
    )


def _dispatch_mcp(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 mcp` verb."""
    from agent6.ui import mcp_server  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import mcp_connect  # noqa: PLC0415

    if args.mcp_command == "serve":
        return mcp_server.run_server(args.config)
    if args.mcp_command == "connect":
        return mcp_connect.cmd_mcp_connect(
            args.name,
            command=args.server_command,
            url=args.url,
            token_env=args.token_env,
            pass_env=args.pass_env,
            to_repo=args.to_repo,
            config_path=args.config,
        )
    if args.mcp_command == "remove":
        return mcp_connect.cmd_mcp_remove(args.name, to_repo=args.to_repo, config_path=args.config)
    return mcp_connect.cmd_mcp_list(args.config)


def _dispatch_machine(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Return the exit code of a `agent6 machine` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.app.machine import create  # noqa: PLC0415  # noqa: PLC0415
    from agent6.app.machine import run as machine_run  # noqa: PLC0415  # noqa: PLC0415
    from agent6.ui.cli import (  # noqa: PLC0415
        machine_check,
        machine_cmds,
    )

    if args.machine_command == "list":
        return machine_cmds._cmd_machine_list()
    if args.machine_command == "check":
        return machine_check._cmd_machine_check(args.file, config_path=args.config)
    if args.machine_command == "test":
        return machine_check._cmd_machine_test(
            args.file, blackboard=args.blackboard, config_path=args.config
        )
    if args.machine_command == "graph":
        return machine_check._cmd_machine_graph(args.file, fmt=args.format)
    if args.machine_command == "run":
        return machine_run.run_machine(
            args.file,
            machine_cmds._machine_frontend(),
            config_path=args.config,
            exit_on_wait=args.exit_on_wait,
            disable_sandbox=args.dangerously_disable_sandbox,
            auto_approve=args.auto_approve,
            no_commands=args.no_commands,
        )
    if args.machine_command == "status":
        return machine_cmds._cmd_machine_status(args.machine_id)
    if args.machine_command == "poke":
        return machine_cmds._cmd_machine_poke(args.machine_id, data=args.data, message=args.message)
    if args.machine_command == "stop":
        return machine_cmds._cmd_machine_stop(args.machine_id)
    if args.machine_command == "replay":
        return machine_cmds._cmd_machine_replay(args.machine_id)
    if args.machine_command == "create":
        return create.create_machine(
            args.task,
            machine_cmds._machine_frontend(),
            output=args.output,
            max_attempts=args.max_attempts,
            config_path=args.config,
        )
    raise AssertionError("unreachable")  # pragma: no cover -- machine subparser is required


def _dispatch_acp(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 acp`."""
    from agent6.ui.acp import serve_acp  # noqa: PLC0415

    return serve_acp(config_path=args.config)


def _dispatch_system(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 system` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import system_cmds  # noqa: PLC0415  # noqa: PLC0415

    if args.system_command == "apparmor":
        return system_cmds._cmd_system_apparmor(args.action)
    raise AssertionError("unreachable")  # pragma: no cover -- system subparser is required


# One handler per top-level command, mirroring the `_*_args.py` parser grouping.
_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "run": _dispatch_run,
    "plan": _dispatch_plan,
    "ask": _dispatch_ask,
    "attach": _dispatch_attach,
    "steer": _dispatch_steer,
    "stop": _dispatch_stop,
    "answer": _dispatch_answer,
    "exec": _dispatch_exec,
    "forward": _dispatch_forward,
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
    "ps": _dispatch_ps,
    "init": _dispatch_init,
    "review": _dispatch_review,
    "machine": _dispatch_machine,
    "mcp": _dispatch_mcp,
    "acp": _dispatch_acp,
    "system": _dispatch_system,
}


def main(argv: list[str] | None = None) -> int:
    """Return the handler's exit code for the command line; unguarded, so tests see tracebacks."""
    parser = cli_parser.build_parser()
    argcomplete.autocomplete(parser)
    raw = sys.argv[1:] if argv is None else argv
    # A bare `agent6` prints help rather than argparse's "required: <command>".
    if not raw:
        parser.print_help()
        return 0
    args = parser.parse_args(cli_parser._inject_default_verb(raw))
    # `system` is host setup that runs as root and runs no LLM, so the root gate exempts it.
    if args.command != "system":
        root_rc = _common._enforce_root_policy(getattr(args, "allow_root", False))
        if root_rc is not None:
            return root_rc
    handler = _DISPATCH.get(args.command)
    if handler is None:  # pragma: no cover -- the top-level subparser is required
        parser.error("unknown command")
    try:
        return handler(args)
    except events.EventWriteError as exc:
        # The run journal could not be appended; the lifecycle's finally already cleaned up.
        _common.error(f"{exc}")
        return 1
