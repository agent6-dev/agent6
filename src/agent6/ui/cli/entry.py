# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
# PYTHON_ARGCOMPLETE_OK
"""The entry point and one dispatcher per command family.

Every dispatcher imports its command module inside the function, so `agent6 --help`
and completion start without importing the engine; the per-file `PLC0415` waiver
in pyproject.toml states that once.
"""

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

from agent6 import errors, event_log, paths
from agent6.sessions import process
from agent6.ui.cli import _common, _terminal_guard
from agent6.ui.cli import parser as cli_parser
from agent6.viewmodel import _session_status


def _plan_title(plan_md: str) -> str:
    """Return the plan's first line, without its Plan prefix, with a mark when clipped."""
    title = "(untitled plan)"
    for raw in plan_md.splitlines():
        line = raw.strip().lstrip("#").lstrip("-*").strip()
        if line:
            title = line
            break
    if title.lower().startswith("plan:"):
        title = title[len("plan:") :].strip() or title
    return _session_status.task_snippet(title, max_chars=80)


def _from_plan_task(plan_md: str, session_id: str) -> str:
    """Return the task for `run --from <plan>`, the plan's title first so listings show it."""
    title = _plan_title(plan_md)
    return f'Execute the prepared plan: {title}\n\n<plan id="{session_id}">\n{plan_md}\n</plan>'


def _unanswered_questions(plan_md: str) -> list[str]:
    """Return the plan's `**Q:**` lines that no answer text follows, in order.

    An answer is text on the question's `**A:**` line or on the line under a blank one.

    Args:
        plan_md: The plan's markdown; its lines may be indented or quoted with `>`.

    Returns:
        Each question's text after its `**Q:**` marker; empty when none is unanswered.
    """
    unanswered: list[str] = []
    pending: str | None = None
    answer_line_open = False
    for line in plan_md.splitlines():
        text = line.lstrip().lstrip(">").strip()
        if text.startswith("**Q:**"):
            if pending is not None:
                unanswered.append(pending)
            pending, answer_line_open = text.removeprefix("**Q:**").strip(), False
        elif pending is not None and text.startswith("**A:**"):
            if text.removeprefix("**A:**").strip():
                pending = None
            answer_line_open = True
        elif pending is not None and answer_line_open:
            if text and not text.startswith("#"):
                pending = None
            answer_line_open = False
    if pending is not None:
        unanswered.append(pending)
    return unanswered


def _warn_unanswered_questions(plan_md: str, session_id: str) -> None:
    """Print one warning naming the plan's unanswered open questions, when it has any.

    Args:
        plan_md: The plan's markdown.
        session_id: The plan's id, named in the warning and its `agent6 plan edit` hint.
    """
    questions = _unanswered_questions(plan_md)
    if not questions:
        return
    named = "; ".join(f'"{q[:120]}"' for q in questions)
    print(
        f"[agent6] WARNING: plan {session_id} has"
        f" {_common.plural(len(questions), 'unanswered open question')}: {named}. The run decides"
        f" {'it' if len(questions) == 1 else 'them'} itself; to answer first, stop the run,"
        f" fill the blank **A:** lines with `agent6 plan edit {session_id}`, and run the plan"
        " again.",
        file=sys.stderr,
    )


def _plan_text_for_run(plan_path: pathlib.Path, session_id: str) -> str | None:
    """Return a plan's non-empty markdown, or None after naming the unusable plan."""
    if not plan_path.is_file():
        from agent6.ui.cli import plan_watch  # the dispatcher's lazy import

        _common.error(
            f"plan {session_id!r} has not written its plan.md yet;"
            f" {plan_watch._plan_next(session_id)}"
        )
        return None
    plan_md = errors.read_operator_file(plan_path)
    if not plan_md.strip():
        _common.error(f"plan {session_id!r} has an empty plan.md")
        return None
    return plan_md


def cli_main(argv: list[str] | None = None) -> int:
    """Run the CLI, sorting failures by fault.

    An `OperatorError` prints as a refusal at exit 2, no traceback. A closed output pipe
    (`agent6 ... | head`) ends quietly at 141, as SIGPIPE would. Anything else
    is a bug: one line plus a saved traceback, exit 1; `AGENT6_DEBUG=1` re-raises it.
    `main` stays unguarded so tests see real tracebacks.

    Args:
        argv: The arguments; `sys.argv[1:]` when None.

    Returns:
        The exit code.
    """
    with _terminal_guard.guarded_terminal():
        try:
            try:
                return main(argv)
            finally:
                # A short answer still sits in stdout's buffer: a closed pipe fails it here, in
                # the handler's reach, not at the interpreter's exit (120 and a diagnostic).
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nagent6: interrupted.", file=sys.stderr)
            return 130
        except errors.OperatorError as exc:
            _common.error(f"{exc}")
            return 2
        except BrokenPipeError:
            # The interpreter flushes both streams at exit, and a failed flush turns the exit
            # into 120: /dev/null takes what the closed pipe cannot.
            sink = os.open(os.devnull, os.O_WRONLY)
            os.dup2(sink, sys.stdout.fileno())
            os.dup2(sink, sys.stderr.fileno())
            return 141
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
    from agent6.app import _setup
    from agent6.ui.cli import (
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
    if parallel and getattr(args, "worktree", False):
        _common.refuse(
            "--parallel cannot combine with --worktree (each lane already works in its own"
            " clone); drop one of the two."
        )
        return 2
    seed_from, source_session_id = getattr(args, "seed_from", ""), ""
    if not args.task and seed_from:
        # A plan id alone runs that plan; seeding the same text again would double it.
        from agent6.sessions import id

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
        _warn_unanswered_questions(plan_md, layout.session_id)
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
        _warn_unanswered_questions(plan_md, last_plan)
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
        worktree=getattr(args, "worktree", False),
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

    Args:
        explicit: The `--session-id` value; "" mints one.
        mode: The session's mode, `run` or `plan`; a failed mint names its bucket.

    Returns:
        The named id, else a fresh one no bucket holds.
    """
    from agent6 import kinds
    from agent6.sessions import id

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

    from agent6.app import _setup
    from agent6.sessions import id
    from agent6.sessions import manifest as sessions_manifest
    from agent6.ui.cli import _session_prompt

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
        config_paths=args.config,
        budget_overrides=_setup.BudgetOverrides.from_args(args),
        sandbox_overrides=_setup.SandboxOverrides.from_args(args),
        model=getattr(args, "model", ""),
    )


def _names_an_existing_plan(cwd: pathlib.Path, task: str) -> bool:
    """Return whether the task is exactly the id of a plan in this repository.

    A task that cannot be a directory name is none: a separator, or a name past the
    filesystem's limit, which the lookup reports as an OSError.

    Args:
        cwd: The checkout.
        task: The `agent6 plan` task, stripped.

    Returns:
        True when the task is a plan's id.
    """
    from agent6.sessions import id
    from agent6.sessions import layout as sessions_layout

    if not sessions_layout.is_safe_session_id(task):
        return False
    try:
        return id.session_id_bucket(paths.state_dir(cwd), task) == "plans"
    except OSError:
        return False


def _dispatch_plan(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 plan` or its show and edit verbs."""
    from agent6.app import _setup
    from agent6.ui.cli import (
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
    if _names_an_existing_plan(pathlib.Path.cwd(), args.task.strip()):
        # `plan run <id>` reads as "run this plan"; the verb for that is `run --from`.
        _common.error(
            f"{args.task.strip()!r} names an existing plan. Execute it with:\n"
            f"    agent6 run --from {args.task.strip()}"
        )
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
    from agent6.app import _setup
    from agent6.ui.cli import (
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
    from agent6.ui.cli import watch

    return watch._cmd_watch_target(
        args.target,
        tui=args.tui,
        json_out=args.json,
        since=args.since,
        raw=args.raw,
        config_paths=args.config,
    )


def _dispatch_steer(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 steer`."""
    from agent6.ui.cli import steer_cmd

    return steer_cmd._cmd_steer(args.target, args.text, now=args.now, config_paths=args.config)


def _dispatch_stop(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 stop`."""
    from agent6.ui.cli import stop_cmd

    return stop_cmd._cmd_stop(args.session_id, all_sessions=args.all, after_step=args.after_step)


def _dispatch_answer(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 answer`."""
    from agent6.ui.cli import answer_cmd

    return answer_cmd._cmd_answer(args.target, tuple(args.answers))


def _dispatch_exec_new(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 exec --new [CMD...]`, run in a fresh sandbox."""
    from agent6.config import (
        ConfigError,
        layer,
    )
    from agent6.ui.cli import exec_cmds

    rest: list[str] = list(args.rest)
    if len(rest) > 1 and rest[0] != "--" and rest[1] == "--":
        _common.error(
            f"`--new` takes no session, and {rest[0]!r} before `--` reads as one: run the"
            " command as `agent6 exec --new -- CMD...`, or in a session as"
            " `agent6 exec SESSION -- CMD...`."
        )
        return 2
    try:
        eff = layer.load_effective(pathlib.Path.cwd(), args.config)
    except ConfigError as exc:
        _common.error(f"{exc}")
        return 2
    argv = tuple(rest[1:] if rest[:1] == ["--"] else rest)
    return exec_cmds.exec_new(eff.config, eff.explicit_leaves, pathlib.Path.cwd(), argv)


def _dispatch_exec(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 exec [SESSION --] CMD...`, run in the session's sandbox.

    Args:
        args: The parsed command line; `--new` runs the command in a fresh sandbox instead.

    Returns:
        The command's exit code; 2 when agent6 cannot run it.
    """
    from agent6.config import (
        ConfigError,
        layer,
    )
    from agent6.ui.cli import net_cmds

    if args.new:
        return _dispatch_exec_new(args)
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
    from agent6.ui.cli import net_cmds

    target, port = args.target, args.port
    if port is None and target.isdigit():
        # A bare number is a port of the newest session; a numeric session id needs both args.
        target, port = "", int(target)

    layout = _common.resolve_target(target)
    if layout is None:
        return 2
    if port is None:
        ports = process.listening_ports(layout.session_dir)
        if not ports:
            reason = (
                f"{layout.session_id} is listening on nothing yet."
                if process.read_session_netns_pid(layout.session_dir) is not None
                else net_cmds.no_session_network_reason(layout)
            )
            _common.refuse(f"{reason}")
            return 2
        print(f"{layout.session_id} is listening on: {', '.join(str(p) for p in ports)}")
        return 0
    return net_cmds.forward(layout, port, args.local_port)


def _dispatch_sessions(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Return the exit code of a `agent6 sessions` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import (
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
            config_paths=args.config,
        )
    if args.sessions_command == "compare":
        return sessions_compare._cmd_compare(
            session_ids=tuple(args.session_ids),
            config_paths=args.config,
            rejudge=args.rejudge,
        )
    if args.sessions_command == "review":
        from agent6.ui.cli import sessions_review

        return sessions_review._cmd_sessions_review(
            args.config, session_id=args.session_id, model=args.model
        )
    if args.sessions_command == "commits":
        return sessions_cmds._cmd_commits(session_id=args.session_id)
    if args.sessions_command == "prune":
        return sessions_merge._cmd_prune(
            delete_squashed=args.delete_squashed, config_paths=args.config
        )
    if args.sessions_command == "dir":
        return sessions_cmds._cmd_sessions_dir(args.session_id)
    if args.sessions_command == "rm":
        return sessions_cmds._cmd_sessions_rm(
            session_id=args.session_id, asks=args.asks, yes=args.yes
        )
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
    from agent6.ui.cli import (
        plan_watch,
        watch,
    )

    if args.target:
        return watch._cmd_watch_target(
            args.target, tui=True, json_out=False, since=None, raw=False, config_paths=args.config
        )
    return plan_watch._cmd_tui(args.config)


def _dispatch_completions(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 completions`."""
    from agent6.ui.cli import completions_cmd

    return completions_cmd.cmd_completions(args.shell, print_only=args.print_only)


def _dispatch_web(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 web`."""
    from agent6.ui.cli import web_cmds

    return web_cmds._cmd_web(
        args.target,
        config_paths=args.config,
        host=args.host,
        port=args.port,
        allow_non_loopback=args.allow_non_loopback,
        secret=args.secret,
    )


def _dispatch_prompt(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 prompt` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import prompt_cmds

    if args.prompt_command == "show":
        return prompt_cmds._cmd_prompt_show(args.config, mode=args.mode, as_json=args.json)
    raise AssertionError("unreachable")  # pragma: no cover -- prompt subparser is required


def _dispatch_resume(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 resume`, on the named session or the newest resumable one."""
    from agent6.app import (
        _setup,
        resume,
    )
    from agent6.sessions import id
    from agent6.ui.cli import _session_prompt
    from agent6.ui.cli import resume as cli_resume
    from agent6.viewmodel import newest_session_dir

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
    from agent6.app import _setup
    from agent6.ui.cli import fork

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
    from agent6.ui.cli import config_cmds

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
            args.key,
            args.value,
            repo=args.repo,
            machine=args.machine_file,
            config_paths=args.config,
        )
    if args.config_command == "unset":
        return config_cmds._cmd_config_unset(
            args.key, repo=args.repo, machine=args.machine_file, config_paths=args.config
        )
    if args.config_command in ("add", "remove"):
        return config_cmds._config_list_edit(
            args.key,
            args.value,
            repo=args.repo,
            machine=args.machine_file,
            add=args.config_command == "add",
            config_paths=args.config,
        )
    if args.config_command == "fix":
        return config_cmds._cmd_config_fix(machine=args.machine_file)
    raise AssertionError("unreachable")  # pragma: no cover -- config subparser is required


def _dispatch_check(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 check`."""
    from agent6.ui.cli import check_cmds

    return check_cmds._cmd_check(args.config, section=args.section)


def _dispatch_connect(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 connect`."""
    from agent6.ui.cli import connect

    return connect._cmd_connect(
        provider=args.provider, to_repo=args.repo, verify=args.verify, logout=args.logout
    )


def _dispatch_model(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 model`."""
    from agent6.ui.cli import model

    return model._cmd_model(
        args.config,
        role=args.role,
        route=args.route,
        effort=args.effort,
        to_repo=args.repo,
    )


def _dispatch_memory(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 memory` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import memory_cmds

    if args.memory_command == "add":
        return memory_cmds._cmd_memory_add(args.name, args.body)
    if args.memory_command == "list":
        return memory_cmds._cmd_memory_list()
    if args.memory_command == "show":
        return memory_cmds._cmd_memory_show(args.name)
    if args.memory_command == "rm":
        return memory_cmds._cmd_memory_rm(args.name, yes=args.yes)
    if args.memory_command == "decisions":
        return memory_cmds._cmd_memory_decisions()
    raise AssertionError("unreachable")  # pragma: no cover -- memory subparser is required


def _dispatch_skills(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 skills` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import skills_cmds

    if args.skills_command == "install":
        return skills_cmds._cmd_skills_install(args.url, force=args.force, config_paths=args.config)
    if args.skills_command == "update":
        return skills_cmds._cmd_skills_update(args.name)
    if args.skills_command == "list":
        return skills_cmds._cmd_skills_list(args.config)
    if args.skills_command == "enable":
        return skills_cmds._cmd_skills_enable(
            args.name, always=args.always, repo=args.repo, config_paths=args.config
        )
    if args.skills_command == "disable":
        return skills_cmds._cmd_skills_disable(args.name, repo=args.repo, config_paths=args.config)
    if args.skills_command == "remove":
        return skills_cmds._cmd_skills_remove(args.name, args.config)
    raise AssertionError("unreachable")  # pragma: no cover -- skills subparser is required


def _dispatch_ps(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 ps`."""
    from agent6.ui.cli import ps_cmd

    return ps_cmd.cmd_ps(as_json=args.json, lanes=args.lanes)


def _dispatch_history(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 history` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import history_cmds

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
    from agent6.ui.cli import init_cmds

    return init_cmds._cmd_init(
        ecosystem=args.ecosystem, assume_yes=args.yes, config_paths=args.config
    )


def _dispatch_review(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 review`."""
    from agent6.ui.cli import review_cmds

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
    from agent6.ui import mcp_server
    from agent6.ui.cli import mcp_connect

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
            config_paths=args.config,
        )
    if args.mcp_command == "remove":
        return mcp_connect.cmd_mcp_remove(args.name, to_repo=args.to_repo, config_paths=args.config)
    return mcp_connect.cmd_mcp_list(args.config)


def _dispatch_machine(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Return the exit code of a `agent6 machine` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.app import _setup
    from agent6.app import reporter as app_reporter
    from agent6.app.machine import create
    from agent6.app.machine import run as machine_run
    from agent6.ui.cli import (
        machine_check,
        machine_cmds,
    )

    if args.machine_command == "list":
        return machine_cmds._cmd_machine_list()
    if args.machine_command in ("check", "test", "graph", "run"):
        args.file = machine_cmds.machine_file(args.file)
    if args.machine_command == "check":
        return machine_check._cmd_machine_check(args.file, config_paths=args.config)
    if args.machine_command == "test":
        return machine_check._cmd_machine_test(
            args.file, blackboard=args.blackboard, config_paths=args.config
        )
    if args.machine_command == "graph":
        return machine_check._cmd_machine_graph(args.file, fmt=args.format)
    if args.machine_command == "run":
        return machine_run.run_machine(
            args.file,
            app_reporter.STDIO_REPORTER,
            config_paths=args.config,
            exit_on_wait=args.exit_on_wait,
            disable_sandbox=args.disable_sandbox,
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
            app_reporter.STDIO_REPORTER,
            output=args.output,
            max_attempts=args.max_attempts,
            config_paths=args.config,
            budget_overrides=_setup.BudgetOverrides.from_args(args),
        )
    raise AssertionError("unreachable")  # pragma: no cover -- machine subparser is required


def _dispatch_acp(args: argparse.Namespace) -> int:
    """Return the exit code of `agent6 acp`."""
    from agent6.ui.acp import serve_acp

    return serve_acp(config_paths=args.config)


def _dispatch_system(args: argparse.Namespace) -> int:
    """Return the exit code of a `agent6 system` verb."""  # noqa: DOC501  # the last raise is unreachable
    from agent6.ui.cli import system_cmds

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
    args.config = cli_parser.config_files(args)
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
    except event_log.EventWriteError as exc:
        # The run journal could not be appended; the lifecycle's finally already cleaned up.
        _common.error(f"{exc}")
        return 1
