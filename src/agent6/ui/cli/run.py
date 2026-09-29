# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`, `plan` and `ask`: adapt argv, build the config and the presentation seam.

The lifecycle is `agent6.app.run.run_task`.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    load_session_config,
)
from agent6.app.frontend import FrontendCapabilities, SessionFrontend
from agent6.app.parallel import build_coordinator_spawner
from agent6.app.preflight import (
    require_git_repo,
    route_preflight,
)
from agent6.app.reporter import STDIO_REPORTER
from agent6.app.run import run_task
from agent6.config import (
    Config,
)
from agent6.errors import OperatorError
from agent6.events import EventSink
from agent6.kinds import ResumableMode, session_kind
from agent6.paths import data_dir
from agent6.skills import operator_skills
from agent6.ui.btw import asks_dir, direct_launch, make_btw_runner
from agent6.ui.cli._ask import (
    build_session_seed,
    run_ask_repl,
    save_ask_transcript,
)
from agent6.ui.cli._common import error, refuse
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._interact import (
    build_approver,
    build_questioner,
    lane_away_mode,
    prompt_detach_away_mode,
)
from agent6.ui.cli._live import (
    loop_logger,
    should_spawn_tui,
    stream_modes,
    tui_session,
)
from agent6.ui.cli._preflight import (
    confirm_replay_after_crash,
    confirm_run_on_run_branch,
    confirm_unconfined_autorun,
)
from agent6.ui.cli._repl import build_repl_hook
from agent6.ui.cli._steer import (
    make_steer_state,
    select_revised_prompt,
)
from agent6.ui.cli._task_refs import (
    expand_task_file_refs,
)
from agent6.ui.cli.parallel import dispatch_parallel, lane_runtime
from agent6.ui.spawn import agent6_exe, spawn_detached_resume
from agent6.ui.steer import SteerState
from agent6.viewmodel import session_policy


def _skills_task_prefix(cfg: Config, names: tuple[str, ...]) -> tuple[str, str]:
    """Return the task-prompt prefix for the `--skill` names, and an error text or ""."""
    resolved = operator_skills(
        cfg.skills.enabled, cfg.skills.extra_dirs, cfg.skills.state, data_dir() / "skills"
    )
    by_name = {s.name: s for s in (*resolved.enabled, *resolved.always)}
    blocks: list[str] = []
    for n in names:
        skill = by_name.get(n)
        if skill is None:
            if not cfg.skills.enabled:
                return "", (
                    f"--skill: skills are disabled, so {n!r} cannot load"
                    " (agent6 config set skills.enabled true)"
                )
            available = ", ".join(sorted(by_name)) or "(none installed)"
            return "", f"--skill: unknown or disabled skill {n!r}; available: {available}"
        blocks.append(f'<skill name="{skill.name}">\n{skill.text.rstrip()}\n</skill>')
    joined = "\n\n".join(blocks)
    return (
        f"Apply the operator-installed skill(s) below to this task.\n\n{joined}\n\n---\n\n",
        "",
    )


def _remember_steer(cell: list[SteerState | None], state: SteerState) -> SteerState:
    """Publish the execution's steer state for the approver's late-bound read.

    Returns:
        The state, unchanged.
    """
    cell[0] = state
    return state


def session_frontend(config_path: Path | None = None) -> SessionFrontend:
    """Return the presentation seam the run and resume lifecycles drive.

    One per invocation: the console-view cell is run-scoped. The console view is created
    lazily on `attach_console_view`, and the builders that need it close over its cell, so
    the lifecycle never holds a UI type. The lifecycle owns egress itself; only the two
    exe-spawn primitives it cannot reach (`ui.spawn`) are injected.

    Args:
        config_path: The `--config` file, if any.
    """
    # Late-bound: the approver and questioner are built before the view or steer state exists.
    console_cell: list[ConsoleView | None] = [None]
    steer_cell: list[SteerState | None] = [None]

    def attach_console_view(events: EventSink) -> None:
        """Create the console view on the run's event sink."""
        # The sink's path is the handle to the run dir, so the layout need not cross the protocol.
        view = ConsoleView(sys.stderr, policy=lambda: session_policy(events.path.parent).line())
        console_cell[0] = view
        events.subscribe(view)

    def close_console_view() -> None:
        """Close the console view, if one was attached."""
        view = console_cell[0]
        if view is not None:
            view.close()

    return SessionFrontend(
        # The CLI asks on the terminal, so a piped stdin means it cannot ask.
        capabilities=FrontendCapabilities(can_ask=sys.stdin.isatty()),
        should_spawn_tui=lambda tui, interactive, mode: should_spawn_tui(
            tui=tui, interactive=interactive, mode=mode
        ),
        stream_modes=lambda tui_enabled: stream_modes(tui_enabled=tui_enabled),
        attach_console_view=attach_console_view,
        close_console_view=close_console_view,
        loop_logger=lambda mode: loop_logger(mode, console_cell[0]),
        tui_session=lambda session_dir, enabled: tui_session(session_dir, enabled=enabled),
        build_approver=lambda session_dir: build_approver(session_dir, console_cell, steer_cell),
        build_questioner=lambda session_dir: build_questioner(session_dir, console_cell),
        make_steer_state=lambda events, session_dir, facts: _remember_steer(
            steer_cell,
            make_steer_state(
                events,
                session_dir,
                console_cell[0],
                facts,
                # The CLI has a terminal, so `/btw` launches directly, unlike a confined lane.
                make_btw_runner(
                    session_dir.name,
                    launch=direct_launch,
                    list_asks=lambda: (
                        [d for d in asks_dir(session_dir).iterdir() if d.is_dir()]
                        if asks_dir(session_dir).is_dir()
                        else []
                    ),
                    events=events,
                ),
                config_path=config_path,
            ),
        ),
        confirm_unconfined_autorun=confirm_unconfined_autorun,
        confirm_run_on_run_branch=confirm_run_on_run_branch,
        confirm_replay_after_crash=confirm_replay_after_crash,
        prompt_detach_away_mode=prompt_detach_away_mode,
        # The choice reads stdin: with it redirected nobody can answer.
        select_revised_prompt=(
            (
                lambda original, revised, questions: select_revised_prompt(
                    original, revised, questions, console_cell[0]
                )
            )
            if sys.stdin.isatty()
            else None
        ),
        build_repl_hook=lambda cwd, budget, session_id, mcp_manager: build_repl_hook(
            cwd,
            budget,
            session_id=session_id,
            mcp_manager=mcp_manager,
            console_view=console_cell[0],
            steer_cell=steer_cell,
        ),
        run_ask_repl=lambda wf, budget, layout, first_question: run_ask_repl(
            wf, budget, layout, first_question=first_question
        ),
        save_ask_transcript=lambda layout, question, answer: save_ask_transcript(
            layout, question=question, answer=answer
        ),
        build_coordinator_spawner=(
            lambda cfg, cwd, state_dir, mode, session_id, max_usd, auto_approve: (
                build_coordinator_spawner(
                    cfg,
                    cwd,
                    state_dir,
                    mode=mode,
                    session_id=session_id,
                    runtime=lane_runtime(),
                    max_usd=max_usd,
                    auto_approve=auto_approve,
                    lane_away=lane_away_mode(),
                )
            )
        ),
        agent6_exe=agent6_exe,
        spawn_detached_resume=lambda cwd, sid, flags: spawn_detached_resume(
            cwd, sid, config_path=config_path, flags=flags
        ),
    )


@dataclass(frozen=True, slots=True)
class ComposedTask:
    """The prompt a session starts from.

    Attributes:
        text: The prompt.
        source_session_id: The session `--from` seeded it from, or "".
    """

    text: str
    source_session_id: str = ""


def _compose_task(
    task: str, cfg: Config, *, skills: tuple[str, ...], seed_from: str
) -> ComposedTask:
    """Assemble the prompt the session starts from.

    The skills prefix, then another session's context when `--from` seeds this one.
    `--from` starts a new session and leaves the source untouched; `fork` keeps a session's
    mode, while this takes the mode from the command the operator typed.

    Args:
        task: The task as typed.
        cfg: The effective config.
        skills: The `--skill` names.
        seed_from: The `--from` session, or "".

    Returns:
        The composed task.

    Raises:
        OperatorError: A skill or the seed cannot be resolved.
    """
    if skills:
        prefix, skills_err = _skills_task_prefix(cfg, skills)
        if skills_err:
            raise OperatorError(skills_err)
        task = prefix + task
    if not seed_from:
        return ComposedTask(task)
    seed = build_session_seed(Path.cwd(), seed_from, latest=False)
    if seed is None:
        raise OperatorError(f"could not seed from {seed_from!r}")
    return ComposedTask(f"{seed.text}\n\n{task}" if task else seed.text, seed.source_session_id)


def _cmd_run(
    config_path: Path | None,
    task: str,
    *,
    session_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    decompose: bool = False,
    mode: ResumableMode = "run",
    seed_from: str = "",
    source_session_id: str = "",
    skills: tuple[str, ...] = (),
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    parallel_spec: str = "",
    standing_goal: str = "",
    pins: tuple[str, ...] = (),
    model: str = "",
) -> int:
    """Adapt `agent6 run`, `plan` and `ask` argv and drive the lifecycle.

    Builds the effective config, applies the flag overrides, resolves skills and `@file`
    references, routes `--parallel`, then hands off to `app.run.run_task` with the
    injected seam.

    Args:
        config_path: The `--config` file, if any.
        task: The task as typed.
        session_id: A session id to use; "" allocates one.
        interactive: Stay attached for a conversation.
        tui: Open the dashboard.
        decompose: Plan first, overriding config.
        mode: `run`, `plan` or `ask`.
        seed_from: The `--from` session, or "".
        source_session_id: The lineage to record when not seeded.
        skills: The `--skill` names.
        budget_overrides: The budget flags.
        sandbox_overrides: The sandbox flags.
        preset: The `--preset` name.
        parallel_spec: The `--parallel` lane spec, or "".
        standing_goal: The `--standing` goal, or "".
        pins: The pinned session ids.
        model: The `--model` override.

    Returns:
        The exit code; 2 on a refusal.
    """
    # The git wall first, so a scratch dir does not clear every other wall before hitting it.
    if mode != "ask" and not require_git_repo(Path.cwd()):
        return 2
    effective = load_session_config(
        Path.cwd(),
        config_path,
        mode=mode,
        preset=preset,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        model=model,
    )
    cfg, explicit_leaves = effective.config, effective.explicit_leaves
    if decompose:
        cfg = cfg.with_decompose("on")
    try:
        composed = _compose_task(task, cfg, skills=skills, seed_from=seed_from)
    except OperatorError as exc:
        error(f"{exc}")
        return 2
    task = composed.text
    source_session_id = composed.source_session_id or source_session_id
    role = session_kind(mode).role

    # `@path` references inline the files verbatim before the harness sees the task.
    task = expand_task_file_refs(task, Path.cwd())

    # `--parallel` routes after the config walls and before the single-run preflight; run mode only.
    if parallel_spec and mode == "run":
        # Depth 1: a subordinate lane (AGENT6_SUBRUN) must never itself fan out.
        if os.environ.get("AGENT6_SUBRUN"):
            refuse(
                "--parallel is unavailable inside a subordinate run (parallel dispatch is depth 1)."
            )
            return 2
        # run_task's route preflight, early so the fan-out refuses before cloning.
        if not route_preflight(cfg, role, reporter=STDIO_REPORTER, model_flag=model):
            return 2
        return dispatch_parallel(
            cfg,
            task,
            parallel_spec,
            cwd=Path.cwd(),
            max_usd=budget_overrides.max_usd if budget_overrides is not None else None,
            auto_approve=sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
            pins=pins,
        )

    return run_task(
        cfg,
        task,
        frontend=session_frontend(config_path),
        started_at=time.time(),
        session_id=session_id,
        source_session_id=source_session_id or None,
        interactive=interactive,
        tui=tui,
        mode=mode,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        pins=pins,
        standing_goal=standing_goal,
        model=model,
        explicit_leaves=explicit_leaves,
    )
