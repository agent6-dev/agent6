# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run` (and its plan/ask modes): adapt argv, build the config and the
presentation seam, and hand the lifecycle to `agent6.app.run.run_task`."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    check_provider_keys,
)
from agent6.app.preflight import (
    require_git_repo,
)
from agent6.app.run import RunFrontend, run_task
from agent6.config import (
    Config,
    ConfigError,
    RoleName,
)
from agent6.config.layer import (
    load_effective,
)
from agent6.events import EventSink
from agent6.git_ops import set_repo_hook_policy
from agent6.paths import data_dir
from agent6.skills import discover_skills, resolve_states, skill_search_dirs
from agent6.ui.cli._ask import (
    run_ask_repl,
    save_ask_transcript,
)
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.cli._interact import (
    build_approver,
    build_questioner,
    prompt_detach_away_mode,
)
from agent6.ui.cli._live import (
    loop_logger,
    should_spawn_tui,
    stream_modes,
    tui_session,
)
from agent6.ui.cli._preflight import (
    choose_branch_start_point,
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
from agent6.ui.cli.parallel import (
    build_coordinator_spawner,
    dispatch_parallel,
)
from agent6.ui.spawn import agent6_exe, spawn_detached_resume


def _skills_task_prefix(cfg: Config, names: tuple[str, ...]) -> tuple[str, str]:
    """Resolve ``--skill`` names to a task-prompt prefix. Returns (prefix, error)."""
    found, _warns = discover_skills(skill_search_dirs(cfg.skills.extra_dirs, data_dir() / "skills"))
    resolved = resolve_states(found, cfg.skills.state)
    by_name = {s.name: s for s in (*resolved.enabled, *resolved.always)}
    blocks: list[str] = []
    for n in names:
        skill = by_name.get(n)
        if skill is None:
            available = ", ".join(sorted(by_name)) or "(none installed)"
            return "", f"--skill: unknown or disabled skill {n!r}; available: {available}"
        blocks.append(f'<skill name="{skill.name}">\n{skill.text.rstrip()}\n</skill>')
    joined = "\n\n".join(blocks)
    return (
        f"Apply the operator-installed skill(s) below to this task.\n\n{joined}\n\n---\n\n",
        "",
    )


def run_frontend() -> RunFrontend:
    """Build the presentation seam `app.run.run_task` / `app.resume.resume_task`
    drive: one per invocation (the console-view cell is run-scoped). The console
    view is created lazily on ``attach_console_view``; the builders that need it
    close over its cell, so the lifecycle never holds a UI type. The lifecycle
    owns egress (`app.egress`) itself; only the two exe-spawn primitives it
    can't reach (`ui.spawn`) are injected."""
    console_cell: list[ConsoleView | None] = [None]

    def attach_console_view(events: EventSink) -> None:
        view = ConsoleView(sys.stderr)
        console_cell[0] = view
        events.subscribe(view)

    def close_console_view() -> None:
        view = console_cell[0]
        if view is not None:
            view.close()

    return RunFrontend(
        should_spawn_tui=lambda tui, interactive, mode: should_spawn_tui(
            tui=tui, interactive=interactive, mode=mode
        ),
        stream_modes=lambda tui_enabled: stream_modes(tui_enabled=tui_enabled),
        attach_console_view=attach_console_view,
        close_console_view=close_console_view,
        loop_logger=lambda mode: loop_logger(mode, console_cell[0]),
        tui_session=lambda run_dir, enabled: tui_session(run_dir, enabled=enabled),
        build_approver=lambda run_dir, events: build_approver(run_dir, events, console_cell[0]),
        build_questioner=lambda run_dir, events: build_questioner(run_dir, events, console_cell[0]),
        make_steer_state=lambda events, run_dir: make_steer_state(events, run_dir, console_cell[0]),
        confirm_unconfined_autorun=confirm_unconfined_autorun,
        confirm_run_on_run_branch=confirm_run_on_run_branch,
        choose_branch_start_point=choose_branch_start_point,
        prompt_detach_away_mode=prompt_detach_away_mode,
        select_revised_prompt=select_revised_prompt,
        build_repl_hook=lambda cwd, budget, run_id, mcp_manager: build_repl_hook(
            cwd, budget, run_id=run_id, mcp_manager=mcp_manager
        ),
        run_ask_repl=lambda wf, budget, layout, first_question: run_ask_repl(
            wf, budget, layout, first_question=first_question
        ),
        save_ask_transcript=lambda layout, question, answer: save_ask_transcript(
            layout, question=question, answer=answer
        ),
        build_coordinator_spawner=(
            lambda cfg, cwd, state_dir, mode, run_id, max_usd, auto_approve, host_lane_launch: (
                build_coordinator_spawner(
                    cfg,
                    cwd,
                    state_dir,
                    mode=mode,
                    run_id=run_id,
                    max_usd=max_usd,
                    auto_approve=auto_approve,
                    host_lane_launch=host_lane_launch,
                )
            )
        ),
        agent6_exe=agent6_exe,
        spawn_detached_resume=spawn_detached_resume,
    )


def _cmd_run(  # noqa: PLR0911
    config_path: Path | None,
    task: str,
    *,
    run_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    decompose: bool = False,
    mode: Literal["run", "plan", "ask"] = "run",
    skills: tuple[str, ...] = (),
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    profile: str = "",
    parallel_spec: str = "",
) -> int:
    """Adapt `agent6 run`/`plan`/`ask` argv: build the effective config, apply
    the flag overrides, resolve skills and @file refs, route ``--parallel``,
    then drive the lifecycle (`app.run.run_task`) with the injected seam."""
    try:
        cfg = load_effective(Path.cwd(), config_path, profile=profile).config
        set_repo_hook_policy(cfg.git.run_repo_hooks)
        if budget_overrides is not None:
            cfg = budget_overrides.apply(cfg)
        if sandbox_overrides is not None:
            cfg = sandbox_overrides.apply(cfg)
        if decompose:  # --decompose: plan-first for this run (overrides config)
            cfg = cfg.model_copy(
                update={"prompt": cfg.prompt.model_copy(update={"decompose": "on"})}
            )
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    if skills:
        prefix, skills_err = _skills_task_prefix(cfg, skills)
        if skills_err:
            print(f"ERROR: {skills_err}", file=sys.stderr)
            return 2
        task = prefix + task
    # Surface the not-a-git-repo wall up front. run/plan need git; ask is
    # read-only and may run outside a repo. Without this, a user in a scratch
    # non-git dir clears the provider, model, and key walls serially only to
    # discover at the end that they also need git. Mirrors the resume path,
    # which already checks git before require_runnable.
    if mode != "ask" and not require_git_repo(Path.cwd()):
        return 2
    role: RoleName = "planner" if mode == "plan" else "worker"
    try:
        cfg.require_runnable(role)
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2

    # Resolve @path references in the task string before the
    # workflow ever sees it. Lets the user write "fix the bug in @src/x.py
    # described in @notes.md" and have those files inlined verbatim.
    task = expand_task_file_refs(task, Path.cwd())

    # Provider key + models-cache preflight, shared by the single run and the
    # --parallel fan-out: resolves each referenced provider's key AND refreshes
    # its models cache, which carries the pricing explicit_usd_flag_error reads.
    # Runs before the --parallel route so dispatch_parallel's own --max-usd check
    # sees the same refreshed cache a plain --max-usd run does.
    missing = check_provider_keys(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2

    # `--parallel`: fan out isolated lanes instead of a single run. Routed here,
    # after config/skills/require_runnable and the key preflight, but BEFORE the
    # single-run sandbox preflight (no branch cut, no run dir on the origin); the
    # orchestrator clones each lane and runs its own `agent6 run`. run mode only.
    if parallel_spec and mode == "run":
        # Depth 1: a subordinate lane (AGENT6_SUBRUN) must never itself fan out.
        if os.environ.get("AGENT6_SUBRUN"):
            print(
                "REFUSING: --parallel is unavailable inside a subordinate run"
                " (parallel dispatch is depth 1).",
                file=sys.stderr,
            )
            return 2
        return dispatch_parallel(
            cfg,
            task,
            parallel_spec,
            cwd=Path.cwd(),
            max_usd=budget_overrides.max_usd if budget_overrides is not None else None,
            auto_approve=sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
        )

    return run_task(
        cfg,
        task,
        frontend=run_frontend(),
        run_id=run_id,
        interactive=interactive,
        tui=tui,
        mode=mode,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        profile=profile,
    )
