# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Define the seam a front-end injects into the run and resume lifecycles.

`SessionFrontend` with its capability and steer contracts, and the away-mode policy applied
when a launcher spawns a run detached.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Protocol

from agent6 import budget, event_log, kinds, portable
from agent6.config import Config
from agent6.harness import _snapshot, loop, subrun
from agent6.sessions import ipc, layout
from agent6.tools import mcp_client, operator_prompts


@dataclasses.dataclass(frozen=True, slots=True)
class SessionFacts:
    """Hold the live facts the CLI pause banner shows.

    Read inside a signal handler, so every field is already in memory.

    Attributes:
        spend_usd: The run's spend so far.
        spend_partial: The spend is a lower bound (a model with no price data contributed).
        model: The model in use.
        run_commands: The run-commands policy word.
        isolation: The isolation level word.
    """

    spend_usd: float
    spend_partial: bool
    model: str
    run_commands: str
    isolation: str


class SteerHooks(Protocol):
    """Declare what the lifecycle needs of the front-end's steer state.

    `ui/cli/_steer.SteerState` satisfies it structurally.
    """

    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]
    abort_pending: Callable[[], bool]
    interrupt: Callable[[], bool]
    reset_stage: Callable[[], None]


def approval_scopes(cfg: Config) -> tuple[str, ...]:
    """Return every scope this run can be asked about: the commands, plus one per MCP server.

    A grant is per scope, so an approve-all with only the command scope granted would block
    on the first MCP call.
    """
    servers = (
        tuple(f"{ipc.MCP_SCOPE_PREFIX}{name}" for name, s in cfg.mcp.servers.items() if s.enabled)
        if cfg.mcp.enabled
        else ()
    )
    return (ipc.COMMAND_SCOPE, *servers)


def settle_away_mode(session_dir: pathlib.Path, cfg: Config) -> None:
    """Settle the away mode at an execution's start.

    A foreground start drops a stale detach answer and every approve-all grant, since the
    operator is back to answer; a spawned start honours the away marker the launcher set.
    """
    if portable.has_controlling_tty():
        ipc.clear_away_mode(session_dir)
        ipc.clear_session_grants(session_dir)
    else:
        apply_spawned_away_default(session_dir, approval_scopes(cfg))


def apply_spawned_away_default(session_dir: pathlib.Path, scopes: tuple[str, ...]) -> None:
    """Honor AGENT6_DETACHED_AWAY, set by a launcher that spawns a run detached.

    Without it a spawned run with no terminal fabricates empty ask_user answers when no viewer
    is live. An away mode already on the run dir is the operator's own detach answer and wins:
    overwriting a chosen 'deny' with 'wait' would block the run on an approval nobody answers.
    """
    away = os.environ.get("AGENT6_DETACHED_AWAY", "")
    if not away or ipc.away_mode(session_dir):
        return
    if away == "approve":
        # approve is never stored in away.mode (deny|wait): it is an allow marker per scope.
        for scope in scopes:
            ipc.set_session_allow(session_dir, scope)
    elif away in ipc.AWAY_MODES:
        ipc.set_away_mode(session_dir, away)


@dataclasses.dataclass(frozen=True, slots=True)
class FrontendCapabilities:
    """Declare what a surface can do, so a headless run denies rather than fabricates answers.

    Attributes:
        can_ask: Approvals and ask_user reach a human; `ui/cli` sets it from `isatty()`.
    """

    can_ask: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class SessionFrontend:
    """Hold the presentation and process-spawn callables `ui/cli` injects into the lifecycles.

    The console view lives cli-side and the lifecycle only signals attach and close; the
    lifecycle owns the run-dir bridge. One value serves both `run_task` and `resume_task`;
    resume never calls the run-only fields.

    Attributes:
        capabilities: What this surface can do at all.
        should_spawn_tui: Whether to open the TUI for this invocation.
        stream_modes: The (stream, live view) pair for a verbosity.
        attach_console_view: Attaches the console view to the event sink.
        close_console_view: Closes the console view.
        loop_logger: Builds the loop's logger for a run id.
        tui_session: The context a TUI-backed run runs inside.
        build_approver: Answers the approvals the gate journals on the run dir's bridge.
        build_questioner: Answers the questions the gate journals on the run dir's bridge.
        make_steer_state: Builds the steer state over the sink, the run dir and the facts.
        confirm_unconfined_autorun: Confirms running commands unconfined.
        confirm_run_on_run_branch: Confirms starting on a run branch.
        confirm_replay_after_crash: (iteration, tool names) -> replay a turn whose tools may
            have partially applied; interactive fronts prompt, headless warns and proceeds.
        prompt_detach_away_mode: Asks the away mode when the run detaches.
        select_revised_prompt: Runs the revise choice; None when the surface cannot, and the
            execution then skips revision instead of reading a selector's None as a quit.
        build_repl_hook: Builds the `run -i` / `ask -i` hook.
        run_ask_repl: Runs the ask REPL.
        save_ask_transcript: Saves an ask's transcript.
        build_coordinator_spawner: Builds the `/parallel` dispatch spawner, or None.
        agent6_exe: The agent6 executable to spawn.
        spawn_detached_resume: (cwd, session_id, flags) -> spawns the detached execution under
            this invocation's overrides as CLI options (see `_setup.override_flags`).
    """

    capabilities: FrontendCapabilities
    should_spawn_tui: Callable[[bool, bool, str], bool]
    stream_modes: Callable[[bool], tuple[bool, bool]]
    attach_console_view: Callable[[event_log.EventSink], None]
    close_console_view: Callable[[], None]
    loop_logger: Callable[[str], Callable[[str], None]]
    tui_session: Callable[[pathlib.Path, bool], contextlib.AbstractContextManager[None]]
    build_approver: Callable[[pathlib.Path], operator_prompts.Approver]
    build_questioner: Callable[[pathlib.Path], operator_prompts.Questioner]
    make_steer_state: Callable[
        [event_log.EventSink, pathlib.Path, Callable[[], SessionFacts]], SteerHooks
    ]
    confirm_unconfined_autorun: Callable[[kinds.IsolationLevel, Config], bool]
    confirm_run_on_run_branch: Callable[[str], bool]
    confirm_replay_after_crash: Callable[[int, tuple[str, ...]], bool]
    prompt_detach_away_mode: Callable[[pathlib.Path, tuple[str, ...]], None]
    select_revised_prompt: Callable[[str, str, tuple[str, ...]], str | None] | None
    build_repl_hook: Callable[
        [pathlib.Path, budget.BudgetTracker, str, mcp_client.MCPManager | None],
        Callable[[int, str], kinds.AutoCommitDirective],
    ]
    run_ask_repl: Callable[
        [loop.Harness, budget.BudgetTracker, layout.SessionLayout, str], _snapshot.SessionResult
    ]
    save_ask_transcript: Callable[[layout.SessionLayout, str, str], None]
    build_coordinator_spawner: Callable[
        [Config, pathlib.Path, pathlib.Path, str, str, float | None, bool],
        subrun.GroupLaneSpawner | None,
    ]
    agent6_exe: Callable[[], str]
    spawn_detached_resume: Callable[[pathlib.Path, str, Sequence[str]], str]
