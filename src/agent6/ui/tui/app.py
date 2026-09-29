# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run views' app: `agent6 run`, `agent6 attach --tui` and the hub open it.

`Agent6TUI` is the data plane: a background thread tails the log into the folded
`SessionState`, and the app owns prompt dispatch, run control and the exit code.
`DashboardScreen` and `ConversationScreen` present that state. The app reads the
log stream and writes only the answer files the harness polls (approvals,
questions, steer, the compact request), the contract every front-end shares.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

try:
    from textual import work
    from textual.app import App, ScreenStackError, SystemCommand
    from textual.binding import Binding
    from textual.css.query import NoMatches
    from textual.screen import ModalScreen, Screen
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.app.fork import create_fork
from agent6.app.reporter import Reporter
from agent6.app.stop import stop_session
from agent6.app.undo import undo_fork
from agent6.config.layer import available_preset_names
from agent6.models.choices import available_routes, resume_defaults
from agent6.paths import mkdir_for_real_user
from agent6.sessions.ipc import (
    register_frontend,
    submit_steer,
    unregister_frontend,
)
from agent6.sessions.layout import LOGS_NAME, bucket_dir, layout_of
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.background import shells_text
from agent6.ui.directives import submit_composer_line
from agent6.ui.spawn import (
    DETACHED_RUN_ENV,
    agent6_argv,
    run_cli_capture,
    run_cli_output,
    spawn_and_locate,
    spawn_detached_resume,
)
from agent6.ui.tui.composer import SteerInput
from agent6.ui.tui.conversation import ConversationScreen
from agent6.ui.tui.dashboard import DashboardScreen
from agent6.ui.tui.modals import (
    ConfirmModal,
    TextModal,
)
from agent6.ui.tui.prompts import PromptDispatcher
from agent6.ui.tui.theme import (
    PALETTE_CSS,
    MuxPointerShapes,
    PlainNotify,
    setup_theme,
)
from agent6.viewmodel import restate
from agent6.viewmodel.events import SESSION_START_EVENTS
from agent6.viewmodel.format import status_label
from agent6.viewmodel.listing import (
    LIVE_STATUS_WORDS,
    finished_needs_new_work,
    needs_new_work,
    status_for_session_dir,
    task_snippet,
)
from agent6.viewmodel.state import (
    STREAM_DELTA_EVENTS,
    SessionState,
    apply_event,
    context_fill,
    fold_session,
    initial_state,
    status_facts,
)
from agent6.viewmodel.tail import tail_events

# An answer submitted after the worker died: the next resume re-asks the prompt.
_ANSWER_LOST = "the session is not live; the answer reached nothing (a resume re-asks the prompt)"

# Events that recompute dir_status at once, not on the ~1s heartbeat: the chip and both
# composer bars route off it, so serving the previous state for a heartbeat lies.
_STATUS_NOW_EVENTS = SESSION_START_EVENTS | {
    "session.end",
    "approval.prompt",
    "approval.answer",
    "question.prompt",
    "question.answer",
}


@dataclass(frozen=True, slots=True)
class TuiExit:
    """How a run view ended.

    Attributes:
        quit_hub: The operator quit the hub, not just the view.
        open_next: A session the hub opens next: the run a plan's Run this plan started.
    """

    quit_hub: bool = False
    open_next: Path | None = None


class Agent6TUI(PlainNotify, MuxPointerShapes, App[TuiExit]):
    """The run views over one session, following its log live."""

    TITLE = "agent6"
    CSS = (
        PALETTE_CSS
        + """
    Screen { layers: base dropdown; background: $surface; }
    /* The flat Screen rule above also matches ModalScreens, which would make
       their backdrops opaque; restore textual's translucent dim (same
       specificity, later rule wins) so the screen shows through behind dialogs. */
    ModalScreen { background: $background 60%; }
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }  /* half the 2-wide default */
    /* A footer that does not fit clips (textual's default); the 1-row widget has no
       room for the scrollbar the universal rule gives it, which replaced every hint. */
    Footer { scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    /* I-beam over anything you can type into (kitty OSC 22; inert elsewhere). */
    Input, TextArea { pointer: text; }
    """
    )

    BINDINGS: ClassVar = [
        # App-level so every screen has it; the hub-aware exit code needs this handler.
        Binding("ctrl+q", "quit_hub", "Quit", show=False),
        # Ctrl-Z steps away, the run keeps going; priority beats the composer's ctrl+z undo.
        Binding("ctrl+z", "detach_exit", "Detach", show=True, priority=True),
    ]

    def __init__(
        self,
        session_dir: Path,
        *,
        exit_on_end: bool = False,
        from_hub: bool = False,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.session_dir = session_dir
        self._review_running = False  # Run > Review this run…: one call at a time
        # The invocation's `--config F`; a detach-resume it spawns re-applies F.
        self.config_path = config_path
        # From the hub loop, Esc returns to it and q quits it; standalone, both close the view.
        self.from_hub = from_hub
        self.logs_path = session_dir / LOGS_NAME
        self.state: SessionState = initial_state()
        self._prompts = PromptDispatcher(
            self, answerable=self.session_controllable, lost=_ANSWER_LOST
        )
        self._seen_steer = 0
        self._dirty = False  # a structural event arrived; _tick coalesces the repaint
        self._light_dirty = False  # only stream deltas / heartbeat: light repaint
        self._stop = threading.Event()
        # The view `agent6 run` spawned ends with the run, holding on the payoff until Ctrl+Q.
        self.exit_on_end = exit_on_end
        # The run ended under exit_on_end and the dashboard is holding.
        self._end_hold = False
        # The fork this view created (/undo, Run > Fork); continue_as routes the follow-up there.
        self._continue_child = ""
        # Set by action_detach_exit; run_tui reads it to print the reattach hint.
        self.detached = False
        # The (word, reason) the hub row shows too, refreshed on the ~1/s heartbeat; derived,
        # never latched, so a crash then a resume reads as running again.
        self.dir_status: tuple[str, str] = status_for_session_dir(
            session_dir, status_facts(self.state)
        )
        # The journal prefix folded at open; the reader starts after it.
        self._seed_log_count = 0
        self._reader_start_at = 0
        # The header's task line before session.start folds: the manifest knows the work.
        self.fallback_task = ""
        # The session's mode (run, plan, ask), the title word; "run" for a manifest-less dir.
        self.mode = "run"
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            self.fallback_task = manifest.user_task
            self.mode = manifest.mode or "run"
        # A run is silent for a whole reasoning turn; the ~1/s repaint tells thinking from hung.
        self.last_event_at = time.monotonic()
        self._heartbeat_at = 0.0
        self.spin = 0
        # The preset and model a composer resume continues under ("" is no flag).
        self.resume_preset = ""
        self.resume_model = ""
        presets = available_preset_names(Path.cwd(), config_path)
        routes = available_routes(Path.cwd(), config_path)
        self._dash = DashboardScreen(presets=presets, routes=routes, prompts=self._prompts)
        self._conv = ConversationScreen(
            self.logs_path,
            title=self.screen_title,
            presets=presets,
            routes=routes,
            prompts=self._prompts,
        )

    def resume_defaults(self, preset: str) -> tuple[str, str]:
        """Return the resume rows' no-flag labels for a preset."""
        return resume_defaults(Path.cwd(), self.config_path, self.session_dir, preset=preset)

    def _task_lead(self) -> str:
        """Return the clipped task for a title, or the session name before a task is known."""
        task = self.state.user_task or self.fallback_task
        return task_snippet(task, max_chars=57) or self.session_dir.name

    def screen_title(self, context: str) -> str:
        """Return a run screen's menu-bar subtitle.

        Screens call this at stamp time, never freezing a string at construction: the
        task name lands after the first fold, and the end hold outlives every stamp.

        Args:
            context: The view's word.

        Returns:
            The context and the task, plus the status and how to leave once the run ended.
        """
        if self._end_hold:
            return (
                f"{context} · {self._task_lead()} · {status_label(*self.dir_status)}"
                " · Ctrl+Q to leave"
            )
        return f"{context} · {self._task_lead()}"

    def run_title(self) -> str:
        """Return the dashboard's subtitle: the mode word and the task."""
        return self.screen_title(self.mode)

    def on_mount(self) -> None:
        """Claim the session, fold the log on disk, push both views and start the reader."""
        setup_theme(self)  # apply the saved theme before the first paint
        # A per-process claim; concurrent web, TUI and attach viewers each hold their own.
        register_frontend(self.session_dir, os.getpid())
        self.sub_title = self.run_title()
        self._seed_from_disk()
        # Pushed, since only the push path loads a screen's CSS; the conversation is installed,
        # so popping it hides it and Ctrl+D toggles the two views.
        self.push_screen(self._dash)
        self.install_screen(self._conv, "conversation")
        self.push_screen(self._conv)
        if self.state.undone_to:
            # An /undo already folded at open: the message it took back is the operator's to edit.
            self.call_after_refresh(self._fill_composers, self.state.undone_text)
        # The exit condition is polled from a timer: exit() from a call_from_thread callback
        # does not take effect, from a timer callback it does.
        self.set_interval(0.2, self._tick)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _seed_from_disk(self) -> None:
        """Fold the log already on disk, before the reader thread starts.

        The first paint reads one complete state; the reader starts at the folded
        prefix's byte boundary. The seed fixes the steer baseline too, so a request
        already in the log does not prompt.
        """
        position = 0

        def heard(end: int) -> None:
            nonlocal position
            position = end

        with contextlib.suppress(OSError):
            seeded = fold_session(tail_events(self.logs_path, follow=False, on_position=heard))
            self.state = seeded
            self._seen_steer = seeded.steer_requests
            self._seed_log_count = seeded.log_count
            self._reader_start_at = position
            self.dir_status = status_for_session_dir(self.session_dir, status_facts(seeded))

    def on_unmount(self) -> None:
        """Stop the reader and drop this process's front-end claim."""
        self._stop.set()
        unregister_frontend(self.session_dir, os.getpid())

    # --- reader thread -----------------------------------------------

    def _reader_loop(self) -> None:
        """Feed the log's new events to the app thread until the view closes."""
        for event in tail_events(
            self.logs_path,
            follow=True,
            # Without this, closing the view on a run that never ends leaks the thread.
            should_stop=self._stop.is_set,
            start_at=self._reader_start_at,
        ):
            if self._stop.is_set():
                return
            self.call_from_thread(self._handle_event, event)

    def seconds_since_event(self) -> int:
        """Return the idle seconds for the heartbeat, anchored to the last event's timestamp.

        An arrival anchor would read a few seconds on attach to a run wedged for
        forty minutes; the fold time is the fallback for a log without timestamps.
        """
        if self.state.last_event_ep is None:
            return int(time.monotonic() - self.last_event_at)
        return max(0, int(time.time() - self.state.last_event_ep))

    def _handle_event(self, event: dict[str, object]) -> None:
        """Fold one event and mark what the next tick repaints."""
        self.state = apply_event(self.state, event)
        self.last_event_at = time.monotonic()
        if event.get("type") == "session.undone" and self.state.undone_to:
            # The fork is the continuation; the message taken back is the operator's to resend.
            self._fill_composers(self.state.undone_text)
            self.notify(f"undone: continue as {self.state.undone_to}; your message is back to edit")
        if event.get("type") in SESSION_START_EVENTS:
            # A boundary restarts the prompt id counters; a stale seen-set would swallow the new
            # session's first prompts and the run would block on a modal that never opens.
            self._prompts.reset()
            self._end_hold = False
            # The task is known and a resumed execution releases the hold: retitle the view.
            screen = self._screen_or_none()
            if screen is self._dash:
                self.sub_title = self.run_title()
            elif screen is self._conv:
                self.sub_title = self.screen_title("conversation")
        if event.get("type") in _STATUS_NOW_EVENTS:
            self._refresh_dir_status()
        # Replaying a finished run floods hundreds of events on open; the 0.2s tick repaints
        # once, and stream deltas take the light repaint.
        if event.get("type") in STREAM_DELTA_EVENTS:
            self._light_dirty = True
        else:
            self._dirty = True

    @property
    def worker_lost(self) -> bool:
        """The worker is gone without a session.end, and the fold has caught up to the log."""
        return self.dir_status[0] == "stale" and self.state.log_count >= self._seed_log_count

    def _refresh_dir_status(self) -> None:
        """Recompute dir_status; a change repaints and relabels both composer bars.

        The covered screen's bar is relabelled too, or the two would disagree until
        its next event-driven paint.
        """
        status = status_for_session_dir(self.session_dir, status_facts(self.state))
        if status != self.dir_status:
            self.dir_status = status
            self._dirty = True
            self._conv.refresh_liveness()

    def _screen_or_none(self) -> Screen[object] | None:
        """Return the active screen, or None while the stack is empty at startup or teardown."""
        try:
            return self.screen
        except ScreenStackError:
            return None

    def _tick(self) -> None:
        """Dispatch prompts, route steer requests, beat the heartbeat and repaint once."""
        # Prompts only while the run can consume an answer: the fold keeps an unanswered prompt
        # across a worker death. Skipped ids are not marked seen, so they pop on a later tick.
        if self.session_controllable() and self._screen_or_none() is not None:
            self._prompts.dispatch(self.session_dir, self.state)
        # An external steer request goes to the composer bar once (steer_requests is monotonic).
        if self.state.steer_requests > self._seen_steer:
            self._seen_steer = self.state.steer_requests
            self._steer_request_to_bar()
        # The ~1/s status refresh is how a death, a parked resume or a revival is noticed.
        now = time.monotonic()
        if now - self._heartbeat_at >= 1.0:
            self._heartbeat_at = now
            self._refresh_dir_status()
            if self.model_call_in_flight():
                self.spin += 1
                self._light_dirty = True
        # Repaint only while the dashboard is on top; covered, dirty stays set. screen_stack, not
        # App.screen: the interval outlives the stack during shutdown.
        stack = self.screen_stack
        if not self._stop.is_set() and stack and stack[-1] is self._dash:
            if self._dirty:
                self._dirty = self._light_dirty = False
                with contextlib.suppress(NoMatches):
                    self._dash.render_state()
            elif self._light_dirty:
                self._light_dirty = False
                with contextlib.suppress(NoMatches):
                    self._dash.render_heartbeat()
        # Once the run ended and no modal is open, hold on the payoff instead of tearing down.
        if (
            self.exit_on_end
            and not self._end_hold
            and (self.state.finished or self.worker_lost)
            and not (stack and isinstance(stack[-1], ModalScreen))
        ):
            self._end_hold = True
            self.sub_title = self.run_title()
            self.notify(
                f"{status_label(*self.dir_status)} · Ctrl+Q to leave, or type below to continue"
                " the session",
                timeout=8.0,
            )
            self._dirty = True

    # --- run control (dispatched from the composer bars, keys, and menus) --

    def submit_instruction(self, text: str) -> None:
        """Act on a composer line: a local directive, a steer to a live run, else a resume.

        Args:
            text: The line as typed.
        """
        local = {
            "/undo": self._undo_session,
            "/restate": self._restate,
            "/shells": self._show_shells,
            "/stop": self._typed_stop,
        }
        if (action := local.get(text.strip())) is not None:
            action()
            return
        if self.session_controllable():
            did, said = submit_composer_line(self.session_dir, text)
            self.notify(said, severity="information" if did else "warning")
        else:
            self.resume_with_instruction(text)

    def _restate(self) -> None:
        """Show what happened since the last operator message, rendered from the journal."""
        rendered = restate(list(tail_events(self.logs_path, follow=False)))
        self.push_screen(TextModal("since your last message", rendered))

    def _show_shells(self) -> None:
        """Show the background commands."""
        self.push_screen(TextModal("background commands", shells_text(self.session_dir)))

    def _typed_stop(self) -> None:
        """Stop the live run without a confirm, as `agent6 stop` does."""
        if self.session_controllable():
            self._stop_session(after_step=False)
        else:
            self.notify("nothing to stop: the session is not live", severity="warning")

    def _undo_session(self) -> None:
        """Fork this run at the state before its last operator message, unstarted.

        A live run takes the request at its next boundary and emits session.undone;
        a finished one is forked here and the undone text handed to the composers.
        """
        if self.session_controllable():
            if submit_steer(self.session_dir, "/undo"):
                self.notify("undo requested; applies at the next step")
            else:
                self.notify("could not write the undo request", severity="warning")
            return
        said: list[str] = []
        result = undo_fork(
            None,
            self.session_dir.name,
            cwd=Path.cwd(),
            reporter=Reporter(out=said.append, err=said.append),
        )
        if result is None:
            self.notify(said[-1].strip() if said else "undo failed", severity="warning")
            return
        child, text = result
        self._continue_child = child
        self._fill_composers(text)
        self.notify(f"undone: continue as {child}; your message is back to edit")

    def plan_md(self) -> str:
        """Return a plan session's plan.md, or "" for a run or an unwritten plan."""
        if self.mode != "plan":
            return ""
        with contextlib.suppress(OSError):
            return (self.session_dir / "plan.md").read_text(encoding="utf-8")
        return ""

    @property
    def continue_as(self) -> str:
        """The fork a typed follow-up resumes (an undo's or this view's), "" for this run."""
        return self.state.undone_to or self._continue_child

    def _fill_composers(self, text: str) -> None:
        """Put the text in both composer bars, the covered view's too, to edit and resend."""
        for screen, bar_id in ((self._conv, "#conv-input"), (self._dash, "#dash-input")):
            with contextlib.suppress(NoMatches):
                screen.query_one(bar_id, SteerInput).load_text(text)

    def resume_with_instruction(self, text: str) -> None:
        """Resume this run, or the fork it continues as, with the text as its first steer.

        The text rides `agent6 resume --steer`, which seeds the steer files after its
        stale-state clear; a pre-seed here would be wiped.

        Args:
            text: The instruction the new session injects at its first boundary.
        """
        target = self.continue_as or self.session_dir.name
        self._spawn_resume(
            target,
            steer=text,
            started=f"resuming {target}{self._resume_under()} with your instruction…",
        )

    def _resume_under(self) -> str:
        """Return the row's picks for a notice: " under preset P, model M", or ""."""
        picks = [
            f"preset {self.resume_preset}" if self.resume_preset else "",
            f"model {self.resume_model}" if self.resume_model else "",
        ]
        picked = ", ".join(p for p in picks if p)
        return f" under {picked}" if picked else ""

    @work(thread=True)
    def _spawn_resume(self, target: str, *, started: str, steer: str = "") -> None:
        """Spawn the detached resume off the UI thread and post its notice.

        The spawn waits until the child owns the run or refuses, a second or more.

        Args:
            target: The session to resume.
            started: The notice on success.
            steer: The first instruction, or "".
        """
        err = spawn_detached_resume(
            Path.cwd(),
            target,
            steer=steer,
            preset=self.resume_preset,
            model=self.resume_model,
            config_path=self.config_path,
        )
        self.call_from_thread(
            self.notify, err or started, severity="error" if err else "information"
        )

    def _focus_composer(self) -> None:
        """Focus the visible composer bar; under a viewer or modal the notice points at it."""
        stack = self.screen_stack
        if not stack:  # a tick after the last screen popped
            return
        if stack[-1] is self._conv:
            self._conv.focus_bar()
        elif stack[-1] is self._dash:
            with contextlib.suppress(NoMatches):
                self._dash.query_one("#dash-input", SteerInput).focus()

    def _steer_request_to_bar(self) -> None:
        """Route an external steer request to the visible composer bar and say why."""
        self._focus_composer()
        self.notify("steering requested: type an instruction and press Enter")

    def action_compact(self) -> None:
        """Submit `/compact` to a live run; a finished one has nothing to compact."""
        if not self.session_controllable():
            self.notify("nothing to compact: the session is not live", severity="warning")
            return
        self.submit_instruction("/compact")

    def action_stop_now(self) -> None:
        """Confirm, then stop the run now through the one stop every surface uses."""
        if not self.session_controllable():
            self.notify("nothing to stop: the session is not live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                self._stop_session(after_step=False)

        self.push_screen(
            ConfirmModal(
                "Stop this session now?",
                "Its model call is cut and a running command is handed back; the run ends "
                "at once and can be resumed later with `agent6 resume`. A worker that does "
                "not answer within 5 s is killed.",
                confirm_label="Stop now",
            ),
            _confirmed,
        )

    def action_stop_step(self) -> None:
        """Confirm, then stop the run after the current step lands."""
        if not self.session_controllable():
            self.notify("nothing to stop: the session is not live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                self._stop_session(after_step=True)

        self.push_screen(
            ConfirmModal(
                "Stop after this step?",
                "The current step finishes (its tool results and auto-commit land), "
                "then the run stops. Resume later with `agent6 resume`.",
                confirm_label="Stop",
            ),
            _confirmed,
        )

    @work(thread=True)
    def _stop_session(self, *, after_step: bool) -> None:
        """Stop the run off the UI thread, since a stop now waits for the run to end."""
        out = stop_session(self.session_dir, after_step=after_step)
        self.call_from_thread(
            self.notify, out.message, severity="information" if out.ok else "warning"
        )

    def action_delete_session(self) -> None:
        """Confirm, delete this run's history and return to the hub; the branch stays git's."""
        if self.session_controllable():
            self.notify("stop the session first: it is still live", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                ok, msg = run_cli_capture(
                    [*agent6_argv(self.config_path), "sessions", "rm", "--", self.session_dir.name],
                    Path.cwd(),
                )
                self.notify(
                    msg or ("removed" if ok else "could not remove"),
                    severity="information" if ok else "error",
                )
                if ok:
                    self.action_to_hub()

        self.push_screen(
            ConfirmModal(
                "Delete this session's history?",
                "Removes its transcripts, events and manifest from the state dir. "
                "The run branch and its commits are kept.",
                confirm_label="Delete",
            ),
            _confirmed,
        )

    def action_resume(self) -> None:
        """Resume a finished run in the background; this view follows straight through.

        A run the agent ended has nothing to continue: the refusal lands here, and the
        composer gives it new work.
        """
        if self.session_controllable():
            self.notify("nothing to resume: the session is still going", severity="warning")
            return
        target = self.continue_as or self.session_dir.name
        if not self.continue_as and finished_needs_new_work(self.session_dir):
            self.notify(
                "this run finished; type what to do next below (Enter resumes it with the"
                " instruction)",
                severity="warning",
            )
            self._focus_composer()
            return
        self._spawn_resume(
            target,
            started=f"resuming {target}{self._resume_under()} in the background…",
        )

    def action_run_plan(self) -> None:
        """Spawn `agent6 run --from <id>` for a finished plan and, from the hub, open it."""
        if self.mode != "plan":
            self.notify("this session is not a plan", severity="warning")
            return
        plan_path = self.session_dir / "plan.md"
        try:
            plan_md = plan_path.read_text(encoding="utf-8")
        except OSError:
            self.notify(
                "no readable plan.md yet (still planning, or never finished)", severity="warning"
            )
            return
        if not plan_md.strip():
            self.notify(f"plan {self.session_dir.name!r} has an empty plan.md", severity="warning")
            return
        runs = bucket_dir(layout_of(self.session_dir).state_dir, "runs")
        mkdir_for_real_user(runs)
        self._spawn_run_plan(runs)

    @work(thread=True)
    def _spawn_run_plan(self, runs: Path) -> None:
        """Spawn the detached `run --from` off the UI thread and post its notice.

        Args:
            runs: The runs bucket the new session lands in.
        """
        new_dir, err = spawn_and_locate(
            [*agent6_argv(self.config_path), "run", "--from", self.session_dir.name],
            Path.cwd(),
            before={p for p in runs.iterdir() if p.is_dir()},
            list_dirs=lambda: [p for p in runs.iterdir() if p.is_dir()],
            env={**os.environ, **DETACHED_RUN_ENV},
        )
        if new_dir is None:
            self.call_from_thread(self.notify, err or "could not start the run", severity="error")
            return
        if self.from_hub:
            self.call_from_thread(self.exit, TuiExit(open_next=new_dir))
            return
        self.call_from_thread(
            self.notify, f"run started: {new_dir.name} (follow it: agent6 attach {new_dir.name})"
        )

    def action_fork(self) -> None:
        """Fork this run at its latest checkpoint, unstarted.

        On a finished run the composer is handed to the fork, as after /undo; on a
        live run it keeps steering this run and the notice says how the fork starts.
        """
        said: list[str] = []
        child, rc = create_fork(
            self.config_path,
            self.session_dir.name,
            cwd=Path.cwd(),
            reporter=Reporter(out=said.append, err=said.append),
        )
        if rc != 0:
            self.notify(said[-1].strip() if said else "fork failed", severity="error")
            return
        if self.session_controllable():
            self.notify(f"forked to {child} (unstarted); start it: agent6 resume {child} --steer …")
            return
        self._continue_child = child
        self.notify(f"forked to {child}; type what it should do below (Enter resumes it)")
        self._conv.refresh_liveness()
        with contextlib.suppress(NoMatches):
            self._dash.render_heartbeat()
        self._focus_composer()

    def action_review_run(self) -> None:
        """Review a finished run through `sessions review`, its markdown in a modal."""
        if self.session_controllable():
            self.notify("the run is live; review it once it has ended", severity="warning")
            return
        if self._review_running:
            self.notify("a review of this run is already running; it opens when it lands")
            return
        self._review_running = True
        self.notify(
            f"reviewing {self.session_dir.name} (a model call; the review opens when it lands)"
        )
        self._review_run()

    @work(thread=True)
    def _review_run(self) -> None:
        """Run the review off the UI thread; the modal or the refusal lands from here."""
        try:
            ok, text = run_cli_output(
                [*agent6_argv(self.config_path), "sessions", "review", "--", self.session_dir.name],
                Path.cwd(),
                timeout_s=900.0,
            )
        finally:
            self.call_from_thread(setattr, self, "_review_running", False)
        if not ok:
            self.call_from_thread(self.notify, text or "review failed", severity="error")
            return
        self.call_from_thread(
            self.push_screen, TextModal(f"review of {self.session_dir.name}", text)
        )

    def context_pct(self) -> int | None:
        """Return the context-window fill in percent at the last completed model call."""
        return context_fill(self.state)

    def session_controllable(self) -> bool:
        """Return whether the run can receive operator input over the file bridge.

        Parked, stale and every end word route the composer to resume instead.
        """
        return self.dir_status[0] in LIVE_STATUS_WORDS

    def finished_green(self) -> bool:
        """Return whether the agent finished over a green tree, when a bare resume has no work."""
        s = self.state
        return needs_new_work(finished=s.finished, end_reason=s.end_reason, all_passed=s.all_passed)

    def model_call_in_flight(self) -> bool:
        """Return whether the live run has a model call awaiting its result."""
        role = self.state.last_role
        return (
            self.session_controllable()
            and self.dir_status[0] != "waiting"
            and role is not None
            and role.in_flight
        )

    def action_toggle_dashboard(self) -> None:
        """Flip between the conversation and the dashboard; a no-op under a modal or viewer."""
        if self.screen is self._conv:
            self.pop_screen()
        elif self.screen is self._dash:
            self.push_screen(self._conv)

    def action_to_hub(self) -> None:
        """Leave the view; from the hub loop, back to the hub."""
        self.exit(TuiExit())

    def action_quit_hub(self) -> None:
        """Leave the view and, from the hub loop, the hub too."""
        self.exit(TuiExit(quit_hub=self.from_hub))

    def action_detach_exit(self) -> None:
        """Leave the view; the run `agent6 run --tui` fronts detaches at its next step."""
        if self.exit_on_end and not submit_steer(self.session_dir, "detach"):
            self.notify("could not write the detach request", severity="warning")
            return
        self.detached = True
        self.exit(TuiExit())

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        """Yield textual's palette commands minus the four the menus replace."""
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme", "Quit"):
                yield cmd


def run_tui(
    session_dir: Path,
    *,
    exit_on_end: bool = False,
    from_hub: bool = False,
    config_path: Path | None = None,
) -> TuiExit:
    """Run the views over a session and print the reattach hint after a detach.

    Args:
        session_dir: The session to follow.
        exit_on_end: Hold on the payoff once the run ends, then exit with it.
        from_hub: The hub loop opened the view.
        config_path: The invocation's `--config`.

    Returns:
        How the view ended.
    """
    app = Agent6TUI(
        session_dir, exit_on_end=exit_on_end, from_hub=from_hub, config_path=config_path
    )
    result = app.run() or TuiExit()
    if app.detached:
        sid = session_dir.name
        if exit_on_end:
            # The parent's lifecycle prints the reattach line once the run is in the background.
            print(f"[agent6] leaving the view: {sid} detaches to the background after this step.")
        else:
            print(f"[agent6] detached: {sid} keeps running.")
            print(f"          reattach:  agent6 attach {sid}")
        print("          (Ctrl+_ undoes typing in the composer)")
    return result
