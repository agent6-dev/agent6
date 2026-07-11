# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 run dashboard (`agent6 run` / `agent6 watch` / `agent6 tui`).

`textual` ships in the base install; importing this module fails clearly if it
has been stripped out. The CLI imports it lazily.

Architecture:
- main thread: textual event loop.
- background thread: tail_events(logs.jsonl) -> apply_event -> call_from_thread.

The dashboard is READ-ONLY on the log stream and only writes the answer files
the workflow polls: `<run_dir>/approvals/<id>.answer` (approve), `.../questions/
<id>.answer` (ask_user), and `<run_dir>/steer.answer` (Ctrl-C steer). Any other
front-end can mirror this contract.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

try:
    from rich.markup import escape
    from rich.text import Text
    from textual.app import App, ComposeResult, SystemCommand
    from textual.binding import Binding
    from textual.command import DiscoveryHit, Hit, Hits, Provider
    from textual.containers import Horizontal, VerticalScroll
    from textual.css.query import NoMatches
    from textual.screen import Screen
    from textual.widgets import (
        DataTable,
        Footer,
        ProgressBar,
        RichLog,
        Static,
        Tree,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.frontend.approval import (
    clear_frontend_pid,
    clear_steer_answer,
    frontend_is_live,
    request_steer,
    write_answer,
    write_frontend_pid,
    write_question_answer,
    write_steer_answer,
)
from agent6.tui.conversation import ConversationScreen
from agent6.tui.logview import LogScreen
from agent6.tui.menubar import HelpScreen, Menu, MenuBar, MenuItem, menu_bindings
from agent6.tui.modals import ApprovalModal, QuestionModal, SteerModal, ToolCallDetailModal
from agent6.tui.theme import PALETTE_CSS, open_theme_picker, setup_theme
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    ApprovalPrompt,
    QuestionPrompt,
    RunState,
    apply_event,
    initial_state,
)
from agent6.viewmodel.tail import tail_events

_TASK_ICONS = {
    "passed": "✓",
    "failed": "✗",
    "in_progress": "▶",
    "skipped": "—",
    "obsolete": "⊘",
    "pending": "·",
}

# How many recent tool calls the inline table shows. The RowSelected handler maps
# a visual row back through the same window, so both must use this one value.
_TOOL_TABLE_ROWS = 20


class _Agent6Commands(Provider):
    """Agent-specific entries for the Ctrl+P command palette (in addition to
    textual's built-in system commands)."""

    @property
    def _tui(self) -> Agent6TUI:
        app = self.app
        assert isinstance(app, Agent6TUI)
        return app

    async def discover(self) -> Hits:
        for name, runnable, help_text in self._tui.palette_commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, runnable, help_text in self._tui.palette_commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), runnable, help=help_text)


# Dashboard exit code meaning "quit the whole hub" (vs 0 == back to the hub).
QUIT_HUB_CODE = 99


class _ScrollPane(VerticalScroll):
    """A scrollable pane that can be tabbed to and maximized (f). VerticalScroll is
    focusable but disables maximize by default, so re-enable it; the content is a
    child Static the dashboard updates in place."""

    ALLOW_MAXIMIZE = True


class Agent6TUI(App[int]):
    TITLE = "agent6"
    CSS = (
        PALETTE_CSS
        + """
    Screen { layers: base dropdown; }
    #top { height: 4; padding: 0 1; }
    /* Top row: the task graph is usually a few nodes, so it stays compact beside
       the model's live output. */
    #head { height: 28%; }
    #plan { width: 32%; border: round $primary; }
    #stream { width: 1fr; border: round $primary; padding: 0 1; }
    /* The tool table spans the full width so all four columns stay visible. */
    #tools { height: 20%; border: round $primary; }
    /* Maximized (press f), a pane fills the screen instead of holding its resting
       size -- textual tags the maximized widget with `-maximized`. The tool table
       drops its 20% height; the task graph drops its 32% width (else it stays a
       narrow column when maximized, like the tool table stayed short). */
    #tools.-maximized { height: 1fr; }
    #plan.-maximized { width: 1fr; }
    /* Log and diff share the tallest row; press f to maximize either full-screen. */
    #body { height: 1fr; }
    #log { width: 1fr; border: round $primary; }
    #diff { width: 1fr; border: round $primary; padding: 0 1; }
    /* The stream/diff bodies fill their scroll pane so long content scrolls. */
    #stream-body, #diff-body { width: 1fr; height: auto; }
    #budget { width: 1fr; height: 3; border: round $primary; padding: 0 1; }
    /* Uniform resting border (matches the home table + config card); the focused
       panel goes $accent. */
    #plan:focus, #stream:focus, #tools:focus, #log:focus, #diff:focus { border: round $accent; }
    """
    )

    COMMANDS: ClassVar = App.COMMANDS | {_Agent6Commands}

    MENUS: ClassVar = (
        Menu(
            "File",
            (MenuItem("Back", "to_hub", "Esc/q"), MenuItem("Quit", "quit_hub", "ctrl+q")),
        ),
        Menu(
            "View",
            (
                MenuItem("Steer the run", "steer", "s"),
                MenuItem("Next pane", "focus_next", "Tab"),
                MenuItem("Prev pane", "focus_previous", "Shift+Tab"),
                MenuItem("Maximize pane", "fullscreen", "f"),
                MenuItem("Full log…", "view_logs", "l"),
                MenuItem("Conversation…", "view_transcript", "t"),
                MenuItem("Log → top", "scroll_log_home", "g"),
                MenuItem("Log → end", "scroll_log_end", "G"),
                MenuItem("Theme…", "choose_theme"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help", "question_mark"),
                MenuItem("Command palette", "command_palette", "ctrl+p"),
            ),
        ),
    )
    BINDINGS: ClassVar = [
        # Footer order: page action, then meta (Help, Back, Menu) -- matching the
        # home + config footers. The dashboard is one level below the hub, so q
        # (like Esc) backs out TO the hub; only the root hub quits on q. Ctrl+Q is
        # the app-wide hard quit. (Esc on an open modal cancels it first -- the
        # modal consumes the key.)
        Binding("s", "steer", "Steer", show=True),
        Binding("l", "view_logs", "Full log", show=True),
        Binding("t", "view_transcript", "Conversation", show=True),
        # g=top / G=end, matching vi and the LogScreen/ConversationScreen viewers
        # (g used to be "end" here, contradicting those screens reached via l/t).
        Binding("g", "scroll_log_home", "Log→top", show=False),
        Binding("G", "scroll_log_end", "Log→end", show=True),
        Binding("f", "fullscreen", "Fullscreen", show=True),
        Binding("question_mark", "help", "Help"),
        # q and Esc both back out to the hub; shown as one "Esc/q Back" footer entry.
        Binding("escape", "to_hub", "Back", key_display="Esc/q"),
        Binding("q", "to_hub", "Back", show=False),
        Binding("ctrl+q", "quit_hub", "Quit", show=False),
        Binding("tab", "focus_next", "Next pane", show=False),
        Binding("shift+tab", "focus_previous", "Prev pane", show=False),
        *menu_bindings(MENUS),
    ]

    def __init__(self, run_dir: Path, *, exit_on_end: bool = False, from_hub: bool = False) -> None:
        super().__init__()
        self.run_dir = run_dir
        # When launched from the hub loop, Esc returns to it and q quits the hub
        # (signalled by the exit code); standalone, both just close the dashboard.
        self.from_hub = from_hub
        self.logs_path = run_dir / "logs.jsonl"
        self.state: RunState = initial_state()
        self._seen_approval_ids: set[str] = set()
        self._seen_question_ids: set[str] = set()
        self._seen_steer = 0
        self._steer_open = False
        self._last_log_count = 0
        self._dirty = False  # an event arrived; _tick coalesces the repaint
        self._stop = threading.Event()
        # When True (the auto-spawned co-process of `agent6 run`), close the
        # dashboard once the run ends so the parent command returns; `agent6
        # watch` leaves this False and keeps following.
        self.exit_on_end = exit_on_end
        self._run_ended = False

    # --- layout -------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)  # the top row: menus + "agent6 — <run>"
        yield Static("", id="top")
        with Horizontal(id="head"):
            yield Tree("tasks", id="plan")
            with _ScrollPane(id="stream"):
                yield Static("", id="stream-body")
        # cursor_type="row": the whole row highlights and Enter opens its full
        # detail (the columns truncate long args/summaries; see RowSelected).
        yield DataTable(id="tools", cursor_type="row")
        with Horizontal(id="body"):
            # markup=False: log lines contain raw tool args like `args=[a,b]` which
            # Rich would otherwise try to parse as markup and crash. auto_scroll off:
            # _render does sticky-bottom itself (snap to the newest line only when the
            # operator is already at the bottom).
            # max_lines == the state log window: a burst that outruns the window
            # between coalesced paints evicts the pre-burst lines, so the inline
            # pane stays a gapless recent window (full history is under `l`).
            yield RichLog(
                id="log",
                highlight=False,
                markup=False,
                wrap=False,
                auto_scroll=False,
                max_lines=MAX_LOG_TAIL,
            )
            with _ScrollPane(id="diff"):
                yield Static("", id="diff-body")
        yield ProgressBar(id="budget", total=100, show_eta=False)
        yield Footer()

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        self._ensure_claim()
        self.sub_title = f"run · {self.run_dir.name}"  # menu-bar title context
        self.query_one("#tools", DataTable).add_columns("tool", "args", "ok", "summary")
        self._render()  # initial paint; later paints are coalesced in _tick
        # Auto-spawn close: the reader thread sets `_run_ended` on `run.end`; we
        # poll it from a timer in the app's OWN loop and exit there. Exit()
        # scheduled from inside a call_from_thread callback does not take effect,
        # but exiting from a timer callback does. The same timer also drives the
        # approval / question / steer modals.
        self.set_interval(0.2, self._tick)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _ensure_claim(self) -> None:
        """Claim frontend.pid only when no live front-end owns it, so a concurrent
        web/TUI viewer on the same run is not clobbered. Re-asserted each tick, so
        if the owner goes away the bridge self-heals to this still-open dashboard
        (the same pattern as MachineWatchScreen)."""
        if not frontend_is_live(self.run_dir):
            write_frontend_pid(self.run_dir, os.getpid())

    def on_unmount(self) -> None:
        self._stop.set()
        # Stop claiming the run's prompts, but only if frontend.pid is still ours
        # (a concurrent viewer may own it).
        try:
            owned = (self.run_dir / "frontend.pid").read_text(encoding="utf-8").strip() == str(
                os.getpid()
            )
        except OSError:
            owned = False
        if owned:
            clear_frontend_pid(self.run_dir)

    # --- reader thread -----------------------------------------------

    def _reader_loop(self) -> None:
        for event in tail_events(self.logs_path, follow=True, stop_when_finished=self.exit_on_end):
            if self._stop.is_set():
                return
            self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: dict[str, object]) -> None:
        self.state = apply_event(self.state, event)
        if event.get("type") == "run.start":
            # A resume appends a new session to the same log and its prompt id
            # counters restart at approval-1/question-1; a stale seen-set would
            # swallow the new session's first prompts.
            self._seen_approval_ids.clear()
            self._seen_question_ids.clear()
        if self.exit_on_end and event.get("type") == "run.end":
            self._run_ended = True
        # Coalesce: mark dirty and let the 0.2s _tick repaint once. Replaying a
        # finished run floods hundreds of events on open; rendering each one would
        # rebuild the whole dashboard per event (UI thrash, and vhs can't capture
        # the burst, so the tour video skipped past the dashboard).
        self._dirty = True

    def _tick(self) -> None:
        self._ensure_claim()  # re-assert the bridge if a peer viewer went away
        for ap in self.state.pending_approvals:
            if not ap.answered and ap.id not in self._seen_approval_ids:
                self._seen_approval_ids.add(ap.id)
                self.push_screen(ApprovalModal(ap.id, ap.prompt), self._on_approval(ap))
        for qp in self.state.pending_questions:
            if not qp.answered and qp.id not in self._seen_question_ids:
                self._seen_question_ids.add(qp.id)
                self.push_screen(
                    QuestionModal(qp.id, qp.question, qp.options), self._on_question(qp)
                )
        # Pop a steer modal once per Ctrl-C (steer_requests is monotonic).
        if self.state.steer_requests > self._seen_steer and not self._steer_open:
            self._seen_steer = self.state.steer_requests
            self._steer_open = True
            self.push_screen(SteerModal(), self._on_steer)
        # Coalesced repaint: once per tick, and only when the dashboard is the
        # active, mounted screen. A modal (transcript/log) or shutdown leaves the
        # dashboard widgets covered or torn down, so querying them raises; defer
        # the paint (dirty stays set) until it is back on top.
        if self._dirty and not self._stop.is_set() and len(self.screen_stack) <= 1:
            self._dirty = False
            with contextlib.suppress(NoMatches):
                self._render()
        # Exit only once the run ended AND nothing is still awaiting an answer,
        # so a final approval/question/steer isn't dropped on the way out.
        if (
            self.exit_on_end
            and self._run_ended
            and not self._steer_open
            and all(ap.answered for ap in self.state.pending_approvals)
            and all(q.answered for q in self.state.pending_questions)
        ):
            self.exit()

    def _on_approval(self, ap: ApprovalPrompt):  # type: ignore[no-untyped-def]
        def cb(approved: bool | None) -> None:
            write_answer(self.run_dir, ap.id, approved=bool(approved))

        return cb

    def _on_question(self, qp: QuestionPrompt):  # type: ignore[no-untyped-def]
        def cb(answer: str | None) -> None:
            write_question_answer(self.run_dir, qp.id, answer or "")

        return cb

    def _on_steer(self, answer: str | None) -> None:
        self._steer_open = False
        write_steer_answer(self.run_dir, answer or "")

    def action_steer(self) -> None:
        """Steer the run WITHOUT Ctrl-C: open the steer box and drop a request
        marker the run picks up at its next safe boundary (after the current step,
        never mid tool-call), then injects your instruction into the next step.
        The run keeps going -- no stop/resume. Submit blank to cancel."""
        if self._steer_open or self._run_ended:
            return
        self._steer_open = True
        clear_steer_answer(self.run_dir)  # discard any stale answer -> run waits for this one
        request_steer(self.run_dir)
        self.push_screen(SteerModal(), self._on_steer)

    # --- command palette ---------------------------------------------

    def palette_commands(self) -> Iterator[tuple[str, Callable[[], Any], str]]:
        """(label, runnable, help) per menu action -- the Ctrl+P palette source, from
        the same MENUS registry as the menu bar, footer, and key bindings, so the
        surfaces never drift (same generator pattern as the home hub + config). Skips
        the palette opener itself (textual provides it)."""
        for menu in self.MENUS:
            for item in menu.items:
                if item.action == "command_palette":
                    continue
                handler = getattr(self, f"action_{item.action}", None)
                if handler is not None:
                    yield (item.label, handler, menu.title)

    def action_to_hub(self) -> None:
        self.exit(0)  # back to the hub loop (or just close, standalone)

    def action_quit_hub(self) -> None:
        # In the hub loop, signal "quit the hub" via the exit code; standalone,
        # there's nothing to return to, so a plain close (0) is the same thing.
        self.exit(QUIT_HUB_CODE if self.from_hub else 0)

    def action_fullscreen(self) -> None:
        """Maximize the focused pane; Esc or f again restores the dashboard."""
        screen = self.screen
        if screen.maximized is not None:
            screen.minimize()
        elif self.focused is not None and self.focused.allow_maximize:
            screen.maximize(self.focused)

    def action_scroll_log_end(self) -> None:
        self.query_one("#log", RichLog).scroll_end(animate=False)

    def action_scroll_log_home(self) -> None:
        self.query_one("#log", RichLog).scroll_home(animate=False)

    def action_view_logs(self) -> None:
        """Open the full, scrollable log of THIS run -- the inline #log pane is a
        small sliding window; this is the whole history, scroll-anchored."""
        self.push_screen(LogScreen(self.logs_path, title=f"logs · {self.run_dir.name}"))

    def action_view_transcript(self) -> None:
        """Open THIS run's full LLM conversation (assistant text + every tool
        call with its result), folded live from the run's event log."""
        self.push_screen(
            ConversationScreen(self.logs_path, title=f"conversation · {self.run_dir.name}")
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a tool-calls row opens its full args + summary in a modal (the
        columns truncate long values). Map the visual row back through the same
        window the table was built from; ignore an out-of-range index from a race
        with a rebuild."""
        if event.data_table.id != "tools":
            return
        window = self.state.tool_calls[-_TOOL_TABLE_ROWS:]
        if 0 <= event.cursor_row < len(window):
            tc = window[event.cursor_row]
            self.push_screen(ToolCallDetailModal(tc.name, tc.ok, tc.args_full, tc.result_summary))

    def action_menu(self, mnemonic: str) -> None:
        self.query_one(MenuBar).open(mnemonic)

    def action_help(self) -> None:
        self.push_screen(HelpScreen(self.MENUS, title="agent6 — keys & actions"))

    def action_choose_theme(self) -> None:
        open_theme_picker(self)

    async def on_menu_bar_selected(self, event: MenuBar.Selected) -> None:
        # action_quit (and other built-ins) are coroutines, so await results.
        handler = getattr(self, f"action_{event.action}", None)
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        # Drop textual's "Keys" panel (our Help page replaces it), "Screenshot" (an
        # unused default whose SVG export is broken in our terminals), "Theme"
        # (replaced by our live-preview Theme… picker), and "Quit" (its plain exit()
        # returns the wrong code here -- our File menu's Back to hub / Quit do). All
        # of these are provided by MENUS via palette_commands, so nothing's added.
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme", "Quit"):
                yield cmd

    # --- rendering ---------------------------------------------------

    def _render(self) -> None:  # noqa: PLR0912, PLR0915
        s = self.state
        role = s.last_role
        role_line = (
            f"{role.role} / {role.model} {'…' if role.in_flight else ''}" if role else "(idle)"
        )
        done_n = sum(1 for t in s.tasks if t.status in ("passed", "skipped"))
        step = f"tasks: {done_n}/{len(s.tasks)}" if s.tasks else "tasks: —"
        finished = (
            "[b green]done[/]"
            if s.finished and s.all_passed
            else ("[b red]done (failed)[/]" if s.finished else "")
        )
        cost_prefix = "~" if s.budget.usd_partial else ""
        cost = f"[b]{cost_prefix}${s.budget.usd_total:.4f}[/]"
        self.query_one("#top", Static).update(
            f"[b]agent6[/]  {step}   role: {escape(role_line)}   cost: {cost}   {finished}\n"
            f"task: {escape(s.user_task[:120])}"
        )

        # Live reasoning / response pane. Built as rich Text so model output is
        # never parsed as markup.
        stream = self.query_one("#stream-body", Static)
        st = Text()
        if role is not None and role.in_flight:
            if role.streamed_thinking:
                st.append("💭 ", style="bold")
                st.append(role.streamed_thinking[-1200:] + "\n", style="dim")
            if role.streamed_text:
                st.append(role.streamed_text[-1200:])
            if not role.streamed_thinking and not role.streamed_text:
                st.append(f"{role.role} working…", style="dim italic")
        elif role is not None:
            st.append(f"{role.role} idle", style="dim")
        else:
            st.append("(waiting for the model…)", style="dim")
        stream.update(st)

        # Task DAG: the worker's live add_task/update_task breakdown (graph.update
        # snapshots), indented by depth, cursor marked.
        tree = self.query_one("#plan", Tree)
        tree.clear()
        for tv in s.tasks:
            icon = _TASK_ICONS.get(tv.status, "·")
            indent = "  " * tv.depth
            marker = "▸ " if tv.is_cursor else ""
            tree.root.add_leaf(Text(f"{indent}{marker}{icon} {tv.title}"))
        tree.root.expand()

        bar = self.query_one("#budget", ProgressBar)
        used = s.budget.input_total + s.budget.output_total
        cap = s.budget.input_cap + s.budget.output_cap
        if cap > 0:
            bar.total = cap
            bar.progress = min(used, cap)
        bar.tooltip = (
            f"in {s.budget.input_total}/{s.budget.input_cap}  "
            f"out {s.budget.output_total}/{s.budget.output_cap}  "
            f"{'~' if s.budget.usd_partial else ''}${s.budget.usd_total:.4f}"
        )

        table = self.query_one("#tools", DataTable)
        table.clear()
        for tc in s.tool_calls[-_TOOL_TABLE_ROWS:]:
            ok = "…" if tc.ok is None else ("✓" if tc.ok else "✗")
            table.add_row(
                Text(tc.name), Text(tc.args_preview[:90]), ok, Text(tc.result_summary[:40])
            )

        # Log. Diff on the monotonic log_count, not len(log_tail): log_tail is a
        # sliding window, so a length-based diff freezes once it saturates.
        # Sticky-bottom: only snap to the newest line if the operator was already
        # at the bottom, so scrolling up to read holds position (the pane no
        # longer "plays through" out from under them). `G` / Full log jump back
        # to the live tail.
        log = self.query_one("#log", RichLog)
        n_new = min(s.log_count - self._last_log_count, len(s.log_tail))
        if n_new > 0:
            at_bottom = (log.max_scroll_y - log.scroll_offset.y) <= 1
            for line in s.log_tail[-n_new:]:
                log.write(line)
            if at_bottom:
                log.scroll_end(animate=False)
        self._last_log_count = s.log_count

        # Diff (the latest auto-commit) or live verify output. Built as rich Text
        # to avoid markup parsing of diff/verify bodies (which contain brackets).
        diff_widget = self.query_one("#diff-body", Static)
        verify = s.last_verify
        dt = Text()
        # A RUNNING or FAILED verify takes precedence so a failure is never
        # hidden behind a stale passing diff. A passed verify yields to the diff.
        if verify is not None and verify.exit_code is None:
            dt.append("verify running: ", style="bold")
            dt.append(" ".join(verify.cmd)[:200] + "\n")
            dt.append("…", style="dim")
            diff_widget.update(dt)
        elif verify is not None and verify.exit_code != 0:
            dt.append(f"verify exit={verify.exit_code} ", style="bold red")
            dt.append(f"({verify.duration_s:.1f}s)  {' '.join(verify.cmd)[:160]}\n")
            dt.append((verify.stderr_tail or verify.stdout_tail)[:2000] or "(no output)")
            diff_widget.update(dt)
        elif s.latest_diff:
            dt.append("latest commit diff\n", style="bold")
            _append_colored_diff(dt, s.latest_diff[:2000])
            diff_widget.update(dt)
        elif verify is not None:
            dt.append(f"verify passed ({verify.duration_s:.1f}s)", style="bold green")
            diff_widget.update(dt)
        else:
            diff_widget.update(Text("(no diffs yet)", style="dim"))


def _append_colored_diff(dt: Text, patch: str) -> None:
    """Append a unified diff with +/- line coloring (no markup parsing)."""
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            dt.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("---"):
            dt.append(line + "\n", style="red")
        elif line.startswith("@@"):
            dt.append(line + "\n", style="cyan")
        else:
            dt.append(line + "\n")


def run_tui(run_dir: Path, *, exit_on_end: bool = False, from_hub: bool = False) -> int:
    return Agent6TUI(run_dir, exit_on_end=exit_on_end, from_hub=from_hub).run() or 0
