# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 run dashboard (`agent6 run` / `agent6 attach` / `agent6 tui`).

`textual` ships in the base install; importing this module fails clearly if it
has been stripped out. The CLI imports it lazily.

Architecture:
- `Agent6TUI` (the App) is the data plane: a background thread tails
  logs.jsonl -> apply_event -> call_from_thread, and the app owns the folded
  RunState, the approval/question/steer prompt dispatch, run control (steer /
  stop / resume / fork), and the exit codes.
- `DashboardScreen` is the presentation: the panes, their key bindings and
  menus, and the coalesced repaint of the app's RunState.

The dashboard is READ-ONLY on the log stream and only writes the answer files
the workflow polls: `<run_dir>/approvals/<id>.answer` (approve), `.../questions/
<id>.answer` (ask_user), `<run_dir>/steer.answer` (steer), and the
`<run_dir>/compact.request` marker (Compact now). Any other front-end can
mirror this contract.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

try:
    from rich.markup import escape
    from rich.text import Text
    from textual.app import App, ComposeResult, SystemCommand
    from textual.binding import Binding
    from textual.command import DiscoveryHit, Hit, Hits, Provider
    from textual.containers import Horizontal, ScrollableContainer, VerticalScroll
    from textual.css.query import NoMatches
    from textual.screen import Screen
    from textual.scroll_view import ScrollView
    from textual.widget import Widget
    from textual.widgets import (
        DataTable,
        Footer,
        RichLog,
        Static,
        Tree,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.models.registry import context_window
from agent6.runs.ipc import (
    clear_frontend_pid,
    clear_steer_answer,
    frontend_is_live,
    request_compact,
    request_steer,
    request_stop,
    set_session_allow,
    write_answer,
    write_frontend_pid,
    write_question_answers,
    write_steer_answer,
)
from agent6.ui.bridge.spawn import agent6_exe, spawn_and_locate, spawn_detached_resume
from agent6.ui.tui import clipboard
from agent6.ui.tui.conversation import RUN_MENU, ConversationScreen, SteerInput
from agent6.ui.tui.copy_method import open_copy_method_picker
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.menubar import HelpScreen, Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import (
    ApprovalModal,
    ConfirmModal,
    QuestionModal,
    ToolCallDetailModal,
)
from agent6.ui.tui.settings import get_copy_method
from agent6.ui.tui.theme import PALETTE_CSS, MuxPointerShapes, open_theme_picker, setup_theme
from agent6.viewmodel import run_compare
from agent6.viewmodel.format import TASK_STATUS_GLYPH, format_compare, format_cost
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    STREAM_DELTA_EVENTS,
    ApprovalPrompt,
    QuestionPrompt,
    RunState,
    ToolCallView,
    apply_event,
    initial_state,
    run_status_label,
)
from agent6.viewmodel.tail import tail_events

_TASK_ICONS = TASK_STATUS_GLYPH

# How many recent tool calls the inline table shows. The RowSelected handler maps
# a visual row back through the same window, so both must use this one value.
_TOOL_TABLE_ROWS = 20


class _DashboardCommands(Provider):
    """The dashboard's menu actions in the Ctrl+P palette, from the same MENUS
    registry as the menu bar and key bindings -- so the surfaces never drift."""

    @property
    def _dash(self) -> DashboardScreen:
        screen = self.screen
        assert isinstance(screen, DashboardScreen)
        return screen

    async def discover(self) -> Hits:
        for name, runnable, help_text in self._dash.palette_commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, runnable, help_text in self._dash.palette_commands():
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


class DashboardScreen(Screen[None]):
    """The run dashboard panes: task graph, live stream, tool table, log window,
    diff/verify, and the composer bar. Presentation only -- it renders the app's
    folded RunState and dispatches run control back through the app (see the
    module docstring)."""

    CSS = """
    /* Top row: the task graph is usually a few nodes, so it stays compact beside
       the model's live output. */
    #top { height: 4; padding: 0 1; }
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
    /* The stream/diff bodies fill their scroll pane so long content scrolls;
       they are selectable text, so the pointer shows an I-beam over them. */
    #stream-body, #diff-body { width: 1fr; height: auto; pointer: text; }
    /* The composer bar (the same widget as the conversation's): auto-grows with
       its content, squeezing the 1fr #body row above. */
    #dash-input { height: auto; max-height: 8; border: round $primary; }
    #dash-input:focus { border: round $accent; }
    /* One card background everywhere. Tree/DataTable/RichLog default to $surface
       but the Static-based stream/diff panes are transparent (screen background),
       so set it explicitly to keep every card the same. */
    #plan, #stream, #tools, #log, #diff, #dash-input { background: $surface; }
    /* Uniform resting border (matches the home table + config card); the focused
       panel goes $accent. */
    #plan:focus, #stream:focus, #tools:focus, #log:focus, #diff:focus { border: round $accent; }
    """

    COMMANDS: ClassVar = Screen.COMMANDS | {_DashboardCommands}

    MENUS: ClassVar = (
        Menu(
            "File",
            (MenuItem("Back", "to_hub"), MenuItem("Quit", "quit_hub", "ctrl+q")),
        ),
        RUN_MENU,  # shared verbatim with the primary conversation view
        Menu(
            "View",
            (
                MenuItem("Next pane", "focus_next_pane", "tab"),
                MenuItem("Prev pane", "focus_prev_pane", "shift+tab"),
                MenuItem("Maximize pane", "fullscreen"),
                MenuItem("Full log…", "view_logs"),
                MenuItem("Conversation…", "toggle_dashboard"),
                MenuItem("Theme…", "choose_theme"),
                MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help"),
                MenuItem("Command palette", "command_palette", "ctrl+p"),
            ),
        ),
    )
    # The composer bar is the default focus, so -- exactly like the conversation
    # view -- there are no plain-letter shortcuts: the same priority-bound set,
    # in the same footer order, on both screens. Run control lives in the Run
    # menu and the palette. `?` opens help when focus is not in the bar.
    BINDINGS: ClassVar = [
        Binding("ctrl+d", "toggle_dashboard", "Conversation", priority=True),
        Binding("ctrl+c", "copy", "Copy", priority=True),
        Binding("escape", "to_hub", "Back", key_display="Esc", priority=True),
        Binding("pageup", "page_up", "Scroll up", priority=True, show=False),
        Binding("pagedown", "page_down", "Scroll down", priority=True, show=False),
        Binding("ctrl+home", "scroll_top", "Top", priority=True, show=False),
        Binding("ctrl+end", "scroll_bottom", "End", priority=True, show=False),
        Binding("question_mark", "help", "Help", show=False),
        *menu_bindings(MENUS),
    ]

    def __init__(self) -> None:
        super().__init__()
        # Select a task in the #plan tree to filter tools/log/diff to it; re-select
        # to clear. _log_filter tracks what the RichLog currently shows so a filter
        # change forces one full re-render (it is append-only otherwise).
        self._selected_task_id: str | None = None
        self._log_filter: str | None = None
        self._last_log_count = 0
        self._visible_tools: tuple[ToolCallView, ...] = ()  # the tool rows on screen now
        self._footer_finished = False  # last state.finished the footer bindings reflect
        # What each pane last rendered (strong refs; the fold's replace() keeps
        # untouched fields identical, so `is` says "nothing to redo"). Rebuilding
        # the tree/table/diff on every structural event was most of the burst cost.
        self._rendered_tree: tuple[object, object] | None = None
        self._rendered_tools: tuple[object, object] | None = None
        self._rendered_diff: tuple[object, object, object, object] | None = None
        self._compare_line: str | None = None  # cached fan-out compare header (terminal state)

    def _compare_top(self) -> str:
        """The fan-out compare outcome for the header's task line (empty for a
        non-lane run). Read from the manifest once it appears (a lane is stamped
        post-import, by which point it is finished) and cached: it never changes."""
        if self._compare_line is not None:
            return self._compare_line
        formatted = format_compare(run_compare(self._tui.run_dir))
        if formatted is None:
            return ""  # not stamped (yet); don't cache -- a live lane may get stamped later
        headline, rationale = formatted
        rat = f" — {rationale[:100]}" if rationale else ""
        self._compare_line = f"\ncompare: {headline}{rat}"
        return self._compare_line

    @property
    def _tui(self) -> Agent6TUI:
        app = self.app
        assert isinstance(app, Agent6TUI)
        return app

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
        # The composer bar (steer a live run / type the follow-up a finished one
        # resumes with) sits where the budget bar used to; the budget readout
        # lives in the top status line now.
        yield SteerInput(id="dash-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tools", DataTable).add_columns("tool", "args", "ok", "summary")
        self.render_state()  # initial paint; later paints are coalesced in the app's tick
        # Like the conversation: open ready to type (Tab moves out to the panes).
        self.query_one("#dash-input", SteerInput).focus()

    # --- actions ------------------------------------------------------

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        self._tui.submit_instruction(message.text)

    def action_toggle_dashboard(self) -> None:
        self._tui.action_toggle_dashboard()

    def action_to_hub(self) -> None:
        self._tui.action_to_hub()

    def action_quit_hub(self) -> None:
        self._tui.action_quit_hub()

    def action_copy(self) -> None:
        """Copy the mouse selection via the copy_method preference (the same
        Ctrl+C the conversation has; textual's built-in copy would emit a bare
        OSC 52, which multiplexers like tmux swallow)."""
        text = self.get_selected_text()
        if not text or not text.strip():
            self.notify("nothing selected")
            return
        driver = self.app._driver  # pyright: ignore[reportPrivateUsage]

        def emit(seq: str) -> None:
            if driver is not None:
                driver.write(seq)

        try:
            status = clipboard.emit_clipboard(
                text, clipboard.resolve_method(get_copy_method()), emit
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            self.notify(f"copy failed: {exc}", severity="error")
            return
        self.notify(f"copied selection ({status})")

    def _scroll_target(self) -> Widget:
        """The pane the shared scroll keys drive: the focused scrollable if any
        (Tab reaches every pane), else the log -- the dashboard's main scrollback."""
        focused = self.focused
        if isinstance(focused, (ScrollView, ScrollableContainer)):
            return focused
        return self.query_one("#log", RichLog)

    def action_page_up(self) -> None:
        self._scroll_target().scroll_page_up(animate=False)  # instant, like the viewers

    def action_page_down(self) -> None:
        self._scroll_target().scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        self._scroll_target().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll_target().scroll_end(animate=False)

    def action_focus_next_pane(self) -> None:
        # Local action wrapping the App's framework action so it resolves from a
        # menu item / palette entry (a namespaced `app.focus_next` does not).
        self.app.action_focus_next()

    def action_focus_prev_pane(self) -> None:
        self.app.action_focus_previous()

    def action_fullscreen(self) -> None:
        """Maximize the focused pane; Esc or f again restores the dashboard."""
        if self.maximized is not None:
            self.minimize()
        elif self.focused is not None and self.focused.allow_maximize:
            self.maximize(self.focused)

    def action_view_logs(self) -> None:
        """Open the full, scrollable log of THIS run -- the inline #log pane is a
        small sliding window; this is the whole history, scroll-anchored. (l again
        inside the view closes it: LogScreen binds l -> close.)"""
        self.app.push_screen(
            LogScreen(self._tui.logs_path, title=f"logs · {self._tui.run_dir.name}")
        )

    def on_screen_resume(self) -> None:
        # The conversation stamps its own sub_title; re-stamp ours when the
        # toggle (or a closing viewer) brings the dashboard back on top.
        self.app.sub_title = f"run · {self._tui.run_dir.name}"

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
                handler = getattr(self, f"action_{item.action}", None) or getattr(
                    self.app, f"action_{item.action}", None
                )
                if handler is not None:
                    yield (item.label, handler, menu.title)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on a tool-calls row opens its full args + summary in a modal (the
        columns truncate long values). Map the visual row back through the same
        window the table was built from; ignore an out-of-range index from a race
        with a rebuild."""
        if event.data_table.id != "tools":
            return
        window = self._visible_tools  # exactly the rows on screen (task filter applied)
        if 0 <= event.cursor_row < len(window):
            tc = window[event.cursor_row]
            self.app.push_screen(
                ToolCallDetailModal(tc.name, tc.ok, tc.args_full, tc.result_summary)
            )

    def on_tree_node_selected(self, event: Tree.NodeSelected[str | None]) -> None:
        """Select a task in the #plan tree to filter tools/log/diff to it; select it
        again (or a different task) to change the filter, clearing when re-selected."""
        if event.control.id != "plan":
            return
        tid = event.node.data
        if not isinstance(tid, str):
            return
        self._selected_task_id = None if tid == self._selected_task_id else tid
        self.render_state()  # a selection, not an event: re-render with the new filter now

    def action_menu(self, mnemonic: str) -> None:
        self.query_one(MenuBar).open(mnemonic)

    def action_help(self) -> None:
        self.app.push_screen(
            HelpScreen(
                self.MENUS,
                self,
                title="agent6 — keys & actions",
                hints=(
                    "Tab focuses a pane · PgUp/PgDn, Home/End scroll it",
                    "Enter on a tool row opens its full detail",
                    "Pickers: ↑↓ highlight · Space selects",
                ),
            )
        )

    def action_choose_theme(self) -> None:
        open_theme_picker(self.app)

    def action_choose_copy_method(self) -> None:
        open_copy_method_picker(self.app)

    async def on_menu_bar_selected(self, event: MenuBar.Selected) -> None:
        # Screen actions first, then app-level built-ins (command_palette), which
        # are coroutines -- await results. Mirrors the hub / config / machines.
        handler = getattr(self, f"action_{event.action}", None) or getattr(
            self.app, f"action_{event.action}", None
        )
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result

    # --- rendering ---------------------------------------------------

    def render_heartbeat(self) -> None:
        """The CHEAP once-a-second repaint: the top status line, the composer
        bar's labels, and the live stream pane. The full pane rebuild
        (render_state) runs only when events actually arrive -- rebuilding the
        task tree and tool table every heartbeat was most of the idle churn."""
        tui = self._tui
        s = tui.state
        # The finished flag gates which run-control keys the footer shows; when it
        # flips (run ends, or a resume un-finishes it), re-ask check_action and
        # relabel the composer bar (steer <-> continue).
        if s.finished != self._footer_finished:
            self._footer_finished = s.finished
            self.refresh_bindings()
        # Relabel every paint: mode flips on finished, and the context readout
        # in the subtitle moves with the run.
        self.query_one("#dash-input", SteerInput).set_mode(
            live=not s.finished, ctx_pct=tui.context_pct()
        )
        role = s.last_role
        # Live heartbeat: a spinner + seconds since the last event, shown while
        # the run is active. Silent thinking / the resume gap now visibly tick.
        active = not s.finished and not tui.run_ended
        beat = ""
        if active and role is not None:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[tui.spin % 10]
            beat = f" {spinner} {int(time.monotonic() - tui.last_event_at)}s"
        role_line = f"{role.role} / {role.model}{beat}" if role else "(idle)"
        done_n = sum(1 for t in s.tasks if t.status in ("passed", "skipped"))
        step = f"tasks: {done_n}/{len(s.tasks)}" if s.tasks else "tasks: —"
        if not s.finished:
            finished = ""
        else:  # colour the shared status label: green passed, yellow stopped, red else
            color = (
                "green" if s.all_passed else "yellow" if s.end_reason == "steer_abort" else "red"
            )
            finished = f"[b {color}]{escape(run_status_label(s))}[/]"
        cost = f"[b]{format_cost(s.budget.usd_total, partial=s.budget.usd_partial)}[/]"
        # The token-budget consumption, up here as a readout (the bottom bar it
        # used to be gave that row to the composer). Labelled "token budget": a
        # bare "budget: 11%" right after "cost: $0.05" read as a dollar cap.
        used = s.budget.input_total + s.budget.output_total
        cap = s.budget.input_cap + s.budget.output_cap
        budget = f"   token budget: {min(used / cap, 1.0):.0%}" if cap > 0 else ""
        pct = tui.context_pct()
        ctx = f"   ctx: {pct}%" if pct is not None else ""
        self.query_one("#top", Static).update(
            f"[b]agent6[/]  {step}   role: {escape(role_line)}   cost: {cost}{budget}{ctx}"
            f"   {finished}\n"
            f"task: {escape(s.user_task[:120])}"
            f"{escape(self._compare_top())}"
        )

        # Live reasoning / response pane. Built as rich Text so model output is
        # never parsed as markup.
        stream = self.query_one("#stream-body", Static)
        st = Text()
        streaming = (
            role is not None and role.in_flight and (role.streamed_thinking or role.streamed_text)
        )
        if s.finished:
            # The end story, not a stale "idle": how it ended + the closing summary.
            color = (
                "green" if s.all_passed else "yellow" if s.end_reason == "steer_abort" else "red"
            )
            st.append(run_status_label(s) + "\n", style=f"bold {color}")
            if s.finish_summary:
                st.append(s.finish_summary, style="dim")
        elif streaming:
            assert role is not None
            if role.streamed_thinking:
                st.append("💭 ", style="bold")
                st.append(role.streamed_thinking[-1200:] + "\n", style="dim")
            if role.streamed_text:
                st.append(role.streamed_text[-1200:])
        elif active and role is not None:
            # No live deltas: the model is thinking, or a resume is rebuilding
            # context. A ticking heartbeat, never a stale "idle" or blank.
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[tui.spin % 10]
            secs = int(time.monotonic() - tui.last_event_at)
            st.append(f"{spinner} {role.role} working… {secs}s", style="dim italic")
        else:
            st.append("(waiting for the model…)", style="dim")
        stream.update(st)

    def render_state(self) -> None:  # noqa: PLR0912, PLR0915
        self.render_heartbeat()
        tui = self._tui
        s = tui.state

        # A task selected in the #plan tree filters tools/log/diff to it. sel=None
        # is the unfiltered live view; the border titles show which task when set.
        sel = self._selected_task_id
        sel_title = next((t.title for t in s.tasks if t.id == sel), "") if sel else ""
        filt = f" · task: {sel_title[:28]}" if sel else ""

        # Task DAG: the worker's live add_task/update_task breakdown (graph.update
        # snapshots), indented by depth, cursor marked. Rebuilt only when the
        # tasks tuple (or the selection highlight) actually changed.
        if self._rendered_tree is None or not (
            self._rendered_tree[0] is s.tasks and self._rendered_tree[1] == sel
        ):
            self._rendered_tree = (s.tasks, sel)
            tree = self.query_one("#plan", Tree)
            tree.clear()
            for tv in s.tasks:
                icon = _TASK_ICONS.get(tv.status, "·")
                indent = "  " * tv.depth
                marker = "▸ " if tv.is_cursor else ""
                label = Text(f"{indent}{marker}{icon} {tv.title}")
                if tv.id == sel:  # the task the panes are filtered to
                    label.stylize("bold reverse")
                tree.root.add_leaf(label, data=tv.id)
            tree.root.expand()

        table = self.query_one("#tools", DataTable)
        if self._rendered_tools is None or not (
            self._rendered_tools[0] is s.tool_calls and self._rendered_tools[1] == sel
        ):
            self._rendered_tools = (s.tool_calls, sel)
            table.clear()
            tools = [tc for tc in s.tool_calls if sel is None or tc.task_id == sel]
            self._visible_tools = tuple(tools[-_TOOL_TABLE_ROWS:])
            for tc in self._visible_tools:
                ok = "…" if tc.ok is None else ("✓" if tc.ok else "✗")
                table.add_row(
                    Text(tc.name), Text(tc.args_preview[:90]), ok, Text(tc.result_summary[:40])
                )
            table.border_title = f"tools{filt}" if sel else ""

        # Log. Diff on the monotonic log_count, not len(log_tail): log_tail is a
        # sliding window, so a length-based diff freezes once it saturates.
        # Sticky-bottom: only snap to the newest line if the operator was already
        # at the bottom, so scrolling up to read holds position (the pane no
        # longer "plays through" out from under them). End (pane focused) / Full
        # log jump back to the live tail. A filter change forces one full
        # re-render (the RichLog is append-only, so it cannot re-window itself
        # incrementally).
        log = self.query_one("#log", RichLog)
        log.border_title = f"log{filt}" if sel else ""
        if sel != self._log_filter:
            log.clear()
            for ln in s.log_tail:
                if sel is None or ln.task_id == sel:
                    log.write(ln.text)
            log.scroll_end(animate=False)
            self._log_filter = sel
            self._last_log_count = s.log_count
        else:
            n_new = min(s.log_count - self._last_log_count, len(s.log_tail))
            if n_new > 0:
                at_bottom = (log.max_scroll_y - log.scroll_offset.y) <= 1
                for ln in s.log_tail[-n_new:]:
                    if sel is None or ln.task_id == sel:
                        log.write(ln.text)
                if at_bottom:
                    log.scroll_end(animate=False)
            self._last_log_count = s.log_count

        # Diff: the latest auto-commit or live verify output -- or, when a task is
        # selected, the commits made while it was in focus. Built as rich Text to
        # avoid markup parsing of diff/verify bodies (which contain brackets).
        # Skipped whenever none of its inputs changed.
        diff_key = (sel, s.recent_diffs, s.last_verify, s.latest_diff)
        if self._rendered_diff is not None and all(
            a is b for a, b in zip(self._rendered_diff, diff_key, strict=True)
        ):
            return
        self._rendered_diff = diff_key
        diff_widget = self.query_one("#diff-body", Static)
        self.query_one("#diff").border_title = f"diff{filt}" if sel else ""
        verify = s.last_verify
        dt = Text()
        if sel is not None:
            task_diffs = [d for d in s.recent_diffs if d.task_id == sel]
            if task_diffs:
                n = len(task_diffs)
                dt.append(f"selected task · {n} commit{'s' if n != 1 else ''}\n", style="bold")
                _append_colored_diff(dt, task_diffs[-1].patch[:2000])
            else:
                dt.append("(no commits during the selected task yet)", style="dim")
            diff_widget.update(dt)
        # A RUNNING or FAILED verify takes precedence so a failure is never
        # hidden behind a stale passing diff. A passed verify yields to the diff.
        elif verify is not None and verify.exit_code is None:
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


class Agent6TUI(MuxPointerShapes, App[int]):
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
    /* I-beam over anything you can type into (kitty OSC 22; inert elsewhere). */
    Input, TextArea { pointer: text; }
    """
    )

    BINDINGS: ClassVar = [
        # App-level so it works from any screen (viewers included); the hub-aware
        # exit code needs our handler, not textual's default quit.
        Binding("ctrl+q", "quit_hub", "Quit", show=False),
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
        self._dirty = False  # a structural event arrived; _tick coalesces the repaint
        self._light_dirty = False  # only stream deltas / heartbeat: light repaint
        self._claim_checked_at = 0.0  # last frontend.pid liveness probe (file IO)
        self._stop = threading.Event()
        # When True (the auto-spawned co-process of `agent6 run`), close the
        # dashboard once the run ends so the parent command returns; `agent6
        # watch` leaves this False and keeps following.
        self.exit_on_end = exit_on_end
        self.run_ended = False
        # Live heartbeat: a run can be silent for a whole reasoning turn (or the
        # resume context-rebuild gap). Track when the last event landed and
        # repaint ~1/s while active so an elapsed timer + spinner visibly tick --
        # the difference between "thinking" and "hung" the user could not see.
        self.last_event_at = time.monotonic()
        self._heartbeat_at = 0.0
        self.spin = 0
        # (provider, model) -> context window (None = unknown); the registry
        # lookup can touch the model-listing cache file, so ask it once.
        self._ctx_windows: dict[tuple[str, str], int | None] = {}
        self._dash = DashboardScreen()
        self._conv = ConversationScreen(
            self.logs_path, title=f"conversation · {run_dir.name}", primary=True
        )

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        self._ensure_claim()
        self.sub_title = f"run · {self.run_dir.name}"  # menu-bar title context
        # A steer request already in the log is historical (e.g. a CLI Ctrl-C that
        # detached, whose run.steer_requested replays on open); only prompt for ones
        # that arrive AFTER we start watching, so opening a run never pops a stale,
        # already-handled steer modal.
        with contextlib.suppress(OSError):
            self._seen_steer = self.logs_path.read_bytes().count(b'"run.steer_requested"')
        # Pushed (not the app's default screen): only the push path loads a
        # screen's CSS, and the hub pushes its HomeScreen the same way. The
        # conversation opens on top -- the primary view -- with the dashboard
        # beneath it; Ctrl+D (or t from the dashboard) toggles between them.
        # Installed, so popping the conversation hides rather than destroys it.
        self.push_screen(self._dash)
        self.install_screen(self._conv, "conversation")
        self.push_screen(self._conv)
        # Auto-spawn close: the reader thread sets `run_ended` on `run.end`; we
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
        self.last_event_at = time.monotonic()  # feeds the live "working… Ns" heartbeat
        if event.get("type") == "run.start":
            # A resume appends a new session to the same log and its prompt id
            # counters restart at approval-1/question-1; a stale seen-set would
            # swallow the new session's first prompts.
            self._seen_approval_ids.clear()
            self._seen_question_ids.clear()
        if self.exit_on_end and event.get("type") == "run.end":
            self.run_ended = True
        # Coalesce: mark dirty and let the 0.2s _tick repaint once. Replaying a
        # finished run floods hundreds of events on open; rendering each one would
        # rebuild the whole dashboard per event (UI thrash, and vhs can't capture
        # the burst, so the tour video skipped past the dashboard). Streaming
        # deltas only move the live stream pane, so they take the LIGHT repaint
        # (a reasoning burst was triggering full tree/table rebuilds 5x/s).
        if event.get("type") in STREAM_DELTA_EVENTS:
            self._light_dirty = True
        else:
            self._dirty = True

    def _tick(self) -> None:
        # Re-assert the bridge if a peer viewer went away. Throttled: the probe
        # reads frontend.pid + signals the process, needless 5x a second.
        now = time.monotonic()
        if now - self._claim_checked_at >= 2.0:
            self._claim_checked_at = now
            self._ensure_claim()
        for ap in self.state.pending_approvals:
            if not ap.answered and ap.id not in self._seen_approval_ids:
                self._seen_approval_ids.add(ap.id)
                self.push_screen(ApprovalModal(ap.id, ap.prompt), self._on_approval(ap))
        for qp in self.state.pending_questions:
            if not qp.answered and qp.id not in self._seen_question_ids:
                self._seen_question_ids.add(qp.id)
                self.push_screen(QuestionModal(qp.id, qp.questions), self._on_question(qp))
        # Route an external steer request to the composer bar, once per Ctrl-C
        # (steer_requests is monotonic).
        if self.state.steer_requests > self._seen_steer:
            self._seen_steer = self.state.steer_requests
            self._steer_request_to_bar()
        # Heartbeat: while the run is active but silent, advance the spinner
        # ~1/s so the "working… Ns" timer ticks -- an attached viewer can see
        # the run is alive (thinking / resuming), not hung.
        if not self.state.finished and not self.run_ended:
            now = time.monotonic()
            if now - self._heartbeat_at >= 1.0:
                self._heartbeat_at = now
                self.spin += 1
                self._light_dirty = True
        # Coalesced repaint: once per tick, and only when the dashboard is the
        # active, mounted screen. A pushed viewer, a modal, or shutdown leaves the
        # dashboard covered or torn down, so querying its widgets raises; defer
        # the paint (dirty stays set) until it is back on top. Structural events
        # rebuild the panes; deltas/heartbeat repaint only the light parts.
        # Read screen_stack, not App.screen: the interval outlives the stack
        # during shutdown, and a tick landing after the last screen pops must be
        # a no-op, not a ScreenStackError crash.
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
        # Exit only once the run ended AND nothing is still awaiting an answer,
        # so a final approval/question isn't dropped on the way out.
        if (
            self.exit_on_end
            and self.run_ended
            and all(ap.answered for ap in self.state.pending_approvals)
            and all(q.answered for q in self.state.pending_questions)
        ):
            self.exit()

    def _on_approval(self, ap: ApprovalPrompt):  # type: ignore[no-untyped-def]
        def cb(answer: str | None) -> None:
            if answer == "session":  # allow every later run_command this run
                set_session_allow(self.run_dir)
            write_answer(self.run_dir, ap.id, approved=answer in ("yes", "session"))

        return cb

    def _on_question(self, qp: QuestionPrompt):  # type: ignore[no-untyped-def]
        def cb(answers: tuple[str, ...] | None) -> None:
            write_question_answers(self.run_dir, qp.id, answers or ())

        return cb

    # --- run control (dispatched from the composer bars, keys, and menus) --

    def _seed_steer(self, text: str) -> None:
        """Write the steer request + instruction over the file bridge (the same
        seam the old steer dialog used): discard any stale answer, mark the
        request, and provide the answer in one shot."""
        clear_steer_answer(self.run_dir)
        request_steer(self.run_dir)
        write_steer_answer(self.run_dir, text)

    def submit_instruction(self, text: str) -> None:
        """A composer-bar line. Live: inject it at the run's next safe boundary
        (after the current step, never mid tool-call) -- the run keeps going.
        Finished: resume THIS run with the instruction as the follow-up."""
        if self.run_controllable():
            self._seed_steer(text)
            self.notify("steering the run…")
        else:
            self.resume_with_instruction(text)

    def resume_with_instruction(self, text: str) -> None:
        """Resume this run with *text* as its first steering instruction (rides
        `agent6 resume --steer`, which seeds the steer files AFTER its stale-
        state clear; a pre-seed here would be wiped by that clear). The new
        session's steer poll injects the text at its first boundary."""
        err = spawn_detached_resume(Path.cwd(), self.run_dir.name, steer=text)
        self.notify(
            err or f"resuming {self.run_dir.name} with your instruction…",
            severity="error" if err else "information",
        )

    def _steer_request_to_bar(self) -> None:
        """An external steer request (a CLI Ctrl-C on an attached run, `agent6
        steer`): route it to the visible composer bar instead of a popup --
        focus it and say why. With a viewer or modal on top, the notice alone
        points the operator at the bar."""
        stack = self.screen_stack
        if not stack:  # shutdown race: the tick fired after the last screen popped
            return
        if stack[-1] is self._conv:
            self._conv.focus_bar()
        elif stack[-1] is self._dash:
            with contextlib.suppress(NoMatches):
                self._dash.query_one("#dash-input", SteerInput).focus()
        self.notify("steering requested: type an instruction and press Enter")

    def action_compact(self) -> None:
        """Ask the run to compact its context now: drop the compact.request
        marker (the same file-bridge pattern as steer); the loop honors it at
        its next safe boundary by forcing a summarise-and-restart."""
        if not self.run_controllable():
            self.notify("run is not live -- nothing to compact", severity="warning")
            return
        request_compact(self.run_dir)
        self.notify("compaction requested; applies at the next safe boundary")

    def action_stop_now(self) -> None:
        """Stop the run immediately: confirm, then write the abort answer over
        the file bridge -- the stream watchdog interrupts the in-flight turn and
        the run ends (resumable)."""
        if not self.run_controllable():
            self.notify("run is not live -- nothing to stop", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                self._seed_steer("abort")

        self.push_screen(
            ConfirmModal(
                "Stop the run now?",
                "Interrupts the current step; the run ends at once and can be resumed "
                "later with `agent6 resume`.",
                confirm_label="Stop now",
            ),
            _confirmed,
        )

    def action_stop_step(self) -> None:
        """Stop AFTER the current step completes: drop the stop.request marker
        the loop honors at its next completed-iteration boundary, so the step's
        tool results and auto-commit land before the run ends (resumable)."""
        if not self.run_controllable():
            self.notify("run is not live -- nothing to stop", severity="warning")
            return

        def _confirmed(yes: bool | None) -> None:
            if yes:
                request_stop(self.run_dir)
                self.notify("stopping after this step…")

        self.push_screen(
            ConfirmModal(
                "Stop after this step?",
                "The current step finishes (its tool results and auto-commit land), "
                "then the run stops. Resume later with `agent6 resume`.",
                confirm_label="Stop",
            ),
            _confirmed,
        )

    def action_resume(self) -> None:
        """Resume a finished/stopped run: it continues in the background (appending
        to the same log) and this dashboard follows straight through."""
        if not self.state.finished:
            self.notify("run is still going -- nothing to resume", severity="warning")
            return
        err = spawn_detached_resume(Path.cwd(), self.run_dir.name)
        self.notify(
            err or f"resuming {self.run_dir.name} in the background…",
            severity="error" if err else "information",
        )

    def action_fork(self) -> None:
        """Fork this run into a NEW run (from its latest checkpoint) that runs in the
        background and shows up in the hub. Spawns off-thread so the UI stays live."""
        self.notify(f"forking {self.run_dir.name}…", severity="information")
        threading.Thread(target=self._do_fork, daemon=True).start()

    def _do_fork(self) -> None:
        runs = self.run_dir.parent  # sibling run dirs under runs/
        new_dir, err = spawn_and_locate(
            [agent6_exe(), "fork", self.run_dir.name],
            Path.cwd(),
            before={p for p in runs.iterdir() if p.is_dir()},
            list_dirs=lambda: [p for p in runs.iterdir() if p.is_dir()],
        )
        msg = (
            f"forked to {new_dir.name} (open it from the hub)"
            if new_dir
            else (err or "fork failed")
        )
        self.call_from_thread(self.notify, msg, severity="information" if new_dir else "error")

    def context_pct(self) -> int | None:
        """Context-window fill (percent) at the last completed model call: the
        call's full prompt tokens over the model's window (bundled priors, else
        the provider listing cache). None until both sides are known."""
        role = self.state.last_role
        if role is None or role.ctx_tokens <= 0 or not role.model:
            return None
        key = (role.provider, role.model)
        if key not in self._ctx_windows:
            self._ctx_windows[key] = context_window(role.provider, role.model)
        window = self._ctx_windows[key]
        if not window:
            return None
        return min(100, round(100 * role.ctx_tokens / window))

    def run_controllable(self) -> bool:
        """Steer/Stop are no-ops once the run is over: finished (the case that
        matters for `agent6 attach`, where `run_ended` never trips) or the
        co-process app closing on run.end."""
        return not self.run_ended and not self.state.finished

    def action_toggle_dashboard(self) -> None:
        """Flip between the conversation (the primary view) and the dashboard
        (Ctrl+D anywhere, t from the dashboard). The conversation is installed,
        so popping it hides it -- both views keep their state. No-op while a
        modal or a pushed viewer is on top."""
        if self.screen is self._conv:
            self.pop_screen()
        elif self.screen is self._dash:
            self.push_screen(self._conv)

    def action_to_hub(self) -> None:
        self.exit(0)  # back to the hub loop (or just close, standalone)

    def action_quit_hub(self) -> None:
        # In the hub loop, signal "quit the hub" via the exit code; standalone,
        # there's nothing to return to, so a plain close (0) is the same thing.
        self.exit(QUIT_HUB_CODE if self.from_hub else 0)

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        # Drop textual's "Keys" panel (our Help page replaces it), "Screenshot" (an
        # unused default whose SVG export is broken in our terminals), "Theme"
        # (replaced by our live-preview Theme… picker), and "Quit" (its plain exit()
        # returns the wrong code here -- our File menu's Back to hub / Quit do). All
        # of these are provided by MENUS via palette_commands, so nothing's added.
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme", "Quit"):
                yield cmd


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
