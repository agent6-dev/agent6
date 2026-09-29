# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run dashboard: the panes, their keys and menus, and the coalesced repaint.

`Agent6TUI` owns the data plane and pushes this screen.
"""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

try:
    from rich.markup import escape
    from rich.text import Text
    from textual import events
    from textual.app import ComposeResult
    from textual.containers import Horizontal, ScrollableContainer
    from textual.css.query import NoMatches
    from textual.screen import Screen
    from textual.scroll_view import ScrollView
    from textual.widget import Widget
    from textual.widgets import (
        DataTable,
        Footer,
        RichLog,
        Static,
        TextArea,
        Tree,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.ui.tui import clipboard
from agent6.ui.tui._dashboard_header import RunHeader
from agent6.ui.tui._diff_pane import DiffPane
from agent6.ui.tui.composer import (
    APPROVAL_KEY_BINDINGS,
    RUN_MENU,
    ApprovalKeys,
    ComposerMode,
    ResumeOptions,
    SteerInput,
    SteerSuggest,
    open_history_search,
)
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.menubar import SCROLL_ITEMS, Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import (
    ToolCallDetailModal,
)
from agent6.ui.tui.prompts import PromptDispatcher
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.settings import get_copy_method
from agent6.ui.tui.theme import (
    status_style,
)
from agent6.ui.tui.widgets import ScrollPane
from agent6.viewmodel.format import (
    clip_cell,
    dead_run_note,
    spinner_frame,
    status_label,
)
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    SessionState,
    ToolCallView,
    fold_until_commit,
)
from agent6.viewmodel.tail import tail_events

if TYPE_CHECKING:
    from agent6.ui.tui.app import Agent6TUI

# The tool calls the inline table shows; the RowSelected handler maps rows through the same window.
_TOOL_TABLE_ROWS = 20

# Below this terminal height the dashboard is compact: one pane row at a time.
_COMPACT_ROWS = 28
_PANE_ROWS = ("head", "tools", "body")


class DashboardScreen(ApprovalKeys, ScreenChrome, Screen[None]):
    """The dashboard panes: tasks, the live stream, the tool table, the log, the diff, the composer.

    Presentation only: it renders the app's folded state and sends run control back
    through the app.
    """

    CSS = """
    /* Top row: the task graph is usually a few nodes, so it stays compact beside
       the model's live output. */
    #top { height: auto; max-height: 7; padding: 0 1; }
    /* Compact (a short terminal): one pane row at a time, the one holding focus
       (the log and diff otherwise), with a summary line for what is folded. A
       folded row keeps zero height, not display: none, so Tab still reaches its
       panes and unfolds them. */
    #summary { display: none; height: 1; padding: 0 1; color: $text-muted; }
    DashboardScreen.-compact #summary { display: block; }
    DashboardScreen.-compact #head, DashboardScreen.-compact #tools,
    DashboardScreen.-compact #body { height: 0; }
    DashboardScreen.-compact #tools { border: none; scrollbar-size: 0 0; }
    DashboardScreen.-compact.-show-tools #tools, DashboardScreen.-compact #tools.-maximized {
        border: round $primary; scrollbar-size: 1 1;
    }
    DashboardScreen.-compact.-show-tools #tools:focus { border: round $accent; }
    DashboardScreen.-compact.-show-head #head, DashboardScreen.-compact.-show-tools #tools,
    DashboardScreen.-compact.-show-body #body, DashboardScreen.-compact #tools.-maximized {
        height: 1fr;
    }
    #head { height: 28%; }
    #plan { width: 32%; border: round $primary; }
    #stream { width: 1fr; border: round $primary; padding: 0 1; }
    /* The tool table spans the full width so all four columns stay visible. */
    #tools { height: 20%; border: round $primary; }
    /* Maximized, a pane fills the screen instead of holding its resting size;
       textual tags the maximized widget with `-maximized`. The tool table drops
       its 20% height; the task graph drops its 32% width (else it stays a
       narrow column when maximized). */
    #tools.-maximized { height: 1fr; }
    #plan.-maximized { width: 1fr; }
    /* Log and diff share the tallest row; either maximizes full-screen. */
    #body { height: 1fr; }
    #log { width: 1fr; border: round $primary; }
    #diff { width: 1fr; border: round $primary; padding: 0 1; }
    /* The stream body fills its scroll pane so long content scrolls; it is
       selectable text, so the pointer shows an I-beam over it. */
    #stream-body { width: 1fr; height: auto; pointer: text; }
    /* The composer bar (the same widget as the conversation's) auto-grows with
       its content, squeezing the 1fr #body row above. */
    /* One card background everywhere. Tree/DataTable/RichLog default to $surface
       but the Static-based stream/diff panes are transparent (screen background),
       so set it explicitly to keep every card the same. */
    #plan, #stream, #tools, #log, #diff, #dash-input { background: $surface; }
    /* Uniform resting border (matches the home table + config card); the focused
       panel goes $accent. */
    #plan:focus, #stream:focus, #tools:focus, #log:focus, #diff:focus { border: round $accent; }
    """

    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    APPROVAL_DOCK_BEFORE: ClassVar = "#dash-suggest"
    APPROVAL_FOCUS_AFTER: ClassVar = "#stream"
    HELP_HINTS: ClassVar = (
        "Tab focuses a pane · PgUp/PgDn, Home/End scroll it",
        "Enter on a tool row opens its full detail",
        "Pickers: ↑↓ highlight · Space selects",
    )

    # The composer holds the focus, so the keys are modified keys and Esc, as priority bindings.
    MENUS: ClassVar = (
        Menu(
            "File",
            (
                MenuItem("Back", "to_hub", priority=True),
                MenuItem("Quit", "quit_hub", priority=True),
            ),
        ),
        RUN_MENU,
        Menu(
            "View",
            (
                MenuItem("Next pane", "focus_next_pane"),
                MenuItem("Prev pane", "focus_prev_pane"),
                MenuItem("Maximize pane", "fullscreen"),
                *SCROLL_ITEMS,
                MenuItem("Full log…", "view_logs"),
                MenuItem("Copy selection", "copy", priority=True),
                MenuItem("Conversation…", "toggle_dashboard", priority=True),
                MenuItem("Theme…", "choose_theme"),
                MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help"),
                MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    # The same footer, in the same order, as the conversation view.
    FOOTER: ClassVar = (
        ("toggle_dashboard", "Conversation"),
        ("copy", "Copy"),
        ("history_search", "History"),
        ("to_hub", "Back"),
    )
    BINDINGS: ClassVar = [
        *menu_bindings("dashboard", MENUS, footer=FOOTER),
        *APPROVAL_KEY_BINDINGS,  # an open approval answers from any non-text focus
    ]

    @property
    def diff(self) -> DiffPane:
        """The diff pane."""
        return self.query_one("#diff", DiffPane)

    def on_diff_pane_step_changed(self, _event: DiffPane.StepChanged) -> None:
        """Repaint for the newly selected step."""
        self.render_state()

    def _details_state(self, s: SessionState) -> tuple[SessionState, str]:
        """Return the state the task tree and the cost line show, and its "as of" suffix.

        Live, or as of the selected step, folded once per selection from the log.
        """
        sha = self.diff.step_sel
        if not sha:
            return s, ""
        if self._step_state is None or self._step_state[0] != sha:
            at = fold_until_commit(tail_events(self._tui.logs_path, follow=False), sha)
            if at is None:
                return s, ""
            self._step_state = (sha, at)
        at = self._step_state[1]
        return at, f" · as of iter {at.steps[-1].iteration}"

    def __init__(
        self,
        *,
        presets: list[str] | None = None,
        routes: list[str] | None = None,
        prompts: PromptDispatcher | None = None,
    ) -> None:
        super().__init__()
        self._presets = presets if presets is not None else []
        self._routes = routes if routes is not None else []
        self._prompts = prompts
        # A task selected in the tree filters the tools, the log and the diff to it.
        self._selected_task_id: str | None = None
        self._log_filter: str | None = None  # what the append-only log shows; a change re-renders
        self._last_log_count = 0
        self._visible_tools: tuple[ToolCallView, ...] = ()  # the tool rows on screen now
        # What each pane last rendered: the fold keeps untouched fields identical, so `is` says
        # nothing to redo, and a burst never rebuilds the tree and table per event.
        self._rendered_tree: tuple[object, ...] | None = None
        self._rendered_tools: tuple[object, object] | None = None
        self._step_state: tuple[str, SessionState] | None = None  # the fold as of the step

    @property
    def _tui(self) -> Agent6TUI:
        """The app; only Agent6TUI pushes this screen."""
        return cast("Agent6TUI", self.app)

    def compose(self) -> ComposeResult:
        """Yield the menu bar, the header, the three pane rows, the composer and the footer."""
        yield MenuBar(self.MENUS)
        yield RunHeader()
        yield Static("", id="summary")  # compact only: the folded rows in one line
        with Horizontal(id="head"):
            yield Tree("tasks", id="plan")
            with ScrollPane(id="stream"):
                yield Static("", id="stream-body")
        yield DataTable(id="tools", cursor_type="row")
        with Horizontal(id="body"):
            # markup=False: raw tool args would parse as markup. auto_scroll off: render_state
            # keeps the bottom itself. max_lines is the state's window, so the pane stays gapless.
            yield RichLog(
                id="log",
                highlight=False,
                markup=False,
                wrap=False,
                auto_scroll=False,
                max_lines=MAX_LOG_TAIL,
            )
            yield DiffPane(self._tui.session_dir, id="diff")
        yield SteerSuggest(id="dash-suggest")  # command hints while typing `/…`
        yield ResumeOptions(self._presets, self._routes, id="dash-resume")  # while resuming
        yield SteerInput(id="dash-input")
        yield Footer()

    def on_resize(self, _event: events.Resize) -> None:
        """Go compact below the row threshold."""
        self.set_class(self.size.height < _COMPACT_ROWS, "-compact")
        self._show_pane_row()

    def on_descendant_focus(self, _event: events.DescendantFocus) -> None:
        """Unfold the focused pane's row when compact."""
        self._show_pane_row()

    def approval_dir(self) -> Path:
        """Return the session dir."""
        return self._tui.session_dir

    def approval_live(self) -> bool:
        """Return whether the run takes an answer."""
        return self._tui.session_controllable()

    def approval_answered(self, verdict: str) -> None:
        """Repaint the row after an answer."""
        self._render_approval()

    def _render_approval(self) -> None:
        """Dock the open approval's row, carrying the command since no transcript does."""
        tui = self._tui
        self.sync_approval(self.open_approval(tui.state) if tui.session_controllable() else None)

    def _show_pane_row(self) -> None:
        """Unfold the row holding the focus, else the log and diff, when compact."""
        focused = self.focused
        row = "body"
        for name in ("head", "tools"):
            pane = self.query_one(f"#{name}")
            if focused is not None and (focused is pane or pane in focused.ancestors):
                row = name
        for name in _PANE_ROWS:
            self.set_class(name == row, f"-show-{name}")

    def on_mount(self) -> None:
        """Set the tool columns, hide the diff for an ask or a plan, paint and focus the bar."""
        self.query_one("#tools", DataTable).add_columns("tool", "args", "ok", "summary")
        self.diff.display = self._tui.mode not in ("ask", "plan")
        self.render_state()  # later paints are coalesced in the app's tick
        self.query_one("#dash-input", SteerInput).focus()

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        """Hand a composer line to the app."""
        self._tui.submit_instruction(message.text)

    def action_history_search(self) -> None:
        """Open the prompt history search over the composer."""
        open_history_search(self, self.query_one("#dash-input", SteerInput), self._tui.logs_path)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Refresh the command hints as the composer's text changes."""
        if event.text_area.id != "dash-input":
            return
        with contextlib.suppress(NoMatches):
            self.query_one("#dash-suggest", SteerSuggest).show_for(
                event.text_area.text,
                mode="steer" if self._tui.session_controllable() else "resume",
            )

    def action_toggle_dashboard(self) -> None:
        """Flip to the conversation through the app."""
        self._tui.action_toggle_dashboard()

    def action_to_hub(self) -> None:
        """Close an open list, else leave the run view through the app."""
        if self.close_open_list():
            return
        self._tui.action_to_hub()

    def action_quit_hub(self) -> None:
        """Leave the view and the hub through the app."""
        self._tui.action_quit_hub()

    def action_copy(self) -> None:
        """Copy the mouse selection by the configured method; a bare OSC 52 is swallowed by tmux."""
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
        """Return the pane the scroll keys drive: the focused scrollable, else the log."""
        focused = self.focused
        if isinstance(focused, (ScrollView, ScrollableContainer)):
            return focused
        return self.query_one("#log", RichLog)

    def action_page_up(self) -> None:
        """Scroll the target pane one page up; instant, like the viewers."""
        self._scroll_target().scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        """Scroll the target pane one page down."""
        self._scroll_target().scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        """Scroll the target pane to the top."""
        self._scroll_target().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll the target pane to the bottom."""
        self._scroll_target().scroll_end(animate=False)

    def action_focus_next_pane(self) -> None:
        """Focus the next pane; a local action, so a menu item can name it."""
        self.app.action_focus_next()

    def action_focus_prev_pane(self) -> None:
        """Focus the previous pane."""
        self.app.action_focus_previous()

    def action_fullscreen(self) -> None:
        """Maximize the focused pane, or restore it."""
        if self.maximized is not None:
            self.minimize()
        elif self.focused is not None and self.focused.allow_maximize:
            self.maximize(self.focused)

    def action_view_logs(self) -> None:
        """Open the whole log; the inline pane is a sliding window."""
        self.app.push_screen(
            LogScreen(self._tui.logs_path, title=lambda: self._tui.screen_title("logs"))
        )

    def on_screen_resume(self) -> None:
        """Re-stamp the title and repaint the light parts on coming back on top."""
        self.app.sub_title = self._tui.run_title()
        with contextlib.suppress(NoMatches):
            self.render_heartbeat()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Open a tool row's full args and summary in a modal on Enter."""
        if event.data_table.id != "tools":
            return
        window = self._visible_tools
        if 0 <= event.cursor_row < len(window):
            tc = window[event.cursor_row]
            self.app.push_screen(
                ToolCallDetailModal(tc.name, tc.ok, tc.args_full, tc.result_summary)
            )

    def on_tree_node_selected(self, event: Tree.NodeSelected[str | None]) -> None:
        """Filter the panes to the selected task; selecting it again clears the filter."""
        if event.control.id != "plan":
            return
        tid = event.node.data
        if not isinstance(tid, str):
            return
        self._selected_task_id = None if tid == self._selected_task_id else tid
        self.render_state()

    def render_heartbeat(self) -> None:
        """Repaint the light parts: the header, the composer's labels and the stream pane."""
        tui = self._tui
        s = tui.state
        mode: ComposerMode = "steer" if tui.session_controllable() else "resume"
        self.query_one("#dash-input", SteerInput).set_mode(
            mode=mode,
            ctx_pct=tui.context_pct(),
            continue_as=tui.continue_as,
            needs_new_work=tui.finished_green(),
        )
        self.query_one("#dash-resume", ResumeOptions).show(mode == "resume")
        self._render_approval()
        active = tui.model_call_in_flight()
        ds, as_of = self._details_state(s)
        self.query_one("#top", RunHeader).refresh_lines(s, ds, as_of, active=active)
        if self.has_class("-compact"):
            calls = f"{len(s.tool_calls)} tool call{'' if len(s.tool_calls) == 1 else 's'}"
            if s.tool_calls:
                last = s.tool_calls[-1]
                ok = "…" if last.ok is None else ("✓" if last.ok else "✗")
                calls += f" · last {last.name} {ok}"
            folded = Text(f"{calls} · Tab unfolds a pane")
            width = self.query_one("#top", RunHeader).content_size.width
            folded.truncate(width or 200, overflow="ellipsis")
            self.query_one("#summary", Static).update(folded)

        # Rich Text, so model output is never parsed as markup.
        self.query_one("#stream-body", Static).update(self._stream_story(s, active=active))

    def _stream_story(self, s: SessionState, *, active: bool) -> Text:
        """Return the stream pane's text: the end story, the live deltas, or the dead state.

        Args:
            s: The folded state.
            active: A model call is in flight.
        """
        tui = self._tui
        role = s.last_role
        st = Text()
        streaming = (
            active
            and role is not None
            and role.in_flight
            and (role.streamed_thinking or role.streamed_text)
        )
        if s.finished:
            # How it ended, the closing summary, and a plan's deliverable.
            word, reason = tui.dir_status
            st.append(status_label(word, reason) + "\n", style=f"bold {status_style(word)}")
            if s.finish_summary and s.end_reason in ("", "finish_session", "finish_planning"):
                st.append(s.finish_summary, style="dim")
            if plan := tui.plan_md():
                st.append("\n\n" + plan)
        elif streaming:
            assert role is not None
            if role.streamed_thinking:
                st.append("💭 ", style="bold")
                st.append(role.streamed_thinking[-1200:] + "\n", style="dim")
            if role.streamed_text:
                st.append(role.streamed_text[-1200:])
        elif tui.dir_status[0] == "waiting":
            st.append(status_label(*tui.dir_status), style="bold yellow")
        elif active and role is not None:
            spinner = spinner_frame(tui.spin)
            secs = tui.seconds_since_event()
            st.append(f"{spinner} {role.role} working… {secs}s", style="dim italic")
        elif tui.dir_status[0] == "starting":
            st.append("starting", style=f"bold {status_style('starting')}")
        elif (dead := dead_run_note(*tui.dir_status))[0]:
            st.append(dead[0] + "\n", style=f"bold {status_style(tui.dir_status[0])}")
            st.append(dead[1], style="dim")
        else:
            st.append("(no model call in flight)", style="dim")
        return st

    def render_state(self) -> None:  # noqa: PLR0912
        """Repaint every pane, rebuilding the tree and table only when their inputs changed."""
        self.render_heartbeat()
        tui = self._tui
        s = tui.state

        sel = self._selected_task_id
        sel_title = next((t.title for t in s.tasks if t.id == sel), "") if sel else ""
        # A border title is markup; the task title is the model's or the user's.
        filt = f" · task: {escape(sel_title[:28])}" if sel else ""

        ds, as_of = self._details_state(s)
        if self._rendered_tree is None or not (
            self._rendered_tree[0] is ds.tasks
            and self._rendered_tree[1] == sel
            and self._rendered_tree[2] == as_of
        ):
            self._rendered_tree = (ds.tasks, sel, as_of)
            tree = self.query_one("#plan", Tree)
            tree.clear()
            tree.border_title = f"tasks{as_of}" if as_of else ""
            for tv in ds.tasks:
                indent = "  " * tv.depth
                # The id leads the line: it is what `/retire` takes.
                label = Text(f"{tv.short_id:>3} {indent}{tv.glyph} {tv.title}")
                if tv.note:
                    label.append(f"  {tv.note}", style="dim italic")
                if tv.id == sel:
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
                    Text(tc.name),
                    Text(clip_cell(tc.args_preview, 90)),
                    ok,
                    Text(clip_cell(tc.result_summary, 40)),
                )
            table.border_title = f"tools{filt}" if sel else ""

        # The diff is on the monotonic log_count: log_tail is a sliding window, so a length diff
        # freezes once it saturates. The bottom is kept only when the operator was there.
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

        self.diff.render_state(s, sel=sel, filt=filt)
