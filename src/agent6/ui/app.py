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
    from textual.containers import Horizontal, Vertical
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

from agent6.ui.approval import (
    clear_tui_pid,
    write_answer,
    write_question_answer,
    write_steer_answer,
    write_tui_pid,
)
from agent6.ui.conversation import ConversationScreen
from agent6.ui.logview import LogScreen
from agent6.ui.menubar import HelpScreen, Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.modals import ApprovalModal, QuestionModal, SteerModal
from agent6.ui.state import (
    ApprovalPrompt,
    QuestionPrompt,
    RunState,
    apply_event,
    initial_state,
)
from agent6.ui.tail import tail_events
from agent6.ui.theme import PALETTE_CSS, open_theme_picker, setup_theme

_TASK_ICONS = {
    "passed": "✓",
    "failed": "✗",
    "in_progress": "▶",
    "skipped": "—",
    "obsolete": "⊘",
    "pending": "·",
}


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


class Agent6TUI(App[int]):
    TITLE = "agent6"
    CSS = (
        PALETTE_CSS
        + """
    Screen { layers: base dropdown; }
    #top { height: 4; padding: 0 1; }
    #mid { height: 1fr; }
    #left { width: 38%; }
    /* Uniform resting border (matches the home table + config card); the focused
       panel goes $accent. (Standardized -- was per-panel colour-coded.) */
    #plan { height: 1fr; border: round $primary; }
    #budget { height: 3; border: round $primary; padding: 0 1; }
    #right { width: 1fr; }
    #stream { height: 28%; border: round $primary; padding: 0 1; }
    #tools { height: 24%; border: round $primary; }
    #log { height: 1fr; border: round $primary; }
    #diff { height: 26%; border: round $primary; padding: 0 1; }
    /* Highlight whichever panel currently has keyboard focus. */
    #plan:focus, #tools:focus, #log:focus { border: round $accent; }
    """
    )

    COMMANDS: ClassVar = App.COMMANDS | {_Agent6Commands}

    MENUS: ClassVar = (
        Menu(
            "File",
            (MenuItem("Back", "to_hub", "Esc"), MenuItem("Quit", "quit_hub", "q")),
        ),
        Menu(
            "View",
            (
                MenuItem("Next pane", "focus_next", "Tab"),
                MenuItem("Prev pane", "focus_previous", "Shift+Tab"),
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
        # Footer order: page action, then meta (Help, Back, Quit, Menu) -- matching
        # the home + config footers. Esc backs out to the hub (consistent with the
        # config screen); q quits the whole hub; Ctrl+Q is the hard quit. (Esc on an
        # open modal cancels it first -- the modal consumes the key.)
        Binding("l", "view_logs", "Full log", show=True),
        Binding("t", "view_transcript", "Conversation", show=True),
        # g=top / G=end, matching vi and the LogScreen/ConversationScreen viewers
        # (g used to be "end" here, contradicting those screens reached via l/t).
        Binding("g", "scroll_log_home", "Log→top", show=False),
        Binding("G", "scroll_log_end", "Log→end", show=True),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "to_hub", "Back"),
        Binding("q", "quit_hub", "Quit"),
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
        with Horizontal(id="mid"):
            with Vertical(id="left"):
                yield Tree("tasks", id="plan")
                yield ProgressBar(id="budget", total=100, show_eta=False)
            with Vertical(id="right"):
                yield Static("", id="stream")
                yield DataTable(id="tools")
                # markup=False: log lines contain raw tool args like `args=[a,b]`
                # which Rich would otherwise try to parse as markup and crash.
                # auto_scroll off: _render does sticky-bottom itself (snap to the
                # newest line only when the operator is already at the bottom).
                yield RichLog(
                    id="log", highlight=False, markup=False, wrap=False, auto_scroll=False
                )
                yield Static("", id="diff")
        yield Footer()

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        write_tui_pid(self.run_dir, os.getpid())
        self.sub_title = f"run · {self.run_dir.name}"  # menu-bar title context
        table = self.query_one("#tools", DataTable)
        table.add_columns("tool", "args", "ok", "summary")
        self.query_one("#plan", Tree).root.expand()
        self.query_one("#stream", Static).update(Text("(waiting for the model…)", style="dim"))
        # Auto-spawn close: the reader thread sets `_run_ended` on `run.end`; we
        # poll it from a timer in the app's OWN loop and exit there. Exit()
        # scheduled from inside a call_from_thread callback does not take effect,
        # but exiting from a timer callback does. The same timer also drives the
        # approval / question / steer modals.
        self.set_interval(0.2, self._tick)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def on_unmount(self) -> None:
        self._stop.set()
        clear_tui_pid(self.run_dir)

    # --- reader thread -----------------------------------------------

    def _reader_loop(self) -> None:
        for event in tail_events(self.logs_path, follow=True, stop_when_finished=self.exit_on_end):
            if self._stop.is_set():
                return
            self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: dict[str, object]) -> None:
        self.state = apply_event(self.state, event)
        self._render()
        if self.exit_on_end and event.get("type") == "run.end":
            self._run_ended = True

    def _tick(self) -> None:
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
        call with full I/O), folded from the lossless per-call transcripts."""
        self.push_screen(
            ConversationScreen(
                self.run_dir / "transcripts", title=f"conversation · {self.run_dir.name}"
            )
        )

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
        stream = self.query_one("#stream", Static)
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
        for tc in s.tool_calls[-20:]:
            ok = "…" if tc.ok is None else ("✓" if tc.ok else "✗")
            table.add_row(
                Text(tc.name), Text(tc.args_preview[:60]), ok, Text(tc.result_summary[:60])
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
        diff_widget = self.query_one("#diff", Static)
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
