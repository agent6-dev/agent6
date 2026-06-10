# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Textual TUI app for `agent6 run` / `agent6 watch`.

`textual` ships in the base install; importing this module fails clearly
if it has been stripped out. Nothing else in `agent6/ui/` imports this
module at import-time; the CLI does a lazy `importlib.import_module`.

Architecture:
- main thread: textual event loop.
- background thread: tail_events(logs.jsonl) -> apply_event -> call_from_thread.

The TUI is read-only on the log stream and only writes to
`<run_dir>/approvals/<id>.answer` (approval modal) and
`<run_dir>/steer.answer` (steer modal) when the user answers.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

try:
    from rich.markup import escape
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.command import DiscoveryHit, Hit, Hits, Provider
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
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
    write_steer_answer,
    write_tui_pid,
)
from agent6.ui.state import (
    ApprovalPrompt,
    RunState,
    apply_event,
    initial_state,
)
from agent6.ui.tail import tail_events

_TASK_ICONS = {
    "passed": "✓",
    "failed": "✗",
    "in_progress": "▶",
    "skipped": "—",
    "obsolete": "⊘",
    "pending": "·",
}


class _ApprovalModal(ModalScreen[bool]):
    DEFAULT_CSS = """
    _ApprovalModal {
        align: center middle;
    }
    #approval-box {
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick $warning;
        padding: 1 2;
        background: $surface;
    }
    #approval-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    #approval-buttons Button {
        margin: 0 2;
        min-width: 16;
    }
    #approval-buttons Button:focus {
        text-style: bold reverse;
    }
    """

    # Keys handled on the MODAL (not the app) so they reach the focused button.
    # Left/right move the highlight; Enter/Space activate the focused button
    # (textual Button default); y/n are shortcuts; Esc denies.
    BINDINGS: ClassVar = [
        Binding("left", "focus_previous", "◀", show=False),
        Binding("right", "focus_next", "▶", show=False),
        Binding("y", "approve", "Allow", show=True),
        Binding("Y", "approve", "Allow", show=False),
        Binding("n", "deny", "Deny", show=True),
        Binding("N", "deny", "Deny", show=False),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, prompt_id: str, prompt: str) -> None:
        super().__init__()
        self.prompt_id = prompt_id
        self.prompt_text = prompt

    def compose(self) -> ComposeResult:
        with Container(id="approval-box"):
            body = Text()
            body.append("Approval requested\n\n", style="bold")
            body.append(self.prompt_text)  # plain append: never parsed as markup
            yield Static(body)
            with Horizontal(id="approval-buttons"):
                yield Button("Allow (y)", id="yes", variant="success")
                yield Button("Deny (n)", id="no", variant="error")

    def on_mount(self) -> None:
        # Focus Allow so arrow keys + Enter work immediately and the user sees
        # which choice is active.
        self.query_one("#yes", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class _SteerModal(ModalScreen[str]):
    """Mid-run Ctrl-C prompt: continue, abort, or inject a steering instruction.

    Result string: "" = continue, "abort" = stop, anything else = instruction.
    """

    DEFAULT_CSS = """
    _SteerModal {
        align: center middle;
    }
    #steer-box {
        width: 80%;
        max-width: 100;
        height: auto;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #steer-input {
        margin-top: 1;
    }
    #steer-buttons {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    #steer-buttons Button {
        margin: 0 2;
        min-width: 14;
    }
    #steer-buttons Button:focus {
        text-style: bold reverse;
    }
    """

    BINDINGS: ClassVar = [
        Binding("left", "focus_previous", "◀", show=False),
        Binding("right", "focus_next", "▶", show=False),
        Binding("escape", "cont", "Continue", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="steer-box"):
            body = Text()
            body.append("Run interrupted\n\n", style="bold")
            body.append("Continue, abort, or type a steering instruction below.")
            yield Static(body)
            yield Input(placeholder="instruction (blank = continue)", id="steer-input")
            with Horizontal(id="steer-buttons"):
                yield Button("Continue", id="continue", variant="success")
                yield Button("Send", id="send", variant="primary")
                yield Button("Abort", id="abort", variant="error")

    def on_mount(self) -> None:
        self.query_one("#steer-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the field: blank continues, text steers.
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "abort":
            self.dismiss("abort")
        elif event.button.id == "send":
            self.dismiss(self.query_one("#steer-input", Input).value)
        else:
            self.dismiss("")

    def action_cont(self) -> None:
        self.dismiss("")


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


class Agent6TUI(App[int]):
    CSS = """
    #top { height: 4; padding: 0 1; }
    #mid { height: 1fr; }
    #left { width: 38%; }
    #plan { height: 1fr; border: round $primary; }
    #budget { height: 3; border: round $secondary; padding: 0 1; }
    #right { width: 1fr; }
    #stream { height: 28%; border: round $success; padding: 0 1; }
    #tools { height: 24%; border: round $primary; }
    #log { height: 1fr; border: round $primary; }
    #diff { height: 26%; border: round $accent; padding: 0 1; }
    /* Highlight whichever panel currently has keyboard focus. */
    #plan:focus, #tools:focus, #log:focus { border: round $accent; }
    """

    COMMANDS: ClassVar = App.COMMANDS | {_Agent6Commands}

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Next pane", show=False),
        Binding("shift+tab", "focus_previous", "Prev pane", show=False),
        Binding("g", "scroll_log_end", "Log→end", show=True),
    ]

    def __init__(self, run_dir: Path, *, exit_on_end: bool = False) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.logs_path = run_dir / "logs.jsonl"
        self.state: RunState = initial_state()
        self._seen_approval_ids: set[str] = set()
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
        yield Header(show_clock=True)
        yield Static("", id="top")
        with Horizontal(id="mid"):
            with Vertical(id="left"):
                yield Tree("plan", id="plan")
                yield ProgressBar(id="budget", total=100, show_eta=False)
            with Vertical(id="right"):
                yield Static("", id="stream")
                yield DataTable(id="tools")
                # markup=False: log lines contain raw tool args like `args=[a,b]`
                # which Rich would otherwise try to parse as markup and crash.
                yield RichLog(id="log", highlight=False, markup=False, wrap=False)
                yield Static("", id="diff")
        yield Footer()

    def on_mount(self) -> None:
        write_tui_pid(self.run_dir, os.getpid())
        table = self.query_one("#tools", DataTable)
        table.add_columns("tool", "args", "ok", "summary")
        self.query_one("#plan", Tree).root.expand()
        self.query_one("#stream", Static).update(Text("(waiting for the model…)", style="dim"))
        # Auto-spawn close: the reader thread sets `_run_ended` on `run.end`; we
        # poll it from a timer in the app's OWN loop and exit there. Exit()
        # scheduled from inside a call_from_thread callback does not take effect,
        # but exiting from a timer callback does. The same timer also drives the
        # approval + steer modals.
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
        # Auto-spawned dashboard (exit_on_end): flag the run as ended; the
        # on_mount timer (running in the app's own loop) sees it and exits.
        if self.exit_on_end and event.get("type") == "run.end":
            self._run_ended = True

    def _tick(self) -> None:
        # Pop an approval modal for any new pending approval.
        for ap in self.state.pending_approvals:
            if not ap.answered and ap.id not in self._seen_approval_ids:
                self._seen_approval_ids.add(ap.id)
                self.push_screen(_ApprovalModal(ap.id, ap.prompt), self._on_approval(ap))
        # Pop a steer modal once per Ctrl-C (steer_requests is monotonic).
        if self.state.steer_requests > self._seen_steer and not self._steer_open:
            self._seen_steer = self.state.steer_requests
            self._steer_open = True
            self.push_screen(_SteerModal(), self._on_steer)
        # Exit only once the run ended AND nothing is still awaiting an answer,
        # so a final approval/steer isn't dropped on the way out.
        if (
            self.exit_on_end
            and self._run_ended
            and not self._steer_open
            and all(ap.answered for ap in self.state.pending_approvals)
        ):
            self.exit()

    def _on_approval(self, ap: ApprovalPrompt):  # type: ignore[no-untyped-def]
        def cb(approved: bool | None) -> None:
            if approved is None:
                approved = False
            write_answer(self.run_dir, ap.id, approved=approved)

        return cb

    def _on_steer(self, answer: str | None) -> None:
        self._steer_open = False
        write_steer_answer(self.run_dir, answer or "")

    # --- command palette ---------------------------------------------

    def palette_commands(self) -> list[tuple[str, Callable[[], Any], str]]:
        return [
            ("Scroll log to end", self.action_scroll_log_end, "Jump the log to the newest line"),
            ("Quit dashboard", self.action_quit, "Close the agent6 dashboard"),
        ]

    def action_scroll_log_end(self) -> None:
        self.query_one("#log", RichLog).scroll_end(animate=False)

    # --- rendering ---------------------------------------------------

    def _render(self) -> None:  # noqa: PLR0912, PLR0915
        s = self.state
        # Top header
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
        # Live cost meter. "~" prefix flags some models weren't in the pricing
        # table so the figure is a lower bound.
        cost_prefix = "~" if s.budget.usd_partial else ""
        cost = f"[b]{cost_prefix}${s.budget.usd_total:.4f}[/]"
        self.query_one("#top", Static).update(
            f"[b]agent6[/]  {step}   role: {escape(role_line)}   cost: {cost}   {finished}\n"
            f"task: {escape(s.user_task[:120])}"
        )

        # Live reasoning / response pane. The "watch it think" window: show the
        # tail of the in-flight reasoning (dim) and visible answer so a long
        # call reads as progress. Built as rich Text so model output never gets
        # parsed as markup.
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

        # Task DAG. The worker's live `add_task`/`update_task` breakdown, emitted
        # as `graph.update` snapshots; indented by depth, cursor marked.
        tree = self.query_one("#plan", Tree)
        tree.clear()
        for tv in s.tasks:
            icon = _TASK_ICONS.get(tv.status, "·")
            indent = "  " * tv.depth
            marker = "▸ " if tv.is_cursor else ""
            # Text (not str): titles are model output and may contain Rich markup
            # metacharacters ('[...]') that would crash add_leaf.
            tree.root.add_leaf(Text(f"{indent}{marker}{icon} {tv.title}"))
        tree.root.expand()

        # Budget
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

        # Tool table
        table = self.query_one("#tools", DataTable)
        table.clear()
        for tc in s.tool_calls[-20:]:
            ok = "…" if tc.ok is None else ("✓" if tc.ok else "✗")
            # Text cells: tool names/args/summaries are model output and would
            # otherwise be parsed as Rich markup (crash on stray brackets).
            table.add_row(
                Text(tc.name), Text(tc.args_preview[:60]), ok, Text(tc.result_summary[:60])
            )

        # Log. Diff on the monotonic log_count, not len(log_tail): log_tail is a
        # sliding window, so once it saturates its length stops growing and a
        # length-based diff would silently freeze the panel.
        log = self.query_one("#log", RichLog)
        n_new = min(s.log_count - self._last_log_count, len(s.log_tail))
        if n_new > 0:
            for line in s.log_tail[-n_new:]:
                log.write(line)
        self._last_log_count = s.log_count

        # Diff (latest) or verify output. Built as rich Text to avoid markup
        # parsing of diff/verify bodies (which contain brackets).
        diff_widget = self.query_one("#diff", Static)
        verify = s.last_verify
        dt = Text()
        if verify is not None and (verify.exit_code is None or not s.diffs):
            if verify.exit_code is None:
                dt.append("verify running: ", style="bold")
                dt.append(" ".join(verify.cmd)[:200] + "\n")
                dt.append("…", style="dim")
            else:
                colour = "green" if verify.exit_code == 0 else "red"
                dt.append(f"verify exit={verify.exit_code} ", style=f"bold {colour}")
                dt.append(f"({verify.duration_s:.1f}s)  {' '.join(verify.cmd)[:160]}\n")
                dt.append((verify.stderr_tail or verify.stdout_tail)[:2000] or "(no output)")
            diff_widget.update(dt)
        elif s.diffs:
            latest_idx = max(s.diffs.keys())
            dt.append(f"diff for step #{latest_idx}\n", style="bold")
            dt.append(s.diffs[latest_idx][:2000])
            diff_widget.update(dt)
        else:
            diff_widget.update(Text("(no diffs yet)", style="dim"))


def run_tui(run_dir: Path, *, exit_on_end: bool = False) -> int:
    return Agent6TUI(run_dir, exit_on_end=exit_on_end).run() or 0
