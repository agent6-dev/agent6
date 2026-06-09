# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Textual TUI app for `agent6 run` / `agent6 watch`.

`textual` is an optional dependency — importing this module fails clearly
if textual is not installed. Nothing else in `agent6/ui/` imports this
module at import-time; the CLI does a lazy `importlib.import_module`.

Architecture:
- main thread: textual event loop.
- background thread: tail_events(logs.jsonl) -> apply_event -> call_from_thread.

The TUI is read-only on the log stream and only writes to
`<run_dir>/approvals/<id>.answer` when the user clicks Yes/No in the
approval modal.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import ClassVar

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        ProgressBar,
        RichLog,
        Static,
        Tree,
    )
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package. Install with: pip install 'agent6[tui]'"
    ) from e

from agent6.ui.approval import (
    clear_tui_pid,
    write_answer,
    write_tui_pid,
)
from agent6.ui.state import (
    ApprovalPrompt,
    RunState,
    apply_event,
    initial_state,
)
from agent6.ui.tail import tail_events


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
        height: 3;
        align: center middle;
    }
    """

    def __init__(self, prompt_id: str, prompt: str) -> None:
        super().__init__()
        self.prompt_id = prompt_id
        self.prompt_text = prompt

    def compose(self) -> ComposeResult:
        with Container(id="approval-box"):
            yield Static(f"[b]Approval requested[/b]\n\n{self.prompt_text}")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow (y)", id="yes", variant="success")
                yield Button("Deny (n)", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key in ("y", "Y"):
            self.dismiss(True)
        elif event.key in ("n", "N", "escape"):
            self.dismiss(False)


class Agent6TUI(App[int]):
    CSS = """
    #top { height: 5; }
    #mid { height: 1fr; }
    #plan { width: 40%; border: round $primary; }
    #right { width: 1fr; }
    #tools { height: 40%; border: round $primary; }
    #log { height: 1fr; border: round $primary; }
    #diff { height: 30%; border: round $accent; }
    #budget { height: 5; border: round $secondary; padding: 0 1; }
    """

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("y", "answer_yes", "Approve", show=False),
        Binding("n", "answer_no", "Deny", show=False),
    ]

    def __init__(self, run_dir: Path, *, exit_on_end: bool = False) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.logs_path = run_dir / "logs.jsonl"
        self.state: RunState = initial_state()
        self._seen_approval_ids: set[str] = set()
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
            yield Tree("plan", id="plan")
            with Vertical(id="right"):
                yield ProgressBar(id="budget", total=100, show_eta=False)
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
        # Auto-spawn close: the reader thread sets `_run_ended` on `run.end`; we
        # poll it from a timer in the app's OWN loop and exit there. Exit()
        # scheduled from inside a call_from_thread callback does not take effect,
        # but exiting from a timer callback does.
        if self.exit_on_end:
            self.set_interval(0.2, self._check_run_ended)
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

    def _check_run_ended(self) -> None:
        if self._run_ended:
            self.exit()
        # Pop approval modal if new pending approval appeared.
        for ap in self.state.pending_approvals:
            if not ap.answered and ap.id not in self._seen_approval_ids:
                self._seen_approval_ids.add(ap.id)
                self.push_screen(_ApprovalModal(ap.id, ap.prompt), self._on_approval(ap))

    def _on_approval(self, ap: ApprovalPrompt):  # type: ignore[no-untyped-def]
        def cb(approved: bool | None) -> None:
            if approved is None:
                approved = False
            write_answer(self.run_dir, ap.id, approved=approved)

        return cb

    # --- rendering ---------------------------------------------------

    def _render(self) -> None:  # noqa: PLR0915
        s = self.state
        # Top header
        role = s.last_role
        role_line = (
            f"{role.role} / {role.model} {'…' if role.in_flight else ''}" if role else "(idle)"
        )
        # Live SSE text deltas. Show the tail of the
        # streaming assistant message in the header while the role
        # call is in-flight; once the call resolves the text is
        # frozen with the role and the in-flight ellipsis drops.
        if role and role.in_flight and role.streamed_text:
            tail = role.streamed_text.replace("\n", " ⏎ ")
            if len(tail) > 80:
                tail = "…" + tail[-79:]
            role_line = f"{role_line}  ▸ {tail}"
        step = (
            f"step {s.current_step_index}/{len(s.steps)}"
            if s.current_step_index
            else f"steps: {len(s.steps)}"
        )
        finished = (
            "[b green]done[/]"
            if s.finished and s.all_passed
            else ("[b red]done (failed)[/]" if s.finished else "")
        )
        # Live cost meter in the top header. "$0.0123" updates
        # after every role.result; "~" prefix flags that some models
        # weren't in the pricing table so the figure is a lower bound.
        cost_prefix = "~" if s.budget.usd_partial else ""
        cost = f"[b]{cost_prefix}${s.budget.usd_total:.4f}[/]"
        self.query_one("#top", Static).update(
            f"[b]agent6[/]  {step}   role: {role_line}   cost: {cost}   {finished}\n"
            f"task: {s.user_task[:120]}"
        )

        # Plan tree
        tree = self.query_one("#plan", Tree)
        tree.clear()
        for sv in s.steps:
            icon = {
                "passed": "✓",
                "failed": "✗",
                "running": "▶",
                "skipped": "—",
                "pending": "·",
            }.get(sv.status, "·")
            label = f"{icon} {sv.index}. {sv.title}"
            tree.root.add_leaf(label)
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
            table.add_row(tc.name, tc.args_preview[:60], ok, tc.result_summary[:60])

        # Log
        log = self.query_one("#log", RichLog)
        if not hasattr(self, "_last_log_len"):
            self._last_log_len = 0
        new = s.log_tail[self._last_log_len :]
        for line in new:
            log.write(line)
        self._last_log_len = len(s.log_tail)

        # Diff (latest) or verify output. When a verify is
        # running or just finished, surface its exit code and tail in
        # the panel - otherwise it's invisible until the next log scroll.
        diff_widget = self.query_one("#diff", Static)
        verify = s.last_verify
        if verify is not None and (verify.exit_code is None or not s.diffs):
            if verify.exit_code is None:
                header = f"[b]verify running:[/] {' '.join(verify.cmd)[:200]}"
                body = "…"
            else:
                colour = "green" if verify.exit_code == 0 else "red"
                header = (
                    f"[b {colour}]verify exit={verify.exit_code}[/] "
                    f"({verify.duration_s:.1f}s)  {' '.join(verify.cmd)[:160]}"
                )
                body = (verify.stderr_tail or verify.stdout_tail)[:2000] or "(no output)"
            diff_widget.update(f"{header}\n{body}")
        elif s.diffs:
            latest_idx = max(s.diffs.keys())
            text = s.diffs[latest_idx]
            diff_widget.update(f"[b]diff for step #{latest_idx}[/]\n" + text[:2000])
        else:
            diff_widget.update("(no diffs yet)")

    # --- bindings ----------------------------------------------------

    def action_answer_yes(self) -> None:
        pass  # handled in modal

    def action_answer_no(self) -> None:
        pass  # handled in modal


def run_tui(run_dir: Path, *, exit_on_end: bool = False) -> int:
    return Agent6TUI(run_dir, exit_on_end=exit_on_end).run() or 0
