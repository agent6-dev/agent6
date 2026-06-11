# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 tui` hub: a home screen to browse recent runs and start new work.

CLI-first by design, the hub never reimplements the workflow. "Start a run /
plan / ask" simply spawns the normal `agent6` CLI as a detached subprocess
(whose non-TTY stdout means it won't try to open its own TUI) and then opens the
read-only dashboard on the run directory it creates. So everything here is a
thin driver over the CLI + the same file/event contract the dashboard reads.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import ClassVar

try:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Footer, Header, Input, Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

# Subdirs (relative to the agent6 dir) that hold watchable run directories.
_RUN_SUBDIRS = ("runs", "asks")
# A "running" run whose logs.jsonl hasn't been touched in this long reads as
# crashed/killed (a long reasoning burst still appends within minutes).
_STALE_AFTER_S = 600.0


def _run_summary(run_dir: Path) -> dict[str, str]:
    """A cheap one-line summary of a run for the listing: mode, task, status.

    Reads only the first event (run.start) + scans for run.end, so it stays fast
    on a directory of many runs."""
    logs = run_dir / "logs.jsonl"
    mode, task, status = "?", "", "running"
    if not logs.is_file():
        return {"id": run_dir.name, "mode": mode, "task": "(no logs)", "status": "—"}
    try:
        with logs.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                etype = ev.get("type")
                if etype == "run.start":
                    mode = str(ev.get("mode", mode))
                    task = str(ev.get("user_task", ""))
                elif etype == "run.end":
                    status = "ok" if ev.get("all_passed") else "done"
        # A "running" run whose log has been quiet for a while is almost
        # certainly a crashed/killed process, not active work, say so rather
        # than showing "running" forever.
        if status == "running" and (time.time() - logs.stat().st_mtime) > _STALE_AFTER_S:
            status = "stale"
    except OSError:
        pass
    return {"id": run_dir.name, "mode": mode, "task": task[:60], "status": status}


def _list_runs(agent6_dir: Path) -> list[Path]:
    """All run directories (runs/ + asks/), newest first by mtime."""
    out: list[Path] = []
    for sub in _RUN_SUBDIRS:
        d = agent6_dir / sub
        if d.is_dir():
            out.extend(p for p in d.iterdir() if p.is_dir())
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out


class _NewWorkModal(ModalScreen[tuple[str, str] | None]):
    """Pick a mode (run/plan/ask) and type a task. Result: (mode, task) or None."""

    DEFAULT_CSS = """
    _NewWorkModal { align: center middle; }
    #new-box {
        width: 80%; max-width: 100; height: auto;
        border: thick $accent; padding: 1 2; background: $surface;
    }
    #new-modes { height: auto; align: center middle; margin-top: 1; }
    #new-modes Button { margin: 0 1; min-width: 12; }
    #new-modes Button.selected { text-style: bold reverse; }
    #new-task { margin-top: 1; }
    """

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self) -> None:
        super().__init__()
        self._mode = "run"

    def compose(self) -> ComposeResult:
        with Vertical(id="new-box"):
            yield Static(Text("Start new work", style="bold"))
            with Horizontal(id="new-modes"):
                yield Button("run", id="mode-run", classes="selected")
                yield Button("plan", id="mode-plan")
                yield Button("ask", id="mode-ask")
            yield Input(placeholder="task / question, then Enter", id="new-task")

    def on_mount(self) -> None:
        self.query_one("#new-task", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("mode-"):
            self._mode = bid.removeprefix("mode-")
            for b in self.query("#new-modes Button").results(Button):
                b.set_class(b.id == bid, "selected")
            self.query_one("#new-task", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        self.dismiss((self._mode, task) if task else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class Agent6HomeApp(App[Path | None]):
    """Home hub. `run()` returns the run directory the user chose to open (to be
    watched by the dashboard), or None to quit."""

    CSS = """
    #runs { height: 1fr; border: round $primary; }
    #hint { height: auto; padding: 0 1; color: $text-muted; }
    """

    BINDINGS: ClassVar = [
        Binding("n", "new_work", "New run/plan/ask"),
        Binding("enter", "open_selected", "Open"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, agent6_dir: Path) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir
        self._runs: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="runs")
        yield Static("enter: open · n: new run/plan/ask · r: refresh · q: quit", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.cursor_type = "row"
        table.add_columns("when", "mode", "status", "id", "task")
        self.action_refresh()
        table.focus()

    def action_refresh(self) -> None:
        self._runs = _list_runs(self.agent6_dir)
        table = self.query_one("#runs", DataTable)
        table.clear()
        for rd in self._runs:
            try:
                mtime = rd.stat().st_mtime
            except OSError:
                continue  # vanished since the listing snapshot — skip it
            s = _run_summary(rd)
            when = time.strftime("%m-%d %H:%M", time.localtime(mtime))
            # Text cells: task is model/user input and may carry markup brackets.
            table.add_row(when, s["mode"], s["status"], Text(s["id"]), Text(s["task"]))

    def action_open_selected(self) -> None:
        table = self.query_one("#runs", DataTable)
        if self._runs and 0 <= table.cursor_row < len(self._runs):
            self.exit(self._runs[table.cursor_row])

    def action_new_work(self) -> None:
        self.push_screen(_NewWorkModal(), self._on_new_work)

    def _on_new_work(self, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        run_dir, error = _spawn_and_locate(self.agent6_dir, *result)
        if run_dir is not None:
            self.exit(run_dir)
        else:
            self.notify(error or "Could not start the run.", severity="error", timeout=8.0)


def _spawn_and_locate(agent6_dir: Path, mode: str, task: str) -> tuple[Path | None, str]:
    """Spawn `agent6 <mode> <task>` detached (non-TTY stdout → no nested TUI) and
    return (run_dir, ""). On failure returns (None, diagnostic). The dir is found
    by snapshotting existing runs and polling for a NEW one; if the child exits
    before producing a run dir (no git repo, bad config, …) its stderr tail is
    surfaced instead of silently waiting out the timeout."""
    cwd = agent6_dir.parent
    argv = [_agent6_exe(), mode, task]
    before = set(_list_runs(agent6_dir))
    err = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in finally
        mode="w+", suffix=".agent6-launch.err", delete=False
    )
    try:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
                start_new_session=True,
            )
        except OSError as exc:
            return None, f"failed to start agent6: {exc}"
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            found = _located_run(agent6_dir, before)
            if found is not None:
                return found, ""
            if proc.poll() is not None:
                # Child exited without a run dir, surface why (recheck once in
                # case the dir landed in the same instant the process exited).
                found = _located_run(agent6_dir, before)
                if found is not None:
                    return found, ""
                err.flush()
                tail = Path(err.name).read_text(encoding="utf-8", errors="replace")[-600:]
                return None, f"agent6 {mode} exited ({proc.returncode}) before starting:\n{tail}"
            time.sleep(0.2)
        return None, f"timed out waiting for `agent6 {mode}` to start"
    finally:
        err.close()
        Path(err.name).unlink(missing_ok=True)


def _located_run(agent6_dir: Path, before: set[Path]) -> Path | None:
    """The newest run dir not present in *before*, once its logs.jsonl exists."""
    for rd in _list_runs(agent6_dir):
        if rd not in before and (rd / "logs.jsonl").exists():
            return rd
    return None


def _agent6_exe() -> str:
    """The agent6 executable that launched this hub (so a spawned child uses the
    same install). Falls back to the entry on PATH."""
    argv0 = Path(sys.argv[0])
    if argv0.name.startswith("agent6") and argv0.exists():
        return str(argv0.resolve())
    import shutil  # noqa: PLC0415

    return shutil.which("agent6") or "agent6"


def run_home(agent6_dir: Path) -> Path | None:
    return Agent6HomeApp(agent6_dir).run()
