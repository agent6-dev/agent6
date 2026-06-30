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

import contextlib
import inspect
import json
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import ClassVar

try:
    from rich.text import Text
    from textual import events, on
    from textual.app import App, ComposeResult, SystemCommand
    from textual.binding import Binding
    from textual.command import DiscoveryHit, Hit, Hits, Provider
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen, Screen
    from textual.widgets import DataTable, Footer, Select, Static, TextArea
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

# Safe at module top: the textual guard above runs first, so this (which also
# needs textual) is only reached when textual is present.
from agent6.ui._spawn import agent6_exe
from agent6.ui.config_page import ConfigScreen
from agent6.ui.conversation import ConversationScreen
from agent6.ui.logview import LogScreen
from agent6.ui.machines import MachinesScreen
from agent6.ui.menubar import HelpScreen, Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.modals import ConfirmModal
from agent6.ui.theme import PALETTE_CSS, open_theme_picker, setup_theme
from agent6.ui.widgets import FORM_CSS, ActionItem

# Subdirs (relative to the agent6 dir) that hold watchable run directories.
_RUN_SUBDIRS = ("runs", "asks")
# A "running" run whose logs.jsonl hasn't been touched in this long reads as
# crashed/killed (a long reasoning burst still appends within minutes).
_STALE_AFTER_S = 600.0
# The new-work profile dropdown's first entry: "" => no --profile, so the run
# uses [workflow].profile from config (or the plain defaults).
_DEFAULT_PROFILE_LABEL = "(config default)"


def _available_profiles(repo_cwd: Path) -> list[str]:
    """Profile names the new-work chooser offers (the built-ins plus the user's
    custom ``[profiles.<name>]`` tables). Delegates to ``config_layer`` -- the
    TUI's config entry point (see config_page.py) -- so the dropdown and the
    ``--profile`` CLI flag resolve against the same source."""
    from agent6.config_layer import available_profile_names  # noqa: PLC0415

    return available_profile_names(repo_cwd, None)


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
    if mode == "ask":
        transcript = run_dir / "transcript.md"
        with contextlib.suppress(OSError):
            task = transcript.read_text(encoding="utf-8")
    return {"id": run_dir.name, "mode": mode, "task": _task_snippet(task)[:60], "status": status}


def _task_snippet(task: str) -> str:
    """First user-authored line after any seeded ask context blocks."""
    skip_until: str | None = None
    for line in task.splitlines():
        s = line.strip()
        if skip_until is not None:
            if s == skip_until:
                skip_until = None
            continue
        if s in {"# agent6 ask", "## Question"}:
            continue
        if s == "## Answer":
            break
        if s.startswith("<file "):
            if "</file>" not in s:
                skip_until = "</file>"
            continue
        if s.startswith("<prior-run "):
            if "</prior-run>" not in s:
                skip_until = "</prior-run>"
            continue
        if s and not s.startswith("<"):
            return s
    return task.strip()


def _list_runs(agent6_dir: Path) -> list[Path]:
    """All run directories (runs/ + asks/), newest first by mtime."""
    out: list[Path] = []
    for sub in _RUN_SUBDIRS:
        d = agent6_dir / sub
        if d.is_dir():
            out.extend(p for p in d.iterdir() if p.is_dir())
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return out


class _NewWorkModal(ModalScreen[tuple[str, str, str] | None]):
    """Type a task, pick an optional config profile, then start it as a run /
    plan / ask. The mode IS the button you pick (flat actions, like the config
    dialogs); Enter in the box runs. The profile dropdown maps to the
    ``--profile`` CLI flag; "(config default)" => no flag (so [workflow].profile
    applies). Result: (mode, task, profile) or None, where profile="" means the
    config default (no --profile)."""

    CSS = (
        FORM_CSS
        + """
    _NewWorkModal { align: center middle; }
    #new-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #new-title { text-style: bold; }
    #new-task {
        margin-top: 1; height: 6; padding: 0 1;
        border: round $primary; background: $surface;
    }
    #new-task:focus { border: round $accent; }
    #new-profile-row { height: auto; padding-top: 1; }
    #new-profile-label { width: auto; padding: 1 1 0 0; color: $text-muted; }
    #new-profile { width: 1fr; }
    #new-actions { padding-top: 1; height: auto; }
    """
    )

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, profiles: list[str] | None = None) -> None:
        # Profile names for the dropdown (built-ins + user [profiles.*]); the
        # caller passes the repo-resolved list (built-ins + user [profiles.*]).
        # None (e.g. a bare test) => only the "(config default)" entry, so the
        # modal stands alone as a pure widget without loading config.
        super().__init__()
        self._profiles = profiles if profiles is not None else []

    def compose(self) -> ComposeResult:
        with Vertical(id="new-box"):
            yield Static("Start new work", id="new-title")
            # A multiline TextArea (not an Input): Enter is a newline, so a task
            # can span lines; it brings undo/redo/select-all for free. Tab (and ↓
            # past the last line) move to the run/plan/ask buttons.
            yield TextArea(id="new-task", placeholder="task / question…")
            with Horizontal(id="new-profile-row"):
                yield Static("profile:", id="new-profile-label")
                # value="" is the "(config default)" sentinel: NO --profile, so
                # the config's [workflow].profile (or plain defaults) applies.
                yield Select(
                    [(_DEFAULT_PROFILE_LABEL, ""), *((p, p) for p in self._profiles)],
                    value="",
                    allow_blank=False,
                    id="new-profile",
                )
            with Horizontal(id="new-actions"):
                yield ActionItem("run", "run")
                yield ActionItem("plan", "plan")
                yield ActionItem("ask", "ask")
            yield Static(
                Text("Tab to run / plan / ask · Enter = newline · Esc cancel", style="dim"),
                classes="edit-label",
            )

    def on_mount(self) -> None:
        self.query_one("#new-task", TextArea).focus()

    def on_key(self, event: events.Key) -> None:
        # Buttons: ←/→ move between run/plan/ask, ↑ goes back to the task. The task
        # is a TextArea (Enter=newline), so Tab — or ↓ once you're on the last line
        # — moves down to the buttons.
        focused = self.focused
        if event.key in ("left", "right") and isinstance(focused, ActionItem):
            items = list(self.query(ActionItem))
            step = 1 if event.key == "right" else -1
            items[(items.index(focused) + step) % len(items)].focus()
            event.stop()
        elif event.key == "up" and isinstance(focused, ActionItem):
            self.query_one("#new-task", TextArea).focus()
            event.stop()
        elif event.key == "down" and isinstance(focused, TextArea):
            row, _ = focused.cursor_location
            if row >= focused.document.line_count - 1:
                next(iter(self.query(ActionItem))).focus()
                event.stop()

    @on(ActionItem.Activated)
    def _start(self, event: ActionItem.Activated) -> None:
        self._submit(event.action)

    def _submit(self, mode: str) -> None:
        task = self.query_one("#new-task", TextArea).text.strip()
        if task:
            # Select.value is the option's value: "" for "(config default)"
            # (no --profile), else the chosen profile name. allow_blank=False
            # plus the leading "" option means it's never Select.BLANK.
            profile = str(self.query_one("#new-profile", Select).value)
            self.dismiss((mode, task, profile))
        else:
            self.notify("Enter a task first.", severity="warning")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:  # click on the backdrop (outside the dialog) = cancel
            self.action_cancel()


class _HomeCommands(Provider):
    """The home hub's menu actions in the Ctrl+P palette, from the same MENUS
    registry as the menu bar and key bindings -- so every action is searchable by
    name (parity with the config screen and the run dashboard)."""

    @property
    def _home(self) -> HomeScreen:
        screen = self.screen
        assert isinstance(screen, HomeScreen)
        return screen

    async def discover(self) -> Hits:
        for name, runnable, help_text in self._home.palette_commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, runnable, help_text in self._home.palette_commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), runnable, help=help_text)


class HomeScreen(Screen[None]):
    """The hub view: browse recent runs, start new work, open the config editor.
    Its bindings live here (not on the App) so the footer of a pushed screen --
    e.g. the config editor -- shows only that screen's keys, not the hub's."""

    MENUS: ClassVar = (
        Menu(
            "File",
            (
                MenuItem("New run/plan/ask", "new_work", "n"),
                MenuItem("Open selected", "open_selected", "enter"),
                MenuItem("Merge selected run", "merge_selected", "m"),
                MenuItem("Refresh", "refresh", "r"),
                MenuItem("Quit", "quit", "q"),
            ),
        ),
        Menu("Config", (MenuItem("Open config", "open_config", "c"),)),
        Menu("Machines", (MenuItem("Open machines", "open_machines", "M"),)),
        Menu(
            "View",
            (
                # Viewing a selected run's logs/transcript is filed under View to
                # match the run dashboard's View menu (the two surfaces stay aligned).
                MenuItem("View logs", "view_logs", "l"),
                MenuItem("View transcript", "view_transcript", "t"),
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
        # Footer order: run-list actions, then Config, then meta (Help, Quit, Menu).
        Binding("n", "new_work", "New run/plan/ask"),
        Binding("enter", "open_selected", "Open"),
        Binding("l", "view_logs", "View logs"),
        Binding("t", "view_transcript", "View transcript"),
        Binding("m", "merge_selected", "Merge run"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "open_config", "Config"),
        Binding("M", "open_machines", "Machines"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
        *menu_bindings(MENUS),
    ]
    COMMANDS: ClassVar = Screen.COMMANDS | {_HomeCommands}

    def palette_commands(self) -> Iterator[tuple[str, Callable[[], None], str]]:
        """(label, runnable, help) per menu action -- the Ctrl+P palette source,
        the same MENUS registry as the menu bar and key bindings. Skips the
        palette itself and Quit (textual provides those)."""
        for menu in self.MENUS:
            for item in menu.items:
                if item.action in ("command_palette", "quit"):
                    continue
                handler = getattr(self, f"action_{item.action}", None)
                if handler is not None:
                    yield (item.label, handler, menu.title)

    def __init__(self, agent6_dir: Path, repo_cwd: Path) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir
        # The repo to launch new runs in. The state dir is out of the workspace,
        # so it can't be derived from agent6_dir; the caller passes it.
        self.repo_cwd = repo_cwd
        self._runs: list[Path] = []

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)  # the top row: menus + "agent6 — <path>"
        yield DataTable(id="runs")
        yield Footer()

    def action_menu(self, mnemonic: str) -> None:
        self.query_one(MenuBar).open(mnemonic)

    async def on_menu_bar_selected(self, event: MenuBar.Selected) -> None:
        # Screen actions first, then app-level built-ins (quit, command_palette).
        # action_quit (and other app actions) are coroutines, so await results.
        handler = getattr(self, f"action_{event.action}", None) or getattr(
            self.app, f"action_{event.action}", None
        )
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result

    def on_mount(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.cursor_type = "row"
        table.add_columns("when", "mode", "status", "id", "task")
        self.action_refresh()
        table.focus()

    def on_screen_resume(self) -> None:
        # Returning from a pushed screen (e.g. config) doesn't re-run on_mount, so
        # refresh -- which also resets the menu-bar sub_title that config changed
        # to "config · …" (otherwise the hub keeps showing "agent6 — config").
        self.action_refresh()

    def action_refresh(self) -> None:
        table = self.query_one("#runs", DataTable)
        table.clear()
        # Keep self._runs 1:1 with the table rows: a run dir that vanished between
        # the listing and its stat() must be dropped from BOTH, or every
        # cursor_row-indexed selection action (open/logs/merge) maps to the wrong
        # run for cursor positions past the gap.
        survivors: list[Path] = []
        for rd in _list_runs(self.agent6_dir):
            try:
                mtime = rd.stat().st_mtime
            except OSError:
                continue  # vanished since the listing snapshot — skip it
            s = _run_summary(rd)
            when = time.strftime("%m-%d %H:%M", time.localtime(mtime))
            # Text cells: task is model/user input and may carry markup brackets.
            table.add_row(when, s["mode"], s["status"], Text(s["id"]), Text(s["task"]))
            survivors.append(rd)
        self._runs = survivors
        # Useful context in the header sub-title rather than a duplicate hint bar.
        self.app.sub_title = f"{self.repo_cwd} · {len(self._runs)} runs"
        # An empty table shouldn't paint a full-height focus cursor over its body.
        table.show_cursor = table.row_count > 0

    def action_open_selected(self) -> None:
        table = self.query_one("#runs", DataTable)
        if self._runs and 0 <= table.cursor_row < len(self._runs):
            self.app.exit(self._runs[table.cursor_row])

    def action_view_logs(self) -> None:
        """Open a scrollable, read-only log of the selected run (current or
        finished) without leaving the hub -- the run list only shows a one-line
        status, so this is how you read what a past run actually did."""
        table = self.query_one("#runs", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        run_dir = self._runs[table.cursor_row]
        self.app.push_screen(LogScreen(run_dir / "logs.jsonl", title=f"logs · {run_dir.name}"))

    def action_view_transcript(self) -> None:
        """Open the full LLM conversation of the selected run (the lossless
        transcripts) -- the deep-dive companion to the terse event log."""
        table = self.query_one("#runs", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        run_dir = self._runs[table.cursor_row]
        self.app.push_screen(
            ConversationScreen(run_dir / "transcripts", title=f"conversation · {run_dir.name}")
        )

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        # Enter / double-click a run row opens it. The DataTable consumes Enter
        # for its own RowSelected, so the screen's `enter` binding never fires --
        # handle the row event itself instead.
        self.action_open_selected()

    def action_quit(self) -> None:
        # On the App, `quit` is a built-in; on a Screen it isn't, and the binding
        # doesn't bubble to it -- so define it here, or the footer's "q Quit"
        # would lie (only Ctrl+Q, an app-level default, would work).
        self.app.exit()

    def action_new_work(self) -> None:
        self.app.push_screen(_NewWorkModal(_available_profiles(self.repo_cwd)), self._on_new_work)

    def action_merge_selected(self) -> None:
        """Merge the selected run's branch into its base, after a confirm. The TUI
        shells out to `agent6 runs merge` (never git_ops directly); the CLI applies
        git.merge_strategy and refuses a dirty tree / unconfigured identity."""
        table = self.query_one("#runs", DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        run_id = self._runs[table.cursor_row].name
        self.app.push_screen(
            ConfirmModal(
                f"Merge run {run_id}?",
                "Runs `agent6 runs merge` to land this run's branch on its base using your "
                "git.merge_strategy. The working tree must be clean.",
                confirm_label="Merge",
            ),
            self._on_merge_confirm(run_id),
        )

    def _on_merge_confirm(self, run_id: str) -> Callable[[bool | None], None]:
        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_merge_cli(self.repo_cwd, run_id)
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        return cb

    def action_open_config(self) -> None:
        self.app.push_screen(ConfigScreen(self.repo_cwd))

    def action_open_machines(self) -> None:
        self.app.push_screen(MachinesScreen(self.repo_cwd))

    def action_choose_theme(self) -> None:
        open_theme_picker(self.app)

    def action_help(self) -> None:
        self.app.push_screen(HelpScreen(self.MENUS, title="agent6 — keys & actions"))

    def _on_new_work(self, result: tuple[str, str, str] | None) -> None:
        if result is None:
            return
        mode, task, profile = result
        run_dir, error = _spawn_and_locate(
            self.agent6_dir, self.repo_cwd, mode, task, profile=profile
        )
        if run_dir is not None:
            self.app.exit(run_dir)
        else:
            self.app.notify(error or "Could not start the run.", severity="error", timeout=8.0)


class Agent6HomeApp(App[Path | None]):
    """Home hub. `run()` returns the run directory the user chose to open (to be
    watched by the dashboard), or None to quit. A thin shell around
    :class:`HomeScreen` so the hub's key bindings stay screen-scoped."""

    TITLE = "agent6"
    CSS = (
        PALETTE_CSS
        + """
    Screen { layers: base dropdown; }
    #runs { height: 1fr; border: round $primary; background: $surface; }
    #runs:focus { border: round $accent; }
    /* Panel-coloured header bar (matches the menu bar + footer + config header),
       and a selection bar only when focused. */
    #runs > .datatable--header { background: $panel; color: $foreground; text-style: bold; }
    #runs > .datatable--cursor { background: transparent; color: $foreground; }
    #runs:focus > .datatable--cursor { background: $primary 40%; color: $text; text-style: bold; }
    """
    )

    def __init__(self, agent6_dir: Path, repo_cwd: Path) -> None:
        super().__init__()
        self.agent6_dir = agent6_dir
        self.repo_cwd = repo_cwd

    def on_mount(self) -> None:
        setup_theme(self)  # apply the saved theme before the first paint
        self.push_screen(HomeScreen(self.agent6_dir, self.repo_cwd))

    def get_system_commands(self, screen: Screen[object]) -> Iterable[SystemCommand]:
        # Drop textual's "Keys" panel (our Help page replaces it), "Screenshot"
        # (an unused default whose SVG export is broken in our terminals), and
        # "Theme" (replaced by our live-preview Theme… picker). Every home action,
        # including Open config / Theme… / Keys & actions, is searchable by name
        # via _HomeCommands now, so nothing is added here.
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme"):
                yield cmd


def _spawn_and_locate(
    agent6_dir: Path, repo_cwd: Path, mode: str, task: str, *, profile: str = ""
) -> tuple[Path | None, str]:
    """Spawn `agent6 <mode> [--profile <name>] <task>` detached (non-TTY stdout →
    no nested TUI) and return (run_dir, ""). On failure returns (None,
    diagnostic). A non-empty *profile* maps to the per-subcommand --profile flag
    (placed after the mode, before the task); "" => no flag, so the config's
    [workflow].profile applies. The dir is found by snapshotting existing runs
    and polling for a NEW one; if the child exits before producing a run dir (no
    git repo, bad config, …) its stderr tail is surfaced instead of silently
    waiting out the timeout."""
    cwd = repo_cwd
    # --profile is a per-subcommand flag, so it goes after <mode> and before the
    # positional <task> -> `agent6 <mode> --profile <name> <task>`.
    argv = [agent6_exe(), mode]
    if profile:
        argv += ["--profile", profile]
    argv.append(task)
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


def _run_merge_cli(repo_cwd: Path, run_id: str) -> tuple[bool, str]:
    """Run `agent6 runs merge <run_id>` (capturing output) and return (ok, message).
    The hub shells out to the same CLI a user would, so merging stays a CLI concern
    and the UI never touches git_ops. Synchronous: a merge is a quick git op."""
    try:
        proc = subprocess.run(
            [agent6_exe(), "runs", "merge", run_id],
            cwd=str(repo_cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"merge failed to run: {exc}"
    message = "\n".join(p for p in (proc.stdout.strip(), proc.stderr.strip()) if p)
    return proc.returncode == 0, message or f"exit {proc.returncode}"


def run_home(agent6_dir: Path, repo_cwd: Path) -> Path | None:
    return Agent6HomeApp(agent6_dir, repo_cwd).run()
