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

import inspect
import os
import re
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
from agent6.config import ConfigError
from agent6.config.layer import load_effective
from agent6.directive import DirectiveError, Segment, parse_directive, parse_spec
from agent6.models.validate import known_models, refusal_message, validate_spec_models
from agent6.ui.spawn import agent6_exe, run_cli_capture, spawn_and_locate
from agent6.ui.tui.config_page import ConfigScreen
from agent6.ui.tui.copy_method import open_copy_method_picker
from agent6.ui.tui.logview import LogScreen
from agent6.ui.tui.machines import MachinesScreen
from agent6.ui.tui.menubar import HelpScreen, Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.modals import ConfirmModal
from agent6.ui.tui.theme import PALETTE_CSS, MuxPointerShapes, open_theme_picker, setup_theme
from agent6.ui.tui.widgets import FORM_CSS, ActionItem
from agent6.viewmodel import (
    RunSummary,
    is_run_husk,
    is_winner,
    run_mtime,
    summarize_run_dir,
    task_snippet,
)
from agent6.viewmodel.format import WINNER_GLYPH, format_cost, status_label

# Subdirs (relative to the agent6 dir) that hold watchable run directories.
_RUN_SUBDIRS = ("runs", "asks")
# The new-work profile dropdown's first entry: "" => no --profile, so the run
# uses [workflow].profile from config (or the plain defaults).
_DEFAULT_PROFILE_LABEL = "(config default)"


def _available_profiles(repo_cwd: Path) -> list[str]:
    """Profile names the new-work chooser offers (the built-ins plus the user's
    custom ``[profiles.<name>]`` tables). Delegates to ``config_layer`` -- the
    TUI's config entry point (see config_page.py) -- so the dropdown and the
    ``--profile`` CLI flag resolve against the same source."""
    from agent6.config.layer import available_profile_names  # noqa: PLC0415

    return available_profile_names(repo_cwd, None)


def _available_models(repo_cwd: Path) -> list[str]:
    """Model ids for the new-work modal's `/parallel` autocomplete: the worker's
    model plus the worker provider's cached listing (lanes inherit the worker
    provider; cache-only, no network) -- exactly the set `run --parallel`
    validation accepts. Empty on any config error; the affordance is best-effort."""
    try:
        cfg = load_effective(repo_cwd, None).config
    except ConfigError:
        return []
    return sorted(known_models(cfg))


# The spec token while it is still being typed: the whole message is
# `/parallel <token>` with nothing after it yet (a following space = task text
# has begun, so stop suggesting). Returns the comma-fragment under construction,
# or None when the caret has left the spec / the token is a bare lane count.
_SPEC_TAIL = re.compile(r"/parallel[^\S\n]+(\S*)\Z")


def _spec_fragment(text: str) -> str | None:
    m = _SPEC_TAIL.match(text)
    if m is None:
        return None
    token = m.group(1)
    if token.isdigit():  # a lane count, not a model id
        return None
    return token.rsplit(",", 1)[-1]


# Colors for the shared status words, so a dead run cannot read as a neutral
# "done" in the listing. Unlisted words ("finished", "created") render plain
# on purpose: neutral outcomes carry no signal worth a color.
_STATUS_STYLE = {
    "starting": "cyan",  # launching (pre-loop): in progress, lighter than running
    "running": "bold cyan",
    "waiting": "yellow",  # blocked on the operator (approval / question)
    "stale": "dim",
    "passed": "green",
    "answered": "green",  # an ask that answered is terminal success
    "planned": "#b48ead",  # informational mauve (matches the web pill); not green, not red
    "stopped": "yellow",
    "failed": "bold red",
}


def _status_cell(summary: RunSummary) -> Text:
    label = status_label(summary.status, summary.reason)
    return Text(label, style=_STATUS_STYLE.get(summary.status, ""))


def _cost_cell(cost_usd: float) -> str:
    return "" if cost_usd <= 0 else format_cost(cost_usd)


def _list_runs(agent6_dir: Path) -> list[Path]:
    """All run directories (runs/ + asks/), newest first by last-activity time.
    Husks (never-started dirs) are skipped, the same rule as `agent6 runs`."""
    out: list[Path] = []
    for sub in _RUN_SUBDIRS:
        d = agent6_dir / sub
        if d.is_dir():
            out.extend(p for p in d.iterdir() if p.is_dir() and not is_run_husk(p))
    out.sort(key=run_mtime, reverse=True)
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
    #new-suggest { height: auto; color: $accent; }
    """
    )

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, profiles: list[str] | None = None, models: list[str] | None = None) -> None:
        # Profile names for the dropdown (built-ins + user [profiles.*]); the
        # caller passes the repo-resolved list (built-ins + user [profiles.*]).
        # None (e.g. a bare test) => only the "(config default)" entry, so the
        # modal stands alone as a pure widget without loading config. `models` are
        # the known model ids offered as `/parallel` spec autocomplete (empty = the
        # affordance is inert, e.g. a bare test or a fresh machine with no cache).
        super().__init__()
        self._profiles = profiles if profiles is not None else []
        self._models = models if models is not None else []

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
            # Split at the phrase boundary so a narrow terminal (the box is 80%
            # wide) never wraps mid-phrase.
            yield Static(
                Text(
                    "Tab to run / plan / ask\n"
                    "/parallel [N|models] <task> fans out lanes (repeat to queue more)\n"
                    "Enter = newline · Esc cancel",
                    style="dim",
                ),
                classes="edit-label",
            )
            # Live `/parallel` model-id suggestions: matches update as you type the
            # spec token, the TUI analogue of the web composer's autocomplete popup.
            yield Static("", id="new-suggest")

    @on(TextArea.Changed, "#new-task")
    def _on_task_changed(self, _event: TextArea.Changed) -> None:
        self._refresh_suggestions()

    def _refresh_suggestions(self) -> None:
        """Show model ids matching the `/parallel` spec fragment under the caret,
        or clear the line when the caret isn't in a spec token."""
        out = self.query_one("#new-suggest", Static)
        frag = _spec_fragment(self.query_one("#new-task", TextArea).text)
        if frag is None or not self._models:
            out.update("")
            return
        q = frag.lower()
        starts = [m for m in self._models if m.lower().startswith(q)]
        rest = [m for m in self._models if q in m.lower() and not m.lower().startswith(q)]
        shown = (starts + rest)[:8]
        if not shown:
            out.update(Text("no matching model ids", style="dim"))
            return
        total = sum(1 for m in self._models if q in m.lower())
        more = f"  (+{total - len(shown)} more, keep typing)" if total > len(shown) else ""
        out.update(Text("models: ", style="dim") + Text("  ".join(shown)) + Text(more, style="dim"))

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
                # Viewing a selected run's raw event log is filed under View to
                # match the run views' View menus. There is no separate
                # transcript viewer: Enter opens the run on its conversation.
                MenuItem("View logs", "view_logs", "l"),
                MenuItem("Theme…", "choose_theme"),
                MenuItem("Copy method…", "choose_copy_method"),
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
        table.add_columns("when", "mode", "status", "cost", "id", "task")
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
            if not rd.is_dir():
                continue  # vanished since the listing snapshot — skip it
            s = summarize_run_dir(rd)
            # last-activity time (logs.jsonl), so opening a run to view it does not
            # bump its "when" the way the run-dir mtime did.
            when = time.strftime("%m-%d %H:%M", time.localtime(run_mtime(rd)))
            # Text cells: task is model/user input and may carry markup brackets.
            run_id = f"{s.run_id} {WINNER_GLYPH}" if is_winner(rd) else s.run_id
            table.add_row(
                when,
                s.mode,
                _status_cell(s),
                _cost_cell(s.cost_usd),
                Text(run_id),
                Text(task_snippet(s.task, max_chars=60)),
            )
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
        self.app.push_screen(
            _NewWorkModal(_available_profiles(self.repo_cwd), _available_models(self.repo_cwd)),
            self._on_new_work,
        )

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
        # An invalid config (e.g. a stale value or a leftover table from a removed
        # feature) would crash the config screen on load. Pre-check so we can point
        # at `agent6 config fix` instead of taking down the TUI.
        from agent6.config import ConfigError  # noqa: PLC0415
        from agent6.config.layer import load_effective  # noqa: PLC0415

        try:
            load_effective(self.repo_cwd, None)
        except ConfigError as exc:
            self.app.notify(
                "Config is invalid, so it can't be opened. Run `agent6 config fix` in a"
                f" terminal to drop invalid entries, then reopen.\n{exc}",
                severity="error",
                timeout=15.0,
            )
            return
        self.app.push_screen(ConfigScreen(self.repo_cwd))

    def action_open_machines(self) -> None:
        self.app.push_screen(MachinesScreen(self.repo_cwd, self.agent6_dir))

    def action_choose_theme(self) -> None:
        open_theme_picker(self.app)

    def action_choose_copy_method(self) -> None:
        open_copy_method_picker(self.app)

    def action_help(self) -> None:
        self.app.push_screen(
            HelpScreen(
                self.MENUS,
                self,
                title="agent6 — keys & actions",
                hints=(
                    "Enter opens the selected run",
                    "Pickers: ↑↓ highlight · Space selects",
                ),
            )
        )

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


class Agent6HomeApp(MuxPointerShapes, App[Path | None]):
    """Home hub. `run()` returns the run directory the user chose to open (to be
    watched by the dashboard), or None to quit. A thin shell around
    :class:`HomeScreen` so the hub's key bindings stay screen-scoped."""

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
    """Spawn `agent6 <mode> [--profile <name>] <task>` detached and return the new
    run dir (to be watched by the dashboard), or (None, diagnostic) on failure. A
    non-empty *profile* maps to the per-subcommand --profile flag (after the mode,
    before the task); "" => no flag, so the config's [workflow].profile applies.

    A `/parallel [spec] <task> ...` message (run mode only) fans out one detached
    `agent6 run --parallel <spec>` per segment (omitted spec = one isolated lane);
    a malformed directive is refused before any spawn (all-or-nothing), and the
    first segment's run dir is returned to watch."""
    segments = None
    if mode == "run":
        try:
            segments = parse_directive(task)
        except DirectiveError as exc:
            return None, str(exc)
    if segments is None:
        return _spawn_run(agent6_dir, repo_cwd, mode, task, profile=profile, spec="")
    refusal = _model_refusal(repo_cwd, segments)
    if refusal is not None:
        return None, refusal
    first: tuple[Path | None, str] = (None, "")
    for seg in segments:
        result = _spawn_run(
            agent6_dir, repo_cwd, "run", seg.task, profile=profile, spec=seg.spec or "1"
        )
        if first[0] is None:
            first = result
    return first


def _model_refusal(repo_cwd: Path, segments: list[Segment]) -> str | None:
    """Refuse a `/parallel` directive that names a model the configured providers'
    cache says doesn't exist, before any spawn (the modal's normal error path,
    nothing spawned). None = every model checks out, or there is no cache to check
    against (a fresh/offline machine proceeds; the detached lane's own preflight
    warns). A malformed spec surfaces its grammar error."""
    try:
        models = [m for seg in segments for m in parse_spec(seg.spec)]
    except DirectiveError as exc:
        return str(exc)
    try:
        cfg = load_effective(repo_cwd).config
    except ConfigError:
        return None  # a broken config is its own separate error; don't mask it here
    verdict = validate_spec_models(models, cfg)
    return refusal_message(verdict, directive=True) if verdict.refused else None


def _spawn_run(
    agent6_dir: Path, repo_cwd: Path, mode: str, task: str, *, profile: str, spec: str
) -> tuple[Path | None, str]:
    """Spawn one detached `agent6 <mode>` (optionally `--parallel <spec>`) and
    return its located run dir, or (None, diagnostic)."""
    # --profile is a per-subcommand flag, so it goes after <mode> and before the
    # positional <task> -> `agent6 <mode> --profile <name> <task>`.
    argv = [agent6_exe(), mode]
    if profile:
        argv += ["--profile", profile]
    if spec:
        argv += ["--parallel", spec]
    # `--` before the task so a task that looks like a flag is never parsed as
    # one; every flag precedes it. Matches the web actions spawn.
    argv += ["--", task]
    return spawn_and_locate(
        argv,
        repo_cwd,
        before=set(_list_runs(agent6_dir)),
        list_dirs=lambda: _list_runs(agent6_dir),
        # The hub watches this run on the dashboard, which renders the model's
        # reasoning + answer from role.*_delta events. Tell the detached (non-TTY)
        # run to emit those deltas to its logs.jsonl; without this it takes the
        # non-streaming path and the dashboard shows only worker status.
        # AGENT6_DETACHED_AWAY=wait: driven from the dashboard over the bridge, so
        # approvals/questions WAIT for the front-end, never fabricate an answer.
        env={**os.environ, "AGENT6_STREAM_TO_LOG": "1", "AGENT6_DETACHED_AWAY": "wait"},
    )


def _run_merge_cli(repo_cwd: Path, run_id: str) -> tuple[bool, str]:
    """Run `agent6 runs merge <run_id>` (capturing output) and return (ok, message).
    The hub shells out to the same CLI a user would, so merging stays a CLI concern
    and the UI never touches git_ops. Synchronous: a merge is a quick git op."""
    return run_cli_capture([agent6_exe(), "runs", "merge", run_id], repo_cwd)


def run_home(agent6_dir: Path, repo_cwd: Path) -> Path | None:
    return Agent6HomeApp(agent6_dir, repo_cwd).run()
