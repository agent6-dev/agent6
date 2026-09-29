# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 tui` hub: browse recent sessions and start new work.

The hub never reimplements the harness: starting a session spawns the `agent6`
CLI detached and opens the read-only dashboard on the directory it creates, and
every other verb shells out to the CLI the same way.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

try:
    from rich import text
    from textual import app, events, widgets
    from textual import screen as textual_screen
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

# Reached only once the textual guard above passed.
from agent6 import git_ops
from agent6.config import ConfigError, layer
from agent6.models import choices
from agent6.sessions import layout
from agent6.ui import spawn
from agent6.ui.tui import (
    config_page,
    logview,
    machines,
    menubar,
    modals,
    new_work,
    screen_chrome,
    theme,
)
from agent6.viewmodel import (
    LIVE_STATUS_WORDS,
    SessionSummary,
    format,
    is_winner,
    session_dirs,
    summarize_session_dir,
    task_snippet,
)
from agent6.viewmodel import listing as viewmodel_listing

# The poll cadence, the web hub's, so a session that ends while watched stops reading as running.
_HUB_POLL_S = 4.0


# Below the first width `updated` shortens and a status drops its reason; below the second the
# cost column hides too.
_NARROW_COLS = 100
_NARROWEST_COLS = 80


def _sync_columns(table: widgets.DataTable[Any], width: int) -> tuple[str, ...]:
    """Set the table's columns for the width, rebuilt only on a change, and empty it for a refill.

    Args:
        table: The hub table.
        width: The terminal width; the cost column hides below `_NARROWEST_COLS`.

    Returns:
        The column labels.
    """
    labels = ("updated", "status", "cost", "id", "task")
    if width < _NARROWEST_COLS:
        labels = ("updated", "status", "id", "task")
    if tuple(str(c.label) for c in table.columns.values()) != labels:
        table.clear(columns=True)
        table.add_columns(*labels)
    table.clear()
    return labels


def _status_cell(summary: SessionSummary, *, narrow: bool = False) -> text.Text:
    """Return the status cell in its colour; a narrow table drops the reason."""
    reason = "" if narrow else summary.reason
    label = format.listing_status_label(
        summary.mode, summary.status, reason, unmerged=summary.unmerged
    )
    return text.Text(label, style=theme.status_style(summary.status))


class HomeScreen(screen_chrome.ScreenChrome, textual_screen.Screen[None]):
    """The hub view: browse recent sessions, start new work, open the config editor.

    Its bindings live on the screen, not the app, so a pushed screen's footer shows
    only that screen's keys.
    """

    MENUS: ClassVar = (
        menubar.Menu(
            "File",
            (
                menubar.MenuItem("New session", "new_work"),
                menubar.MenuItem("Open selected", "open_selected"),
                menubar.MenuItem("Merge selected run", "merge_selected"),
                menubar.MenuItem("Delete selected run…", "delete_selected"),
                menubar.MenuItem("Prune merged runs…", "prune"),
                menubar.MenuItem("Prune merged runs, squash-merged too…", "prune_squashed"),
                menubar.MenuItem("Clear saved asks…", "clear_asks"),
                menubar.MenuItem("Refresh", "refresh"),
                menubar.MenuItem("Quit", "quit"),
            ),
        ),
        menubar.Menu("Config", (menubar.MenuItem("Open config", "open_config"),)),
        menubar.Menu("Machines", (menubar.MenuItem("Open machines", "open_machines"),)),
        menubar.Menu(
            "View",
            (
                menubar.MenuItem("View logs", "view_logs"),
                menubar.MenuItem("Fold/expand lanes", "toggle_lanes"),
                menubar.MenuItem("Theme…", "choose_theme"),
                menubar.MenuItem("Copy method…", "choose_copy_method"),
            ),
        ),
        menubar.Menu(
            "Help",
            (
                menubar.MenuItem("Keys & actions", "help"),
                menubar.MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    FOOTER: ClassVar = (
        ("new_work", "New session"),
        ("open_selected", "Open"),
        ("toggle_lanes", "Lanes"),
        ("merge_selected", "Merge run"),
        ("delete_selected", "Delete run"),
        ("open_config", "Config"),
        ("open_machines", "Machines"),
        ("help", "Help"),
        ("quit", "Quit"),
    )
    BINDINGS: ClassVar = menubar.menu_bindings("hub", MENUS, footer=FOOTER)
    COMMANDS: ClassVar = textual_screen.Screen.COMMANDS | {screen_chrome.MenuCommands}
    HELP_HINTS: ClassVar = (
        "Enter opens the selected run",
        "Space folds or expands a fan-out's lanes",
        "Pickers: ↑↓ highlight · Space selects",
    )

    def __init__(
        self,
        agent6_dir: pathlib.Path,
        repo_cwd: pathlib.Path,
        config_path: pathlib.Path | None = None,
    ) -> None:
        """Bind the hub to the state dir, the repo new sessions launch in and the config path."""
        super().__init__()
        self.agent6_dir = agent6_dir
        self.repo_cwd = repo_cwd  # not derivable from the state dir, which is out of the workspace
        self.config_path = config_path  # stamped into everything the hub spawns or loads
        self._runs: list[pathlib.Path] = []
        # The poll's rows by session id, so `check_action` never re-folds a session.
        self._summaries: dict[str, SessionSummary] = {}
        # The fan-outs listed, by coordinator id, and the ones the operator expanded.
        self._fanouts: dict[str, viewmodel_listing.ListingRow] = {}
        self._expanded: set[str] = set()

    def compose(self) -> app.ComposeResult:
        """Lay out the hub.

        Yields:
            The menu bar, the sessions table and the footer.
        """
        yield menubar.MenuBar(self.MENUS)
        yield widgets.DataTable(id="sessions")
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Fill the table, focus it and start the poll."""
        table = self.query_one("#sessions", widgets.DataTable)
        table.cursor_type = "row"
        self.action_refresh()
        table.focus()
        self.set_interval(_HUB_POLL_S, self._poll)

    def on_resize(self, _event: events.Resize) -> None:
        """Refill: the task column is sized to the width, which the mount-time fill lacked."""
        self.action_refresh()

    def on_screen_resume(self) -> None:
        """Refill on return from a pushed screen, which also restores the sub-title."""
        self.action_refresh()

    def _poll(self) -> None:
        """Refill while the hub is the top screen; a rebuild under a modal would shift its rows."""
        if self.app.screen is self:
            self.action_refresh()

    def action_refresh(self) -> None:
        """Rebuild the table, keeping the selection by session id, since activity reorders rows."""
        table = self.query_one("#sessions", widgets.DataTable)
        selected = ""
        if self._runs and 0 <= table.cursor_row < len(self._runs):
            selected = self._runs[table.cursor_row].name
        # A zero width is the mount-time fill before layout: laid out as wide.
        narrow = (self.size.width or _NARROW_COLS) < _NARROW_COLS
        labels = _sync_columns(table, self.size.width or _NARROW_COLS)
        # `_runs` stays 1:1 with the rows: a dir that vanished since the listing is dropped from
        # both, or every cursor-indexed action past the gap maps to the wrong session.
        survivors: list[pathlib.Path] = []
        rows: dict[str, SessionSummary] = {}
        tips = git_ops.run_ref_tips(self.repo_cwd)
        dirs = {rd.name: rd for rd in session_dirs(self.agent6_dir) if rd.is_dir()}
        listing = viewmodel_listing.nested_rows(
            summarize_session_dir(rd, branch_tips=tips) for rd in dirs.values()
        )
        self._fanouts = {row.summary.session_id: row for row in listing if row.lanes}

        pending: list[tuple[str | text.Text, ...]] = []

        def add(row: viewmodel_listing.ListingRow, id_cell: str) -> None:
            # The time is the row's: a fan-out's latest lane activity, as `sessions list` shows it.
            s = row.summary
            cells: dict[str, str | text.Text] = {
                "updated": format.format_when(row.mtime, short=narrow),
                "status": _status_cell(s, narrow=narrow),
                "cost": s.cost_cell,
                "id": text.Text(id_cell),
            }
            pending.append((*(cells[label] for label in labels[:-1]), s.task))
            survivors.append(dirs[s.session_id])
            rows[s.session_id] = s

        def emit(row: viewmodel_listing.ListingRow, depth: int) -> None:
            s = row.summary
            marked = format.winner_id(s.session_id, winner=is_winner(dirs[s.session_id]))
            if depth:
                add(row, format.lane_id_cell(marked, depth))
                for lane in row.lanes:
                    emit(lane, depth + 1)
                return
            expanded = s.session_id in self._expanded
            folded = f" ({format.lane_count(len(row.lanes))})" if row.lanes and not expanded else ""
            add(row, marked + folded)
            if expanded:
                for lane in row.lanes:
                    emit(lane, 1)

        for row in listing:
            emit(row, 0)
        # The task column takes what the other columns leave, floor 24, as `sessions list` sizes it.
        fixed = sum(
            max(len(label), max((len(str(cells[i])) for cells in pending), default=0)) + 2
            for i, label in enumerate(labels[:-1])
        )
        task_w = max(24, table.scrollable_content_region.width - fixed - 2)
        for *cells, task in pending:
            # A Text cell: the task is typed input and may carry markup brackets.
            table.add_row(*cells, text.Text(task_snippet(str(task), max_chars=task_w)))
        self._runs = survivors
        self._summaries = rows
        if selected:
            row = next((i for i, rd in enumerate(survivors) if rd.name == selected), None)
            if row is not None:
                table.move_cursor(row=row)
        count = len(dirs)
        # The empty state says what to do next, as the CLI and the web do.
        tally = (
            'no sessions yet (n starts one, or: agent6 run "<task>")'
            if not count
            else f"{count} session{'' if count == 1 else 's'}"
        )
        self.app.sub_title = f"{self.repo_cwd.name} · {tally}"
        table.show_cursor = table.row_count > 0  # no full-height cursor over an empty body

    def action_toggle_lanes(self) -> None:
        """Expand or fold the selected fan-out's lanes; the cursor stays on the fan-out's row."""
        target = self._selected_fanout()
        if target is None:
            return
        if target in self._expanded:
            self._expanded.discard(target)
        else:
            self._expanded.add(target)
        self.action_refresh()
        table = self.query_one("#sessions", widgets.DataTable)
        row = next((i for i, rd in enumerate(self._runs) if rd.name == target), None)
        if row is not None:
            table.move_cursor(row=row)

    def _selected_fanout(self) -> str | None:
        """Return the fan-out the cursor's row belongs to, its own or a lane's; else None."""
        rd = self._selected_dir()
        if rd is None:
            return None
        sid = rd.name
        while sid not in self._fanouts:
            s = self._summaries.get(sid)
            if s is None or not s.coordinator or s.coordinator == sid:
                return None
            sid = s.coordinator
        return sid

    def action_open_selected(self) -> None:
        """Exit the hub with the selected session, for the run view to open."""
        table = self.query_one("#sessions", widgets.DataTable)
        if self._runs and 0 <= table.cursor_row < len(self._runs):
            self.app.exit(self._runs[table.cursor_row])

    def action_view_logs(self) -> None:
        """Push the selected session's log view."""
        table = self.query_one("#sessions", widgets.DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        session_dir = self._runs[table.cursor_row]
        self.app.push_screen(
            logview.LogScreen(
                session_dir / layout.LOGS_NAME, title=lambda: f"logs · {session_dir.name}"
            )
        )

    def on_data_table_row_selected(self, _event: widgets.DataTable.RowSelected) -> None:
        """Open the row on Enter or a double click; the table consumes Enter itself."""
        self.action_open_selected()

    def action_quit(self) -> None:
        """Quit; the app's built-in `quit` is not reached from a screen binding."""
        self.app.exit()

    def action_new_work(self) -> None:
        """Push the new-session screen."""
        self.app.push_screen(
            new_work.NewWorkScreen(
                self.repo_cwd,
                self.config_path,
                presets=layer.available_preset_names(self.repo_cwd, self.config_path),
                routes=choices.available_routes(self.repo_cwd, self.config_path),
            )
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Grey Merge and Delete out on a live session, and hide Lanes without a fan-out.

        None greys a key where False would hide it, and a missing key reads as a
        capability the hub lacks. The other refusals (no commits, already merged)
        are the CLI's to make: the summary's `unmerged` mark is branch-derived and
        reads False for a session whose commits live only on its chain ref.

        Returns:
            True, None to grey the key, or False to hide it.
        """
        del parameters
        if action == "toggle_lanes":
            return self._selected_fanout() is not None
        if action in ("merge_selected", "delete_selected"):
            rd = self._selected_dir()
            if rd is None:
                return None
            s = self._summaries.get(rd.name)
            return None if s is not None and s.status in LIVE_STATUS_WORDS else True
        return True

    def on_data_table_row_highlighted(self, _event: widgets.DataTable.RowHighlighted) -> None:
        """Re-ask `check_action`, which textual does only on a bindings refresh."""
        self.refresh_bindings()

    def _selected_dir(self) -> pathlib.Path | None:
        """Return the session dir under the cursor, or None on an empty table."""
        table = self.query_one("#sessions", widgets.DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return None
        return self._runs[table.cursor_row]

    def action_merge_selected(self) -> None:
        """Merge the selected session's branch into its base through the CLI, after a confirm."""
        table = self.query_one("#sessions", widgets.DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        session_id = self._runs[table.cursor_row].name
        self.app.push_screen(
            modals.ConfirmModal(
                f"Merge run {session_id}?",
                "Runs `agent6 sessions merge` to land this run's branch on its base using your "
                "git.merge_strategy. Ref plumbing only: the checkout never moves.",
                confirm_label="Merge",
            ),
            self._on_merge_confirm(session_id),
        )

    def action_delete_selected(self) -> None:
        """Delete the selected session's history through the CLI, after a confirm.

        History only: the branch and its commits are `sessions prune`'s.
        """
        table = self.query_one("#sessions", widgets.DataTable)
        if not (self._runs and 0 <= table.cursor_row < len(self._runs)):
            return
        session_id = self._runs[table.cursor_row].name
        self.app.push_screen(
            modals.ConfirmModal(
                f"Delete run {session_id}'s history?",
                "Runs `agent6 sessions rm`: removes its transcripts, events and manifest from"
                " the state dir. The run branch and its commits are kept.",
                confirm_label="Delete",
            ),
            self._on_delete_confirm(session_id),
        )

    def _on_delete_confirm(self, session_id: str) -> Callable[[bool | None], None]:
        """Return the confirm callback that runs the delete and refreshes."""

        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_delete_cli(self.repo_cwd, session_id, self.config_path)
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        return cb

    def _on_merge_confirm(self, session_id: str) -> Callable[[bool | None], None]:
        """Return the confirm callback that runs the merge and refreshes."""

        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_merge_cli(self.repo_cwd, session_id, self.config_path)
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        return cb

    def action_prune(self) -> None:
        """Run `agent6 sessions prune` after a confirm; the dialog names what it removes."""
        self.app.push_screen(
            modals.ConfirmModal(
                "Prune merged runs?",
                "Runs `agent6 sessions prune`: deletes run branches git can remove as merged"
                " (reachable from the checked-out branch), merged runs' chain refs, the"
                " worktrees of merged forks and fan-out clone dirs whose lanes are all here."
                " Squash-merged branches and ones merged elsewhere are kept and named;"
                " unmerged ones are never touched.",
                confirm_label="Prune",
            ),
            self._on_prune_confirm(delete_squashed=False),
        )

    def action_prune_squashed(self) -> None:
        """Run `agent6 sessions prune --delete-squashed` after a confirm."""
        self.app.push_screen(
            modals.ConfirmModal(
                "Prune merged runs, squash-merged too?",
                "Runs `agent6 sessions prune --delete-squashed`: also force-deletes branches"
                " and chain refs the manifest confirms were squash-merged into a base that"
                " still holds what the merge landed. Each deletion prints its undelete command.",
                confirm_label="Prune",
            ),
            self._on_prune_confirm(delete_squashed=True),
        )

    def _on_prune_confirm(self, *, delete_squashed: bool) -> Callable[[bool | None], None]:
        """Return the confirm callback that runs the prune and refreshes."""

        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_prune_cli(
                self.repo_cwd, delete_squashed=delete_squashed, config_path=self.config_path
            )
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        return cb

    def action_clear_asks(self) -> None:
        """Run `agent6 sessions rm --asks` after a confirm."""

        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            ok, msg = _run_clear_asks_cli(self.repo_cwd, self.config_path)
            self.app.notify(msg, severity="information" if ok else "error", timeout=10.0)
            self.action_refresh()

        self.app.push_screen(
            modals.ConfirmModal(
                "Clear this directory's saved asks?",
                "Runs `agent6 sessions rm --asks`: removes every saved `agent6 ask` transcript"
                " for this directory. Asks run elsewhere are untouched.",
                confirm_label="Clear",
            ),
            cb,
        )

    def action_open_config(self) -> None:
        """Push the config editor, or name `agent6 config fix` when the config is invalid."""
        try:
            layer.load_effective(self.repo_cwd, self.config_path)
        except ConfigError as exc:
            self.app.notify(
                "Config is invalid, so it can't be opened. Run `agent6 config fix` in a"
                f" terminal to drop invalid entries, then reopen.\n{exc}",
                severity="error",
                timeout=15.0,
            )
            return
        self.app.push_screen(config_page.ConfigScreen(self.repo_cwd, self.config_path))

    def action_open_machines(self) -> None:
        """Push the machines screen."""
        self.app.push_screen(
            machines.MachinesScreen(self.agent6_dir, self.repo_cwd, self.config_path)
        )


class Agent6HomeApp(theme.PlainNotify, theme.MuxPointerShapes, app.App[pathlib.Path | None]):
    """The hub app; `run()` returns the session dir to open, or None to quit.

    A thin shell around `HomeScreen`, so the hub's key bindings stay screen-scoped.
    """

    TITLE = "agent6"
    CSS = (
        theme.PALETTE_CSS
        + """
    Screen { layers: base dropdown; background: $surface; }
    /* The Screen rule matches modals too; this restores their translucent backdrop. */
    ModalScreen { background: $background 60%; }
    * { scrollbar-size-vertical: 1; scrollbar-size-horizontal: 1; }
    /* The one-row footer has no room for a scrollbar, which would replace every hint. */
    Footer { scrollbar-size-vertical: 0; scrollbar-size-horizontal: 0; }
    Input, TextArea { pointer: text; }
    #sessions { height: 1fr; border: round $primary; background: $surface; }
    #sessions:focus { border: round $accent; }
    /* A panel-coloured header, and a selection bar only when focused. */
    #sessions > .datatable--header { background: $panel; color: $foreground; text-style: bold; }
    #sessions > .datatable--cursor { background: transparent; color: $foreground; }
    #sessions:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    """
    )

    def __init__(
        self,
        agent6_dir: pathlib.Path,
        repo_cwd: pathlib.Path,
        config_path: pathlib.Path | None = None,
    ) -> None:
        """Bind the app to the state dir, the repo and the config path."""
        super().__init__()
        self.agent6_dir = agent6_dir
        self.repo_cwd = repo_cwd
        self.config_path = config_path

    def on_mount(self) -> None:
        """Apply the saved theme before the first paint, then push the hub."""
        theme.setup_theme(self)
        self.push_screen(HomeScreen(self.agent6_dir, self.repo_cwd, self.config_path))

    def get_system_commands(
        self, screen: textual_screen.Screen[object]
    ) -> Iterable[app.SystemCommand]:
        """Yield textual's palette commands minus the ones the hub's own menus replace."""
        for cmd in super().get_system_commands(screen):
            if cmd.title not in ("Keys", "Screenshot", "Theme"):
                yield cmd


def _run_merge_cli(
    repo_cwd: pathlib.Path, session_id: str, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Return whether `agent6 sessions merge` succeeded on the session, and its output."""
    return spawn.run_cli_capture(
        [*spawn.agent6_argv(config_path), "sessions", "merge", session_id], repo_cwd
    )


def _run_delete_cli(
    repo_cwd: pathlib.Path, session_id: str, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Return whether `agent6 sessions rm` succeeded on the session, and its output."""
    ok, msg = spawn.run_cli_capture(
        [*spawn.agent6_argv(config_path), "sessions", "rm", "--", session_id], repo_cwd
    )
    return ok, msg or ("removed" if ok else "could not remove")


def _run_prune_cli(
    repo_cwd: pathlib.Path, *, delete_squashed: bool, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Return whether `agent6 sessions prune` succeeded, and its output."""
    argv = [*spawn.agent6_argv(config_path), "sessions", "prune"]
    if delete_squashed:
        argv.append("--delete-squashed")
    ok, msg = spawn.run_cli_capture(argv, repo_cwd)
    return ok, msg or ("pruned" if ok else "prune failed")


def _run_clear_asks_cli(
    repo_cwd: pathlib.Path, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Return whether `agent6 sessions rm --asks` succeeded, and its output."""
    ok, msg = spawn.run_cli_capture(
        [*spawn.agent6_argv(config_path), "sessions", "rm", "--asks"], repo_cwd
    )
    return ok, msg or ("saved asks cleared" if ok else "could not clear the saved asks")


def run_home(
    agent6_dir: pathlib.Path, repo_cwd: pathlib.Path, config_path: pathlib.Path | None = None
) -> pathlib.Path | None:
    """Return the session dir the hub chose to open, or None to quit."""
    return Agent6HomeApp(agent6_dir, repo_cwd, config_path).run()
