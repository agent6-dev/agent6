# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 tui` Machines page: browse, view, run, and create state machines.

A separate page from the run hub (machines are not runs). Like the hub's run
actions, it never drives a machine in-process: Run and Create shell out to
`agent6 machine run|create` (detached). View is the one in-process step -- it
parses the .asm.toml via agent6.machine to show the machine's structure,
validation, and graph -- which is why ui depends on agent6.machine for this page
(see tach.toml).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

try:
    from rich.text import Text
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.command import DiscoveryHit, Hit, Hits, Provider
    from textual.containers import Container, VerticalScroll
    from textual.screen import ModalScreen, Screen
    from textual.widgets import DataTable, Footer, Input, Static
except ImportError as e:  # pragma: no cover - clear runtime message
    raise ImportError(
        "agent6 TUI requires the 'textual' package (part of the base install)."
        " Reinstall agent6, or `pip install textual`."
    ) from e

from agent6.machine import MachineError, load_machine, render, validate_semantics
from agent6.ui._spawn import agent6_exe, spawn_detached
from agent6.ui.menubar import HelpScreen, Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.modals import ConfirmModal
from agent6.ui.theme import PALETTE_CSS


def find_machine_files(repo_cwd: Path) -> list[Path]:
    """Authored .asm.toml machine files: the cwd top level (where `machine create`
    writes by default) plus a conventional machines/ subdir. Sorted by path."""
    found: set[Path] = set(repo_cwd.glob("*.asm.toml"))
    sub = repo_cwd / "machines"
    if sub.is_dir():
        found.update(sub.glob("*.asm.toml"))
    return sorted(found)


def _machine_row(path: Path) -> tuple[str, str, str]:
    """(name, state count, status) for the list -- parsed if it loads, else flagged."""
    try:
        spec = load_machine(path)
    except (MachineError, OSError):
        return (path.stem, "-", "invalid")
    problems = validate_semantics(spec)
    return (
        spec.machine,
        str(len(spec.states)),
        "ok" if not problems else f"{len(problems)} issue(s)",
    )


def machine_detail_text(path: Path) -> str:
    """A read-only text view of a parsed machine: name, initial, states, validation,
    and the mermaid graph. On a load error, the error itself (so the page never
    crashes on a half-written file)."""
    try:
        spec = load_machine(path)
    except (MachineError, OSError) as exc:
        return f"failed to load {path.name}:\n\n{exc}"
    lines = [
        f"machine: {spec.machine}",
        f"initial: {spec.initial}",
        "",
        f"states ({len(spec.states)}):",
    ]
    lines.extend(f"  {name}  ({type(state).__name__})" for name, state in spec.states.items())
    problems = validate_semantics(spec)
    lines.append("")
    if problems:
        lines.append(f"validation: {len(problems)} problem(s)")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("validation: OK")
    lines += ["", "graph (mermaid):", render(spec, "mermaid")]
    return "\n".join(lines)


class MachineDetailScreen(Screen[None]):
    """Read-only view of one parsed machine (structure + validation + graph)."""

    CSS = """
    MachineDetailScreen { background: $surface; }
    #machine-detail-title {
        dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold;
    }
    #machine-detail-body { height: 1fr; padding: 0 1; }
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def compose(self) -> ComposeResult:
        yield Static(f"machine · {self._path.name}", id="machine-detail-title")
        with VerticalScroll(id="machine-detail-body"):
            yield Static(Text(machine_detail_text(self._path)))  # plain Text: no markup parsing
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#machine-detail-body", VerticalScroll).focus()

    def action_close(self) -> None:
        self.dismiss()


class CreateMachineModal(ModalScreen[str]):
    """Prompt for a natural-language task to author a machine. Result: the task text
    (or "" if cancelled)."""

    DEFAULT_CSS = """
    CreateMachineModal { align: center middle; }
    #create-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #create-input { margin-top: 1; }
    """

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Container(id="create-box"):
            text = Text()
            text.append("Create a machine\n\n", style="bold")
            text.append("Describe the loop to author; agent6 drafts a .asm.toml in this repo.")
            yield Static(text)
            yield Input(
                placeholder="e.g. nightly: pull, run tests, open an issue on failure",
                id="create-input",
            )

    def on_mount(self) -> None:
        self.query_one("#create-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss("")


class _MachineCommands(Provider):
    """The Machines-page actions in the Ctrl+P palette, from the same MENUS as the
    menu bar and key bindings -- so the surfaces never drift."""

    @property
    def _page(self) -> MachinesScreen:
        screen = self.screen
        assert isinstance(screen, MachinesScreen)
        return screen

    async def discover(self) -> Hits:
        for name, runnable, help_text in self._page.palette_commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, runnable, help_text in self._page.palette_commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), runnable, help=help_text)


class MachinesScreen(Screen[None]):
    """List authored state machines; view (parsed), run, or create them. Run/Create
    shell out to the CLI (detached); View parses in-process."""

    CSS = (
        PALETTE_CSS
        + """
    MachinesScreen { layers: base dropdown; }
    #machines { height: 1fr; border: round $primary; background: $surface; }
    #machines:focus { border: round $accent; }
    #machines > .datatable--header { background: $panel; color: $foreground; text-style: bold; }
    #machines > .datatable--cursor { background: transparent; color: $foreground; }
    #machines:focus > .datatable--cursor {
        background: $primary 40%; color: $text; text-style: bold;
    }
    """
    )
    MENUS: ClassVar = (
        Menu(
            "Machines",
            (
                MenuItem("View", "view", "v"),
                MenuItem("Run", "run", "r"),
                MenuItem("Create…", "create", "c"),
                MenuItem("Refresh", "refresh", "f"),
                MenuItem("Back", "close", "Esc/q"),
                MenuItem("Quit", "quit", "ctrl+q"),
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
        Binding("v", "view", "View"),
        Binding("r", "run", "Run"),
        Binding("c", "create", "Create"),
        Binding("f", "refresh", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        *menu_bindings(MENUS),
    ]
    COMMANDS: ClassVar = Screen.COMMANDS | {_MachineCommands}

    def __init__(self, repo_cwd: Path) -> None:
        super().__init__()
        self.repo_cwd = repo_cwd
        self._machines: list[Path] = []

    def palette_commands(self) -> Iterator[tuple[str, Callable[[], None], str]]:
        for menu in self.MENUS:
            for item in menu.items:
                if item.action in ("command_palette", "quit"):
                    continue
                handler = getattr(self, f"action_{item.action}", None)
                if handler is not None:
                    yield (item.label, handler, menu.title)

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)
        yield DataTable(id="machines")
        yield Footer()

    def action_menu(self, mnemonic: str) -> None:
        self.query_one(MenuBar).open(mnemonic)

    async def on_menu_bar_selected(self, event: MenuBar.Selected) -> None:
        # Screen actions first, then app-level built-ins (quit, command_palette),
        # which are coroutines -- await results. Mirrors the hub + config page.
        handler = getattr(self, f"action_{event.action}", None) or getattr(
            self.app, f"action_{event.action}", None
        )
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result

    def on_mount(self) -> None:
        table = self.query_one("#machines", DataTable)
        table.cursor_type = "row"
        table.add_columns("machine", "states", "status", "file")
        self.app.sub_title = f"machines · {self.repo_cwd}"
        self._reload()

    def _reload(self) -> None:
        table = self.query_one("#machines", DataTable)
        table.clear()
        self._machines = find_machine_files(self.repo_cwd)
        for path in self._machines:
            name, states, status = _machine_row(path)
            table.add_row(Text(name), states, status, Text(path.name))
        table.show_cursor = table.row_count > 0

    def _selected(self) -> Path | None:
        table = self.query_one("#machines", DataTable)
        if self._machines and 0 <= table.cursor_row < len(self._machines):
            return self._machines[table.cursor_row]
        return None

    def on_data_table_row_selected(self, _event: DataTable.RowSelected) -> None:
        # Enter on a row opens the parsed view (the DataTable consumes Enter, so the
        # screen's binding never fires -- handle the row event itself, like the hub).
        self.action_view()

    def action_view(self) -> None:
        path = self._selected()
        if path is not None:
            self.app.push_screen(MachineDetailScreen(path))

    def action_run(self) -> None:
        path = self._selected()
        if path is None:
            return
        self.app.push_screen(
            ConfirmModal(
                f"Run machine {path.name}?",
                "Runs `agent6 machine run` (detached): it drives the machine toward a "
                "terminal or waiting state. Watch it with `agent6 machine status`.",
                confirm_label="Run",
            ),
            self._on_run_confirm(path),
        )

    def _on_run_confirm(self, path: Path) -> Callable[[bool | None], None]:
        def cb(confirmed: bool | None) -> None:
            if not confirmed:
                return
            err = spawn_detached([agent6_exe(), "machine", "run", str(path)], self.repo_cwd)
            self.app.notify(
                err or f"started: machine run {path.name}",
                severity="error" if err else "information",
                timeout=8.0,
            )

        return cb

    def action_create(self) -> None:
        self.app.push_screen(CreateMachineModal(), self._on_create)

    def _on_create(self, task: str | None) -> None:
        if not task:
            return
        err = spawn_detached([agent6_exe(), "machine", "create", task], self.repo_cwd)
        self.app.notify(
            err or "creating machine… press Refresh (f) when it finishes.",
            severity="error" if err else "information",
            timeout=8.0,
        )

    def action_refresh(self) -> None:
        self._reload()

    def action_help(self) -> None:
        self.app.push_screen(HelpScreen(self.MENUS, title="agent6 machines — keys & actions"))

    def action_quit(self) -> None:
        self.app.exit()

    def action_close(self) -> None:
        self.dismiss()
