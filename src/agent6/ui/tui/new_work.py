# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The conversation view before there is a conversation.

The hub's `n` opens it: a run's chrome with the transcript pane empty and a
mode, preset and model row above the composer. Enter starts the session
detached and hands it to the live view; a refusal renders where the transcript
will be, and the typed text stays in the composer to fix and resend.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer, Select, Static, TextArea

from agent6.directive import spec_fragment
from agent6.kinds import OPERATOR_MODES
from agent6.models.choices import default_label, default_preset, default_route
from agent6.ui.spawn import spawn_new_work
from agent6.ui.tui.composer import SteerInput, SteerSuggest
from agent6.ui.tui.menubar import Menu, MenuBar, MenuItem, menu_bindings
from agent6.ui.tui.screen_chrome import MenuCommands, ScreenChrome
from agent6.ui.tui.widgets import Picker, PickerRow

_INTRO = (
    "Describe the task (or the question, for ask). Enter starts it; Ctrl-J adds a line.\n"
    "Tab reaches the mode, preset and model pickers below.\n"
    "/parallel [N|models] <task> fans out isolated lanes (repeat to queue more)."
)


def model_suggestions(models: list[str], text: str, *, limit: int = 8) -> Text | None:
    """Return the suggestion line for a `/parallel` spec fragment under the caret.

    Args:
        models: The routes the config can run.
        text: The composer's text.
        limit: How many matches to show.

    Returns:
        The matching routes, prefix matches first; None when the caret is not in a
        spec token or there is nothing to offer.
    """
    frag = spec_fragment(text)
    if frag is None or not models:
        return None
    q = frag.lower()
    starts = [m for m in models if m.lower().startswith(q)]
    rest = [m for m in models if q in m.lower() and not m.lower().startswith(q)]
    shown = (starts + rest)[:limit]
    if not shown:
        return Text("no matching model ids", style="dim")
    total = len(starts) + len(rest)
    more = f"  (+{total - len(shown)} more, keep typing)" if total > len(shown) else ""
    return Text("models: ", style="dim") + Text("  ".join(shown)) + Text(more, style="dim")


class NewWorkScreen(ScreenChrome, Screen[None]):
    """Type a task, pick a mode, a preset and a model; Enter starts it.

    Lives in the hub app: the located session dir is the hub's return value. The
    preset and model pickers open on the config's own choice, which adds no flag;
    any other pick rides as `--preset` or `--model`.
    """

    CSS = """
    NewWorkScreen { background: $surface; }
    #draft-main { height: 1fr; }
    #draft-scroll { height: 1fr; }
    #draft-notice { height: auto; padding: 0 1; pointer: text; }
    #draft-options { padding: 0 1; }
    """

    # The composer has the focus: Esc and Ctrl+Q fire before it.
    MENUS: ClassVar = (
        Menu(
            "File",
            (
                MenuItem("Back", "close", priority=True),
                MenuItem("Quit", "quit_hub", priority=True),
            ),
        ),
        Menu(
            "View",
            (MenuItem("Theme…", "choose_theme"), MenuItem("Copy method…", "choose_copy_method")),
        ),
        Menu(
            "Help",
            (MenuItem("Keys & actions", "help"), MenuItem("Command palette", "command_palette")),
        ),
    )
    BINDINGS: ClassVar = menu_bindings("new work", MENUS, footer=(("close", "Back"),))
    COMMANDS: ClassVar = Screen.COMMANDS | {MenuCommands}
    HELP_TITLE: ClassVar = "agent6 — new session"
    HELP_HINTS: ClassVar = (
        "Enter starts the task; Ctrl-J or Shift+Enter inserts a newline",
        "Tab moves between the text, the mode, the preset and the model",
    )

    def __init__(
        self,
        repo_cwd: Path,
        config_path: Path | None = None,
        *,
        presets: list[str] | None = None,
        routes: list[str] | None = None,
    ) -> None:
        """Bind the screen to the repo, its config and the preset and route choices."""
        super().__init__()
        self.repo_cwd = repo_cwd
        self.config_path = config_path
        self._presets = presets if presets is not None else []
        self._routes = routes if routes is not None else []
        self._starting = False

    def compose(self) -> ComposeResult:
        """Lay out the screen.

        Yields:
            The menu bar, the notice pane, the suggestion line, the picker row, the
            composer and the footer.
        """
        yield MenuBar(self.MENUS)
        # The empty pane is no tab stop, so Tab reaches the pickers as the intro says.
        with Vertical(id="draft-main"), VerticalScroll(id="draft-scroll", can_focus=False):
            yield Static(Text(_INTRO, style="dim italic"), id="draft-notice")
        yield SteerSuggest(id="draft-suggest")
        with PickerRow(id="draft-options"):
            yield Static("mode", classes="picker-label")
            yield Picker(
                [(m, m) for m in OPERATOR_MODES], value="run", allow_blank=False, id="draft-mode"
            )
            yield Static("preset", classes="picker-label")
            preset = default_preset(self.repo_cwd, self.config_path)
            yield Picker(
                [(default_label(preset), ""), *((p, p) for p in self._presets)],
                value="",
                allow_blank=False,
                id="draft-preset",
            )
            yield Static("model", classes="picker-label")
            route = default_route(self.repo_cwd, self.config_path, "run", "")
            yield Picker(self._model_options(route), value="", allow_blank=False, id="draft-model")
        yield SteerInput(id="draft-input")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the composer in start mode."""
        self.app.sub_title = "new session"
        bar = self.query_one("#draft-input", SteerInput)
        bar.set_mode(mode="start")
        bar.focus()

    def _model_options(self, route: str) -> list[tuple[str, str]]:
        """Return the model picker's rows: the config default, then every route."""
        return [(default_label(route), ""), *((r, r) for r in self._routes)]

    @on(Select.Changed, "#draft-mode")
    @on(Select.Changed, "#draft-preset")
    def _follow_route(self) -> None:
        """Reset the model picker to the config default for the mode and preset."""
        mode = str(self.query_one("#draft-mode", Select).value)
        preset = str(self.query_one("#draft-preset", Select).value)
        route = default_route(self.repo_cwd, self.config_path, mode, preset)
        self.query_one("#draft-model", Select).set_options(self._model_options(route))

    def action_close(self) -> None:
        """Close an open list, else return to the hub."""
        if not self.close_open_list():
            self.dismiss(None)

    def action_quit_hub(self) -> None:
        """Quit the hub."""
        self.app.exit(None)

    @on(TextArea.Changed, "#draft-input")
    def _on_task_changed(self, event: TextArea.Changed) -> None:
        """Refresh the model suggestions under the caret."""
        self.query_one("#draft-suggest", SteerSuggest).show_text(
            model_suggestions(self._routes, event.text_area.text)
        )

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        """Start the session with the picked mode, preset and model."""
        if self._starting:
            return
        mode = str(self.query_one("#draft-mode", Select).value)
        preset = str(self.query_one("#draft-preset", Select).value)
        model = str(self.query_one("#draft-model", Select).value)
        self._starting = True
        self._notice(Text(f"starting the {mode}…", style="bold cyan"))
        self._start(self.app, mode, message.text, preset, model)

    @work(thread=True, exclusive=True)
    def _start(self, app: App[object], mode: str, task: str, preset: str, model: str) -> None:
        """Spawn the session detached and locate it, off the UI thread.

        The locate waits for the first event, which can take seconds. The app is
        passed in: a screen dismissed mid-spawn has no parent to reach it through.
        """
        session_dir, err = spawn_new_work(
            self.repo_cwd, mode, task, preset=preset, model=model, config_path=self.config_path
        )
        app.call_from_thread(self._started, session_dir, err, task)

    def _started(self, session_dir: Path | None, err: str, task: str) -> None:
        self._starting = False
        if session_dir is not None:
            self.app.exit(session_dir)
            return
        if not self.is_attached:  # left mid-spawn: no composer to hand the text back to
            self.app.notify(err or "could not start", severity="error", timeout=8.0)
            return
        # The refusal renders selectable, above the text it refused, back in the composer.
        self._notice(Text(err or "could not start", style="bold red"))
        with contextlib.suppress(NoMatches):
            bar = self.query_one("#draft-input", SteerInput)
            bar.load_text(task)
            bar.move_cursor(bar.document.end)
            bar.focus()

    def _notice(self, text: Text) -> None:
        self.query_one("#draft-notice", Static).update(text)
