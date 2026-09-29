# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The conversation view: a run's transcript, live-following and selectable.

The log folds through the shared `TranscriptFold` with the glyphs the CLI stream
uses. Completed turns scroll in the main pane; a docked live pane streams the turn
in progress, since a reasoning model can think for a minute before its first tool
call. The scrollback is `Static` chunks, not a `RichLog`: a `RichLog` renders as
line strips the text selection cannot extract.
"""

from __future__ import annotations

import bisect
import contextlib
import os
import pathlib
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, cast

from rich import text as rich_text
from textual import app, containers, geometry, screen, timer, widgets
from textual.css import query

from agent6.ui.tui import clipboard, composer, logview, menubar, screen_chrome, settings
from agent6.ui.tui import prompts as tui_prompts
from agent6.viewmodel import approval_parts, events, format, policy, transcript, transcript_style
from agent6.viewmodel import state as viewmodel_state
from agent6.viewmodel import tail as viewmodel_tail

if TYPE_CHECKING:
    from agent6.ui.tui import app as tui_app

_LIVE_TAIL = 1600  # chars of the in-progress turn kept in the live pane
# The sealed-chunk size: one Static re-wraps the whole transcript every poll (185ms at
# ~1800 lines), while only the tail chunk ever changes.
_CHUNK_LINES = 200

# The single detail shortcut cycles through these in order.
_DETAIL_CYCLE: dict[transcript_style.DetailLevel, transcript_style.DetailLevel] = {
    "hidden": "collapsed",
    "collapsed": "expanded",
    "expanded": "hidden",
}


def _tail(text: str, n: int) -> str:
    """Return the last n chars of the text, with an ellipsis when it was cut."""
    return text if len(text) <= n else "…" + text[-n:]


# Semantic style name to Rich style; the CLI has the sibling ANSI map over the same lines.
_STYLE_RICH: dict[transcript_style.StyleName, str] = {
    "thinking": "#6C7086",
    "think-marker": "blue",
    "text": "",
    "call": "bold cyan",
    "verify": "bold yellow",
    "arg": "dim",
    "ok": "green",
    "fail": "red",
    "detail": "dim",
    "more": "dim italic",
    "tail": "dim",
    "commit": "magenta",
    "marker": "dim italic",
    "done-ok": "bold green",
    "done-fail": "bold yellow",
    "done-neutral": "bold",
    "body": "",
    "done-detail": "dim",
    "operator": "bold green",
}


def _rich_line(line: transcript_style.Line) -> rich_text.Text:
    """Return one styled line of item_lines() as a Rich Text."""
    text = rich_text.Text()
    for chunk, style in line:
        text.append(chunk, style=_STYLE_RICH[style] or None)
    return text


def _item_renderables(
    item: transcript.TranscriptItem, *, detail: transcript_style.DetailLevel
) -> list[rich_text.Text]:
    """Return an item's lines as Rich Texts, with a blank line after the item."""
    lines = transcript_style.item_lines(item, detail=detail)
    if not lines:
        return []
    out = [_rich_line(line) for line in lines]
    out.append(rich_text.Text(""))
    return out


def empty_conversation_note(word: str, detail: str, *, ended: bool) -> str:
    """Return the empty-conversation placeholder, naming the state the session is in.

    A crashed run and a run that never started must not both read as an ordinary
    empty one.

    Args:
        word: The session's status word.
        detail: The status reason.
        ended: The session is known to have ended.
    """
    state, action = format.dead_run_note(word, detail)
    if state:
        return f"{state}: {action}" if action else state
    if ended:
        return "this session made no conversation"
    return "(no conversation yet; it appears as the session streams)"


class _ChromeStatic(widgets.Static):
    """A Static that never joins a text selection; only the transcript body is copyable."""

    ALLOW_SELECT = False


_JUMP_LABEL = "↓ bottom · Ctrl+End"


class _JumpButton(widgets.Static):
    """The floating jump-to-bottom pill, shown while the transcript is scrolled up."""

    ALLOW_SELECT = False
    DEFAULT_CSS = """
    _JumpButton {
        layer: dropdown; overlay: screen; constrain: none inside;
        display: none; width: auto; height: 1; padding: 0 1;
        background: $panel; color: $accent;
    }
    _JumpButton:hover { background: $primary 30%; }
    """

    def on_click(self) -> None:
        """Snap the screen back to the live tail."""
        handler = getattr(self.screen, "action_scroll_bottom", None)
        if callable(handler):
            handler()


class ConversationScreen(composer.ApprovalKeys, screen_chrome.ScreenChrome, screen.Screen[None]):
    """The run app's main screen: the transcript over a composer bar."""

    CSS = """
    ConversationScreen { background: $surface; }
    #conv-main { height: 1fr; }
    #conv-scroll { height: 1fr; }
    .conv-chunk { height: auto; padding: 0 1; pointer: text; }  /* selectable: I-beam */
    #conv-live {
        height: auto; max-height: 12; padding: 0 1;
        border-top: solid $border; background: $surface;
    }
    /* The composer: steer a live run, resume a finished one (see _sync_input). */
    #conv-input { display: none; }
    """

    # The composer owns plain letters and Enter, so these are modified keys, Esc, and priority.
    _VIEW_ITEMS: ClassVar = (
        menubar.MenuItem("Detail: hidden / collapsed / expanded", "cycle_detail", priority=True),
        *menubar.SCROLL_ITEMS,
        menubar.MenuItem("Reload the log", "reload"),
        menubar.MenuItem("Full log…", "view_logs"),
        menubar.MenuItem("Copy selection / all", "copy", priority=True),
        menubar.MenuItem("Copy via terminal", "suspend_copy"),
        menubar.MenuItem("Copy via pager", "pager"),
        menubar.MenuItem("Save transcript to file", "write_file"),
        menubar.MenuItem("Theme…", "choose_theme"),
        menubar.MenuItem("Copy method…", "choose_copy_method"),
    )
    _HELP_MENU: ClassVar = menubar.Menu(
        "Help",
        (
            menubar.MenuItem("Keys & actions", "help"),
            menubar.MenuItem("Command palette", "command_palette"),
        ),
    )
    MENUS: ClassVar = (
        menubar.Menu(
            "File",
            (
                menubar.MenuItem("Back", "close", priority=True),
                menubar.MenuItem("Quit", "quit_hub", priority=True),
            ),
        ),
        composer.RUN_MENU,
        menubar.Menu(
            "View",
            (*_VIEW_ITEMS, menubar.MenuItem("Dashboard…", "toggle_dashboard", priority=True)),
        ),
        _HELP_MENU,
    )
    # The same footer, in the same order, as the dashboard, plus the detail cycle.
    FOOTER: ClassVar = (
        ("toggle_dashboard", "Dashboard"),
        ("copy", "Copy"),
        ("history_search", "History"),
        ("cycle_detail", "Detail"),
        ("close", "Back"),
    )
    BINDINGS: ClassVar = [
        *menubar.menu_bindings("conversation", MENUS, footer=FOOTER),
        *composer.APPROVAL_KEY_BINDINGS,  # an open approval answers from any non-text focus
    ]
    COMMANDS: ClassVar = {screen_chrome.MenuCommands}
    HELP_TITLE: ClassVar = "agent6 — conversation"
    APPROVAL_DOCK_BEFORE: ClassVar = "#conv-suggest"
    APPROVAL_FOCUS_AFTER: ClassVar = "#conv-scroll"  # the transcript, where the command is
    APPROVAL_ROW_SHOWS_PROMPT: ClassVar = False  # the transcript item carries it
    HELP_HINTS: ClassVar = (
        "Steer bar: Enter sends the instruction",
        "Ctrl-J or Shift+Enter inserts a newline",
    )

    def __init__(
        self,
        logs_path: pathlib.Path,
        *,
        title: Callable[[str], str],
        presets: list[str] | None = None,
        routes: list[str] | None = None,
        prompts: tui_prompts.PromptDispatcher | None = None,
    ) -> None:
        """Build the screen over a log.

        Args:
            logs_path: The session's log.
            title: The menu-bar subtitle for a view word, called at stamp time.
            presets: The config presets a resume may continue under.
            routes: The `provider/model` routes a resume may continue under.
            prompts: The host's dispatcher, the one record of which prompts a surface took.
        """
        super().__init__()
        self._logs_path = logs_path
        self._prompts = prompts
        self._title = title
        self._presets = presets if presets is not None else []
        self._routes = routes if routes is not None else []
        self._detail: transcript_style.DetailLevel = (
            "collapsed"  # one shortcut cycles hidden/collapsed/expanded
        )
        self._tail = viewmodel_tail.LogTail(logs_path)
        self._fold = transcript.TranscriptFold()
        self._content = rich_text.Text()  # the whole transcript, for copy and anchors
        self._item_starts: list[int] = []  # the logical start line of each rendered item
        self._content_lines = 0
        # The unsealed tail: the only widget content the live appends re-render.
        self._tail_text = rich_text.Text()
        self._tail_lines = 0
        # Tool calls in flight by call_id; the live pane shows them until the settled item lands.
        self._pending: dict[str, transcript.TranscriptItem] = {}
        self._approval: tuple[str, str, bool] | None = None  # (id, prompt, standing)
        self._approval_done: str | None = None
        self._live_think: list[str] = []
        self._live_text: list[str] = []
        self._spin = 0  # live-pane spinner tick, advanced by _poll
        self._timer: timer.Timer | None = None  # the 0.3s poll; paused while covered

    @property
    def _host(self) -> tui_app.Agent6TUI:
        """The app that mounts this screen; its fold and dir status are the one record."""
        return cast("tui_app.Agent6TUI", self.app)

    def compose(self) -> app.ComposeResult:
        """Yield the menu bar, the transcript with its live pane, the composer and the footer."""
        yield menubar.MenuBar(self.MENUS)
        with containers.Vertical(id="conv-main"):
            with containers.VerticalScroll(id="conv-scroll"):
                # Sealed chunks mount above this tail as the log grows; only the tail re-renders.
                yield widgets.Static(id="conv-tail", classes="conv-chunk")
            yield _ChromeStatic("", id="conv-live")
        yield widgets.Static(id="conv-approval", classes="conv-chunk")  # the open approval, inline
        yield composer.SteerSuggest(id="conv-suggest")  # command hints while typing `/…`
        yield composer.ResumeOptions(
            self._presets, self._routes, id="conv-resume"
        )  # while resuming
        yield composer.SteerInput(id="conv-input")
        yield _JumpButton(_JUMP_LABEL, id="conv-jump")
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Stamp the title, fold the log and start the poll."""
        self.app.sub_title = self._title("conversation")
        self._reload()
        self._timer = self.set_interval(0.3, self._poll)
        # The jump pill follows the scroll position; the poll calls _sync_jump on growth.
        self.watch(self._scroll(), "scroll_y", self._sync_jump, init=False)

    def _sync_jump(self, *_: object) -> None:
        """Show the jump-to-bottom pill while scrolled up, pinned to the scroll area's corner."""
        with contextlib.suppress(query.NoMatches):
            jump = self.query_one("#conv-jump", _JumpButton)
            scroll = self._scroll()
            show = scroll.max_scroll_y > 0 and not self._at_bottom(scroll)
            if jump.display != show:  # a same-value write still costs a relayout
                jump.display = show
            if show:
                region = scroll.region
                width = len(_JUMP_LABEL) + 2  # + the 1-cell padding each side
                jump.absolute_offset = geometry.Offset(
                    max(region.x, region.right - width - 2), max(region.y, region.bottom - 2)
                )

    def on_screen_suspend(self) -> None:
        """Stop polling while another screen covers this one."""
        if self._timer is not None:
            self._timer.pause()

    def on_screen_resume(self) -> None:
        """Re-stamp the title, catch up on events that landed while covered, resume the poll."""
        self.app.sub_title = self._title("conversation")
        if self._timer is not None:
            self._poll()
            self._timer.resume()

    def _scroll(self) -> containers.VerticalScroll:
        """Return the transcript's scroll container."""
        return self.query_one("#conv-scroll", containers.VerticalScroll)

    def _append(self, item: transcript.TranscriptItem) -> bool:
        """Append a settled item to the transcript and the tail; a call in flight waits.

        Args:
            item: The item the fold produced.

        Returns:
            Whether any line was written.
        """
        if item.kind == "tool":
            if item.ok is None:
                self._pending[item.call_id] = item
                return False
            self._pending.pop(item.call_id, None)
        self._item_starts.append(self._content_lines)
        wrote = False
        for line in _item_renderables(item, detail=self._detail):
            self._content.append_text(line)
            self._content.append("\n")
            self._content_lines += 1
            self._tail_text.append_text(line)
            self._tail_text.append("\n")
            self._tail_lines += 1
            wrote = True
        return wrote

    def _tail_widget(self) -> widgets.Static:
        """Return the unsealed tail chunk's widget."""
        return self.query_one("#conv-tail", widgets.Static)

    def _flush_tail(self) -> None:
        """Push the tail to its widget, sealing it into a chunk above once it is big enough."""
        tail = self._tail_widget()
        tail.update(self._tail_text)
        if self._tail_lines >= _CHUNK_LINES:
            sealed = widgets.Static(self._tail_text, classes="conv-chunk conv-sealed")
            self._scroll().mount(sealed, before=tail)
            self._tail_text = rich_text.Text()
            self._tail_lines = 0
            tail.update(self._tail_text)

    def _track_event(self, event: dict[str, object]) -> None:
        """Keep what this screen tracks beyond the fold: the open approval and the live buffers.

        The approval as last rendered here keeps an answer given here from being
        re-offered on a reload before the worker journals it.
        """
        etype = event.get("type")
        if etype in events.SESSION_START_EVENTS:
            # An unanswered approval belongs to the execution that ended.
            self._approval = None
            self._approval_done = None
        if etype == "approval.prompt":
            self._approval = (
                str(event.get("id", "")),
                str(event.get("prompt", "")),
                bool(event.get("standing", True)),
            )
            self._approval_done = None
        elif etype == "approval.answer":
            if self._approval is not None and self._approval[0] == str(event.get("id", "")):
                self._note_answered("allowed" if event.get("approved") else "denied")
        if etype in ("role.call", "role.result"):
            self._live_think.clear()
            self._live_text.clear()
            self._approval_done = None
        elif etype == "role.thinking_delta":
            self._live_think.append(str(event.get("text", "")))
        elif etype == "role.text_delta":
            self._live_text.append(str(event.get("text", "")))

    def _open_approval(self) -> viewmodel_state.ApprovalPrompt | None:
        """Return the approval awaiting an answer, from the host's fold."""
        state = self._host.state
        if self._approval is not None:
            aid = self._approval[0]
            answered = next((a for a in state.pending_approvals if a.id == aid), None)
            if answered is not None and answered.answered:
                self._note_answered("allowed" if answered.approved else "denied")
        return self.open_approval(state)

    def approval_dir(self) -> pathlib.Path:
        """Return the session dir the answer file is written under."""
        return self._logs_path.parent

    def approval_live(self) -> bool:
        """Return whether the run can read an answer."""
        return self._host_live()

    def approval_answered(self, verdict: str) -> None:
        """Collapse the answered approval and repaint it."""
        self._note_answered(verdict)
        self._render_approval()

    def _note_answered(self, verdict: str) -> None:
        """Collapse the open approval to one dim line: the verdict and the command it judged."""
        if self._approval is None:
            return
        head, payload = approval_parts(self._approval[1])
        self._approval_done = f"{verdict} · {(payload or head).splitlines()[0][:60]}"
        self._approval = None

    def _render_approval(self) -> None:
        """Render the open approval as a tail item with the answer row docked above the composer.

        After the answer the item collapses to one dim line until the next model turn.
        """
        item = self.query_one("#conv-approval", widgets.Static)
        current = self._open_approval()
        live = self._host_live()
        self.sync_approval(current if live else None)
        if current is not None:
            self._approval = (current.id, current.prompt, current.standing)
            # On a dead run the fact stays visible; the key row, whose answer reaches nothing, goes.
            note = "approval needed" if live else "approval pending when the run ended"
            item.update(composer.approval_text(current.prompt, note, dim=not live))
            item.display = True
            return
        if self._approval is not None:
            return  # answered here before the worker journaled it: as rendered
        if self._approval_done:
            item.update(rich_text.Text(f"? {self._approval_done}", style="dim"))
            item.display = True
        else:
            item.display = False

    def _render_live(self) -> None:
        """Repaint the live pane: the approval, the calls in flight, or the streaming turn."""
        self._render_approval()
        live = self.query_one("#conv-live", widgets.Static)
        if not self._host_live():
            # A killed worker's deltas sit in the buffers forever; the pane would say thinking.
            live.display = False
            self._settle_dead()
            return
        if self._host_waiting():
            # Blocked on the operator: a thinking pulse would lie under the very prompt asking.
            live.display = True
            live.update(
                rich_text.Text(
                    format.status_label(*self._host.dir_status),
                    style="bold yellow",
                )
            )
            return
        call_in_flight = self._host.model_call_in_flight()
        if not call_in_flight:
            rows = [
                ln
                for it in self._pending.values()
                for ln in transcript_style.item_lines(it, detail=self._detail)
            ]
            if not rows:
                live.display = False
                return
            body = rich_text.Text()
            for i, row in enumerate(rows):
                if i:
                    body.append("\n  ")
                body.append_text(_rich_line(row))
            live.display = True
            live.update(body)
            return
        think = "".join(self._live_think).strip()
        text = "".join(self._live_text).strip()
        body = rich_text.Text()
        if not think and not text:
            body.append(f"{format.spinner_frame(self._spin)} working… ", style="bold cyan")
        if think:
            # The thinking indicator always shows; the reasoning streams only when expanded.
            body.append(f"{format.spinner_frame(self._spin)} thinking… ", style="bold cyan")
            if self._detail == "expanded":
                body.append(_tail(think, _LIVE_TAIL), style="#6C7086")
        if text:
            if think:
                body.append("\n\n")
            body.append(_tail(text, _LIVE_TAIL))
        live.display = True
        live.update(body)

    def _settle_dead(self) -> None:
        """Settle the calls a dead worker left in flight into the scrollback."""
        if not self._pending:
            return
        wrote = False
        for item in self._fold.settle_open_calls("the run died"):
            wrote = self._append(item) or wrote
        if wrote:
            self._flush_tail()

    def _reload(self) -> None:
        """Re-read the whole log from scratch: on mount, a reload, a detail cycle."""
        self._tail = viewmodel_tail.LogTail(self._logs_path)
        self._fold = transcript.TranscriptFold()
        self._content = rich_text.Text()
        self._item_starts = []
        self._content_lines = 0
        self._tail_text = rich_text.Text()
        self._tail_lines = 0
        self._pending = {}
        self.query(".conv-sealed").remove()
        self._live_think.clear()
        self._live_text.clear()
        wrote = False
        items = 0
        for event in self._tail.read():
            self._track_event(event)
            for item in self._fold.feed(event):
                items += 1
                wrote = self._append(item) or wrote
        if wrote:
            self._flush_tail()
        elif items and self._detail == "hidden":
            # Reasoning and tool calls alone render no line at this level; the empty placeholder
            # would lie over them.
            note = "(reasoning and tool calls are hidden at this detail level; Ctrl+T shows them)"
            self._tail_widget().update(rich_text.Text(note, style="dim italic"))
        else:
            # Past tense only when the host positively knows the session ended.
            ended = not self._host.session_controllable()
            word, detail = self._host.dir_status
            empty = empty_conversation_note(word, detail, ended=ended)
            self._tail_widget().update(rich_text.Text(empty, style="dim italic"))
        self._render_live()
        self._sync_input()
        self._scroll().scroll_end(animate=False)
        self._focus_default()

    def _at_bottom(self, scroll: containers.VerticalScroll) -> bool:
        """Return whether the view follows the log: at the bottom within a one-line nudge."""
        return scroll.max_scroll_y - scroll.scroll_y <= 2.0

    def _poll(self) -> None:
        """Append the newly completed turns, keeping the bottom, and refresh the live pane."""
        if self._host.model_call_in_flight():
            self._spin += 1
        self._render_approval()
        new_events = self._tail.read()
        if not new_events:
            # The host's fold may have advanced; the composer must agree with it within one poll.
            self._sync_input()
            # The spinner is the only sign of life between events; a repaint can resize the
            # viewport, so the follow is re-pinned as on the data path.
            if self._host_live():
                scroll = self._scroll()
                following = self._at_bottom(scroll)
                self._render_live()
                if following:
                    scroll.scroll_end(animate=False)
            return
        scroll = self._scroll()
        following = self._at_bottom(scroll)
        wrote = False
        for event in new_events:
            self._track_event(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        if wrote:
            self._flush_tail()
        self._render_live()
        self._sync_input()
        # Re-pin after the live pane and the bar resized: growing them nudges off the bottom.
        if following:
            scroll.scroll_end(animate=False)
        self._sync_jump()

    def _host_waiting(self) -> bool:
        """Return whether the run is blocked on the operator, per the host's dir status."""
        return self._host.dir_status[0] == "waiting"

    def _host_live(self) -> bool:
        """Return whether the run is live per the host's dir status, which knows a dead worker."""
        return self._host.session_controllable()

    def _sync_input(self) -> None:
        """Show the composer bar in the run's mode, steer when live and resume when finished."""
        with contextlib.suppress(query.NoMatches):
            bar = self.query_one("#conv-input", composer.SteerInput)
            if not bar.display:  # a same-value write still costs a relayout
                bar.display = True
            mode: composer.ComposerMode = "steer" if self._host_live() else "resume"
            self.query_one("#conv-resume", composer.ResumeOptions).show(mode == "resume")
            if not bar.policy:  # the manifest does not change mid-run
                bar.policy = policy.session_policy(self._logs_path.parent).short()
            bar.set_mode(
                mode=mode,
                ctx_pct=self._host.context_pct(),
                continue_as=self._host.continue_as,
                needs_new_work=self._host.finished_green(),
            )

    def refresh_liveness(self) -> None:
        """Relabel the composer for a liveness change with no event, covered or not."""
        self._sync_input()
        with contextlib.suppress(query.NoMatches):
            self._render_live()

    def focus_bar(self) -> None:
        """Focus the composer bar."""
        with contextlib.suppress(query.NoMatches):
            self.query_one("#conv-input", composer.SteerInput).focus()

    def _focus_default(self) -> None:
        """Focus the composer bar, or the scrollback when there is none."""
        with contextlib.suppress(query.NoMatches):
            self.query_one("#conv-input", composer.SteerInput).focus()
            return
        self._scroll().focus()

    def on_steer_input_submitted(self, message: composer.SteerInput.Submitted) -> None:
        """Hand a composer line to the host, which routes it by the run's state."""
        self._host.submit_instruction(message.text)

    def action_history_search(self) -> None:
        """Open the prompt history search over the composer."""
        composer.open_history_search(
            self, self.query_one("#conv-input", composer.SteerInput), self._logs_path
        )

    def on_text_area_changed(self, event: widgets.TextArea.Changed) -> None:
        """Refresh the command hints as the composer's text changes."""
        if event.text_area.id != "conv-input":
            return
        with contextlib.suppress(query.NoMatches):
            self.query_one("#conv-suggest", composer.SteerSuggest).show_for(
                event.text_area.text, mode="steer" if self._host_live() else "resume"
            )

    def _emit(self, seq: str) -> None:
        """Write a raw terminal escape (an OSC 52 clipboard set) through the driver."""
        driver = self.app._driver  # pyright: ignore[reportPrivateUsage]
        if driver is not None:
            driver.write(seq)

    def _selected_or_all(self) -> tuple[str, str]:
        """Return the body selection or the whole transcript, with the word naming which."""
        body_selection = self._body_selection()
        if body_selection and body_selection.strip():
            return body_selection, "selection"
        return self._content.plain, "whole transcript"

    def _body_selection(self) -> str | None:
        """Return the selected text from the transcript body's chunks only, in order."""
        parts: list[str] = []
        for chunk in self.query(".conv-chunk"):
            selection = self.selections.get(chunk)
            if selection is None:
                continue
            grabbed = chunk.get_selection(selection)
            if grabbed is not None:
                parts.append(grabbed[0])
        return "\n".join(parts) if parts else None

    def get_selected_text(self) -> str | None:
        """Return the selection for textual's copy, restricted to the transcript body."""
        return self._body_selection()

    def _copy_text(self, text: str, *, method: str) -> str:
        """Return a short status after copying the text by the resolved method."""
        return clipboard.emit_clipboard(text, clipboard.resolve_method(method), self._emit)

    def action_copy(self) -> None:
        """Copy the selection or the whole transcript by the configured method."""
        text, what = self._selected_or_all()
        if not text:
            self.notify("nothing to copy yet")
            return
        try:
            status = self._copy_text(text, method=settings.get_copy_method())
        except (OSError, subprocess.CalledProcessError) as exc:
            self.notify(f"copy failed: {exc}", severity="error")
            return
        self.notify(f"copied {what} ({status})")

    def action_write_file(self) -> None:
        """Write the whole transcript to a file."""
        path = clipboard.write_transcript_file(self._content.plain)
        self.notify(f"wrote transcript to {path}")

    def action_suspend_copy(self) -> None:
        """Drop to the terminal and print the text to select and copy there; Enter returns."""
        text, what = self._selected_or_all()
        with self.app.suspend():
            print(f"\n===== COPY BELOW ({what}): select and copy in your terminal =====\n")
            print(text)
            print("\n===== END (press Enter to return) =====")
            with contextlib.suppress(EOFError):
                input()

    def action_pager(self) -> None:
        """Open the text in $PAGER."""
        text, _ = self._selected_or_all()
        pager = os.environ.get("PAGER") or "less"
        cmd = [pager, "-R"] if pathlib.Path(pager).name.startswith("less") else [pager]
        with self.app.suspend():
            try:
                subprocess.run(cmd, input=text, text=True, check=False)
            except OSError as exc:
                print(f"pager {pager!r} failed: {exc}\nPress Enter to return.")
                with contextlib.suppress(EOFError):
                    input()

    def action_reload(self) -> None:
        """Re-read the log."""
        self._reload()

    def action_view_logs(self) -> None:
        """Open the run's raw event log."""
        self.app.push_screen(logview.LogScreen(self._logs_path, title=lambda: self._title("logs")))

    def action_cycle_detail(self) -> None:
        """Cycle the detail level, keeping the block at the top of the viewport anchored."""
        self._reload_keeping_place(lambda: setattr(self, "_detail", _DETAIL_CYCLE[self._detail]))

    def _item_visual_starts(self) -> list[int]:
        """Return the wrapped row where each rendered item begins, at the current body width."""
        width = max(1, self._tail_widget().content_size.width)
        starts: list[int] = []
        visual = 0
        nxt = 0
        for logical, line in enumerate(self._content.split("\n")):
            while nxt < len(self._item_starts) and self._item_starts[nxt] == logical:
                starts.append(visual)
                nxt += 1
            visual += max(1, -(-line.cell_len // width))
        starts.extend([visual] * (len(self._item_starts) - nxt))
        return starts

    def _reload_keeping_place(self, flip: Callable[[], None]) -> None:
        """Apply the flip, re-render and restore the reading position.

        Pinned to the bottom when following, else anchored to the block at the top of
        the viewport, at the same offset, so a block expanding above keeps the place.

        Args:
            flip: The change to apply before the re-render.
        """
        scroll = self._scroll()
        following = self._at_bottom(scroll)
        top = scroll.scroll_y
        old_visual = self._item_visual_starts()
        anchor = bisect.bisect_right(old_visual, top) - 1 if old_visual else -1
        offset = top - old_visual[anchor] if 0 <= anchor < len(old_visual) else 0.0
        flip()
        self._reload()
        if following or not (0 <= anchor < len(self._item_starts)):
            return
        self._scroll().scroll_to(y=self._item_visual_starts()[anchor] + offset, animate=False)

    def action_scroll_top(self) -> None:
        """Scroll to the top."""
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll to the live tail."""
        self._scroll().scroll_end(animate=False)

    def action_page_up(self) -> None:
        """Scroll one page up; instant, since animation reads as lag."""
        self._scroll().scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        """Scroll one page down."""
        self._scroll().scroll_page_down(animate=False)

    def action_close(self) -> None:
        """Close an open list, else leave the run view through the host."""
        if self.close_open_list():
            return
        self._host.action_to_hub()

    def action_quit_hub(self) -> None:
        """Leave the view and the hub through the host."""
        self._host.action_quit_hub()

    def action_toggle_dashboard(self) -> None:
        """Flip to the dashboard through the host."""
        self._host.action_toggle_dashboard()
