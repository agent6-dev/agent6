# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The composer every conversation surface shares.

The steer input and its mode labels, the slash-command suggestions, the resume
picker row, the history search, and the inline approval row.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input, Select, Static, TextArea

from agent6.directive import LIVE_RUN_COMMANDS, STEER_COMMANDS
from agent6.sessions.ipc import ANSWERED_ELSEWHERE, write_answer
from agent6.ui.keymap import (
    APPROVAL_ANSWERS,
    answer_entry,
)
from agent6.ui.tui.menubar import (
    Menu,
    MenuItem,
)
from agent6.ui.tui.modals import HistorySearchModal
from agent6.ui.tui.widgets import Picker, PickerRow
from agent6.viewmodel import approval_parts
from agent6.viewmodel.state import ApprovalPrompt, SessionState, open_approval_of
from agent6.viewmodel.tail import tail_events
from agent6.viewmodel.transcript import (
    operator_inputs,
)

if TYPE_CHECKING:
    from agent6.ui.tui.prompts import PromptDispatcher

ComposerMode = Literal["steer", "resume", "start", "draft"]


def composer_labels(
    mode: ComposerMode, *, continue_as: str = "", needs_new_work: bool = False
) -> tuple[str, str]:
    """Return the composer's border title and key hint for a mode.

    One conversation view serves runs, plans and asks, so it says "session".

    Args:
        mode: The composer's mode.
        continue_as: The fork an undone run continues as; Enter resumes that session.
        needs_new_work: The agent finished green, so a bare resume has nothing to do.
    """
    if mode == "steer":
        return ("steer this session (/pin, /compact [focus])", "Enter sends · Ctrl-J newline")
    if mode == "draft":
        return ("describe the machine", "Enter drafts it · Ctrl-J newline · Esc cancels")
    if mode == "resume":
        if continue_as:
            title = f"continue as {continue_as}"
        else:
            title = "what should it do next" if needs_new_work else "continue this session"
        return (title, "Enter resumes · Ctrl-J newline")
    return ("new session", "Enter starts · Ctrl-J newline")


def steer_suggestion_rows(text: str, *, mode: ComposerMode) -> list[tuple[str, str]]:
    """Return the (command, help) rows matching a `/…` first word still being typed.

    A resume composer withholds the live-run commands, a start offers only
    /parallel, and a draft offers none.

    Args:
        text: The composer's text.
        mode: The composer's mode.
    """
    if not text.startswith("/") or any(ch.isspace() for ch in text):
        return []
    if mode == "draft":
        return []
    if mode == "start":
        offered = {c: h for c, h in STEER_COMMANDS.items() if c == "/parallel"}
    elif mode == "resume":
        offered = {c: h for c, h in STEER_COMMANDS.items() if c not in LIVE_RUN_COMMANDS}
    else:
        offered = STEER_COMMANDS
    return [(c, h) for c, h in offered.items() if c.startswith(text)]


def complete_steer(text: str, *, mode: ComposerMode) -> str | None:
    """Return the Tab completion of a command word, or None when Tab keeps its focus meaning.

    A unique match completes with a trailing space; several advance to their common
    prefix, or return the text unchanged so Tab never moves the focus mid-command.

    Args:
        text: The composer's text.
        mode: The composer's mode.
    """
    rows = steer_suggestion_rows(text, mode=mode)
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0][0] + " "
    lcp = os.path.commonprefix([c for c, _h in rows])
    return lcp if len(lcp) > len(text) else text


class SteerSuggest(Static):
    """The command hints above a composer: one row per matching directive, else hidden."""

    ALLOW_SELECT = False
    DEFAULT_CSS = """
    SteerSuggest { display: none; height: auto; padding: 0 1; background: $surface; }
    """

    def show_for(self, text: str, *, mode: ComposerMode) -> None:
        """Show the rows matching the composer's text, or hide the line."""
        rows = steer_suggestion_rows(text, mode=mode)
        body: Text | None = None
        if rows:
            body = Text()
            for i, (cmd, help_) in enumerate(rows):
                if i:
                    body.append("\n")
                body.append(cmd, style="bold")
                body.append(f"  {help_}", style="dim")
        self.show_text(body)

    def show_text(self, body: Text | None) -> None:
        """Show the body as the hint line, or hide the line for None."""
        if body is not None:
            self.update(body)
        show = body is not None
        if self.display != show:
            self.display = show


_INPUT_MAX_ROWS = 6  # the steer bar grows to this many rows, then scrolls internally


class ResumeHost(Protocol):
    """What a resume row reads and writes on its host app."""

    resume_preset: str
    resume_model: str

    def resume_defaults(self, preset: str) -> tuple[str, str]:
        """Return the no-flag labels for the preset and the model under a preset."""
        ...


class ResumeOptions(PickerRow):
    """The row above a resume composer: the preset and the model the next execution runs under.

    Shown only while the composer resumes; the picks live on the host app, so both
    run views agree. Each first entry adds no flag and names what the resume runs
    under, relabelled when the row reappears and when the preset pick changes.
    """

    DEFAULT_CSS = """
    ResumeOptions { display: none; padding: 0 1; }
    """

    def __init__(self, presets: list[str], routes: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._presets = presets
        self._routes = routes
        self._labels = ("", "")
        self._labelled: str | None = None  # the preset pick the labels name; None when stale

    def compose(self) -> ComposeResult:
        """Yield the two labelled pickers."""
        host = self._host()
        self._labels, self._labelled = host.resume_defaults(host.resume_preset), host.resume_preset
        yield Static("continue under preset", classes="picker-label")
        yield Picker(self._options(0), value="", allow_blank=False, id="resume-preset")
        yield Static("model", classes="picker-label")
        yield Picker(self._options(1), value="", allow_blank=False, id="resume-model")

    def _host(self) -> ResumeHost:
        """Return the app the picks live on."""
        return cast(ResumeHost, self.app)

    def _options(self, index: int) -> list[tuple[str, str]]:
        """Return a picker's options: the no-flag label, then the choices."""
        choices = self._routes if index else self._presets
        return [(self._labels[index], ""), *((c, c) for c in choices)]

    def on_select_changed(self, event: Select.Changed) -> None:
        """Write a pick to the host; a preset pick relabels the model."""
        host, value = self._host(), str(event.value)
        if event.select.id == "resume-model":
            host.resume_model = value
        else:
            host.resume_preset = value
            self._relabel()

    def show(self, shown: bool) -> None:
        """Show or hide the row; shown, it relabels and syncs after the refresh."""
        if self.display != shown:
            self.display = shown
        if not shown:  # a running execution may pin a preset or a model
            self._labelled = None
        else:
            # After the refresh: a value written before the Selects mount leaves a label blank.
            if self._labelled != self._host().resume_preset:
                self.call_after_refresh(self._relabel)
            self.call_after_refresh(self._sync)

    def _relabel(self) -> None:
        """Rename the no-flag entries for the host's preset pick."""
        host = self._host()
        if self._labelled == host.resume_preset:
            return
        old = self._labels
        self._labels, self._labelled = host.resume_defaults(host.resume_preset), host.resume_preset
        with self.prevent(Select.Changed):  # a relabel is no pick
            for index, picker_id in enumerate(("#resume-preset", "#resume-model")):
                if self._labels[index] != old[index]:
                    self.query_one(picker_id, Select).set_options(self._options(index))
        self._sync()

    def _sync(self) -> None:
        """Move the pickers to the host's picks."""
        host = self._host()
        with self.prevent(Select.Changed):
            for wanted, picker_id, choices in (
                (host.resume_preset, "#resume-preset", self._presets),
                (host.resume_model, "#resume-model", self._routes),
            ):
                picker = self.query_one(picker_id, Select)
                if picker.value != wanted and wanted in ("", *choices):
                    picker.value = wanted


# The run-control menu both run views share; every action resolves on the app.
RUN_MENU = Menu(
    "Run",
    (
        MenuItem("Search past messages…", "history_search", priority=True),
        MenuItem("Compact context now", "compact"),
        MenuItem("Stop after this step", "stop_step"),
        MenuItem("Stop now", "stop_now"),
        MenuItem("Resume this session", "resume"),
        MenuItem("Run this plan", "run_plan"),
        MenuItem("Fork this session", "fork"),
        MenuItem("Review this run…", "review_run"),
        MenuItem("Delete this session…", "delete_session"),
    ),
)


# The row's colour per answer; the keymap owns the keys and the words.
_ANSWER_STYLES: dict[str, str] = {
    "yes": "bold green",
    "session": "green",
    "no": "bold red",
    "session-deny": "red",
}


# A run view lists these in its own BINDINGS; textual takes bindings only from DOM classes.
APPROVAL_KEY_BINDINGS: tuple[Binding, ...] = tuple(
    Binding(entry.key, f"answer('{entry.answer}')", entry.label, show=False)
    for entry in APPROVAL_ANSWERS
)


class ApprovalKeys:
    """The approval row's lifecycle, mixed into a session view before its Screen base.

    The view lists APPROVAL_KEY_BINDINGS in its BINDINGS. The open approval docks as
    an `ApprovalRow` before the widget `APPROVAL_DOCK_BEFORE` names; its letters
    answer from any focus but a text field, and each answer is a tab stop where
    Enter answers. The host supplies `approval_dir` and `approval_live`.

    Attributes:
        APPROVAL_DOCK_BEFORE: The selector the row mounts before.
        APPROVAL_FOCUS_AFTER: Where the focus goes after an answer from the row; "" keeps it.
        APPROVAL_ROW_SHOWS_PROMPT: The row carries the command when the screen shows it nowhere.
        APPROVAL_LOST: The notice for an answer to a run that no longer takes one.
        APPROVAL_ROW_HINT: The row's hint on reaching the keys.
    """

    APPROVAL_DOCK_BEFORE: ClassVar[str] = ""
    APPROVAL_FOCUS_AFTER: ClassVar[str] = ""
    APPROVAL_ROW_SHOWS_PROMPT: ClassVar[bool] = True
    APPROVAL_LOST: ClassVar[str] = "the run is gone: the answer reached nothing"
    APPROVAL_ROW_HINT: ClassVar[str] = "(Tab out of the bar for the keys; or click)"

    _prompts: PromptDispatcher | None = None
    _row: ApprovalRow | None = None
    _row_id: str = ""
    _answered_from_row: bool = False  # the focus stayed on the approval

    def approval_dir(self) -> Path:
        """Return the session dir an answer is written to."""
        raise NotImplementedError

    def approval_live(self) -> bool:
        """Return whether the run still takes an answer, read at the answer."""
        raise NotImplementedError

    def approval_answered(self, verdict: str) -> None:
        """Follow up a written answer: "allowed", "denied" or "answered elsewhere"."""

    def open_approval(self, state: SessionState) -> ApprovalPrompt | None:
        """Return the oldest unanswered approval this view has not answered already."""
        prompts = self._prompts
        if prompts is None:
            return open_approval_of(state)
        session_dir = self.approval_dir()
        return open_approval_of(state, taken=lambda aid: prompts.seen(session_dir, aid))

    def sync_approval(self, current: ApprovalPrompt | None) -> None:
        """Dock one row for the open approval, or none.

        A new id gets a fresh row, since a resumed execution reuses prompt ids and the
        old row may still be unmounting. After an answer from the row, the focus goes
        to the next row too.

        Args:
            current: The open approval; None for none, or a run that takes no answer.
        """
        screen = cast(Screen[Any], self)
        if current is None:
            if self._row is not None:
                self._row.remove()
                self._row, self._row_id = None, ""
            return
        if self._row is not None and self._row_id == current.id:
            return
        if self._row is not None:
            self._row.remove()
        prompt = current.prompt if self.APPROVAL_ROW_SHOWS_PROMPT else ""
        self._row = ApprovalRow(
            standing=current.standing, prompt=prompt, hint=self.APPROVAL_ROW_HINT
        )
        self._row_id = current.id
        screen.mount(self._row, before=screen.query_one(self.APPROVAL_DOCK_BEFORE))
        if self._answered_from_row:
            self._row.call_after_refresh(self._row.focus_answers)

    def on_approval_row_answered(self, message: ApprovalRow.Answered) -> None:
        """Deliver an answer from a label's click or its key."""
        if self._row is None:
            return
        # Answered from the row, the focus stays out of the composer for the next approval.
        self._answered_from_row = self._row.holds_focus()
        screen = cast(Screen[Any], self)
        verdict = deliver_answer(
            screen,
            session_dir=self.approval_dir(),
            prompt_id=self._row_id,
            answer=message.answer,
            prompts=self._prompts,
            live=self.approval_live(),
            lost=self.APPROVAL_LOST,
        )
        if not verdict:
            return
        self.approval_answered(verdict)
        if self._answered_from_row and self.APPROVAL_FOCUS_AFTER:
            with contextlib.suppress(NoMatches):
                screen.query_one(self.APPROVAL_FOCUS_AFTER).focus()

    def action_answer(self, answer: str) -> None:
        """Answer the open approval from a key."""
        cast(Screen[Any], self).post_message(ApprovalRow.Answered(answer))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Return whether an answer key is live: a row offers it and no text field has focus."""
        if action != "answer":
            return True
        screen = cast(Screen[Any], self)
        rows = screen.query(ApprovalRow)
        if not rows or isinstance(screen.focused, TextArea | Input):
            return False  # a text field keeps its letters
        return rows.first().offers(str(parameters[0]))


class SteerInput(TextArea):
    """The composer bar: submits on Enter, Ctrl+J inserts a newline, grows with its content.

    Its mode says what Enter does: steer a live run, resume a finished one, start
    or draft. An open approval's letters never answer while the focus is here.
    """

    ALLOW_MAXIMIZE = False

    # One style for every screen's composer; a screen adds only its placement.
    DEFAULT_CSS = f"""
    SteerInput {{
        height: auto; max-height: {_INPUT_MAX_ROWS + 2};
        border: round $primary; background: $surface;
    }}
    SteerInput:focus {{ border: round $accent; }}
    """

    BINDINGS: ClassVar = [
        # TextArea's own undo; ctrl+z is the app's Detach.
        Binding("ctrl+underscore", "undo", "Undo", show=False),
    ]

    class Submitted(Message):
        """A line the operator sent."""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_mount(self) -> None:
        """Label for the mode and size to the content."""
        self.set_mode(mode=self.mode)
        self._resize()

    policy = ""  # the session policy's short form, set once the run dir is known
    mode: ComposerMode = "steer"

    def set_mode(
        self,
        *,
        mode: ComposerMode,
        ctx_pct: int | None = None,
        continue_as: str = "",
        needs_new_work: bool = False,
    ) -> None:
        """Relabel the bar for the session's state; a same-value write still costs a refresh.

        Args:
            mode: What Enter does.
            ctx_pct: The context-window fill, when known.
            continue_as: The fork an undone run resumes as.
            needs_new_work: The run finished green, so a bare resume has nothing to do.
        """
        self.mode = mode
        title, keys = composer_labels(mode, continue_as=continue_as, needs_new_work=needs_new_work)
        ctx = f"ctx {ctx_pct}% · " if ctx_pct is not None else ""
        policy = f"{self.policy} · " if self.policy else ""
        # Border titles are markup: a bracket in a label or a model id would vanish unescaped.
        title = escape(title)
        subtitle = escape(f"{policy}{ctx}{keys}")
        if self.border_title != title:
            self.border_title = title
        if self.border_subtitle != subtitle:
            self.border_subtitle = subtitle

    def on_key(self, event: events.Key) -> None:
        """Submit on Enter, insert a newline on Ctrl+J, complete a command on Tab."""
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
        elif event.key == "tab":
            completed = complete_steer(self.text, mode=self.mode)
            if completed is not None:
                event.prevent_default()
                event.stop()
                if completed != self.text:
                    self.load_text(completed)
                    self.move_cursor(self.document.end)

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        """Grow or shrink with the content."""
        self._resize()

    def _resize(self) -> None:
        """Set the height to the line count plus the border, only on a real change."""
        rows = min(max(self.document.line_count, 1), _INPUT_MAX_ROWS)
        height = rows + 2
        current = self.styles.height
        if current is None or current.value != height:
            self.styles.height = height


def open_history_search(screen: Screen[Any], field: SteerInput, logs_path: Path) -> None:
    """Pick one of the session's past messages into the composer for editing.

    The task, then every steer, read from the journal; newest first, one line
    each, repeats collapsed, the same list every surface's search shows.

    Args:
        screen: The screen the modal is pushed over.
        field: The composer to fill.
        logs_path: The session's log.
    """
    if not field.display:
        screen.notify("this view has no composer to fill", severity="warning")
        return
    recorded = operator_inputs(tail_events(logs_path, follow=False))
    entries = list(dict.fromkeys(" ".join(t.split()) for t in reversed(recorded)))
    if not entries:
        screen.notify("no past messages this session yet", severity="warning")
        return

    def fill(text: str | None) -> None:
        if text:
            field.load_text(text)
            field.move_cursor(field.document.end)
            field.focus()

    screen.app.push_screen(HistorySearchModal(entries), fill)


def approval_text(prompt: str, note: str = "approval needed", *, dim: bool = False) -> Text:
    """Return `? <head>: <note>` over the payload's lines, the command as every view shows it.

    Args:
        prompt: The approval's prompt.
        note: The words after the head.
        dim: Nobody can answer this one.
    """
    head, payload = approval_parts(prompt)
    if dim:
        body = Text(f"? {head}: {note}", style="dim")
    else:
        body = Text("? ", style="bold yellow")
        body.append(f"{head}: {note}", style="bold")
    if payload:
        body.append("\n" + "\n".join(f"    {ln}" for ln in payload.splitlines()))
    return body


class _AnswerLabel(Static, can_focus=True):
    """One answer of the row, `[key] label`: a click answers, and Enter or Space when focused."""

    BINDINGS: ClassVar = [
        Binding("enter", "answer", "Answer", show=False),
        Binding("space", "answer", "Answer", show=False),
    ]

    def __init__(self, key: str, answer: str, label: str, style: str) -> None:
        super().__init__(Text(f"[{key}] {label}", style=style), classes=f"answer-{answer}")
        self.answer = answer

    def on_click(self) -> None:
        """Answer on a click."""
        self.post_message(ApprovalRow.Answered(self.answer))

    def action_answer(self) -> None:
        """Answer on Enter or Space."""
        self.post_message(ApprovalRow.Answered(self.answer))


class ApprovalRow(Vertical):
    """The open approval docked above the composer: the command, when carried, over the answers.

    Nothing here takes the focus; Tab or a click moves it in, where every answer is
    a tab stop, and answering leaves it there for the next approval.
    """

    DEFAULT_CSS = """
    ApprovalRow { height: auto; padding: 0 1; background: $surface; }
    ApprovalRow #approval-answers { height: auto; }
    ApprovalRow Static { width: auto; padding: 0 2 0 0; }
    ApprovalRow _AnswerLabel:focus { background: $primary; color: $text; text-style: bold; }
    ApprovalRow _AnswerLabel:hover { background: $primary 30%; }
    """

    class Answered(Message):
        """An answer chosen on the row."""

        def __init__(self, answer: str) -> None:
            super().__init__()
            self.answer = answer

    def __init__(
        self,
        *,
        standing: bool,
        prompt: str = "",
        hint: str = ApprovalKeys.APPROVAL_ROW_HINT,
    ) -> None:
        super().__init__()  # no fixed id: a superseded row may still be unmounting
        self._standing = standing
        self._prompt = prompt
        self._hint = hint

    def compose(self) -> ComposeResult:
        """Yield the command, when carried, and the answers the prompt offers."""
        if self._prompt:
            yield Static(approval_text(self._prompt))
        with Horizontal(id="approval-answers"):
            for entry in APPROVAL_ANSWERS:
                if self.offers(entry.answer):
                    yield _AnswerLabel(
                        entry.key, entry.answer, entry.label, _ANSWER_STYLES[entry.answer]
                    )
            yield Static(Text(self._hint, style="dim"))

    def offers(self, answer: str) -> bool:
        """Return whether the prompt offers the answer; no scope means no session answer."""
        scoped = {e.answer for e in APPROVAL_ANSWERS if e.standing}
        return self._standing or answer not in scoped

    def focus_answers(self) -> None:
        """Put the focus on the first answer, where the keys work."""
        labels = self.query(_AnswerLabel)
        if labels:
            labels.first().focus()

    def holds_focus(self) -> bool:
        """Return whether the focus is on the row or one of its answers."""
        screen = self.screen if self.is_attached else None
        focused = screen.focused if screen is not None else None
        return focused is not None and (focused is self or self in focused.ancestors)


def deliver_answer(
    screen: Screen[Any],
    *,
    session_dir: Path,
    prompt_id: str,
    answer: str,
    prompts: Any = None,
    live: bool = True,
    lost: str = ApprovalKeys.APPROVAL_LOST,
) -> str:
    """Write an approval answer and notify the screen; one owner, so every view answers alike.

    Args:
        screen: The screen to notify.
        session_dir: The dir the answer file is written under.
        prompt_id: The approval's id.
        answer: The answer word.
        prompts: The dispatcher that records the claim, when there is one.
        live: The run still takes an answer.
        lost: The notice when it does not.

    Returns:
        "allowed", "denied", "answered elsewhere", or "" for a run that no longer takes it.
    """
    if not live:
        screen.notify(lost, severity="warning")
        return ""
    if prompts is not None:
        prompts.claim(session_dir, prompt_id)
    if write_answer(session_dir, prompt_id, answer):
        entry = answer_entry(answer)
        screen.notify(f"answered: {entry.label}")
        return "allowed" if entry.grants else "denied"
    screen.notify(ANSWERED_ELSEWHERE, severity="warning")
    return "answered elsewhere"
