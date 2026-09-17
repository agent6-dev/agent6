# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The composer every conversation surface shares: the steer input and its
mode labels, the slash-command suggestions, the resume preset picker, the
history search, and the inline approval row."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Select, Static, TextArea

from agent6.directive import LIVE_RUN_COMMANDS, STEER_COMMANDS
from agent6.sessions.ipc import ANSWERED_ELSEWHERE, write_answer
from agent6.ui.tui.menubar import (
    Menu,
    MenuItem,
)
from agent6.ui.tui.modals import HistorySearchModal
from agent6.ui.tui.widgets import Picker
from agent6.viewmodel import approval_parts
from agent6.viewmodel.tail import tail_events
from agent6.viewmodel.transcript import (
    operator_inputs,
)

ComposerMode = Literal["steer", "resume", "start", "draft"]


def composer_labels(
    mode: ComposerMode, *, continue_as: str = "", needs_new_work: bool = False
) -> tuple[str, str]:
    """(border title, key hint) for the composer.

    One conversation view serves runs, plans and asks, so it says "session".
    *continue_as* names the fork an undone run continues as (Enter resumes that
    session); *needs_new_work* is a run the agent finished green, which a bare
    resume has nothing to do for (the web composer asks the same question);
    *draft* is the machine description the create dialog takes.
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
    return ("new task", "Enter starts · Ctrl-J newline")


def steer_suggestion_rows(text: str, *, mode: ComposerMode) -> list[tuple[str, str]]:
    """The steer directives matching the composer's first word while it is
    still being typed (`/…`, no whitespace yet): (command, help) rows, empty
    for ordinary text. A resume composer withholds `LIVE_RUN_COMMANDS`; a
    draft offers only /parallel (the fan-out is the one directive a start
    understands)."""
    if not text.startswith("/") or any(ch.isspace() for ch in text):
        return []
    if mode == "draft":
        return []  # a machine description takes no directives
    if mode == "start":
        offered = {c: h for c, h in STEER_COMMANDS.items() if c == "/parallel"}
    elif mode == "resume":
        offered = {c: h for c, h in STEER_COMMANDS.items() if c not in LIVE_RUN_COMMANDS}
    else:
        offered = STEER_COMMANDS
    return [(c, h) for c, h in offered.items() if c.startswith(text)]


def complete_steer(text: str, *, mode: ComposerMode) -> str | None:
    """Tab in a composer: the completed command word, or None when Tab should
    keep its focus-move meaning. A unique match completes with a trailing
    space; several matches advance to their longest common prefix, returning
    *text* unchanged when there is no progress so Tab never yanks focus away
    mid-command."""
    rows = steer_suggestion_rows(text, mode=mode)
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0][0] + " "
    lcp = os.path.commonprefix([c for c, _h in rows])
    return lcp if len(lcp) > len(text) else text


class SteerSuggest(Static):
    """The command hints above a composer (the run views' analogue of the
    hub's model-suggestion line): one row per matching steer directive while
    the first word is being typed, hidden otherwise. Tab in the composer
    completes (see SteerInput.on_key)."""

    ALLOW_SELECT = False
    DEFAULT_CSS = """
    SteerSuggest { display: none; height: auto; padding: 0 1; background: $surface; }
    """

    def show_for(self, text: str, *, mode: ComposerMode) -> None:
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
        """Show *body* as the hint line, or hide the line for None."""
        if body is not None:
            self.update(body)
        show = body is not None
        if self.display != show:
            self.display = show


_INPUT_MAX_ROWS = 6  # the steer bar grows to this many rows, then scrolls internally


class ResumeHost(Protocol):
    """What a resume row reads and writes on its host app (`Agent6TUI`)."""

    resume_preset: str
    resume_model: str

    def resume_defaults(self, preset: str) -> tuple[str, str]: ...


class ResumeOptions(Horizontal):
    """The row above a resume composer: the config preset and the model the
    next leg continues under (`agent6 resume --preset`, `--model`). Both
    change only between legs, so the row shows only while the composer
    resumes; the choices live on the host app, so the conversation and the
    dashboard composers agree. Each first entry adds no flag and names what
    the resume runs under (`ResumeHost.resume_defaults`), relabelled when the
    row reappears and, for the model, when the preset pick changes."""

    DEFAULT_CSS = """
    ResumeOptions { display: none; height: 1; padding: 0 1; }
    ResumeOptions .resume-label { width: auto; padding: 0 1 0 0; color: $text-muted; }
    ResumeOptions Picker { margin-right: 2; }
    """

    def __init__(self, presets: list[str], routes: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._presets = presets
        self._routes = routes
        self._labels = ("", "")
        self._labelled: str | None = None  # the preset pick the labels name; None = stale

    def compose(self) -> ComposeResult:
        host = self._host()
        self._labels, self._labelled = host.resume_defaults(host.resume_preset), host.resume_preset
        yield Static("continue under preset", classes="resume-label")
        yield Picker(self._options(0), value="", allow_blank=False, id="resume-preset")
        yield Static("model", classes="resume-label")
        yield Picker(self._options(1), value="", allow_blank=False, id="resume-model")

    def _host(self) -> ResumeHost:
        return cast(ResumeHost, self.app)

    def _options(self, index: int) -> list[tuple[str, str]]:
        choices = self._routes if index else self._presets
        return [(self._labels[index], ""), *((c, c) for c in choices)]

    def on_select_changed(self, event: Select.Changed) -> None:
        host, value = self._host(), str(event.value)
        if event.select.id == "resume-model":
            host.resume_model = value
        else:
            host.resume_preset = value
            self._relabel()

    def show(self, shown: bool) -> None:
        if self.display != shown:
            self.display = shown
        if not shown:  # a leg is running: it may pin a preset or a model
            self._labelled = None
        else:
            # After the refresh: on the first paint the Selects are not mounted
            # yet, and a value written before the mount leaves a label blank.
            # Relabel after a leg, or after the other view's row moved the pick.
            if self._labelled != self._host().resume_preset:
                self.call_after_refresh(self._relabel)
            self.call_after_refresh(self._sync)

    def _relabel(self) -> None:
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
        host = self._host()
        with self.prevent(Select.Changed):
            for wanted, picker_id, choices in (
                (host.resume_preset, "#resume-preset", self._presets),
                (host.resume_model, "#resume-model", self._routes),
            ):
                picker = self.query_one(picker_id, Select)
                if picker.value != wanted and wanted in ("", *choices):
                    picker.value = wanted


# The run-control menu, shared verbatim by the two run views (this primary
# conversation and the dashboard) so they cannot drift. Every action resolves on
# the Agent6TUI app (the menu bar's dispatcher falls back to app actions).
RUN_MENU = Menu(
    "Run",
    (
        MenuItem("Search past messages…", "history_search", "ctrl+r"),
        MenuItem("Compact context now", "compact"),
        MenuItem("Stop after this step", "stop_step"),
        MenuItem("Stop now", "stop_now"),
        MenuItem("Resume this session", "resume"),
        MenuItem("Run this plan", "run_plan"),
        MenuItem("Fork this session", "fork"),
        MenuItem("Delete this session…", "delete_session"),
    ),
)


# The answers an open approval offers, with the CLI prompt's and the modal's
# keys ("yes" / "no" / "session" / "session-deny"): (key, answer, label, style).
# The row renders them and answers a click; the composer binds the keys.
APPROVAL_ANSWERS: tuple[tuple[str, str, str, str], ...] = (
    ("y", "yes", "allow", "bold green"),
    ("a", "session", "allow all (session)", "green"),
    ("n", "no", "deny", "bold red"),
    ("d", "session-deny", "deny all", "red"),
)
# Offered only by a standing approval (one the operator may answer for the session).
_STANDING_ANSWERS = frozenset({"session", "session-deny"})


class SteerInput(TextArea):
    """The bottom composer bar: a TextArea that submits on Enter (Ctrl+J /
    Shift+Enter insert a newline instead) and grows with its content up to
    _INPUT_MAX_ROWS. Two modes (set_mode): steer a live run, or type the
    follow-up instruction a finished run is resumed with. An open approval
    never takes the keys: they answer only with the focus moved into its row."""

    ALLOW_MAXIMIZE = False  # a full-screen composer is never what Maximize means

    BINDINGS: ClassVar = [
        # TextArea's own undo stack; ctrl+z is the app's Detach (see Agent6TUI).
        Binding("ctrl+underscore", "undo", "Undo", show=False),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_mount(self) -> None:
        self.set_mode(mode=self.mode)
        self._resize()

    policy = ""  # viewmodel.session_policy(...).short(), set once the run dir is known
    mode: ComposerMode = "steer"  # which directives apply (see steer_suggestion_rows)

    def set_mode(
        self,
        *,
        mode: ComposerMode,
        ctx_pct: int | None = None,
        continue_as: str = "",
        needs_new_work: bool = False,
    ) -> None:
        """Relabel for the session's state: steering (live), resuming
        (finished; *continue_as* names the fork an undone run resumes as,
        *needs_new_work* a run finished green), or starting (a draft), plus the
        context-window fill when known, right where you type. Only writes on
        a real change: this runs on every heartbeat, and same-value style
        writes still cost a refresh."""
        self.mode = mode
        title, keys = composer_labels(mode, continue_as=continue_as, needs_new_work=needs_new_work)
        ctx = f"ctx {ctx_pct}% · " if ctx_pct is not None else ""
        # The run's policy sits where the eye already goes for status, from the
        # same fold the CLI banner and the web header read.
        policy = f"{self.policy} · " if self.policy else ""
        # Border titles are markup: `[focus]` in a label or a bracket in a
        # model id would vanish (or crash) unescaped.
        title = escape(title)
        subtitle = escape(f"{policy}{ctx}{keys}")
        if self.border_title != title:
            self.border_title = title
        if self.border_subtitle != subtitle:
            self.border_subtitle = subtitle

    def on_key(self, event: events.Key) -> None:
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
            if completed is not None:  # else Tab keeps its focus-move meaning
                event.prevent_default()
                event.stop()
                if completed != self.text:
                    self.load_text(completed)
                    self.move_cursor(self.document.end)

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self._resize()

    def _resize(self) -> None:
        rows = min(max(self.document.line_count, 1), _INPUT_MAX_ROWS)
        height = rows + 2  # + the rounded border
        current = self.styles.height
        if current is None or current.value != height:  # only relayout on a real change
            self.styles.height = height


def open_history_search(screen: Screen[Any], field: SteerInput, logs_path: Path) -> None:
    """Ctrl-R on a composer: pick one of this session's past messages (the
    task, then every steer, journal-read, so resumes and other surfaces' steers
    appear) into *field* for editing. Newest first, flattened to one
    line each, repeats collapsed: the same list every surface's search shows."""
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


class _AnswerLabel(Static, can_focus=True):
    """One answer of the row: `[key] label`. A click answers from any focus;
    Tab reaches it and Enter or Space answers, like a button."""

    BINDINGS: ClassVar = [
        Binding("enter", "answer", "Answer", show=False),
        Binding("space", "answer", "Answer", show=False),
    ]

    def __init__(self, key: str, answer: str, label: str, style: str) -> None:
        super().__init__(Text(f"[{key}] {label}", style=style), classes=f"answer-{answer}")
        self.answer = answer

    def on_click(self) -> None:
        self.post_message(ApprovalRow.Answered(self.answer))

    def action_answer(self) -> None:
        self.post_message(ApprovalRow.Answered(self.answer))


class ApprovalRow(Vertical):
    """The open approval, docked above the composer: the command under judgment
    (when the screen does not show it itself) over the answers.

    Nothing here takes focus: the composer keeps it, and a message typed as an
    approval arrives is a message. Tab (or a click) moves focus into the row,
    where every answer is a tab stop and its key answers; answering leaves the
    focus there, so the next approval is answerable at once."""

    DEFAULT_CSS = """
    ApprovalRow { height: auto; padding: 0 1; background: $surface; }
    ApprovalRow #approval-answers { height: auto; }
    ApprovalRow Static { width: auto; padding: 0 2 0 0; }
    ApprovalRow _AnswerLabel:focus { background: $primary; color: $text; text-style: bold; }
    ApprovalRow _AnswerLabel:hover { background: $primary 30%; }
    """

    BINDINGS: ClassVar = [
        *(
            Binding(key, f"answer('{answer}')", label, show=False)
            for key, answer, label, _style in APPROVAL_ANSWERS
        ),
    ]

    class Answered(Message):
        def __init__(self, answer: str) -> None:
            super().__init__()
            self.answer = answer

    def __init__(self, *, standing: bool, prompt: str = "") -> None:
        super().__init__()  # no fixed id: a superseded row may still be unmounting
        self._standing = standing
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        if self._prompt:
            head, payload = approval_parts(self._prompt)
            body = Text("? ", style="bold yellow")
            body.append(f"{head}: approval needed", style="bold")
            if payload:
                body.append("\n" + "\n".join(f"    {ln}" for ln in payload.splitlines()))
            yield Static(body)
        with Horizontal(id="approval-answers"):
            for key, answer, label, style in APPROVAL_ANSWERS:
                if self.offers(answer):
                    yield _AnswerLabel(key, answer, label, style)
            yield Static(Text("(Tab here for the keys; or click)", style="dim"))

    def offers(self, answer: str) -> bool:
        return self._standing or answer not in _STANDING_ANSWERS

    def action_answer(self, answer: str) -> None:
        self.post_message(self.Answered(answer))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """A prompt with no scope offers no session answer, key included."""
        return self.offers(str(parameters[0])) if action == "answer" else True

    def focus_answers(self) -> None:
        """Put the focus on the first answer, where the keys work."""
        labels = self.query(_AnswerLabel)
        if labels:
            labels.first().focus()

    def holds_focus(self) -> bool:
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
) -> str:
    """Write an approval answer a row collected, notify the screen, and say what
    happened: "allowed", "denied", "answered elsewhere", or "" for a run that
    can no longer take it. One owner, so both run views answer alike."""
    if not live:
        screen.notify("the run is gone: the answer reached nothing", severity="warning")
        return ""
    if prompts is not None:
        prompts.claim(session_dir, prompt_id)
    if write_answer(session_dir, prompt_id, answer):
        screen.notify(f"answered: {answer}")
        return "allowed" if answer in ("yes", "session") else "denied"
    screen.notify(ANSWERED_ELSEWHERE, severity="warning")
    return "answered elsewhere"
