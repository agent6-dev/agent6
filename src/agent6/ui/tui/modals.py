# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI's modal screens: confirm, steer, question, history search and the read-only views.

Each takes a prompt and dismisses a result; the app wires the result to the file
bridge. A consequential prompt closes only by a key or a button, never a backdrop
click, so an accidental click cannot answer it.
"""

from __future__ import annotations

from typing import ClassVar

from rich import text as rich_text
from textual import app, binding, containers, events, screen, widgets

from agent6.ui.tui import widgets as tui_widgets
from agent6.viewmodel import state

# The arrows move focus like Tab; a focused Input consumes left and right for its cursor.
_ARROW_NAV = (
    binding.Binding("down", "app.focus_next", "next", show=False),
    binding.Binding("up", "app.focus_previous", "prev", show=False),
    binding.Binding("right", "app.focus_next", "next", show=False),
    binding.Binding("left", "app.focus_previous", "prev", show=False),
)


# A modal's frame is the focused accent border: a modal always owns the focus.
class ConfirmModal(screen.ModalScreen[bool]):
    """A yes/no confirmation; focus defaults to Cancel, so an accidental Enter is safe."""

    DEFAULT_CSS = """
    ConfirmModal { align: center middle; }
    #confirm-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #confirm-buttons { height: auto; align: center middle; margin-top: 1; }
    #confirm-buttons Button {
        margin: 0 2; min-width: 16; height: 1; border: none;
        background: transparent; color: $accent;
    }
    #confirm-buttons Button:focus { background: $primary; color: $text; text-style: bold; }
    """

    BINDINGS: ClassVar = [
        *_ARROW_NAV,
        binding.Binding("y", "confirm", "Yes", show=True),
        binding.Binding("Y", "confirm", "Yes", show=False),
        binding.Binding("n", "cancel", "No", show=True),
        binding.Binding("N", "cancel", "No", show=False),
        binding.Binding("escape", "cancel", "No", show=False),
        binding.Binding(
            "q", "cancel", "No", show=False
        ),  # the footer under a modal reads "Esc/q Back"
    ]

    def __init__(self, title: str, body: str, *, confirm_label: str = "Confirm") -> None:
        """Create the dialog with its title, body and the confirm button's label."""
        super().__init__()
        self._title = title
        self._body = body
        self._confirm_label = confirm_label

    def compose(self) -> app.ComposeResult:
        """Lay out the dialog.

        Yields:
            The text and the two buttons.
        """
        with containers.Container(id="confirm-box"):
            text = rich_text.Text()
            text.append(f"{self._title}\n\n", style="bold")
            text.append(self._body)  # never parsed as markup
            yield widgets.Static(text)
            with containers.Horizontal(id="confirm-buttons"):
                yield widgets.Button(f"{self._confirm_label} (y)", id="yes", variant="success")
                yield widgets.Button("Cancel (n)", id="no", variant="error")

    def on_mount(self) -> None:
        """Focus Cancel."""
        self.query_one("#no", widgets.Button).focus()

    def on_button_pressed(self, event: widgets.Button.Pressed) -> None:
        """Dismiss with the button's answer."""
        self.dismiss(event.button.id == "yes")

    def action_confirm(self) -> None:
        """Dismiss with yes."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with no."""
        self.dismiss(False)


class SteerModal(screen.ModalScreen[str]):
    """Steer the run with a multi-line instruction, or continue as is.

    The result is the instruction, or "" to continue; the dialog never stops the run.
    """

    DEFAULT_CSS = """
    SteerModal { align: center middle; }
    #steer-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #steer-input { height: 8; margin-top: 1; border: round $primary; background: $surface; }
    #steer-buttons { height: auto; align: center middle; margin-top: 1; }
    #steer-buttons Button {
        margin: 0 2; min-width: 16; height: 1; border: none;
        background: transparent; color: $accent;
    }
    #steer-buttons Button:focus { background: $primary; color: $text; text-style: bold; }
    """

    BINDINGS: ClassVar = [
        *_ARROW_NAV,
        binding.Binding("ctrl+s", "send", "Send", show=False),
        binding.Binding("escape", "cont", "Continue", show=False),
        binding.Binding(
            "ctrl+underscore", "undo_text", "Undo", show=False
        ),  # the composer's undo key
    ]

    def compose(self) -> app.ComposeResult:
        """Lay out the dialog.

        Yields:
            The text, the input and the two buttons.
        """
        with containers.Container(id="steer-box"):
            body = rich_text.Text()
            body.append("Steer this run\n\n", style="bold")
            # Split at the clause, so a narrow terminal never wraps mid-phrase.
            body.append("Type an instruction (multi-line) then Send it,\nor Continue as-is.")
            yield widgets.Static(body)
            yield widgets.TextArea(id="steer-input", soft_wrap=True)
            with containers.Horizontal(id="steer-buttons"):
                yield widgets.Button("Send (Ctrl+S)", id="send", variant="primary")
                yield widgets.Button("Continue", id="continue", variant="success")

    def on_mount(self) -> None:
        """Focus the input."""
        self.query_one("#steer-input", widgets.TextArea).focus()

    def _text(self) -> str:
        """Return the typed instruction, stripped."""
        return self.query_one("#steer-input", widgets.TextArea).text.strip()

    def on_button_pressed(self, event: widgets.Button.Pressed) -> None:
        """Send or continue, by the button."""
        self.dismiss(self._text() if event.button.id == "send" else "")

    def action_send(self) -> None:
        """Dismiss with the instruction."""
        self.dismiss(self._text())

    def action_cont(self) -> None:
        """Dismiss with continue."""
        self.dismiss("")

    def action_undo_text(self) -> None:
        """Undo the last edit."""
        self.query_one("#steer-input", widgets.TextArea).undo()


class ToolCallDetailModal(screen.ModalScreen[None]):
    """The full args and summary of one tool call, selectable; Esc or the backdrop closes."""

    DEFAULT_CSS = """
    ToolCallDetailModal { align: center middle; }
    #toolcall-box {
        width: 90%; max-width: 120; height: auto; max-height: 85%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #toolcall-box .tc-label { color: $accent; text-style: bold; margin-top: 1; }
    #toolcall-box TextArea {
        height: auto; max-height: 24; border: round $primary; background: $surface;
    }
    /* No caret: it would read as an editable field. */
    #toolcall-box TextArea .text-area--cursor { background: transparent; color: $foreground; }
    """

    BINDINGS: ClassVar = [
        binding.Binding(
            "escape", "close", "Close", show=True
        ),  # the one key the text area never swallows
        binding.Binding("enter", "close", "Close", show=False),
        binding.Binding("q", "close", "Close", show=False),
    ]

    def __init__(self, name: str, ok: bool | None, args: str, summary: str) -> None:
        """Create the view for a tool call's name, verdict, args and summary."""
        super().__init__()
        self._name = name
        self._ok = ok
        self._args = args or "(no args)"
        self._summary = summary or "(no summary)"

    def compose(self) -> app.ComposeResult:
        """Lay out the view.

        Yields:
            The header, then the args and the summary, each labelled.
        """
        status = "… in flight" if self._ok is None else ("✓ ok" if self._ok else "✗ failed")
        with containers.Vertical(id="toolcall-box"):
            header = rich_text.Text()
            header.append(self._name, style="bold")
            header.append(f"   {status}", style="dim")
            yield widgets.Static(header)
            yield widgets.Static("args", classes="tc-label")
            yield widgets.TextArea(self._args, read_only=True, soft_wrap=True, id="tc-args")
            yield widgets.Static("summary", classes="tc-label")
            yield widgets.TextArea(self._summary, read_only=True, soft_wrap=True, id="tc-summary")

    def on_mount(self) -> None:
        """Focus the args, so the page keys scroll them at once."""
        self.query_one("#tc-args", widgets.TextArea).focus()

    def on_click(self, event: events.Click) -> None:
        """Close on a click outside the box."""
        if event.widget is self:
            self.dismiss(None)

    def action_close(self) -> None:
        """Close."""
        self.dismiss(None)


class TextModal(screen.ModalScreen[None]):
    """A titled read-only text view, selectable; Esc or the backdrop closes."""

    DEFAULT_CSS = """
    TextModal { align: center middle; }
    #text-box {
        width: 90%; max-width: 120; height: auto; max-height: 85%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #text-box TextArea {
        height: auto; max-height: 32; border: round $primary; background: $surface;
    }
    /* No caret: it would read as an editable field. */
    #text-box TextArea .text-area--cursor { background: transparent; color: $foreground; }
    """

    BINDINGS: ClassVar = [
        binding.Binding("escape", "close", "Close", show=True),
        binding.Binding("q", "close", "Close", show=False),
    ]

    def __init__(self, title: str, text: str) -> None:
        """Create the view for a title and its text."""
        super().__init__()
        self._title = title
        self._text = text

    def compose(self) -> app.ComposeResult:
        """Lay out the view.

        Yields:
            The title and the text.
        """
        with containers.Vertical(id="text-box"):
            yield widgets.Static(rich_text.Text(self._title, style="bold"))
            yield widgets.TextArea(self._text, read_only=True, soft_wrap=True, id="text-view")

    def on_mount(self) -> None:
        """Focus the text."""
        self.query_one("#text-view", widgets.TextArea).focus()

    def on_click(self, event: events.Click) -> None:
        """Close on a click outside the box."""
        if event.widget is self:
            self.dismiss(None)

    def action_close(self) -> None:
        """Close."""
        self.dismiss(None)


class TextInputModal(screen.ModalScreen[str | None]):
    """A one-line text prompt; Enter submits the text, Esc dismisses with None."""

    DEFAULT_CSS = """
    TextInputModal { align: center middle; }
    #ti-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #ti-input { margin-top: 1; }
    """

    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, placeholder: str = "") -> None:
        """Create the prompt with its title and the input's placeholder."""
        super().__init__()
        self._title = title
        self._placeholder = placeholder

    def compose(self) -> app.ComposeResult:
        """Lay out the prompt.

        Yields:
            The title and the input.
        """
        with containers.Container(id="ti-box"):
            yield widgets.Static(rich_text.Text(self._title, style="bold"))
            yield widgets.Input(placeholder=self._placeholder, id="ti-input")

    def on_mount(self) -> None:
        """Focus the input."""
        self.query_one("#ti-input", widgets.Input).focus()

    def on_input_submitted(self, event: widgets.Input.Submitted) -> None:
        """Dismiss with the text."""
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        """Dismiss with None."""
        self.dismiss(None)


class HistorySearchModal(screen.ModalScreen[str | None]):
    """Pick one of the session's past messages to edit and resend.

    Enter keeps the highlighted match, or the typed text when none is; Esc or the
    backdrop cancels, since a pick is not consequential: sending still takes Enter
    in the composer.
    """

    DEFAULT_CSS = """
    HistorySearchModal { align: center middle; }
    #hs-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #hs-field { margin-top: 1; }
    #hs-hint { margin-top: 1; color: $text-muted; }
    """

    BINDINGS: ClassVar = [
        binding.Binding("escape", "cancel", "Cancel", show=False),
        binding.Binding("enter", "submit", "Use", show=False),
    ]

    def __init__(self, entries: list[str]) -> None:
        """Create the search over the past messages."""
        super().__init__()
        self._entries = entries

    def compose(self) -> app.ComposeResult:
        """Lay out the search.

        Yields:
            The title, the typeahead field and the hint.
        """
        with containers.Container(id="hs-box"):
            yield widgets.Static(rich_text.Text("Search past messages", style="bold"))
            yield tui_widgets.TypeaheadField("", self._entries, id="hs-field")
            yield widgets.Static(
                "↑↓ highlight · Enter fills the composer · Esc closes", id="hs-hint"
            )

    def on_mount(self) -> None:
        """Focus the field."""
        self.query_one("#hs-field", tui_widgets.TypeaheadField).focus()

    def on_click(self, event: events.Click) -> None:
        """Cancel on a click outside the box."""
        if event.widget is self:
            self.dismiss(None)

    def action_submit(self) -> None:
        """Dismiss with the field's value, or None when empty."""
        self.dismiss(self.query_one("#hs-field", tui_widgets.TypeaheadField).value or None)

    def action_cancel(self) -> None:
        """Dismiss with None."""
        self.dismiss(None)


class QuestionModal(screen.ModalScreen["tuple[str, ...] | None"]):
    """An `ask_user` prompt: related questions answered together and submitted at once.

    Each question has an answer field its option buttons fill. Submit returns the
    answers aligned to the questions; Esc submits empties, so the agent gets its
    defaults.
    """

    DEFAULT_CSS = """
    QuestionModal { align: center middle; }
    #question-box {
        width: 80%; max-width: 100; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #question-list { height: auto; }
    .q-text { margin-top: 1; text-style: bold; }
    /* One-row pieces, like the config dialogs: flat option chips, flat fields, a flat action. */
    .q-opts { height: auto; margin-top: 1; }
    #question-box .q-opts Button {
        width: auto; min-width: 0; height: 1; margin: 0 1 0 0; padding: 0 1;
        background: $panel; color: $foreground; text-style: none;
    }
    #question-box .q-opts Button:hover { background: $primary 30%; }
    #question-box .q-opts Button:focus { background: $primary; color: $text; text-style: bold; }
    #question-box .q-ans {
        height: 1; margin-top: 1; padding: 0 1; border: none; background: $panel;
    }
    #question-box .q-ans:focus { border: none; background: $primary 25%; }
    #question-box #question-submit {
        width: auto; min-width: 0; height: 1; margin-top: 1; padding: 0 2;
        background: transparent; color: $accent;
    }
    #question-box #question-submit:hover { background: $primary 30%; }
    #question-box #question-submit:focus { background: $primary; color: $text; text-style: bold; }
    #question-hint { margin-top: 1; color: $text-muted; }
    """

    BINDINGS: ClassVar = [
        *_ARROW_NAV,
        binding.Binding("ctrl+s", "submit", "Submit", show=True),
        binding.Binding("escape", "skip", "Skip", show=True),
    ]

    def __init__(
        self, question_id: str, questions: tuple[state.Question, ...], *, from_harness: bool = False
    ) -> None:
        """Create the prompt for a question id and its questions."""
        super().__init__()
        self.question_id = question_id
        self.questions = questions
        self.from_harness = from_harness

    def compose(self) -> app.ComposeResult:
        """Lay out the prompt.

        Yields:
            The header, then per question its text, its option chips and its answer
            field, then Submit and the hint.
        """
        multi = len(self.questions) > 1
        with containers.Vertical(id="question-box"):
            head = rich_text.Text()
            head.append(
                "agent6 is asking" if self.from_harness else "The agent is asking", style="bold"
            )
            head.append(". Answer, then Submit (ctrl+s):" if multi else ":")
            yield widgets.Static(head)
            with containers.VerticalScroll(id="question-list"):
                for qi, q in enumerate(self.questions):
                    body = rich_text.Text()
                    if multi:
                        body.append(f"{qi + 1}. ", style="bold")
                    body.append(q.question)  # never parsed as markup
                    yield widgets.Static(body, classes="q-text")
                    if q.options:
                        # Text labels, so an option holding brackets is not parsed as markup.
                        with containers.Horizontal(classes="q-opts"):
                            for oi, opt in enumerate(q.options):
                                yield widgets.Button(
                                    rich_text.Text(opt), id=f"opt-{qi}-{oi}", compact=True
                                )
                    yield widgets.Input(
                        placeholder="pick above or type an answer",
                        id=f"ans-{qi}",
                        classes="q-ans",
                    )
            yield widgets.Button("Submit (ctrl+s)", id="question-submit", compact=True)
            yield widgets.Static("Enter next field · Ctrl+S submit · Esc skip", id="question-hint")

    def on_mount(self) -> None:
        """Focus the first answer field."""
        self.query_one("#ans-0", widgets.Input).focus()

    def on_button_pressed(self, event: widgets.Button.Pressed) -> None:
        """Submit, or fill a question's field with the pressed option."""
        bid = event.button.id or ""
        if bid == "question-submit":
            self.action_submit()
        elif bid.startswith("opt-"):
            _, qi, oi = bid.split("-")
            self.query_one(f"#ans-{qi}", widgets.Input).value = self.questions[int(qi)].options[
                int(oi)
            ]

    def on_input_submitted(self, event: widgets.Input.Submitted) -> None:
        """Advance to the next field on Enter, or submit from the last one."""
        idx = int((event.input.id or "ans-0").removeprefix("ans-"))
        if idx + 1 < len(self.questions):
            self.query_one(f"#ans-{idx + 1}", widgets.Input).focus()
        else:
            self.action_submit()

    def action_submit(self) -> None:
        """Dismiss with every answer, stripped."""
        answers = tuple(
            self.query_one(f"#ans-{qi}", widgets.Input).value.strip()
            for qi in range(len(self.questions))
        )
        self.dismiss(answers)

    def action_skip(self) -> None:
        """Dismiss with an empty answer per question."""
        self.dismiss(tuple("" for _ in self.questions))
