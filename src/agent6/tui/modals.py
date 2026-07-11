# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Modal screens for the agent6 TUI: approval (y/n), steer (free text), and
question (selectable options + free text).

These are pure textual widgets, they take a prompt and `dismiss()` a result.
The app wires the result back through the file bridge (see frontend.approval); nothing
here touches the workflow, so any other front-end can drop them in or replace
them.

Unlike the theme/edit/provider/help overlays, these consequential prompts have
NO backdrop-click-to-close: an accidental click outside must not silently
approve/deny/answer -- dismissal is explicit (buttons / keys) only.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from agent6.viewmodel.state import Question

# Uniform arrow-key focus navigation for every consequential modal: Tab already
# moves focus; these make the arrows do the same, so the dialogs navigate the way
# the rest of the TUI does. left/right in a focused Input still move the cursor
# (the Input consumes them), so only up/down bubble to focus there.
_ARROW_NAV = (
    Binding("down", "app.focus_next", "next", show=False),
    Binding("up", "app.focus_previous", "prev", show=False),
    Binding("right", "app.focus_next", "next", show=False),
    Binding("left", "app.focus_previous", "prev", show=False),
)


# Modal frames pin a static round $accent (focused) border: a modal always owns
# focus, so it always shows the focused accent -- the $primary<->$accent
# resting/focus toggle is only for non-modal cards where focus actually moves.
class ApprovalModal(ModalScreen[str]):
    """Dismisses "yes", "no", or "session" (allow every later run_command this run)."""

    DEFAULT_CSS = """
    ApprovalModal { align: center middle; }
    #approval-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #approval-buttons { height: auto; align: center middle; margin-top: 1; }
    #approval-buttons Button {
        margin: 0 1; min-width: 18; height: 1; border: none;
        background: transparent; color: $accent;
    }
    #approval-buttons Button:focus { background: $primary; color: $text; text-style: bold; }
    """

    # Keys handled on the MODAL (not the app) so they reach the focused button.
    BINDINGS: ClassVar = [
        *_ARROW_NAV,
        Binding("y", "approve", "Allow", show=True),
        Binding("Y", "approve", "Allow", show=False),
        Binding("a", "approve_session", "Allow session", show=True),
        Binding("n", "deny", "Deny", show=True),
        Binding("N", "deny", "Deny", show=False),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, prompt_id: str, prompt: str) -> None:
        super().__init__()
        self.prompt_id = prompt_id
        self.prompt_text = prompt

    def compose(self) -> ComposeResult:
        with Container(id="approval-box"):
            body = Text()
            body.append("Approval requested\n\n", style="bold")
            body.append(self.prompt_text)  # plain append: never parsed as markup
            yield Static(body)
            with Horizontal(id="approval-buttons"):
                yield Button("Allow (y)", id="yes", variant="success")
                yield Button("Allow session (a)", id="session", variant="success")
                yield Button("Deny (n)", id="no", variant="error")

    def on_mount(self) -> None:
        self.query_one("#yes", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id or "no")  # button ids ARE the answer values

    def action_approve(self) -> None:
        self.dismiss("yes")

    def action_approve_session(self) -> None:
        self.dismiss("session")

    def action_deny(self) -> None:
        self.dismiss("no")


class ConfirmModal(ModalScreen[bool]):
    """A generic yes/no confirmation (title + body). y confirms; n / Esc cancels.
    No backdrop-click dismissal, matching the other consequential modals. Defaults
    focus to Cancel so an accidental Enter is safe."""

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
        Binding("y", "confirm", "Yes", show=True),
        Binding("Y", "confirm", "Yes", show=False),
        Binding("n", "cancel", "No", show=True),
        Binding("N", "cancel", "No", show=False),
        Binding("escape", "cancel", "No", show=False),
    ]

    def __init__(self, title: str, body: str, *, confirm_label: str = "Confirm") -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Container(id="confirm-box"):
            text = Text()
            text.append(f"{self._title}\n\n", style="bold")
            text.append(self._body)  # plain append: never parsed as markup
            yield Static(text)
            with Horizontal(id="confirm-buttons"):
                yield Button(f"{self._confirm_label} (y)", id="yes", variant="success")
                yield Button("Cancel (n)", id="no", variant="error")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()  # default to the safe choice

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class SteerModal(ModalScreen[str]):
    """Steer the run: inject a multi-line instruction, or continue as-is. Stopping
    is a separate action -- this dialog never stops the run.

    Result string: "" = continue, anything else = the steering instruction.
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
        Binding("ctrl+s", "send", "Send", show=False),
        Binding("escape", "cont", "Continue", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="steer-box"):
            body = Text()
            body.append("Steer this run\n\n", style="bold")
            body.append("Type an instruction (multi-line) then Send it, or Continue as-is.")
            yield Static(body)
            yield TextArea(id="steer-input", soft_wrap=True)
            with Horizontal(id="steer-buttons"):
                yield Button("Send  (Ctrl+S)", id="send", variant="primary")
                yield Button("Continue", id="continue", variant="success")

    def on_mount(self) -> None:
        self.query_one("#steer-input", TextArea).focus()

    def _text(self) -> str:
        return self.query_one("#steer-input", TextArea).text.strip()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self._text() if event.button.id == "send" else "")

    def action_send(self) -> None:
        self.dismiss(self._text())

    def action_cont(self) -> None:
        self.dismiss("")


class ToolCallDetailModal(ModalScreen[None]):
    """Read-only detail of one tool-call row: the full args + summary the inline
    table truncates to fit its columns. Informational, so clicking the backdrop
    closes it (unlike the consequential approval/steer modals). The text areas are
    read-only but selectable, so a long command, path, or payload can be copied.
    Esc closes; the args area is focused so arrow/page keys scroll it at once.
    """

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
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Close", show=True),
        # enter/q also close, but the focused read-only TextArea may swallow them;
        # Esc is the one that always bubbles, so it is the advertised key.
        Binding("enter", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
    ]

    def __init__(self, name: str, ok: bool | None, args: str, summary: str) -> None:
        super().__init__()
        self._name = name
        self._ok = ok
        self._args = args or "(no args)"
        self._summary = summary or "(no summary)"

    def compose(self) -> ComposeResult:
        status = "… in flight" if self._ok is None else ("✓ ok" if self._ok else "✗ failed")
        with Vertical(id="toolcall-box"):
            header = Text()
            header.append(self._name, style="bold")
            header.append(f"   {status}", style="dim")
            yield Static(header)
            yield Static("args", classes="tc-label")
            yield TextArea(self._args, read_only=True, soft_wrap=True, id="tc-args")
            yield Static("summary", classes="tc-label")
            yield TextArea(self._summary, read_only=True, soft_wrap=True, id="tc-summary")

    def on_mount(self) -> None:
        self.query_one("#tc-args", TextArea).focus()

    def on_click(self, event: events.Click) -> None:
        if event.widget is self:  # click on the backdrop (outside the box) = close
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class TextInputModal(ModalScreen[str | None]):
    """A one-line text prompt (title + input). Enter submits the text; Esc
    dismisses with None (cancelled). Used for the machine `poke` message box."""

    DEFAULT_CSS = """
    TextInputModal { align: center middle; }
    #ti-box {
        width: 80%; max-width: 100; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #ti-input { margin-top: 1; }
    """

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, title: str, placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Container(id="ti-box"):
            yield Static(Text(self._title, style="bold"))
            yield Input(placeholder=self._placeholder, id="ti-input")

    def on_mount(self) -> None:
        self.query_one("#ti-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuestionModal(ModalScreen["tuple[str, ...] | None"]):
    """An `ask_user` prompt: one or more related questions the operator answers
    together and reviews before submitting. Each question has an answer field; its
    option buttons fill that field (or type free text). Submit (ctrl+s) returns all
    answers aligned to the questions; Esc submits empties (the agent gets defaults).
    """

    DEFAULT_CSS = """
    QuestionModal { align: center middle; }
    #question-box {
        width: 80%; max-width: 100; height: auto; max-height: 90%;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #question-list { height: auto; }
    .q-text { margin-top: 1; text-style: bold; }
    /* Options are chips: clicking one fills that question's answer field below.
       A visible border + panel fill reads as pressable (a borderless full-width
       label read as a heading); auto width so short options sit compact, and
       they wrap across rows rather than one tall column. Keep the default height
       (a borderless height:1 button collapses its label to nothing). */
    .q-opts { height: auto; }
    .q-opts Button {
        width: auto; min-width: 8; margin: 0 1 0 0;
        border: round $primary; background: $panel; color: $foreground;
    }
    .q-opts Button:focus { border: round $accent; background: $primary; text-style: bold; }
    .q-ans { margin-top: 0; }
    #question-submit {
        margin-top: 1; background: $primary; color: $text; text-style: bold;
    }
    #question-submit:focus { background: $accent; }
    """

    BINDINGS: ClassVar = [
        *_ARROW_NAV,
        Binding("ctrl+s", "submit", "Submit", show=True),
        Binding("escape", "skip", "Skip", show=True),
    ]

    def __init__(self, question_id: str, questions: tuple[Question, ...]) -> None:
        super().__init__()
        self.question_id = question_id
        self.questions = questions

    def compose(self) -> ComposeResult:
        multi = len(self.questions) > 1
        with Vertical(id="question-box"):
            head = Text()
            head.append("The agent is asking", style="bold")
            head.append(". Answer, then Submit (ctrl+s):" if multi else ":")
            yield Static(head)
            with VerticalScroll(id="question-list"):
                for qi, q in enumerate(self.questions):
                    body = Text()
                    if multi:
                        body.append(f"{qi + 1}. ", style="bold")
                    body.append(q.question)  # plain append: never parsed as markup
                    yield Static(body, classes="q-text")
                    if q.options:
                        # Buttons carry Text so an option with '[...]' can't crash
                        # markup parsing; pressing one fills that answer field. A
                        # Horizontal row of auto-width chips wraps compactly.
                        with Horizontal(classes="q-opts"):
                            for oi, opt in enumerate(q.options):
                                yield Button(Text(opt), id=f"opt-{qi}-{oi}")
                    yield Input(
                        placeholder="pick above or type an answer",
                        id=f"ans-{qi}",
                        classes="q-ans",
                    )
            yield Button("Submit (ctrl+s)", id="question-submit")

    def on_mount(self) -> None:
        self.query_one("#ans-0", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "question-submit":
            self.action_submit()
        elif bid.startswith("opt-"):  # opt-{qi}-{oi}: fill that question's field
            _, qi, oi = bid.split("-")
            self.query_one(f"#ans-{qi}", Input).value = self.questions[int(qi)].options[int(oi)]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter advances to the next field, or submits on the last one.
        idx = int((event.input.id or "ans-0").removeprefix("ans-"))
        if idx + 1 < len(self.questions):
            self.query_one(f"#ans-{idx + 1}", Input).focus()
        else:
            self.action_submit()

    def action_submit(self) -> None:
        answers = tuple(
            self.query_one(f"#ans-{qi}", Input).value.strip() for qi in range(len(self.questions))
        )
        self.dismiss(answers)

    def action_skip(self) -> None:
        self.dismiss(tuple("" for _ in self.questions))
