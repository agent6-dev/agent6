# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Modal screens for the agent6 TUI: approval (y/n), steer (free text), and
question (selectable options + free text).

These are pure textual widgets, they take a prompt and `dismiss()` a result.
The app wires the result back through the file bridge (see ui.approval); nothing
here touches the workflow, so any other front-end can drop them in or replace
them.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class ApprovalModal(ModalScreen[bool]):
    DEFAULT_CSS = """
    ApprovalModal { align: center middle; }
    #approval-box {
        width: 80%; max-width: 100; height: auto;
        border: thick $warning; padding: 1 2; background: $surface;
    }
    #approval-buttons { height: auto; align: center middle; margin-top: 1; }
    #approval-buttons Button { margin: 0 2; min-width: 16; }
    #approval-buttons Button:focus { text-style: bold reverse; }
    """

    # Keys handled on the MODAL (not the app) so they reach the focused button.
    BINDINGS: ClassVar = [
        Binding("left", "focus_previous", "◀", show=False),
        Binding("right", "focus_next", "▶", show=False),
        Binding("y", "approve", "Allow", show=True),
        Binding("Y", "approve", "Allow", show=False),
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
                yield Button("Deny (n)", id="no", variant="error")

    def on_mount(self) -> None:
        self.query_one("#yes", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class SteerModal(ModalScreen[str]):
    """Mid-run Ctrl-C prompt: continue, abort, or inject a steering instruction.

    Result string: "" = continue, "abort" = stop, anything else = instruction.
    """

    DEFAULT_CSS = """
    SteerModal { align: center middle; }
    #steer-box {
        width: 80%; max-width: 100; height: auto;
        border: thick $accent; padding: 1 2; background: $surface;
    }
    #steer-input { margin-top: 1; }
    #steer-buttons { height: auto; align: center middle; margin-top: 1; }
    #steer-buttons Button { margin: 0 2; min-width: 14; }
    #steer-buttons Button:focus { text-style: bold reverse; }
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cont", "Continue", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="steer-box"):
            body = Text()
            body.append("Run interrupted\n\n", style="bold")
            body.append("Continue, abort, or type a steering instruction below.")
            yield Static(body)
            yield Input(placeholder="instruction (blank = continue)", id="steer-input")
            with Horizontal(id="steer-buttons"):
                yield Button("Continue", id="continue", variant="success")
                yield Button("Send", id="send", variant="primary")
                yield Button("Abort", id="abort", variant="error")

    def on_mount(self) -> None:
        self.query_one("#steer-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "abort":
            self.dismiss("abort")
        elif event.button.id == "send":
            self.dismiss(self.query_one("#steer-input", Input).value)
        else:
            self.dismiss("")

    def action_cont(self) -> None:
        self.dismiss("")


class QuestionModal(ModalScreen[str]):
    """An agent->user question (`ask_user`): pick a numbered option (keys 1-9)
    or type a free-text answer. Esc submits empty (the agent gets the default).

    Result string = the chosen option text or the typed answer.
    """

    DEFAULT_CSS = """
    QuestionModal { align: center middle; }
    #question-box {
        width: 80%; max-width: 100; height: auto; max-height: 80%;
        border: thick $primary; padding: 1 2; background: $surface;
    }
    #question-options { height: auto; }
    #question-options Button {
        width: 100%; margin-top: 1; text-align: left;
    }
    #question-options Button:focus { text-style: bold reverse; }
    #question-input { margin-top: 1; }
    """

    BINDINGS: ClassVar = [
        Binding("escape", "skip", "Skip", show=True),
    ]

    def __init__(self, question_id: str, question: str, options: tuple[str, ...]) -> None:
        super().__init__()
        self.question_id = question_id
        self.question_text = question
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="question-box"):
            body = Text()
            body.append("The agent is asking:\n\n", style="bold")
            body.append(self.question_text)  # plain append: never parsed as markup
            yield Static(body)
            with Vertical(id="question-options"):
                # Buttons (not str labels) carry Text so a model-authored option
                # with '[...]' can't crash markup parsing.
                for i, opt in enumerate(self.options[:9], start=1):
                    label = Text(f"{i}. ")
                    label.append(opt)
                    yield Button(label, id=f"opt-{i}")
            yield Input(placeholder="…or type your own answer (Enter)", id="question-input")

    def on_mount(self) -> None:
        # Focus the first option if any, else the free-text field.
        if self.options:
            self.query_one("#opt-1", Button).focus()
        else:
            self.query_one("#question-input", Input).focus()

    def on_key(self, event: object) -> None:
        # Number keys 1-9 pick the matching option directly.
        key = getattr(event, "key", "")
        if key.isdigit() and key != "0":
            idx = int(key)
            if idx <= len(self.options):
                self.dismiss(self.options[idx - 1])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("opt-"):
            idx = int(bid.removeprefix("opt-"))
            self.dismiss(self.options[idx - 1])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_skip(self) -> None:
        self.dismiss("")
