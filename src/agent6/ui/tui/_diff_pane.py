# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dashboard's diff pane.

A step picker and a cumulative toggle over the latest commit's patch, the verify
output, a selected step's patch, or the commits made while a selected task was
in focus.
"""

from __future__ import annotations

import contextlib
import pathlib

from rich import text
from textual import app, message, widgets

from agent6 import git_ops
from agent6.sessions import manifest
from agent6.ui.tui import forms
from agent6.viewmodel import format, state


class DiffPane(forms.ScrollPane):
    """The diff pane; a picker change posts `StepChanged`, since the header follows the step.

    Attributes:
        step_sel: The selected step's sha; "" for the latest commit.
        cumulative: Whether the patch runs from the session's base to the step.
    """

    DEFAULT_CSS = """
    /* The picker and the toggle share a row; a focus border would make it three lines. */
    DiffPane #diff-cumulative { margin-left: 2; background: transparent; }
    DiffPane #diff-cumulative:focus { border: none; }
    /* The body fills the pane so long content scrolls; selectable text shows an I-beam. */
    DiffPane #diff-body { width: 1fr; height: auto; pointer: text; }
    """

    class StepChanged(message.Message):
        """The selected step or the cumulative toggle changed."""

    def __init__(self, session_dir: pathlib.Path, *, id: str) -> None:
        """Bind the pane to a session dir."""
        super().__init__(id=id)
        self._session_dir = session_dir
        self.step_sel = ""
        self.cumulative = False
        self._nav_steps = -1  # how many steps the selector lists
        self._rendered: tuple[object, ...] | None = None  # the inputs last painted

    def compose(self) -> app.ComposeResult:
        """Lay out the pane.

        Yields:
            The picker row and the body.
        """
        with forms.PickerRow(id="diff-nav"):
            yield forms.Picker(
                [("latest commit", "")],
                value="",
                allow_blank=False,
                opens="down",
                id="diff-step",
            )
            yield widgets.Checkbox("cumulative", compact=True, id="diff-cumulative", disabled=True)
        yield widgets.Static("", id="diff-body")

    def git_control(self) -> str:
        """Return the manifest's `git_control`; "agent6" when the manifest is unreadable."""
        with contextlib.suppress(manifest.ManifestError):
            return manifest.read_manifest(self._session_dir).git_control
        return "agent6"

    def sync_nav(self, s: state.SessionState) -> None:
        """Refresh the step selector from the state's commits, newest first.

        Hidden while nothing is committed, and under `[git].control = "model"`, which
        has no chain to select from.
        """
        nav = self.query_one("#diff-nav", forms.PickerRow)
        if self.git_control() == "model" or not s.steps:
            nav.display = False
            return
        nav.display = True
        if len(s.steps) != self._nav_steps:
            self._nav_steps = len(s.steps)
            options = [("latest commit", "")]
            options.extend((st.label, st.sha) for st in reversed(s.steps))
            select = self.query_one("#diff-step", widgets.Select)
            select.set_options(options)
            select.value = self.step_sel if any(v == self.step_sel for _, v in options) else ""
            self._sync_cumulative()

    def _step_patch(self, sha: str) -> str:
        if self.cumulative:
            with contextlib.suppress(manifest.ManifestError):
                base = manifest.read_manifest(self._session_dir).base_sha
                if base:
                    return git_ops.diff_range(pathlib.Path.cwd(), base, sha) or "(no diff)"
        return git_ops.commit_diff(pathlib.Path.cwd(), sha) or "(no diff)"

    def on_select_changed(self, event: widgets.Select.Changed) -> None:
        """Follow the step picker."""
        if event.select.id != "diff-step":
            return
        event.stop()
        self.step_sel = str(event.value or "")
        self._sync_cumulative()
        self.post_message(self.StepChanged())

    def _sync_cumulative(self) -> None:
        """Disable and clear the cumulative box unless a step is chosen, as the web does."""
        box = self.query_one("#diff-cumulative", widgets.Checkbox)
        box.disabled = not self.step_sel
        if not self.step_sel and box.value:
            box.value = False

    def on_checkbox_changed(self, event: widgets.Checkbox.Changed) -> None:
        """Follow the cumulative toggle."""
        if event.checkbox.id != "diff-cumulative":
            return
        event.stop()
        self.cumulative = bool(event.value)
        self.post_message(self.StepChanged())

    def render_state(self, s: state.SessionState, *, sel: str | None, filt: str) -> None:
        """Paint the pane for the state, skipping when none of its inputs changed.

        Built as rich Text, so a diff or verify body holding brackets is never parsed
        as markup.

        Args:
            s: The session's fold.
            sel: The task the panes are filtered to; None for the live view.
            filt: The border title's suffix for the filter.
        """
        self.sync_nav(s)
        key = (sel, s.recent_diffs, s.last_verify, s.latest_diff, self.step_sel, self.cumulative)
        if self._rendered is not None and all(
            a is b for a, b in zip(self._rendered, key, strict=True)
        ):
            return
        self._rendered = key
        self.border_title = (
            "diff · the model owns git"
            if self.git_control() == "model"
            else (f"diff{filt}" if sel else "")
        )
        self.query_one("#diff-body", widgets.Static).update(self._story(s, sel))

    def _story(self, s: state.SessionState, sel: str | None) -> text.Text:
        verify = s.last_verify
        dt = text.Text()
        if self.step_sel:
            step = next((st for st in s.steps if st.sha == self.step_sel), None)
            if step is not None:
                what = "cumulative to" if self.cumulative else "step"
                dt.append(f"{what} {step.label}\n", style="bold")
                append_colored_diff(dt, self._step_patch(step.sha), cap=4000)
                return dt
        if sel is not None:
            task_diffs = [d for d in s.recent_diffs if d.task_id == sel]
            if task_diffs:
                n = len(task_diffs)
                dt.append(f"selected task · {n} commit{'s' if n != 1 else ''}\n", style="bold")
                append_colored_diff(dt, task_diffs[-1].patch, cap=2000)
            else:
                dt.append("(no commits during the selected task yet)", style="dim")
            return dt
        # A running or failed verify outranks the diff, so a failure never hides behind a stale one.
        if verify is not None and verify.exit_code is None:
            dt.append("verify running: ", style="bold")
            dt.append(format.clip_cell(" ".join(verify.cmd), 200) + "\n")
            dt.append("…", style="dim")
        elif verify is not None and verify.exit_code != 0:
            dt.append(f"verify exit={verify.exit_code} ", style="bold red")
            dt.append(
                f"({verify.duration_s:.1f}s)  {format.clip_cell(' '.join(verify.cmd), 160)}\n"
            )
            out = verify.stderr_tail or verify.stdout_tail
            dt.append(out[:2000] or "(no output)")
            if len(out) > 2000:
                dt.append("\n… (truncated)", style="dim")
        elif s.latest_diff:
            dt.append("latest commit diff\n", style="bold")
            append_colored_diff(dt, s.latest_diff, cap=2000)
        elif verify is not None:
            dt.append(f"verify passed ({verify.duration_s:.1f}s)", style="bold green")
        else:
            dt.append("(no diffs yet)", style="dim")
        return dt


def append_colored_diff(dt: text.Text, patch: str, *, cap: int = 0) -> None:
    """Append a unified diff with its lines coloured, without markup parsing.

    Args:
        dt: The text to append to.
        patch: The unified diff.
        cap: Clip the patch to this many characters and mark the cut; 0 for no cap.
    """
    shown = patch if not cap or len(patch) <= cap else patch[:cap]
    for line in shown.splitlines():
        if line.startswith("+") and not line.startswith("+++ "):
            dt.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("--- "):
            dt.append(line + "\n", style="red")
        elif line.startswith("@@"):
            dt.append(line + "\n", style="cyan")
        else:
            dt.append(line + "\n")
    if cap and len(patch) > cap:
        dt.append("… (truncated; `sessions diff` for the full patch)\n", style="dim")
