# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dashboard's diff pane: a step picker and a cumulative toggle over the
latest commit's patch, the verify output, a selected step's patch, or the
commits made while a selected task was in focus."""

from __future__ import annotations

import contextlib
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Checkbox, Select, Static

from agent6.git_ops import commit_diff, diff_range
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.ui.tui.widgets import Picker, PickerRow, ScrollPane
from agent6.viewmodel.format import clip_cell
from agent6.viewmodel.state import SessionState


class DiffPane(ScrollPane):
    """`step_sel` is the selected step's sha ("" = latest), `cumulative` the
    toggle; a change posts `StepChanged`, since the header's task count and
    cost are as of the selected step too."""

    DEFAULT_CSS = """
    /* The step picker and the cumulative toggle share a picker row; the compact
       toggle's focus border would make it three lines. */
    DiffPane #diff-cumulative { margin-left: 2; background: transparent; }
    DiffPane #diff-cumulative:focus { border: none; }
    /* The body fills the pane so long content scrolls; it is selectable text,
       so the pointer shows an I-beam over it. */
    DiffPane #diff-body { width: 1fr; height: auto; pointer: text; }
    """

    class StepChanged(Message):
        """The selected step or the cumulative toggle changed."""

    def __init__(self, session_dir: Path, *, id: str) -> None:
        super().__init__(id=id)
        self._session_dir = session_dir
        self.step_sel = ""
        self.cumulative = False
        self._nav_steps = -1  # how many steps the selector lists
        self._rendered: tuple[object, ...] | None = None  # the inputs last painted

    def compose(self) -> ComposeResult:
        with PickerRow(id="diff-nav"):
            yield Picker(
                [("latest commit", "")],
                value="",
                allow_blank=False,
                opens="down",
                id="diff-step",
            )
            yield Checkbox("cumulative", compact=True, id="diff-cumulative", disabled=True)
        yield Static("", id="diff-body")

    def git_control(self) -> str:
        with contextlib.suppress(ManifestError):
            return read_manifest(self._session_dir).git_control
        return "agent6"

    def sync_nav(self, s: SessionState) -> None:
        """The step selector lists the run's commits (newest first) behind
        "latest commit"; hidden while nothing is committed, and under
        `[git].control = "model"` the pane says so (no chain to select from)."""
        nav = self.query_one("#diff-nav", PickerRow)
        if self.git_control() == "model" or not s.steps:
            nav.display = False
            return
        nav.display = True
        if len(s.steps) != self._nav_steps:
            self._nav_steps = len(s.steps)
            options = [("latest commit", "")]
            options.extend((st.label, st.sha) for st in reversed(s.steps))
            select = self.query_one("#diff-step", Select)
            select.set_options(options)
            select.value = self.step_sel if any(v == self.step_sel for _, v in options) else ""
            self._sync_cumulative()

    def _step_patch(self, sha: str) -> str:
        if self.cumulative:
            with contextlib.suppress(ManifestError):
                base = read_manifest(self._session_dir).base_sha
                if base:
                    return diff_range(Path.cwd(), base, sha) or "(no diff)"
        return commit_diff(Path.cwd(), sha) or "(no diff)"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "diff-step":
            return
        event.stop()
        self.step_sel = str(event.value or "")
        self._sync_cumulative()
        self.post_message(self.StepChanged())

    def _sync_cumulative(self) -> None:
        """Cumulative applies to a chosen step: with "latest commit" picked the
        box is off and disabled, as the web's is."""
        box = self.query_one("#diff-cumulative", Checkbox)
        box.disabled = not self.step_sel
        if not self.step_sel and box.value:
            box.value = False

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "diff-cumulative":
            return
        event.stop()
        self.cumulative = bool(event.value)
        self.post_message(self.StepChanged())

    def render_state(self, s: SessionState, *, sel: str | None, filt: str) -> None:
        """Paint the pane for the state: *sel* is the task the panes are
        filtered to (None = the live view), *filt* its border-title suffix.
        Built as rich Text, so a diff or verify body (which holds brackets) is
        never parsed as markup. Skipped whenever none of its inputs changed."""
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
        self.query_one("#diff-body", Static).update(self._story(s, sel))

    def _story(self, s: SessionState, sel: str | None) -> Text:
        verify = s.last_verify
        dt = Text()
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
        # A running or failed verify takes precedence so a failure is never
        # hidden behind a stale passing diff. A passed verify yields to the diff.
        if verify is not None and verify.exit_code is None:
            dt.append("verify running: ", style="bold")
            dt.append(clip_cell(" ".join(verify.cmd), 200) + "\n")
            dt.append("…", style="dim")
        elif verify is not None and verify.exit_code != 0:
            dt.append(f"verify exit={verify.exit_code} ", style="bold red")
            dt.append(f"({verify.duration_s:.1f}s)  {clip_cell(' '.join(verify.cmd), 160)}\n")
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


def append_colored_diff(dt: Text, patch: str, *, cap: int = 0) -> None:
    """Append a unified diff with +/- line coloring (no markup parsing). With
    *cap*, clip to it and mark the cut, so a truncated patch never reads as the
    whole one (the pane is a preview; `sessions diff` prints the full patch)."""
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
