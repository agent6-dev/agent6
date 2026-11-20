# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Show a session's children in the run dashboard."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Literal, cast

from rich import text
from textual import widgets

from agent6 import budget
from agent6.app import stop
from agent6.sessions import ipc, layout, manifest, process
from agent6.ui.tui import modals, theme
from agent6.viewmodel import children
from agent6.viewmodel.session import status as session_status

if TYPE_CHECKING:
    from agent6.ui.tui import app


def takes_action(child: children.ChildView) -> bool:
    """Return whether the displayed row represents a live child."""
    return child.launched and not child.outcome and child.status in session_status.LIVE_STATUS_WORDS


def child_cells(child: children.ChildView) -> tuple[text.Text, ...]:
    """Return the child row as literal text, including the reason for its status.

    Args:
        child: The parent's view of the child.

    Returns:
        One line per cell, in column order.
    """
    status = child.status
    result: list[str] = []
    if child.source == "panel":
        status = f"reviewer: {status}"
    if not child.launched and child.status == "failed":
        result.append("not started")
    if child.outcome in ("stopped", "retained"):
        if child.outcome != child.status:
            status += f" · {child.outcome}"
        result.append(f"{child.commits} commit{'' if child.commits == 1 else 's'}")
        if child.imported_ref:
            result.append(child.imported_ref)
    if child.error:
        result.append(child.error)
    if child.verdict_order:
        result.append(" > ".join(child.verdict_order))
    if child.verdict_summary:
        result.append(child.verdict_summary)
    return (
        text.Text(str(child.number)),
        text.Text(child.session_id),
        text.Text(status, style=theme.status_style(child.status)),
        text.Text(session_status.task_snippet(child.task, max_chars=28)),
        text.Text(child.current_tool),
        text.Text(str(child.turn) if child.turn is not None else ""),
        text.Text(str(child.tokens) if child.tokens is not None else ""),
        text.Text(
            budget.format_usd(child.spend_usd, partial=child.usd_partial)
            if child.launched or child.settled
            else ""
        ),
        text.Text(f"{child.elapsed_s}s" if child.elapsed_s is not None else ""),
        text.Text(child.route or child.model or ""),
        text.Text(" ".join(" · ".join(result).split())),
    )


class ChildTable(widgets.DataTable[text.Text]):
    """Keep the dashboard's child rows current even while the parent's journal is quiet."""

    DEFAULT_CSS = """
    ChildTable { border: round $primary; max-height: 9; }
    ChildTable:focus { border: round $accent; }
    DashboardScreen.-compact ChildTable { max-height: 7; }
    """

    def __init__(self, session_dir: pathlib.Path) -> None:
        """Point the table at the parent whose dashboard is open."""
        super().__init__(id="children", cursor_type="row", fixed_columns=1)
        self.session_dir = session_dir
        self._views: dict[str, children.ChildView] = {}

    def on_mount(self) -> None:
        """Start polling for children, including the first dispatch after mount."""
        self.add_columns(
            "#",
            "id",
            "status",
            "task",
            "tool",
            "turn",
            "tokens",
            "spend",
            "elapsed",
            "model",
            "result",
        )
        self.border_subtitle = "Enter opens · ← → columns"
        self.load_rows()
        self.set_interval(1.0, self.load_rows)

    @property
    def selected_child(self) -> children.ChildView | None:
        """Return the highlighted child, or None in an empty table."""
        if not self.row_count:
            return None
        key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key.value
        return self._views.get(key) if key is not None else None

    def load_rows(self) -> None:
        """Refresh changed cells in place and keep the selected id when rows move."""
        selected = self.selected_child
        views = {c.session_id: c for c in children.children_of(self.session_dir)}
        self.display = bool(views)
        if tuple(views) != tuple(self._views):
            self.clear()
            for sid, child in views.items():
                self.add_row(*child_cells(child), key=sid)
            if selected is not None and selected.session_id in views:
                self.move_cursor(row=self.get_row_index(selected.session_id))
            self.border_title = f"children: {len(views)}"
            self.styles.height = len(views) + 4
        else:
            for sid, child in views.items():
                if child != self._views[sid]:
                    for key, cell in zip(self.columns, child_cells(child), strict=True):
                        if cell != self.get_cell(sid, key):
                            self.update_cell(sid, key, cell, update_width=True)
        self._views = views
        current = self.selected_child
        if (takes_action(selected) if selected else None) != (
            takes_action(current) if current else None
        ):
            self.screen.refresh_bindings()

    def on_data_table_row_highlighted(self, event: widgets.DataTable.RowHighlighted) -> None:
        """Update the footer when the cursor moves between live and ended children."""
        event.stop()
        self.screen.refresh_bindings()

    def _directory(self, child: children.ChildView) -> pathlib.Path | None:
        """Return the child's exact directory for inspection, even with only its log."""
        if not child.launched or not layout.is_safe_session_id(child.session_id):
            return None
        bucket = layout.EVIDENCE_BUCKET if child.source == "panel" else "runs"
        target = (
            layout.bucket_dir(layout.layout_of(self.session_dir).state_dir, bucket)
            / child.session_id
        )
        return target if target != self.session_dir and target.is_dir() else None

    def _live_target(
        self, child: children.ChildView
    ) -> tuple[pathlib.Path, process.ProcessIdentity] | None:
        """Check both the parent's current roster and the child's worker identity.

        Args:
            child: The child named by the prompt.

        Returns:
            The verified directory and worker, or None when either has gone.
        """
        current = next(
            (
                row
                for row in children.children_of(self.session_dir)
                if row.session_id == child.session_id
                and row.group == child.group
                and row.source == child.source
            ),
            None,
        )
        if current is None or not takes_action(current):
            return None
        directory = self._directory(current)
        if directory is None:
            return None
        try:
            record = manifest.read_manifest(directory)
        except manifest.ManifestError:
            return None
        # A reviewer's lineage names its batch. Its row belongs to the whole panel.
        if (
            record.session_id != child.session_id
            or record.parallel is None
            or record.parallel.parent != self.session_dir.name
        ):
            return None
        worker = process.read_live_worker_identity(directory)
        return (directory, worker) if worker is not None else None

    def prompt_child(self, action: Literal["stop_child", "message_child"]) -> None:
        """Name the selected child in the prompt and check the same worker before writing.

        Args:
            action: The dashboard action for the highlighted row.
        """
        child = self.selected_child
        if child is None or not takes_action(child):
            return
        target = self._live_target(child)
        if target is None:
            self.notify(
                f"Cannot verify a live worker for child {child.session_id}.", severity="warning"
            )
            return
        directory, _worker = target

        def submit(answer: str | bool | None) -> None:
            if not answer or (isinstance(answer, str) and not answer.strip()):
                return
            if self._live_target(child) != target:
                self.notify(
                    f"Child {child.session_id} changed or ended. Nothing sent.", severity="warning"
                )
            elif answer is True:
                outcome = stop.stop_session(directory, request_only=True)
                self.notify(outcome.message, severity="information" if outcome.ok else "warning")
            # A message is prose even when its first word is a steer verb or a directive.
            elif ipc.submit_steer(directory, f"Message from the operator:\n{answer.strip()}"):
                self.notify(f"Message queued for child {child.session_id}.")
            else:
                self.notify(f"Could not message child {child.session_id}.", severity="warning")

        if action == "stop_child":
            self.app.push_screen(
                modals.ConfirmModal(
                    f"Stop child {child.session_id}?",
                    "Request an immediate stop for this child.",
                    confirm_label="Stop child",
                ),
                submit,
            )
        else:
            self.app.push_screen(
                modals.TextInputModal(
                    f"Message child {child.session_id}", "Instruction for its next turn"
                ),
                submit,
            )

    def on_data_table_row_selected(self, event: widgets.DataTable.RowSelected) -> None:
        """Open the exact session, never an id that a missing child prefixes."""
        event.stop()
        sid = event.row_key.value
        child = self._views.get(sid) if sid is not None else None
        if child is not None and (target := self._directory(child)) is not None:
            cast("app.Agent6TUI", self.app).open_session(target)
            return
        self.notify(f"No session to open for child {sid}.", severity="warning")
