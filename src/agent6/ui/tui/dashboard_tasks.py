# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dashboard's task tree."""

from __future__ import annotations

from rich import text
from textual import widgets

from agent6 import terminal
from agent6.graph import order
from agent6.viewmodel.session import records


class TaskTree(widgets.Tree[str]):
    """Show the graph at the chosen commit and mark the operator's selection."""

    def __init__(self) -> None:
        super().__init__("tasks", id="plan")
        self.show_root = False
        self._tasks: tuple[records.TaskNodeView, ...] | None = None
        self._selection: tuple[str | None, str] = (None, "")

    def show_tasks(
        self, tasks: tuple[records.TaskNodeView, ...], selected: str | None, as_of: str
    ) -> None:
        """Rebuild only when the graph, selection or commit changes."""
        if tasks is self._tasks and (selected, as_of) == self._selection:
            return
        self._tasks, self._selection = tasks, (selected, as_of)
        cursor = self.cursor_node.data if self.cursor_node is not None else None
        self.clear()
        done = sum(t.status in order.DONE_STATUSES for t in tasks)
        self.border_title = (f"tasks: {done}/{len(tasks)}" if tasks else "tasks") + as_of
        self.root.expand()
        for task in tasks:
            indent = "  " * task.depth
            title = terminal.scrub_terminal_controls(task.title)
            label = text.Text(f"{task.short_id:>3} {indent}{task.glyph} {title}")
            if task.note:
                label.append(f"  {task.note}", style="dim italic")
            if task.id == selected:
                label.stylize("bold reverse")
            node = self.root.add_leaf(label, data=task.id)
            if task.id == cursor:
                self.move_cursor(node)
