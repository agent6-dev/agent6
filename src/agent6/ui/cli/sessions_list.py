# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""List sessions from the read model without starting the run engine."""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

from agent6 import kinds
from agent6 import paths as agent6_paths
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import common, sessions_children
from agent6.viewmodel import format
from agent6.viewmodel import listing as viewmodel_listing
from agent6.viewmodel.session import status as session_status


def cmd_list(*, as_json: bool = False, children: bool = False) -> int:
    """List this repo's sessions, newest activity first.

    Columns: updated, status (the mode folded in when the word does not imply it, the
    failure reason, the unmerged mark), usage, id, task. A group's children nest under
    its row: folded into a count, or listed indented with `children`; the JSON row always
    nests them. Every bucket, unlike the TUI and web hubs, which give `machine create`
    drafts their own card.

    Args:
        as_json: Print the rows as JSON.
        children: List each group's children.

    Returns:
        The exit code, 0.
    """
    listing, winners = viewmodel_listing.rows_for(
        pathlib.Path.cwd(), buckets=sessions_layout.SESSION_BUCKETS
    )
    if not listing:
        print(
            "[]" if as_json else common.nothing_yet()
        )  # the empty listing is output, not an error
        return 0
    state = agent6_paths.state_dir(pathlib.Path.cwd())
    rosters: dict[str, tuple[sessions_children.ChildRow, ...]] = {}
    for row in listing:
        sid = row.summary.session_id
        kind = kinds.SESSION_KINDS.get(row.summary.mode)
        parent = sessions_layout.bucket_dir(state, kind.bucket if kind else "runs") / sid
        rosters[sid] = sessions_children.read_children(parent, listed=row)
    if as_json:
        payload = [
            {
                **viewmodel_listing.row_json(row, winners=winners),
                "children": [child.as_dict() for child in rosters[row.summary.session_id]],
            }
            for row in listing
        ]
        print(json.dumps(payload, indent=2))
        return 0
    color = sys.stdout.isatty()

    def cells(
        row: viewmodel_listing.ListingRow, id_cell: str
    ) -> tuple[str, str, str, str, str, session_status.TaskLabel]:
        """Return a row's cells with the id cell as given."""
        s = row.summary
        label = s.merge.status_label(s.mode, s.status, s.detail)
        note = s.merge_note()
        styled, plain = common.styled_status(
            s.status, s.detail, color=color, label=f"{label} · {note}" if note else label
        )
        return (
            format.format_when(row.group_last_activity),
            styled,
            plain,
            s.usage_cell,
            id_cell,
            s.task_label,
        )

    rows: list[tuple[str, str, str, str, str, session_status.TaskLabel]] = []

    for row in listing:
        s = row.summary
        id_cell = s.id_cell(winner=s.session_id in winners)
        if (nested := rosters[s.session_id]) and not children:
            id_cell += f" ({format.child_count(len(nested))})"
        rows.append(cells(row, id_cell))
    status_w = max(6, *(len(plain) for _, _, plain, *_ in rows))
    id_w = max(2, *(len(r[4]) for r in rows))
    # The task column takes what a tty has left (floor 24); piped output keeps a fixed 60.
    fixed = 11 + 2 + status_w + 2 + 8 + 2 + id_w + 2
    task_w = max(24, shutil.get_terminal_size().columns - fixed) if color else 60
    print(f"{'updated':<11}  {'status':<{status_w}}  {'usage':<8}  {'id':<{id_w}}  task")
    for row, (when, styled, plain, usage, session_id, task) in zip(listing, rows, strict=True):
        pad = " " * (status_w - len(plain))
        task_lines = task.line(max_chars=task_w).splitlines()
        snip = task_lines[0] if task_lines else ""
        print(f"{when:<11}  {styled}{pad}  {usage:<8}  {session_id:<{id_w}}  {snip}")
        for continuation in task_lines[1:]:
            print(" " * fixed + continuation)
        if children:
            for child in rosters[row.summary.session_id]:
                print("\n".join(child.lines(depth=1)))
    return 0
