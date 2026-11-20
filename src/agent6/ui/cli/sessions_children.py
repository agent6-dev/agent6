# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a parent's child roster in the CLI."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from agent6 import budget, terminal
from agent6.sessions import layout, manifest
from agent6.viewmodel import children, comparison, format, listing
from agent6.viewmodel.session import status


@dataclasses.dataclass(frozen=True, slots=True)
class ChildRow:
    """A dispatch view with the listing facts of a lineage-only group."""

    view: children.ChildView
    lineage: listing.SummaryRow | None
    placement: manifest.PlacementStamp | None
    nested: tuple[ChildRow, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the dispatch facts or the enriched lineage row."""
        facts = dataclasses.asdict(self.view)
        if self.lineage is not None:
            # A lineage row's model is its route. Its status and unmerged mark concern its base.
            facts.update(self.lineage)
        facts["children"] = [row.as_dict() for row in self.nested]
        return facts

    def lines(self, *, depth: int = 0) -> list[str]:
        """Return the child's status, task, meter and result as indented lines.

        Args:
            depth: The listing depth, or zero for a session header.

        Returns:
            The lines, including any nested children.
        """
        c = self.view
        who = f"{'reviewer' if c.source == 'panel' else 'child'} {c.number}"
        ident = self.lineage["id_cell"] if self.lineage is not None else c.session_id
        rank = f"rank {self.placement.rank}/{self.placement.of}  " if self.placement else ""
        heading = (
            f"{format.child_id_cell(ident, depth)}  {rank}{who}"
            if depth
            else f"  {rank}{who}  {ident}"
        )
        route = c.route or c.model or "?"
        if c.model and c.route and c.model != c.route:
            route += f" ({c.model})"
        state = self.lineage["status_label"] if self.lineage is not None else c.status
        if not c.launched and c.status == "failed":
            state += " (never started)"
        if c.outcome == "joined":
            state = c.join_label
        elif c.outcome and c.outcome != state:
            state += f" · {c.outcome.replace('_', ' ')}"
        if c.error and c.error not in state:
            state += f": {' '.join(c.error.split())}"
        lines = [f"{heading}  {state}  {route}"]
        indent = "  " * (max(1, depth) + 1)
        if c.task:
            lines.append(indent + status.task_snippet(c.task))
        spend = budget.format_usd(c.spend_usd, partial=c.usd_partial)
        meter = [f"turn {c.turn}" if c.turn is not None else "turn ?"]
        if c.current_tool:
            meter.append(c.current_tool)
        meter += [
            f"{c.tokens if c.tokens is not None else '?'} tokens",
            spend,
            f"elapsed {c.elapsed_s}s" if c.elapsed_s is not None else "elapsed ?",
        ]
        lines.append(indent + " · ".join(meter))
        if c.imported_ref or c.outcome in ("stopped", "retained"):
            count = (
                f" · {c.commits} {'commit' if c.commits == 1 else 'commits'}"
                if c.outcome in ("stopped", "retained")
                else ""
            )
            lines.append(f"{indent}ref: {c.imported_ref or '?'}{count}")
        if c.verdict_order or c.verdict_summary:
            verdict = ": ".join(filter(None, (" > ".join(c.verdict_order), c.verdict_summary)))
            lines.append(f"{indent}verdict: {verdict}")
        for row in self.nested:
            lines.extend(row.lines(depth=depth + 1))
        return [terminal.scrub_terminal_controls(line) for line in lines]


def read_children(
    parent_dir: pathlib.Path,
    *,
    tips: Mapping[str, str] | None = None,
    listed: listing.ListingRow | None = None,
) -> tuple[ChildRow, ...]:
    """Read the parent's dispatch roster or enrich a lineage-only group.

    Args:
        parent_dir: The parent's session directory.
        tips: The current branch tips for a lineage-only group.
        listed: The parent's listing row when nesting its subtree.

    Returns:
        Children in dispatch order. A session header includes only its own children.
    """
    dispatches = children.dispatches_of(parent_dir)
    if listed is not None and not listed.children and not dispatches:
        return ()
    by_id = {row.summary.session_id: row for row in listed.children} if listed else {}
    rows: list[ChildRow] = []
    for view in children.children_of(parent_dir, dispatches):
        listed_child = by_id.get(view.session_id)
        legacy = None
        placement = None
        nested: tuple[ChildRow, ...] = ()
        if view.launched:
            bucket = layout.EVIDENCE_BUCKET if view.source == "panel" else "runs"
            child_dir = parent_dir.parent.parent / bucket / view.session_id
            if not view.source:
                placement = comparison.session_placement(child_dir)
                summary = (
                    listed_child.summary
                    if listed_child
                    else listing.summarize_session_dir(child_dir, branch_tips=tips)
                )
                legacy = listing.summary_row(summary, winner=bool(placement and placement.winner))
            if listed_child is not None:
                nested = read_children(child_dir, listed=listed_child)
        rows.append(ChildRow(view, legacy, placement, nested))
    return tuple(rows)


def print_children(rows: Sequence[ChildRow]) -> None:
    """Print the parent header's child section when it has children."""
    if rows:
        print(f"children:      {len(rows)}")
        for row in rows:
            print("\n".join(row.lines()))
