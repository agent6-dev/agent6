from __future__ import annotations

import json
import pathlib
import uuid

import pydantic

from agent6 import review_records
from agent6.vcs import git_read


class ReviewHistoryError(Exception):
    """The journal's structured review evidence cannot be read."""


def load_panels(path: pathlib.Path) -> tuple[review_records.ReviewPanel, ...]:
    """Read structured panels in append order.

    Args:
        path: The session journal.

    Returns:
        Structured panels; legacy counts carry no restorable findings.

    Raises:
        ReviewHistoryError: The journal or a structured panel cannot be read.
    """
    panels: list[review_records.ReviewPanel] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (
                    isinstance(row, dict)
                    and row.get("type") == "loop.review.panel"
                    and "target" in row
                ):
                    panels.append(review_records.ReviewPanel.model_validate(row))
    except FileNotFoundError:
        return ()
    except (OSError, pydantic.ValidationError) as exc:
        raise ReviewHistoryError(f"cannot read review evidence at {path}: {exc}") from exc
    return tuple(panels)


def _blob(
    root: pathlib.Path, target: review_records.ReviewTarget, finding: review_records.Finding
) -> str:
    tree = target.base_tree if finding.location.side == "base" else target.head_tree
    entry = git_read.tree_entry(root, tree, finding.location.path).split()
    return entry[2] if len(entry) == 3 and entry[1] == "blob" else ""


def matching_findings(
    root: pathlib.Path,
    target: review_records.ReviewTarget,
    panels: tuple[review_records.ReviewPanel, ...],
) -> tuple[dict[str, str], tuple[review_records.Finding, ...]]:
    """Return known identities and injected findings whose cited blobs still match."""
    known: dict[str, str] = {}
    shown: dict[str, review_records.Finding] = {}
    injected = {fid for panel in panels for fid in panel.injected_finding_ids}
    for panel in panels:
        for observation in panel.findings:
            finding = observation.finding
            old_blob = _blob(root, panel.target, finding)
            if not old_blob or old_blob != _blob(root, target, finding):
                continue
            key = review_records.finding_key(finding)
            known[key] = observation.finding_id
            if observation.finding_id in injected:
                shown[key] = finding
    return known, tuple(shown.values())


def observations(
    panel_id: str,
    verdicts: tuple[review_records.ReviewVerdict, ...],
    known: dict[str, str],
) -> tuple[review_records.FindingObservation, ...]:
    """Return every grounded finding, including repeats omitted from feedback."""
    grouped: dict[str, tuple[review_records.Finding, list[int]]] = {}
    for verdict in verdicts:
        for finding in verdict.findings:
            _, seats = grouped.setdefault(review_records.finding_key(finding), (finding, []))
            if verdict.seat_index not in seats:
                seats.append(verdict.seat_index)
    return tuple(
        review_records.FindingObservation(
            finding_id=known.get(key) or uuid.uuid4().hex,
            panel_id=panel_id,
            finding=finding,
            seat_indexes=tuple(seats),
        )
        for key, (finding, seats) in grouped.items()
    )
