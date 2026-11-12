"""Read lane placements and their parent-owned comparison records."""

from __future__ import annotations

import pathlib

from agent6.sessions import layout
from agent6.sessions import manifest as sessions_manifest


def session_placement(session_dir: pathlib.Path) -> sessions_manifest.PlacementStamp | None:
    """Return the placement stamp on a lane's manifest.

    The event fold does not carry it, so every run view reads it from here.

    Args:
        session_dir: The session's state dir.

    Returns:
        The stamp, or None for a run outside a compared group or with an unreadable
        manifest.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
    except sessions_manifest.ManifestError:
        return None
    return manifest.placement


def is_winner(session_dir: pathlib.Path) -> bool:
    """Return whether a run is its group's compare winner."""
    placement = session_placement(session_dir)
    return placement is not None and placement.winner


def session_comparison(session_dir: pathlib.Path) -> sessions_manifest.ComparisonStamp | None:
    """Read a parent's comparison, directly or through a lane's group reference.

    Args:
        session_dir: The parent or lane's session directory.

    Returns:
        The group record, or None when the owner is missing, unreadable or names another group.
    """
    try:
        manifest = sessions_manifest.read_manifest(session_dir)
        if manifest.comparison is not None:
            return manifest.comparison
        if manifest.placement is None or not manifest.placement.group:
            return None
        group = manifest.placement.group
        owner_id = manifest.parallel.parent if manifest.parallel is not None else group
        owner = layout.session_layout(layout.layout_of(session_dir).state_dir, owner_id)
        if owner is None or owner.session_id != owner_id:
            return None
        record = sessions_manifest.read_manifest(owner.session_dir).comparison
        return record if record is not None and record.group == group else None
    except sessions_manifest.ManifestError:
        return None
