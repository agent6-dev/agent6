# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Discover child sessions across imported records and live clone links."""

from __future__ import annotations

import pathlib
from collections.abc import Collection

from agent6.sessions import layout, manifest


def children_of(session_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Find direct children in both the visible state and a live clone's resolved state.

    Args:
        session_dir: A session directory or its live link in an ancestor repository.

    Returns:
        Actual child paths across every session bucket, including ended children.
    """
    states = {layout.layout_of(p).state_dir for p in (session_dir, session_dir.resolve())}
    found: dict[str, pathlib.Path] = {}
    for state in sorted(states):
        for bucket in layout.SESSION_BUCKETS:
            directory = layout.bucket_dir(state, bucket)
            if not directory.is_dir():
                continue
            for child in sorted(directory.iterdir()):
                try:
                    saved = manifest.read_manifest(child)
                except manifest.ManifestError:
                    continue
                if saved.parallel is not None and saved.parallel.coordinator == session_dir.name:
                    found[child.name] = child
    return tuple(found[key] for key in sorted(found))


def descendants_of(session_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Find descendants through live links and dead intermediaries, without revisiting a record.

    Args:
        session_dir: The ancestor directory.

    Returns:
        Descendants in parent-before-child order.
    """
    seen = {session_dir.resolve()}
    found: list[pathlib.Path] = []
    pending = list(children_of(session_dir))
    while pending:
        child = pending.pop(0)
        resolved = child.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(child)
        pending.extend(children_of(child))
    return tuple(found)


def seat_records(state_dir: pathlib.Path, parents: Collection[str]) -> tuple[pathlib.Path, ...]:
    """Find the bounded read-only conversations the named sessions dispatched.

    No listing shows a seat's record, so whatever removes its parent removes it too.

    Args:
        state_dir: The repository's state dir.
        parents: The dispatching sessions' ids.

    Returns:
        The seat records, in name order.
    """
    directory = layout.bucket_dir(state_dir, layout.EVIDENCE_BUCKET)
    if not parents or not directory.is_dir():
        return ()
    found: list[pathlib.Path] = []
    for child in sorted(directory.iterdir()):
        try:
            saved = manifest.read_manifest(child)
        except manifest.ManifestError:
            continue
        if saved.parallel is not None and saved.parallel.coordinator in parents:
            found.append(child)
    return tuple(found)
