# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child's journal supplies its commit previews until the parent has the commit."""

from __future__ import annotations

import dataclasses
import pathlib
import re

from agent6.sessions import journal, layout, manifest
from agent6.vcs.git import read as git_read
from agent6.viewmodel import events, tail


@dataclasses.dataclass(frozen=True, slots=True)
class RecordedPatch:
    """The journaled patch and its cut notice, or an empty patch and the reason."""

    patch: str
    note: str


def child_patch(
    repo: pathlib.Path,
    session_dir: pathlib.Path,
    run: manifest.SessionManifest,
    sha: str,
    *,
    cumulative: bool,
) -> RecordedPatch | None:
    """Look up the selected child checkpoint in its journal, including older picks.

    Args:
        repo: The parent's repository.
        session_dir: The session record in the parent's state directory.
        run: The session's manifest.
        sha: The selected commit id.
        cumulative: Whether the selection asks for a range from the base.

    Returns:
        The preview or a diagnostic. None means to read the patch from the repository.
    """
    if run.parallel is None or not run.parallel.parent:
        return None
    if not re.fullmatch(r"[0-9a-f]{7,64}", sha):
        return RecordedPatch("", f"not a commit sha: {sha!r}")
    selected = ""
    for row in tail.tail_events(session_dir / layout.LOGS_NAME, follow=False):
        match row:
            case {"type": "checkpoint.committed", "sha": str(commit)} if commit.startswith(sha):
                tip = git_read.chain_tip(repo, sha)
                if tip is not None and tip.startswith(commit):
                    return None
                if cumulative:
                    return RecordedPatch(
                        "",
                        "Cumulative diff available after the parent imports this child's commits",
                    )
                selected = commit
            case {"type": "diff.updated", "sha": str(commit)} if selected and commit == selected:
                event = events.parse_event(row)
                assert isinstance(event, journal.DiffUpdated)
                if not event.patch:
                    break
                note = ""
                if event.truncated:
                    note = "Recorded patch truncated"
                elif event.truncated is None and len(event.patch) == journal.DiffUpdated.MAX_CHARS:
                    note = "Recorded patch possibly truncated"
                return RecordedPatch(event.patch, note)
            case _:
                continue
    return RecordedPatch("", f"No patch recorded for {sha}")
