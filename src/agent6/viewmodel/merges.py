# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read a run's merge state from its checkout stamp and its parent's dispatch."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping

from agent6.sessions import child_dispatch, layout, manifest
from agent6.vcs.git import read as git_read
from agent6.viewmodel import format


@dataclasses.dataclass(frozen=True, slots=True)
class MergeState:
    """The unmerged mark or the child's recorded merge into its parent's chain."""

    unmerged: bool = False
    merged_into_parent: str | None = None
    parent_merge_sha: str | None = None

    def status_label(self, mode: str, status: str, reason: format.StatusReason) -> str:
        """Return the session's status with its merge mark."""
        return format.listing_status_label(
            mode, status, reason, unmerged=self.unmerged, merged_into_parent=self.merged_into_parent
        )


def read_merge_state(
    session_dir: pathlib.Path,
    record: manifest.SessionManifest | None,
    word: str,
    branch_tips: Mapping[str, str] | None,
) -> MergeState:
    """Read the merge facts for a session.

    Args:
        session_dir: The session's directory.
        record: Its manifest, or None when unreadable.
        word: Its status word. An undone run's child holds its work.
        branch_tips: The repository's run refs, or None without a snapshot.

    Returns:
        The recorded parent-chain merge, or whether a run tip lacks a checkout merge.
    """
    if record is None or record.git_control == "model" or word == "undone":
        return MergeState()
    lineage = record.parallel
    if lineage and all(layout.is_safe_session_id(s) for s in (lineage.parent, lineage.group)):
        parent = session_dir.parent / lineage.parent
        try:
            dispatch = child_dispatch.read_record(child_dispatch.record_path(parent, lineage.group))
            merge = dispatch.child(session_dir.name).merge
        except (KeyError, child_dispatch.DispatchRecordError):
            pass
        else:
            if dispatch.identity == lineage.group and merge and merge.status == "merged":
                return MergeState(
                    merged_into_parent=parent.name, parent_merge_sha=merge.sha or None
                )
    if not branch_tips:
        return MergeState()
    tips = {
        branch_tips[ref]
        for ref in (git_read.chain_ref_for(record.session_id), record.run_branch or "")
        if branch_tips.get(ref)
    }
    # A checkout merge stamps the tip it merged, which can be on the run branch or the chain.
    return MergeState(
        unmerged=bool(tips - {record.base_sha})
        and (record.merged is None or record.merged.tip not in tips)
    )
