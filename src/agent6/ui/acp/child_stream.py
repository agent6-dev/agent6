# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Show a child's journal in the editor that owns its parent."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from agent6 import child_records, config
from agent6.app import children, reporter
from agent6.sessions import event_log, layout
from agent6.ui import spawn
from agent6.viewmodel import tail, transcript


def parent_spawner(
    cfg: config.Config,
    explicit_leaves: frozenset[str],
    cwd: pathlib.Path,
    state_dir: pathlib.Path,
    mode: str,
    session_id: str,
    max_usd: float | None,
    auto_approve: bool,
    parent_events: event_log.EventSink,
    watch: Callable[[pathlib.Path, str, str], bool],
    *,
    say: Callable[[str], None],
) -> children.GroupSpawner | None:
    """Observe the guarded spawner without changing its budget or approval watch.

    Args:
        cfg: The parent config.
        explicit_leaves: The settings the operator set explicitly.
        cwd: The parent checkout.
        state_dir: The repository state directory.
        mode: The parent mode.
        session_id: The parent run id.
        max_usd: The spend ceiling.
        auto_approve: The operator's approval override.
        parent_events: The parent execution's journal.
        watch: The parent's budget and approval watch.
        say: Send a notice to the parent's client.

    Returns:
        The observed spawner, or None when this session cannot dispatch children.
    """

    def observe(
        spec: child_records.ChildSpec, directory: pathlib.Path, continuing: bool
    ) -> Callable[[], None]:
        reader = tail.LogTail(directory / layout.LOGS_NAME)
        if continuing:
            reader.read()  # The next execution appends to the same journal.
        fold = transcript.TranscriptFold()
        label = (
            f"{'reviewer' if spec.mode == 'reviewer' else 'child'} {spec.number} {spec.session_id}"
        )

        def drain() -> None:
            for row in reader.read():
                if row.get("type") == "model.call":
                    say(f"{label}: calling {row.get('provider')}/{row.get('model')}")
                for item in fold.feed(row):
                    detail = " ".join(filter(None, (item.name, item.arg, item.detail, item.body)))
                    action = "failed" if item.ok is False else item.kind
                    say(f"{label}: {action}: {' '.join(detail.splitlines())}")

        return drain

    return children.build_parent_spawner(
        cfg,
        cwd,
        state_dir,
        explicit_leaves=explicit_leaves,
        mode=mode,
        session_id=session_id,
        runtime=children.ChildRuntime(spawn=spawn.spawn_child, observe=observe),
        max_usd=max_usd,
        auto_approve=auto_approve,
        watch=watch,
        parent_events=parent_events,
        reporter=reporter.Reporter(out=say, err=say, receipt=say),
    )
