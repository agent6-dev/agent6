# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A live run with the same task ids on disk and in the dashboard's journal."""

from __future__ import annotations

import os
import pathlib

from agent6 import paths
from agent6.graph import curator, models
from agent6.sessions import event_log, journal, layout, process


def live_graph(repo: pathlib.Path, *, child_title: str = "Link pages") -> layout.SessionLayout:
    """Write an open root, an open child and a passed child under the repo's state directory."""
    run = layout.SessionLayout(paths.state_dir(repo), "task-run")
    graph = curator.GraphCurator(run)
    for parent, title in ((None, "Write index"), ("0001", child_title), ("0001", "Check paths")):
        graph.add_subtask(
            models.AddSubtaskIntent(
                parent_id=parent, draft=models.TaskNodeDraft(title=title, created_by="worker")
            )
        )
    graph.update_status(models.UpdateStatusIntent(id="0003", new_status="passed"))
    sink = event_log.EventSink(run.logs_path)
    sink.emit(
        journal.SessionStart(
            execution_id="first", session_id=run.session_id, mode="run", user_task="Write index"
        )
    )
    sink.emit(
        journal.GraphUpdate(
            nodes={k: v.model_dump(mode="json") for k, v in graph.nodes().items()}, cursor="0001"
        )
    )
    process.write_worker_pid(run.session_dir, os.getpid())
    return run
