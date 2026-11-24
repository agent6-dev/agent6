# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An edit reaches the editor as a reviewable diff, the same live and on replay."""

from __future__ import annotations

import io
import pathlib
import shutil
from typing import Any

from agent6.config import Config
from agent6.sessions import event_log, journal
from agent6.tools import dispatch, results
from agent6.ui.acp import runner, updates
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as session_mod
from agent6.viewmodel import tail
from tests.acp import schema_check

_RUN = "run-x"


class _Run:
    """A recorded run: a workspace, and a session dir a real dispatcher journals into."""

    def __init__(self, tmp_path: pathlib.Path, config: Config | None = None) -> None:
        self.repo, self.session_dir = tmp_path / "repo", tmp_path / "session"
        self.repo.mkdir()
        self.session_dir.mkdir()
        self.logs, self.objects = self.session_dir / "logs.jsonl", self.session_dir / "objects"
        sink = event_log.EventSink(self.logs)
        self.tools = dispatch.ToolDispatcher(
            root=self.repo, config=config or Config(), session_dir=self.session_dir, events=sink
        )
        sink.emit(
            journal.SessionStart(mode="run", user_task="t", execution_id=self.tools.execution_id)
        )

    def edit(self, path: str, **edit: str) -> None:
        self.tools.dispatch("apply_edit", {"path": path, "edits": [edit]})

    def live(self) -> list[dict[str, Any]]:
        """Return the tool updates the tail sends while the run writes its journal."""
        sent: list[dict[str, Any]] = []
        server = acp_server.ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
        server.notify_raw = sent.append  # pyrefly: ignore[bad-assignment, method-assign]
        session = session_mod.Session(acp_id=_RUN, cwd=self.repo, session_id=_RUN)
        runner.RunBridge(server=server)._stream(
            session, self.logs, lambda: True, 0, runner.Announced(), objects_dir=self.objects
        )
        return _settled(sent)

    def replayed(self) -> list[dict[str, Any]]:
        """Return the tool updates `session/load` sends for the recorded journal."""
        sent: list[dict[str, Any]] = []
        updates.replay(
            tail.tail_events(self.logs, follow=False),
            acp_session_id=_RUN,
            cwd=self.repo,
            objects_dir=self.objects,
            notify=sent.append,
        )
        return _settled(sent)


def _settled(sent: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return each settled call's update, every frame checked against the protocol schema."""
    for body in sent:
        assert not schema_check.errors("SessionNotification", body["params"]), body
    frames = [body["params"]["update"] for body in sent]
    return [frame for frame in frames if frame["sessionUpdate"] == "tool_call_update"]


def _diffs(update: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in update["content"] if entry["type"] == "diff"]


def _text(update: dict[str, Any]) -> str:
    return "\n".join(e["content"]["text"] for e in update["content"] if e["type"] == "content")


def test_two_writes_to_one_file_show_two_different_diffs(tmp_path: pathlib.Path) -> None:
    """Each call shows the text it read and the text it wrote, not the file's last state."""
    run = _Run(tmp_path)
    run.edit("a.py", kind="create", new_string="one\n")
    run.edit("a.py", kind="replace", old_string="one", new_string="two")
    path = str((run.repo / "a.py").resolve())

    first, second = run.live()

    assert _diffs(first) == [{"type": "diff", "path": path, "oldText": None, "newText": "one\n"}]
    assert _diffs(second) == [
        {"type": "diff", "path": path, "oldText": "one\n", "newText": "two\n"}
    ]
    # The summary the call carried before the diff stays, as the first entry.
    assert first["content"][0]["type"] == "content" and "path=a.py" in _text(first)
    assert first["locations"] == [{"path": path}]


def test_a_patch_sends_one_diff_per_file_with_a_create_and_a_delete(
    tmp_path: pathlib.Path,
) -> None:
    run = _Run(tmp_path)
    (run.repo / "keep.py").write_text("old\n", encoding="utf-8")
    (run.repo / "gone.py").write_text("bye\n", encoding="utf-8")
    patch = (
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+fresh\n"
        "diff --git a/keep.py b/keep.py\n--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
    )
    run.tools.dispatch("apply_patch", {"patch": patch})
    names = ("new.py", "keep.py", "gone.py")
    new, keep, gone = (str((run.repo / name).resolve()) for name in names)

    (update,) = run.live()

    assert _diffs(update) == [
        {"type": "diff", "path": new, "oldText": None, "newText": "fresh\n"},
        {"type": "diff", "path": keep, "oldText": "old\n", "newText": "new\n"},
        {"type": "diff", "path": gone, "oldText": "bye\n", "newText": ""},
    ]
    assert "deleted" in _text(update)
    # The locations name the diffs' paths: the deleted file is in no journaled `paths`.
    assert sorted(loc["path"] for loc in update["locations"]) == sorted([new, keep, gone])


def test_a_write_outside_the_workspace_shows_the_file_it_wrote(tmp_path: pathlib.Path) -> None:
    """A write under a grant shows the granted file, not the workspace's file of that name."""
    granted = tmp_path / "granted"
    granted.mkdir()
    config = Config.model_validate({"sandbox": {"extra_write_paths": [str(granted)]}})
    run = _Run(tmp_path, config)
    (run.repo / "note.txt").write_text("the workspace's own\n", encoding="utf-8")
    run.edit(str(granted / "note.txt"), kind="create", new_string="x\n")

    (update,) = run.live()

    path = str((granted / "note.txt").resolve())
    assert _diffs(update) == [{"type": "diff", "path": path, "oldText": None, "newText": "x\n"}]
    assert update["locations"] == [{"path": path}]


def test_a_call_over_the_budget_sends_its_summary_and_no_diff(tmp_path: pathlib.Path) -> None:
    run = _Run(tmp_path)
    run.edit("big.txt", kind="create", new_string="x" * (results.DIFF_BUDGET_BYTES + 1))

    (update,) = run.live()

    assert _diffs(update) == []
    assert "diff omitted for size" in _text(update)
    assert update["locations"] == [{"path": str((run.repo / "big.txt").resolve())}]


def test_a_replayed_session_shows_the_diffs_the_live_one_showed(tmp_path: pathlib.Path) -> None:
    """The replay reads the recorded text: the working tree has moved on since."""
    run = _Run(tmp_path)
    run.edit("a.py", kind="create", new_string="one\n")
    run.edit("a.py", kind="replace", old_string="one", new_string="two")
    run.edit("big.txt", kind="create", new_string="x" * (results.DIFF_BUDGET_BYTES + 1))
    live = run.live()
    (run.repo / "a.py").write_text("changed after the run\n", encoding="utf-8")

    assert run.replayed() == live
    assert [len(_diffs(update)) for update in live] == [1, 1, 0]


def test_a_replay_after_the_objects_were_pruned_says_the_diff_is_gone(
    tmp_path: pathlib.Path,
) -> None:
    run = _Run(tmp_path)
    run.edit("a.py", kind="create", new_string="one\n")
    shutil.rmtree(run.objects)

    (update,) = run.replayed()

    assert _diffs(update) == []
    assert "diff unavailable" in _text(update)
