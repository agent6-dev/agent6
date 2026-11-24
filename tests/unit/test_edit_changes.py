# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An edit records each file's text before and after, at the write, for every surface."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6.config import Config
from agent6.sessions import event_log, objects
from agent6.tools import dispatch, results, tool_errors
from agent6.viewmodel import transcript, transcript_items


def _dispatcher(tmp_path: pathlib.Path) -> tuple[dispatch.ToolDispatcher, pathlib.Path]:
    """Return a dispatcher over a workspace, journaling into a session dir beside it."""
    root, session_dir = tmp_path / "repo", tmp_path / "session"
    root.mkdir()
    session_dir.mkdir()
    sink = event_log.EventSink(session_dir / "logs.jsonl")
    return dispatch.ToolDispatcher(
        root=root, config=Config(), session_dir=session_dir, events=sink
    ), session_dir


def _results(session_dir: pathlib.Path) -> list[dict[str, Any]]:
    lines = (session_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    return [event for event in map(json.loads, lines) if event["type"] == "tool.result"]


def _texts(session_dir: pathlib.Path, event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the event's changes with each reference replaced by its stored text."""
    return objects.materialize(event["changes"], session_dir / "objects")


def _edit(path: str, *edits: dict[str, str]) -> dict[str, Any]:
    return {"path": path, "edits": list(edits)}


def test_two_writes_to_one_file_each_record_their_own_before_and_after(
    tmp_path: pathlib.Path,
) -> None:
    """A diff rebuilt from the working tree would show both calls the file's last text."""
    d, session_dir = _dispatcher(tmp_path)
    d.dispatch("apply_edit", _edit("a.py", {"kind": "create", "new_string": "one\n"}))
    d.dispatch(
        "apply_edit", _edit("a.py", {"kind": "replace", "old_string": "one", "new_string": "two"})
    )
    d.dispatch(
        "apply_edit", _edit("a.py", {"kind": "replace", "old_string": "two", "new_string": "three"})
    )

    first, second, third = (_texts(session_dir, event) for event in _results(session_dir))
    assert first == [{"path": "a.py", "before": None, "after": "one\n"}]
    assert second == [{"path": "a.py", "before": "one\n", "after": "two\n"}]
    assert third == [{"path": "a.py", "before": "two\n", "after": "three\n"}]


def test_a_multi_file_patch_records_a_create_a_change_and_a_delete(tmp_path: pathlib.Path) -> None:
    d, session_dir = _dispatcher(tmp_path)
    (tmp_path / "repo" / "keep.py").write_text("old\n", encoding="utf-8")
    (tmp_path / "repo" / "gone.py").write_text("bye\n", encoding="utf-8")
    patch = (
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+fresh\n"
        "diff --git a/keep.py b/keep.py\n--- a/keep.py\n+++ b/keep.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
    )

    result = d.dispatch("apply_patch", {"patch": patch})

    (event,) = _results(session_dir)
    assert _texts(session_dir, event) == [
        {"path": "new.py", "before": None, "after": "fresh\n"},
        {"path": "keep.py", "before": "old\n", "after": "new\n"},
        {"path": "gone.py", "before": "bye\n", "after": None},
    ]
    assert "deleted" in event["summary"]
    # The model's view of the result carries no file text.
    assert "changes" not in result.to_wire()


def test_a_call_over_the_budget_records_no_pair_and_says_so(tmp_path: pathlib.Path) -> None:
    """Both sides count, across every file of the call; a side is never cut to fit."""
    d, session_dir = _dispatcher(tmp_path)
    half = "x" * (results.DIFF_BUDGET_BYTES // 2)
    d.dispatch("apply_edit", _edit("fits.txt", {"kind": "create", "new_string": half + half}))
    d.dispatch("apply_edit", _edit("fits.txt", {"kind": "overwrite", "new_string": half + "y"}))
    two_files = (
        f"diff --git a/a.txt b/a.txt\n--- /dev/null\n+++ b/a.txt\n@@ -0,0 +1 @@\n+{half}\n"
        f"diff --git a/b.txt b/b.txt\n--- /dev/null\n+++ b/b.txt\n@@ -0,0 +1 @@\n+{half}\n"
    )
    d.dispatch("apply_patch", {"patch": two_files})

    at_budget, over, patch = _results(session_dir)
    assert _texts(session_dir, at_budget)[0]["after"] == half + half
    assert "omitted" not in at_budget["summary"]
    for event in (over, patch):
        assert "changes" not in event
        assert event["summary"].endswith("(diff omitted for size)")
    assert (tmp_path / "repo" / "fits.txt").read_text(encoding="utf-8") == half + "y"
    assert (tmp_path / "repo" / "b.txt").read_text(encoding="utf-8") == half + "\n"


def test_multibyte_text_is_measured_in_utf8_bytes(tmp_path: pathlib.Path) -> None:
    d, session_dir = _dispatcher(tmp_path)
    text = "é" * (results.DIFF_BUDGET_BYTES // 2 + 1)  # two bytes each: one byte over
    d.dispatch("apply_edit", _edit("accents.txt", {"kind": "create", "new_string": text}))
    assert "changes" not in _results(session_dir)[0]


def test_a_write_under_a_grant_outside_the_workspace_records_its_absolute_path(
    tmp_path: pathlib.Path,
) -> None:
    """A path relative to the grant names another file when read against the workspace."""
    root, session_dir, granted = tmp_path / "repo", tmp_path / "session", tmp_path / "granted"
    for directory in (root, session_dir, granted):
        directory.mkdir()
    (root / "note.txt").write_text("the workspace's own\n", encoding="utf-8")
    config = Config.model_validate({"sandbox": {"extra_write_paths": [str(granted)]}})
    sink = event_log.EventSink(session_dir / "logs.jsonl")
    d = dispatch.ToolDispatcher(root=root, config=config, session_dir=session_dir, events=sink)

    d.dispatch(
        "apply_edit", _edit(str(granted / "note.txt"), {"kind": "create", "new_string": "x"})
    )
    d.dispatch("apply_edit", _edit("note.txt", {"kind": "overwrite", "new_string": "y"}))

    outside, inside = _results(session_dir)
    assert [change["path"] for change in outside["changes"]] == [str(granted / "note.txt")]
    assert [change["path"] for change in inside["changes"]] == ["note.txt"]
    assert (outside["paths"], inside["paths"]) == ([str(granted / "note.txt")], ["note.txt"])


def test_a_patch_under_a_grant_journals_absolute_paths(tmp_path: pathlib.Path) -> None:
    """A resume reads `paths` against the workspace, where a grant's own path names another file."""
    root, session_dir, granted = tmp_path / "repo", tmp_path / "session", tmp_path / "granted"
    for directory in (root, session_dir, granted):
        directory.mkdir()
    (granted / "old.txt").write_text("bye\n", encoding="utf-8")
    config = Config.model_validate({"sandbox": {"extra_write_paths": [str(granted)]}})
    sink = event_log.EventSink(session_dir / "logs.jsonl")
    d = dispatch.ToolDispatcher(root=root, config=config, session_dir=session_dir, events=sink)

    result = d.dispatch(
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Add File: a.txt\n+in\n"
            f"*** Add File: {granted}/a.txt\n+out\n"
            f"*** Delete File: {granted}/old.txt\n*** End Patch\n"
        },
    )

    assert isinstance(result, results.PatchResult)
    assert result.deleted == (str(granted / "old.txt"),)
    assert [path for path, _ in result.files] == ["a.txt", str(granted / "a.txt")]
    assert _results(session_dir)[0]["paths"] == ["a.txt", str(granted / "a.txt")]


def test_a_preview_and_a_failed_edit_record_no_change(tmp_path: pathlib.Path) -> None:
    d, session_dir = _dispatcher(tmp_path)
    d.dispatch("apply_edit", {**_edit("a.py", {"kind": "create", "new_string": "x"}), "preview": 1})
    with pytest.raises(tool_errors.ToolError):
        d.dispatch(
            "apply_edit", _edit("a.py", {"kind": "replace", "old_string": "x", "new_string": "y"})
        )
    assert all("changes" not in event for event in _results(session_dir))


def test_the_fold_carries_a_settled_edits_changes_and_reads_an_older_journal() -> None:
    """A `tool.result` written before the field folds as it always did."""
    ref = {"$object": "a" * 64}
    call: dict[str, Any] = {"type": "tool.call", "name": "apply_edit", "call_id": 1}
    result: dict[str, Any] = {"type": "tool.result", "name": "apply_edit", "call_id": 1, "ok": True}

    def settled(extra: dict[str, Any]) -> transcript_items.TranscriptItem:
        return transcript.fold_transcript([call, {**result, **extra}])[-1]

    assert settled({"paths": ["a.py"]}).changes == ()
    assert settled({"changes": [{"path": "a.py", "before": None, "after": ref}]}).changes == (
        transcript_items.StoredChange("a.py", None, "a" * 64),
    )
    # A side that is neither null nor a reference is no create and no delete: the entry goes.
    garbled = [{"path": "a.py", "before": "text", "after": ref}, {"after": ref}, "x"]
    assert settled({"changes": garbled}).changes == ()
