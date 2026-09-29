# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `GraphCurator` mutations."""

from __future__ import annotations

import pathlib

import pytest

from agent6.graph import curator, models
from agent6.sessions import layout as sessions_layout


def _layout(tmp_path: pathlib.Path) -> sessions_layout.SessionLayout:
    return sessions_layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")


def _draft(title: str = "do thing", deps: tuple[str, ...] = ()) -> models.TaskNodeDraft:
    return models.TaskNodeDraft(title=title, depends_on=deps, created_by="planner")


def test_curator_startup_tolerates_torn_journal_line(tmp_path: pathlib.Path) -> None:
    # A torn line on graph.jsonl must not crash curator startup, or the run is unresumable.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    with layout.journal_path.open("a", encoding="utf-8") as fh:
        fh.write('{"op": "add_subtask", "graph_v')  # torn: no newline, invalid JSON
    reopened = curator.GraphCurator(layout)  # must not raise
    assert reopened.graph_version >= 1
    assert len(reopened.nodes()) == 1


def test_add_subtask_with_no_parent_creates_root(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    assert n.parent_id is None
    assert n.status == "pending"
    assert c.graph_version >= 1


def test_add_subtask_unknown_parent_raises(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    with pytest.raises(curator.CuratorError, match="unknown parent"):
        c.add_subtask(models.AddSubtaskIntent(parent_id="X" * 26, draft=_draft()))


def test_add_subtask_links_child_to_parent(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    p = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    ch = c.add_subtask(models.AddSubtaskIntent(parent_id=p.id, draft=_draft("child")))
    assert c.get(p.id).children == (ch.id,)


def test_update_status_passed_then_obsolete_ok_other_rejected(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="passed"))
    # passed -> obsolete is fine
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="obsolete"))
    # but passed -> anything else would be rejected, set it back first
    n2 = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(models.UpdateStatusIntent(id=n2.id, new_status="passed"))
    with pytest.raises(curator.CuratorError):
        c.update_status(models.UpdateStatusIntent(id=n2.id, new_status="failed"))


def test_a_retired_task_stays_retired(tmp_path: pathlib.Path) -> None:
    """`passed -> obsolete -> pending` is refused: it walks around the passed-node guard."""
    c = curator.GraphCurator(_layout(tmp_path))
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="passed"))
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="obsolete"))
    with pytest.raises(curator.CuratorError, match="stays retired"):
        c.update_status(models.UpdateStatusIntent(id=n.id, new_status="pending"))
    # A note on a retired task still lands: the status is unchanged.
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="obsolete", note="superseded"))
    assert "superseded" in c.get(n.id).notes

    skipped = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(models.UpdateStatusIntent(id=skipped.id, new_status="skipped"))
    with pytest.raises(curator.CuratorError, match="stays retired"):
        c.update_status(models.UpdateStatusIntent(id=skipped.id, new_status="in_progress"))


def test_add_dependency_detects_cycle(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    a = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("a")))
    b = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("b")))
    c.add_dependency(models.AddDependencyIntent(id=b.id, depends_on=a.id))
    with pytest.raises(curator.CuratorError, match="cycle"):
        c.add_dependency(models.AddDependencyIntent(id=a.id, depends_on=b.id))


def test_cycle_check_survives_dangling_depends_on(tmp_path: pathlib.Path) -> None:
    # A depends_on edge to an unloaded id is not a cycle, not a KeyError.
    c = curator.GraphCurator(_layout(tmp_path))
    a = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("a")))
    b = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("b")))
    # add_dependency rejects an unknown target, so the in-memory node is corrupted directly.
    c._nodes[b.id] = b.model_copy(update={"depends_on": ("ghost-id",)})  # pyright: ignore[reportPrivateUsage]
    # add_dependency(a -> b) walks b's deps (incl. the ghost); must not raise KeyError.
    updated = c.add_dependency(models.AddDependencyIntent(id=a.id, depends_on=b.id))
    assert b.id in updated.depends_on


def test_a_container_with_open_children_cannot_pass(tmp_path: pathlib.Path) -> None:
    """A parent with open children is a container and cannot be passed.

    Every dependency on it counts as satisfied once it passes, while the work its children name goes
    undone.
    """
    c = curator.GraphCurator(_layout(tmp_path))
    root = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("run root")))
    parent = c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("phase")))
    child = c.add_subtask(models.AddSubtaskIntent(parent_id=parent.id, draft=_draft("step")))

    with pytest.raises(curator.CuratorError, match="unresolved children"):
        c.update_status(models.UpdateStatusIntent(id=parent.id, new_status="passed"))

    c.update_status(models.UpdateStatusIntent(id=child.id, new_status="passed"))
    assert (
        c.update_status(models.UpdateStatusIntent(id=parent.id, new_status="passed")).status
        == "passed"
    )
    # The root is the whole job: nothing depends on it, so an open subtask still passes it.
    open_child = c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("later")))
    assert (
        c.update_status(models.UpdateStatusIntent(id=root.id, new_status="passed")).status
        == "passed"
    )
    assert c.get(open_child.id).status == "pending"


def test_a_container_with_a_failed_child_cannot_pass(tmp_path: pathlib.Path) -> None:
    """A failed child is neither open nor done, so its container cannot be passed over it."""
    c = curator.GraphCurator(_layout(tmp_path))
    root = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    parent = c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("phase")))
    child = c.add_subtask(models.AddSubtaskIntent(parent_id=parent.id, draft=_draft("step")))
    c.update_status(models.UpdateStatusIntent(id=child.id, new_status="failed"))
    with pytest.raises(curator.CuratorError, match="unresolved children"):
        c.update_status(models.UpdateStatusIntent(id=parent.id, new_status="passed"))


def test_retire_as_obsolete_and_record_commit(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.record_commit(models.RecordCommitIntent(id=n.id, sha="abcd1234"))
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="obsolete", note="user-canceled"))
    final = c.get(n.id)
    assert final.commit_sha == "abcd1234"
    assert final.status == "obsolete"
    assert "user-canceled" in final.notes


def test_set_cursor_persists_and_validates(tmp_path: pathlib.Path) -> None:
    c = curator.GraphCurator(_layout(tmp_path))
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.set_cursor(models.SetCursorIntent(id=n.id))
    assert c.cursor() == n.id
    with pytest.raises(curator.CuratorError):
        c.set_cursor(models.SetCursorIntent(id="Z" * 26))
    c.set_cursor(models.SetCursorIntent(id=None))
    assert c.cursor() is None


def test_curator_reload_preserves_state(tmp_path: pathlib.Path) -> None:
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("persist me")))
    c.update_status(models.UpdateStatusIntent(id=n.id, new_status="in_progress"))
    v_before = c.graph_version
    c2 = curator.GraphCurator(layout)
    again = c2.get(n.id)
    assert again.status == "in_progress"
    assert c2.graph_version == v_before


def test_journal_entry_shapes_are_pinned(tmp_path: pathlib.Path) -> None:
    # The per-op key sets match the pre-typed writer's format, so old journal dirs read the same.
    import json

    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    root = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    a = c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("a")))
    b = c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("b")))
    c.update_status(models.UpdateStatusIntent(id=a.id, new_status="in_progress"))
    c.add_dependency(models.AddDependencyIntent(id=b.id, depends_on=a.id))
    c.record_commit(models.RecordCommitIntent(id=a.id, sha="abcd1234"))
    c.update_status(models.UpdateStatusIntent(id=b.id, new_status="obsolete", note="dropped"))
    c.set_cursor(models.SetCursorIntent(id=a.id))

    lines = [
        json.loads(raw)
        for raw in layout.journal_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    for entry in lines:
        assert entry.pop("ts")  # storage stamps it; not part of the typed shape
    assert lines == [
        {
            "op": "add_subtask",
            "id": root.id,
            "parent_id": None,
            "by": "planner",
            "graph_version": 1,
        },
        {
            "op": "add_subtask",
            "id": a.id,
            "parent_id": root.id,
            "by": "planner",
            "graph_version": 2,
        },
        {
            "op": "add_subtask",
            "id": b.id,
            "parent_id": root.id,
            "by": "planner",
            "graph_version": 3,
        },
        {"op": "update_status", "id": a.id, "new_status": "in_progress", "graph_version": 4},
        {"op": "add_dependency", "id": b.id, "depends_on": a.id, "graph_version": 5},
        {"op": "record_commit", "id": a.id, "sha": "abcd1234", "graph_version": 6},
        {"op": "update_status", "id": b.id, "new_status": "obsolete", "graph_version": 7},
        {"op": "set_cursor", "id": a.id, "graph_version": 8},
    ]


def test_every_write_in_one_mutation_carries_the_journaled_version(tmp_path: pathlib.Path) -> None:
    """add_subtask writes the child and relinks the parent, both stamped with the same version.

    A journal that lost its tail is then detectable from the nodes alone.
    """
    import json as _json

    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    root = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    child = c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("child")))
    assert root.graph_version == 1
    assert child.graph_version == 2
    nodes = c.nodes()
    assert nodes[root.id].graph_version == 2  # relinked by the child's mutation
    assert nodes[child.id].graph_version == 2
    entries = [
        _json.loads(line)
        for line in layout.journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [e["graph_version"] for e in entries] == [1, 2]


def test_a_lost_journal_tail_resyncs_the_version_and_says_so(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Boot resyncs the version counter past a node stamp newer than the journal's tail.

    A death between a node write and its journal append loses the entry; reusing the number would
    make two operations share one version.
    """
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    root = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    c.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("child")))
    # Simulate the crash: drop the journal's last line (the v2 entry).
    lines = layout.journal_path.read_text(encoding="utf-8").splitlines()
    layout.journal_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    reopened = curator.GraphCurator(layout)
    err = capsys.readouterr().err
    assert "lost its tail" in err and "v2" in err
    assert reopened.graph_version == 2
    third = reopened.add_subtask(models.AddSubtaskIntent(parent_id=root.id, draft=_draft("late")))
    assert third.graph_version == 3  # the lost number is never reused
