# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The graph curator's resilience pins.

`load_graph` skips a corrupt node file with a warning; `add_subtask` writes the child before the
parent link; a write-path fault reloads from disk so no unpersisted node is observed; re-rooting an
orphan removes its stale nested file.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6.graph import curator, models, storage
from agent6.sessions import layout as sessions_layout


def _layout(tmp_path: pathlib.Path) -> sessions_layout.SessionLayout:
    return sessions_layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")


def _draft(title: str = "do thing") -> models.TaskNodeDraft:
    return models.TaskNodeDraft(title=title, depends_on=(), created_by="planner")


# ---- in-process disk-fault fail-safe (replaces the subprocess die->reload) ---


def test_mutation_write_fault_reraises_and_reloads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A write fault after the in-memory update re-raises and reloads, so no phantom status survives.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    node = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("task")))
    assert c.get(node.id).status == "pending"

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("ENOSPC during status write")

    monkeypatch.setattr("agent6.graph.curator.write_node", boom)
    with pytest.raises(OSError, match="ENOSPC"):
        c.update_status(models.UpdateStatusIntent(id=node.id, new_status="in_progress"))
    monkeypatch.undo()

    # Reloaded from disk: the phantom "in_progress" never persisted.
    assert c.get(node.id).status == "pending"
    assert storage.load_graph(layout)[node.id].status == "pending"


def test_mutation_non_oserror_fault_also_reloads(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any non-CuratorError write-path fault fails safe the same way.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    node = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("task")))

    def boom(*_a: object, **_k: object) -> None:
        raise ValueError("serialization glitch")

    monkeypatch.setattr("agent6.graph.curator.write_node", boom)
    with pytest.raises(ValueError, match="serialization glitch"):
        c.update_status(models.UpdateStatusIntent(id=node.id, new_status="passed"))
    monkeypatch.undo()
    assert c.get(node.id).status == "pending"


def test_curator_error_reject_does_not_reload(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A CuratorError is a pre-mutation reject: it propagates without the disk reload.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    calls = {"n": 0}
    real = storage.load_graph

    def counting_load(lyt: sessions_layout.SessionLayout) -> dict[str, models.TaskNode]:
        calls["n"] += 1
        return real(lyt)

    monkeypatch.setattr("agent6.graph.curator.load_graph", counting_load)
    with pytest.raises(curator.CuratorError, match="unknown node"):
        c.update_status(models.UpdateStatusIntent(id="01" + "Z" * 24, new_status="passed"))
    assert calls["n"] == 0  # no reload on a clean validation reject


# ---- #16: corrupt node file must not brick the whole graph ----------------


def test_load_graph_skips_single_corrupt_node_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    a = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("alpha")))
    b = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("bravo")))

    # Corrupt one node file with frontmatter that fails to parse.
    bad_path = storage.node_md_path(layout, c.nodes(), b.id)
    bad_path.write_text("---\nnot-valid-frontmatter-no-colon\n---\n", encoding="utf-8")

    nodes = storage.load_graph(layout)
    # The good node still loads; the corrupt one is skipped, not fatal.
    assert a.id in nodes
    assert b.id not in nodes
    captured = capsys.readouterr()
    assert "skipping malformed node file" in captured.err

    # And a fresh curator can still start (resume is not bricked).
    reopened = curator.GraphCurator(layout)
    assert a.id in reopened.nodes()


def test_add_subtask_writes_child_before_parent_link(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A crash during the parent write: the child is on disk, the parent does not reference it yet.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    parent = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("parent")))

    real_write_node = storage.write_node
    parent_path = storage.node_md_path(layout, c.nodes(), parent.id)
    crash = {"armed": True}

    def crashing_write_node(layout_, nodes_, node_):  # type: ignore[no-untyped-def]
        # Crash specifically when re-writing the parent (the link update).
        if crash["armed"] and node_.id == parent.id:
            raise OSError("simulated ENOSPC during parent link write")
        return real_write_node(layout_, nodes_, node_)

    monkeypatch.setattr("agent6.graph.curator.write_node", crashing_write_node)

    with pytest.raises(OSError, match="simulated ENOSPC"):
        c.add_subtask(models.AddSubtaskIntent(parent_id=parent.id, draft=_draft("child")))

    monkeypatch.undo()

    # Reloaded purely from disk: no dangling child reference.
    on_disk = storage.load_graph(layout)
    # The child was written first, so it persists as a recoverable orphan.
    assert len(on_disk) == 2, "child node was not persisted before the parent link"
    orphan = next(n for n in on_disk.values() if n.id != parent.id)
    assert orphan.parent_id == parent.id  # it's the child, recorded as an orphan
    # The parent on disk is the pre-link version, because its rewrite crashed.
    assert parent_path.exists()
    reloaded_parent = on_disk[parent.id]
    for child_id in reloaded_parent.children:
        assert child_id in on_disk, f"dangling child reference {child_id}"


def test_add_subtask_normal_path_still_links(tmp_path: pathlib.Path) -> None:
    # The reordering must not regress the happy path.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    p = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    ch = c.add_subtask(models.AddSubtaskIntent(parent_id=p.id, draft=_draft("child")))
    assert c.get(p.id).children == (ch.id,)
    on_disk = storage.load_graph(layout)
    assert on_disk[p.id].children == (ch.id,)
    assert ch.id in on_disk


# ---- fix-review HIGH: corrupt PARENT must not orphan its surviving child ----


def test_load_graph_reroots_orphan_when_parent_corrupt(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Skipping a corrupt parent left its child dangling; the child is re-rooted and paths resolve.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    parent = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    child = c.add_subtask(models.AddSubtaskIntent(parent_id=parent.id, draft=_draft("child")))

    # Corrupt only the PARENT file; the (nested) child file stays valid.
    storage.node_md_path(layout, c.nodes(), parent.id).write_text("garbage", encoding="utf-8")

    nodes = storage.load_graph(layout)
    assert parent.id not in nodes  # parent skipped
    assert child.id in nodes  # child survives
    assert nodes[child.id].parent_id is None  # re-rooted, no dangling parent
    err = capsys.readouterr().err
    assert "re-rooting orphan node" in err

    # The exact KeyError trigger, and a fresh curator resolves the orphan without raising.
    assert storage.node_md_path(layout, nodes, child.id) == layout.graph_dir / f"{child.id}.md"
    reopened = curator.GraphCurator(layout)
    storage.node_md_path(layout, reopened.nodes(), child.id)  # must not KeyError


def test_ancestor_chain_terminates_on_missing_parent(tmp_path: pathlib.Path) -> None:
    # A dangling parent_id set directly must not KeyError: the chain ends at the present node.

    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    parent = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("p")))
    child = c.add_subtask(models.AddSubtaskIntent(parent_id=parent.id, draft=_draft("c")))
    nodes = dict(c.nodes())
    del nodes[parent.id]  # simulate the parent gone
    assert storage._ancestor_chain(nodes, child.id) == [child.id]  # terminates, no KeyError


# ---- graph-resilience #2: re-rooted node must not leave a stale duplicate ----


def test_rerooted_node_mutation_leaves_single_md_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-rooting moves the canonical path; the stale nested file must go, or rglob finds two.

    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    parent = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    child = c.add_subtask(models.AddSubtaskIntent(parent_id=parent.id, draft=_draft("child")))

    nested_path = storage.node_md_path(layout, c.nodes(), child.id)
    assert nested_path == layout.graph_dir / parent.id / f"{child.id}.md"
    assert nested_path.exists()

    # Corrupt the parent so the child is re-rooted on reload.
    storage.node_md_path(layout, c.nodes(), parent.id).write_text("garbage", encoding="utf-8")

    # Fresh curator: child is re-rooted (parent_id None) in its in-memory graph.
    c2 = curator.GraphCurator(layout)
    assert c2.get(child.id).parent_id is None
    # load_graph re-roots in memory only, so two .md for child.id sit on disk for now.
    assert nested_path.exists()

    # write_node now targets the root path and prunes the stale nested file.
    fsynced_dirs: list[pathlib.Path] = []
    monkeypatch.setattr(storage, "fsync_dir", fsynced_dirs.append)
    c2.update_status(models.UpdateStatusIntent(id=child.id, new_status="in_progress"))

    root_path = layout.graph_dir / f"{child.id}.md"
    assert root_path.exists()
    assert not nested_path.exists(), "stale nested .md must be pruned after re-root"
    assert nested_path.parent in fsynced_dirs

    # Exactly one .md on disk for this id, and load_graph yields exactly one node.
    remaining = list(layout.graph_dir.rglob(f"{child.id}.md"))
    assert remaining == [root_path]
    reloaded = storage.load_graph(layout)
    assert reloaded[child.id].status == "in_progress"
    assert reloaded[child.id].parent_id is None


def test_write_node_keeps_normal_path_file(tmp_path: pathlib.Path) -> None:
    # The prune must not delete the file it just wrote: a root node round-trips with one file.
    layout = _layout(tmp_path)
    c = curator.GraphCurator(layout)
    n = c.add_subtask(models.AddSubtaskIntent(parent_id=None, draft=_draft("solo")))
    path = storage.node_md_path(layout, c.nodes(), n.id)
    assert path.exists()
    assert list(layout.graph_dir.rglob(f"{n.id}.md")) == [path]
