# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for graph curator resilience bugfixes.

Covers:
  #16 load_graph must skip-with-warning on a single corrupt node file instead
      of aborting the whole graph.
  #17 add_subtask must write the child node before the parent->child link so a
      crash in between can't leave a dangling child reference.

  in-process fail-safe: a write-path fault in a MUTATING op (which runs AFTER
      self._nodes is updated in memory) must re-raise AND reload from disk (the
      source of truth), so a later read never observes a node that was never
      persisted. This replaced the old subprocess die->reload fail-safe; a clean
      CuratorError validation reject propagates without a reload.
  graph-resilience #2: re-rooting an orphan node changes its canonical .md path;
      the stale nested file must be removed so load_graph never sees two .md
      files for one id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.graph import storage
from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNode, TaskNodeDraft, UpdateStatusIntent
from agent6.graph.storage import load_graph, node_md_path
from agent6.runs.layout import RunLayout


def _layout(tmp_path: Path) -> RunLayout:
    return RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")


def _draft(title: str = "do thing") -> TaskNodeDraft:
    return TaskNodeDraft(title=title, depends_on=(), created_by="planner")


# ---- in-process disk-fault fail-safe (replaces the subprocess die->reload) ---


def test_mutation_write_fault_reraises_and_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A mutation updates self._nodes IN MEMORY before write_node(). An OSError on
    # the write path (ENOSPC/EROFS) leaves in-memory state ahead of disk. The
    # curator must re-raise the fault AND reload from disk, so the phantom status
    # change is gone from a later read -- not surfaced as if it had persisted.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    node = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("task")))
    assert c.get(node.id).status == "pending"

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("ENOSPC during status write")

    monkeypatch.setattr("agent6.graph.curator.write_node", boom)
    with pytest.raises(OSError, match="ENOSPC"):
        c.update_status(UpdateStatusIntent(id=node.id, new_status="in_progress"))
    monkeypatch.undo()

    # Reloaded from disk: the phantom "in_progress" never persisted, so both the
    # in-memory graph and a fresh load see the pre-mutation "pending".
    assert c.get(node.id).status == "pending"
    assert load_graph(layout)[node.id].status == "pending"


def test_mutation_non_oserror_fault_also_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not just OSError: any non-CuratorError write-path fault (e.g. a
    # serialization error surfacing from write_node) fails-safe the same way.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    node = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("task")))

    def boom(*_a: object, **_k: object) -> None:
        raise ValueError("serialization glitch")

    monkeypatch.setattr("agent6.graph.curator.write_node", boom)
    with pytest.raises(ValueError, match="serialization glitch"):
        c.update_status(UpdateStatusIntent(id=node.id, new_status="passed"))
    monkeypatch.undo()
    assert c.get(node.id).status == "pending"


def test_curator_error_reject_does_not_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A CuratorError is a pre-mutation validation reject (nothing applied): it
    # propagates untouched, WITHOUT the disk reload the fault path does.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    calls = {"n": 0}
    real = storage.load_graph

    def counting_load(lyt: RunLayout) -> dict[str, TaskNode]:
        calls["n"] += 1
        return real(lyt)

    monkeypatch.setattr("agent6.graph.curator.load_graph", counting_load)
    with pytest.raises(CuratorError, match="unknown node"):
        c.update_status(UpdateStatusIntent(id="01" + "Z" * 24, new_status="passed"))
    assert calls["n"] == 0  # no reload on a clean validation reject


# ---- #16: corrupt node file must not brick the whole graph ----------------


def test_load_graph_skips_single_corrupt_node_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    a = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("alpha")))
    b = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("bravo")))

    # Corrupt one node file with frontmatter that fails to parse.
    bad_path = node_md_path(layout, c.nodes(), b.id)
    bad_path.write_text("---\nnot-valid-frontmatter-no-colon\n---\n", encoding="utf-8")

    nodes = load_graph(layout)
    # The good node still loads; the corrupt one is skipped, not fatal.
    assert a.id in nodes
    assert b.id not in nodes
    captured = capsys.readouterr()
    assert "skipping malformed node file" in captured.err

    # And a fresh curator can still start (resume is not bricked).
    reopened = GraphCurator(layout)
    assert a.id in reopened.nodes()


def test_load_graph_raises_without_fix_would_have(tmp_path: Path) -> None:
    # Sanity: a wholly corrupt-but-loadable graph (every other file valid)
    # still returns the valid nodes rather than aborting.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    ids = [
        c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft(f"n{i}"))).id for i in range(3)
    ]
    nodes_map = c.nodes()
    node_md_path(layout, nodes_map, ids[1]).write_text("garbage", encoding="utf-8")
    nodes = load_graph(layout)
    assert ids[0] in nodes and ids[2] in nodes
    assert ids[1] not in nodes


# ---- #17: child node written before parent->child link --------------------


def test_add_subtask_writes_child_before_parent_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a crash that happens AFTER the child write but DURING the parent
    # write. With the fix, the child .md already exists on disk and the parent
    # on disk does NOT yet reference it -- so no dangling reference. (Before the
    # fix, the parent link was written first, so a crash here would persist a
    # parent referencing a child whose .md never existed.)
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    parent = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("parent")))

    real_write_node = storage.write_node
    parent_path = node_md_path(layout, c.nodes(), parent.id)
    crash = {"armed": True}

    def crashing_write_node(layout_, nodes_, node_):  # type: ignore[no-untyped-def]
        # Crash specifically when re-writing the parent (the link update).
        if crash["armed"] and node_.id == parent.id:
            raise OSError("simulated ENOSPC during parent link write")
        return real_write_node(layout_, nodes_, node_)

    monkeypatch.setattr("agent6.graph.curator.write_node", crashing_write_node)

    with pytest.raises(OSError, match="simulated ENOSPC"):
        c.add_subtask(AddSubtaskIntent(parent_id=parent.id, draft=_draft("child")))

    monkeypatch.undo()

    # Reload purely from disk. The parent must NOT reference any child whose
    # .md is missing -- i.e. no dangling references.
    on_disk = load_graph(layout)
    # The child .md was written BEFORE the parent link, so the crash during the
    # link write leaves the child on disk as a recoverable orphan. This is the
    # observable that distinguishes the fix from the reverted order (which writes
    # the parent link first and never reaches the child write, losing it): under
    # the fix two nodes persist, under the revert only the parent.
    assert len(on_disk) == 2, "child node was not persisted before the parent link"
    orphan = next(n for n in on_disk.values() if n.id != parent.id)
    assert orphan.parent_id == parent.id  # it's the child, recorded as an orphan
    # The original parent file on disk should still be the pre-link version
    # (children empty), because its re-write crashed.
    assert parent_path.exists()
    reloaded_parent = on_disk[parent.id]
    for child_id in reloaded_parent.children:
        assert child_id in on_disk, f"dangling child reference {child_id}"


def test_add_subtask_normal_path_still_links(tmp_path: Path) -> None:
    # The reordering must not regress the happy path.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    p = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    ch = c.add_subtask(AddSubtaskIntent(parent_id=p.id, draft=_draft("child")))
    assert c.get(p.id).children == (ch.id,)
    on_disk = load_graph(layout)
    assert on_disk[p.id].children == (ch.id,)
    assert ch.id in on_disk


# ---- fix-review HIGH: corrupt PARENT must not orphan its surviving child ----


def test_load_graph_reroots_orphan_when_parent_corrupt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Skipping a corrupt PARENT (the #16 fix) left its child with a dangling
    # parent_id, which then KeyErrored in node_md_path/_ancestor_chain on the
    # next mutation (masked by the #6 broad except). The child must be re-rooted
    # and node-path resolution must not raise.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    parent = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    child = c.add_subtask(AddSubtaskIntent(parent_id=parent.id, draft=_draft("child")))

    # Corrupt only the PARENT file; the (nested) child file stays valid.
    node_md_path(layout, c.nodes(), parent.id).write_text("garbage", encoding="utf-8")

    nodes = load_graph(layout)
    assert parent.id not in nodes  # parent skipped
    assert child.id in nodes  # child survives
    assert nodes[child.id].parent_id is None  # re-rooted, no dangling parent
    err = capsys.readouterr().err
    assert "re-rooting orphan node" in err

    # The exact KeyError trigger (node_md_path -> _ancestor_chain) must now work,
    # and a fresh curator must start + resolve the orphan without raising.
    assert node_md_path(layout, nodes, child.id) == layout.graph_dir / f"{child.id}.md"
    reopened = GraphCurator(layout)
    node_md_path(layout, reopened.nodes(), child.id)  # must not KeyError


def test_ancestor_chain_terminates_on_missing_parent(tmp_path: Path) -> None:
    # Defensive: even a node carrying a dangling parent_id (not via load) must
    # not KeyError in path resolution -- the chain terminates at the present node.
    from agent6.graph.storage import _ancestor_chain  # pyright: ignore[reportPrivateUsage]

    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    parent = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("p")))
    child = c.add_subtask(AddSubtaskIntent(parent_id=parent.id, draft=_draft("c")))
    nodes = dict(c.nodes())
    del nodes[parent.id]  # simulate the parent gone
    assert _ancestor_chain(nodes, child.id) == [child.id]  # terminates, no KeyError


# ---- graph-resilience #2: re-rooted node must not leave a stale duplicate ----


def test_rerooted_node_mutation_leaves_single_md_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An orphan node (its parent file was corrupt -> skipped) is re-rooted by
    # load_graph (parent_id -> None), which moves its canonical .md path from the
    # nested <parent>/<child>.md to the root <child>.md. Mutating it then writes
    # the new root path; the OLD nested file must be removed, otherwise
    # load_graph's rglob finds TWO .md for the same id (nondeterministic winner).
    from agent6.graph.models import UpdateStatusIntent

    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    parent = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    child = c.add_subtask(AddSubtaskIntent(parent_id=parent.id, draft=_draft("child")))

    nested_path = node_md_path(layout, c.nodes(), child.id)
    assert nested_path == layout.graph_dir / parent.id / f"{child.id}.md"
    assert nested_path.exists()

    # Corrupt the parent so the child is re-rooted on reload.
    node_md_path(layout, c.nodes(), parent.id).write_text("garbage", encoding="utf-8")

    # Fresh curator: child is re-rooted (parent_id None) in its in-memory graph.
    c2 = GraphCurator(layout)
    assert c2.get(child.id).parent_id is None
    # The stale nested file still exists at this point (load_graph only re-roots
    # in memory) -- so there are momentarily two .md for child.id on disk.
    assert nested_path.exists()

    # Mutate the re-rooted child: write_node now targets the ROOT path and must
    # prune the stale nested file.
    fsynced_dirs: list[Path] = []
    monkeypatch.setattr(storage, "fsync_dir", fsynced_dirs.append)
    c2.update_status(UpdateStatusIntent(id=child.id, new_status="in_progress"))

    root_path = layout.graph_dir / f"{child.id}.md"
    assert root_path.exists()
    assert not nested_path.exists(), "stale nested .md must be pruned after re-root"
    assert nested_path.parent in fsynced_dirs

    # Exactly one .md on disk for this id, and load_graph yields exactly one node.
    remaining = list(layout.graph_dir.rglob(f"{child.id}.md"))
    assert remaining == [root_path]
    reloaded = load_graph(layout)
    assert reloaded[child.id].status == "in_progress"
    assert reloaded[child.id].parent_id is None


def test_write_node_keeps_normal_path_file(tmp_path: Path) -> None:
    # The prune must not delete the file it just wrote on the normal (no re-root)
    # path: a plain root node round-trips with exactly one file.
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("solo")))
    path = node_md_path(layout, c.nodes(), n.id)
    assert path.exists()
    assert list(layout.graph_dir.rglob(f"{n.id}.md")) == [path]
