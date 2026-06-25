# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for graph curator resilience bugfixes.

Covers:
  #6  a non-CuratorError exception in a READ op must NOT crash the curator
      subprocess -- _serve_connection degrades it to a rejected request.
  #16 load_graph must skip-with-warning on a single corrupt node file instead
      of aborting the whole graph.
  #17 add_subtask must write the child node before the parent->child link so a
      crash in between can't leave a dangling child reference.

  graph-resilience #1: a non-OSError write-path fault in a MUTATING op (which
      runs AFTER self._nodes is updated in memory) must fail-safe (die) like the
      OSError case, not stay alive with in-memory state ahead of disk.
  graph-resilience #2: re-rooting an orphan node changes its canonical .md path;
      the stale nested file must be removed so load_graph never sees two .md
      files for one id.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from pathlib import Path

import pytest

from agent6.graph import storage
from agent6.graph.curator import GraphCurator
from agent6.graph.ipc import recv_message, send_message
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.graph.server import _serve_connection  # pyright: ignore[reportPrivateUsage]
from agent6.graph.storage import RunLayout, load_graph, node_md_path


def _layout(tmp_path: Path) -> RunLayout:
    return RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")


def _draft(title: str = "do thing") -> TaskNodeDraft:
    return TaskNodeDraft(title=title, depends_on=(), created_by="planner")


# ---- #6: internal fault must not kill the connection ----------------------


def test_serve_connection_survives_non_curator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-CuratorError/ValidationError fault on a READ op (which cannot leave
    # in-memory state ahead of disk) must be caught and reported in-band, leaving
    # the connection alive for the next request -- not propagate out and kill the
    # subprocess. (A fault on a MUTATING op instead re-raises to fail-safe-reload;
    # see test_mutating_op_write_fault_reraises.)
    import agent6.graph.server as server_mod

    curator = GraphCurator(_layout(tmp_path))

    calls = {"n": 0}

    def boom(_curator: object, _intent: dict[str, object]) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError("transient-read-glitch")  # not a CuratorError
        return {"ok": "second-request-handled"}

    monkeypatch.setattr(server_mod, "_handle_one", boom)

    cli, srv = socket.socketpair()
    try:
        t = threading.Thread(target=_serve_connection, args=(curator, srv))
        t.start()

        # First request (a read) triggers the internal fault.
        send_message(cli, {"id": 1, "intent": {"op": "get_state"}})
        resp1 = recv_message(cli)
        assert resp1 is not None
        assert resp1["id"] == 1
        assert resp1["ok"] is False
        assert "curator internal error" in resp1["error"]
        assert "transient-read-glitch" in resp1["error"]

        # The connection must still be alive: a second request gets served.
        send_message(cli, {"id": 2, "intent": {"op": "get_state"}})
        resp2 = recv_message(cli)
        assert resp2 is not None
        assert resp2["id"] == 2
        assert resp2["ok"] is True

        cli.close()
        t.join(timeout=5)
        assert not t.is_alive()
    finally:
        srv.close()
        with contextlib.suppress(OSError):
            cli.close()


def test_mutating_op_write_fault_reraises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # graph-resilience #1. A mutating handler updates self._nodes IN MEMORY
    # before write_node(). If write_node raises a NON-OSError (e.g. a
    # serialization TypeError, or an _ancestor_chain cycle ValueError) AFTER the
    # in-memory write, the in-memory graph is ahead of disk. The broad except
    # used to keep the subprocess alive in that skewed state -- the client could
    # then observe a node that was never persisted (a phantom that vanishes on
    # restart). With the fix a mutating-op fault re-raises (the subprocess dies
    # so the next spawn reloads consistent on-disk state), matching the OSError
    # fail-safe -- it must NOT silently stay alive and reply ok=False.
    import agent6.graph.server as server_mod

    curator = GraphCurator(_layout(tmp_path))

    def boom(_curator: object, _intent: dict[str, object]) -> object:
        # Models a non-OSError surfacing from the write path after the in-memory
        # self._nodes mutation already happened.
        raise ValueError("cycle in parent chain at <id>")

    monkeypatch.setattr(server_mod, "_handle_one", boom)

    cli, srv = socket.socketpair()
    with cli, srv:
        # A real mutating op (add_subtask): the fault must propagate (die), not
        # be swallowed as an in-band "curator internal error" reply.
        send_message(cli, {"id": 1, "intent": {"op": "add_subtask"}})
        with pytest.raises(ValueError, match="cycle in parent chain"):
            _serve_connection(curator, srv)


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


# ---- fix-review MEDIUM: a disk fault mid-mutation must NOT be masked ---------


def test_disk_fault_during_mutation_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A disk OSError inside a mutation, surfaced through _serve_connection, must
    # propagate (crash -> respawn reloads consistent on-disk state) rather than
    # be swallowed as "curator internal error" with in-memory state ahead of disk.
    curator = GraphCurator(_layout(tmp_path))

    def disk_full(_curator: object, _intent: dict[str, object]) -> object:
        raise OSError("ENOSPC: simulated disk fault during mutation")

    monkeypatch.setattr("agent6.graph.server._handle_one", disk_full)

    a, b = socket.socketpair()
    with a, b:
        send_message(a, {"id": 1, "intent": {"kind": "get_state"}})
        with pytest.raises(OSError, match="ENOSPC"):
            _serve_connection(curator, b)


# ---- graph-resilience #2: re-rooted node must not leave a stale duplicate ----


def test_rerooted_node_mutation_leaves_single_md_file(
    tmp_path: Path,
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
    c2.update_status(UpdateStatusIntent(id=child.id, new_status="in_progress"))

    root_path = layout.graph_dir / f"{child.id}.md"
    assert root_path.exists()
    assert not nested_path.exists(), "stale nested .md must be pruned after re-root"

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
