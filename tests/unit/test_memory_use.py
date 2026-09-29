# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The memory use record.

The loop counts each fact an execution writes and reads through the in-process tools (the jail never
sees the store), and the execution's end persists them to `<state-dir>/memory-use.json`, the record
`agent6 memory list` shows under each entry.
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

from agent6 import memory
from agent6.config import Config
from agent6.harness import _chain, _conversation, _guards, _loop_state, _snapshot, loop
from agent6.tools import results


def _wf(state_dir: pathlib.Path | None, **kw: Any) -> loop.Harness:
    events = mock.MagicMock()
    events.path = pathlib.Path("/x/sessions/runs/run-a/logs.jsonl")
    return loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate({}),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        state_dir=state_dir,
        events=events,
        **kw,
    )


def _state() -> _loop_state.LoopState:
    return _loop_state.LoopState(original_task="t", tool_calls=0)


def _turn() -> _loop_state.TurnState:
    return _loop_state.TurnState(
        iteration=1, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )


def _read(wf: loop.Harness, state: _loop_state.LoopState, path: str) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        _turn(),
        "read_file",
        results.ReadFileResult(content="x", size=1, lines_total=1),
        {"path": path},
    )


def _edit(
    wf: loop.Harness, state: _loop_state.LoopState, path: str, *, created: bool = False
) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        _turn(),
        "apply_edit",
        results.EditResult(
            applied=("create" if created else "replace",),
            path=pathlib.Path(path).name,
            created=created,
        ),
        {"path": path, "edits": []},
    )


def _patch(wf: loop.Harness, state: _loop_state.LoopState, text: str) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        _turn(),
        "apply_patch",
        results.PatchResult(path="x", bytes_written=1),
        {"patch": text},
    )


def test_a_read_under_the_store_counts_the_fact(tmp_path: pathlib.Path) -> None:
    wf = _wf(tmp_path)
    state = _state()
    store = memory.memory_dir(tmp_path)
    _read(wf, state, str(store / "quirk.md"))
    _read(wf, state, str(store / "quirk.md"))
    _read(wf, state, str(store / "other.md"))
    assert state.memory.read == {"quirk": 2, "other": 1}
    # The index and the rulings are not facts; a workspace read is not memory.
    _read(wf, state, str(store / "MEMORY.md"))
    _read(wf, state, str(store / "DECISIONS.md"))
    _read(wf, state, "src/app.py")
    assert state.memory.read == {"quirk": 2, "other": 1}
    assert state.memory.wrote == []


def test_an_edit_under_the_store_records_the_fact_name(tmp_path: pathlib.Path) -> None:
    wf = _wf(tmp_path)
    state = _state()
    store = memory.memory_dir(tmp_path)
    _edit(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "quirk.md"))
    assert state.memory.wrote == ["quirk"]
    assert state.memory.created == []
    assert state.memory.written is True
    # An index-only edit is a memory write for the nudges, but names no fact.
    state = _state()
    _edit(wf, state, str(store / "MEMORY.md"))
    assert state.memory.written is True
    assert state.memory.wrote == []


def test_the_edit_tools_create_marks_the_fact_created(tmp_path: pathlib.Path) -> None:
    """`memory list` said `written by <session>` for a fact the session only edited.

    The record tells a create (the edit tool's result says the file is new, whichever kind wrote it;
    a patch's `--- /dev/null` or `*** Add File:`) from an edit.
    """
    wf = _wf(tmp_path)
    state = _state()
    store = memory.memory_dir(tmp_path)
    _edit(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "fresh.md"), created=True)
    _patch(wf, state, f"*** Begin Patch\n*** Add File: {store}/added.md\n+x\n*** End Patch")
    _patch(wf, state, f"--- {store}/quirk.md\n+++ {store}/quirk.md\n@@ -1 +1 @@\n-a\n+b\n")
    _patch(wf, state, f"--- /dev/null\n+++ {store}/MEMORY.md\n@@ -0,0 +1 @@\n+- x\n")
    assert state.memory.wrote == ["quirk", "fresh", "added"]
    assert state.memory.created == ["fresh", "added"]


def test_a_deletion_ends_the_fact_for_the_record(tmp_path: pathlib.Path) -> None:
    """A deletion ends the fact for the record.

    A fact the model deleted through a patch must not stay in the record as an edit, or `memory
    list` keeps a line for a file that is gone and a later re-creation is not seen as one.
    """
    wf = _wf(tmp_path)
    state = _state()
    store = memory.memory_dir(tmp_path)
    _read(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "fresh.md"), created=True)
    _patch(wf, state, f"*** Begin Patch\n*** Delete File: {store}/quirk.md\n*** End Patch")
    _patch(wf, state, f"--- {store}/fresh.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
    assert (state.memory.wrote, state.memory.created) == ([], [])
    assert state.memory.deleted == ["quirk", "fresh"]
    # The read before the delete belonged to the fact's old life: a live run
    # read a fact, deleted it, and the execution's end put an entry back for it.
    assert state.memory.read == {}
    _edit(wf, state, str(store / "quirk.md"), created=True)
    assert (state.memory.wrote, state.memory.created) == (["quirk"], ["quirk"])
    assert state.memory.deleted == ["quirk", "fresh"]


def test_the_execution_end_drops_a_deleted_fact_and_starts_a_recreated_one_afresh(
    tmp_path: pathlib.Path,
) -> None:
    memory.record_use(
        tmp_path,
        session="run-0",
        wrote=("gone", "again"),
        created=("gone", "again"),
        read={"gone": 5, "again": 1},
        when=0.0,
    )
    wf = _wf(tmp_path)
    state = _state()
    state.memory = _guards.MemoryState(
        wrote=["again"], created=["again"], deleted=["gone", "again"]
    )
    wf._record_memory_use(state)  # pyright: ignore[reportPrivateUsage]
    use = memory.read_use(tmp_path)
    assert "gone" not in use
    assert use["again"].created is not None and use["again"].created.session == "run-a"
    assert (use["again"].reads, len(use["again"].writes)) == (0, 1)


def test_a_state_dir_behind_a_symlink_still_counts(tmp_path: pathlib.Path) -> None:
    """A state dir behind a symlink still counts.

    The model is told the store's unresolved path (a symlinked XDG_STATE_HOME); a check that
    resolves the model's path against the unresolved store counts every memory edit as workspace
    work.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    wf = _wf(link)
    state = _state()
    store = memory.memory_dir(link)
    _read(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "quirk.md"))
    assert state.memory.read == {"quirk": 1}
    assert state.memory.wrote == ["quirk"]
    assert state.memory.written is True


def test_only_names_the_store_accepts_are_counted(tmp_path: pathlib.Path) -> None:
    """A read of `<store>/Quirk.md` recorded a `Quirk` entry the list can never show.

    The name rule (`memory add`'s) filters the facts a call touched.
    """
    wf = _wf(tmp_path)
    state = _state()
    store = memory.memory_dir(tmp_path)
    _read(wf, state, str(store / "Quirk.md"))
    _read(wf, state, str(store / "notes.txt"))
    _read(wf, state, str(store / "ok-name.md"))
    assert state.memory.read == {"ok-name": 1}


def test_no_store_means_nothing_is_counted() -> None:
    wf = _wf(None)
    state = _state()
    _read(wf, state, "/anywhere/memory/quirk.md")
    assert state.memory.read == {}


def test_the_execution_end_persists_what_it_wrote_and_read(tmp_path: pathlib.Path) -> None:
    wf = _wf(tmp_path)
    state = _state()
    state.memory = _guards.MemoryState(
        wrote=["quirk", "edited"], created=["quirk"], read={"quirk": 2, "other": 1}
    )
    wf._record_memory_use(state)  # pyright: ignore[reportPrivateUsage]
    use = memory.read_use(tmp_path)
    assert use["quirk"].writers == ("run-a",)
    assert use["quirk"].created is not None and use["quirk"].created.session == "run-a"
    assert use["quirk"].reads == 2
    assert use["quirk"].last_read is not None and use["quirk"].last_read.session == "run-a"
    assert use["edited"].writers == ("run-a",)
    assert use["edited"].created is None
    assert use["other"].created is None
    assert use["other"].reads == 1


def test_a_execution_that_touched_nothing_writes_no_record(tmp_path: pathlib.Path) -> None:
    wf = _wf(tmp_path)
    wf._record_memory_use(_state())  # pyright: ignore[reportPrivateUsage]
    assert not memory.use_path(tmp_path).exists()


def test_a_resumed_execution_starts_its_own_count_with_the_nudge_flags_carried() -> None:
    """The nudge flags are run-lifetime (the snapshot); the touched facts are execution-local.

    A resumed execution records only what it touches itself.
    """
    snap = _snapshot.SessionSnapshot(
        system="s",
        messages=[],
        tool_calls=0,
        next_iteration=3,
        root_task_id=None,
        original_task="go",
        verify_command=(),
        memory_written=True,
        memory_flip_nudged=True,
    )
    state = _state()
    _loop_state.restore_completion_state(state, snap)
    assert (state.memory.written, state.memory.flip_nudged) == (True, True)
    assert (state.memory.wrote, state.memory.read) == ([], {})
    # The shape itself: the snapshot carries the three flags and nothing
    # else of the memory bookkeeping, so no restore can bring an execution's
    # touched facts into the next one.
    assert {f for f in _snapshot.SessionSnapshot.model_fields if f.startswith("memory_")} == {
        "memory_written",
        "memory_flip_nudged",
        "memory_finish_nudged",
    }


def test_finish_records_the_use(tmp_path: pathlib.Path) -> None:
    """Finish records the use.

    Every end goes through `_finish`, so the record lands whichever way an execution ends; a write
    fault there must not break the end.
    """
    wf = _wf(tmp_path)
    state = _state()
    state.memory = _guards.MemoryState(read={"quirk": 1})
    end = _snapshot.End(reason="finish_session", summary="done", completed=True, verdict="passed")
    result = wf._finish(state, end, iteration=3)  # pyright: ignore[reportPrivateUsage]
    assert result.completed is True
    assert memory.read_use(tmp_path)["quirk"].reads == 1


def test_a_record_that_cannot_be_written_logs_and_lets_the_end_stand(
    tmp_path: pathlib.Path,
) -> None:
    """A read-only state dir must not turn a finished run into a crash.

    The end stands and the log names the fault.
    """
    import os
    import stat

    import pytest

    if os.geteuid() == 0:
        pytest.skip("root writes anywhere")
    logs: list[str] = []
    events = mock.MagicMock()
    events.path = pathlib.Path("/x/sessions/runs/run-a/logs.jsonl")
    wf = loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate({}),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=logs.append,
        state_dir=tmp_path,
        events=events,
    )
    state = _state()
    state.memory = _guards.MemoryState(read={"quirk": 1})
    tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        end = _snapshot.End(
            reason="finish_session", summary="done", completed=True, verdict="passed"
        )
        result = wf._finish(state, end, iteration=1)  # pyright: ignore[reportPrivateUsage]
    finally:
        tmp_path.chmod(stat.S_IRWXU)
    assert result.completed is True
    assert any("memory use record failed" in line for line in logs)
    assert not memory.use_path(tmp_path).exists()
