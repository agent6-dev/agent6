# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The memory use record: the loop counts each fact a leg writes and reads
through the in-process tools (the jail never sees the store), and the leg's
end persists them to `<state-dir>/memory-use.json`, the record
`agent6 memory list` shows under each entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.config import Config
from agent6.memory import memory_dir, read_use, use_path
from agent6.tools.results import EditResult, ReadFileResult
from agent6.workflows._chain import RunChain
from agent6.workflows._conversation import AssistantTurn
from agent6.workflows._guards import MemoryState
from agent6.workflows._session_state import End
from agent6.workflows.loop import LoopState, TurnState, Workflow


def _wf(state_dir: Path | None, **kw: Any) -> Workflow:
    events = MagicMock()
    events.path = Path("/x/sessions/runs/run-a/logs.jsonl")
    return Workflow(
        chain=RunChain(Path("/tmp")),
        config=Config.model_validate({}),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        state_dir=state_dir,
        events=events,
        **kw,
    )


def _state() -> LoopState:
    return LoopState(original_task="t", tool_calls=0)


def _turn() -> TurnState:
    return TurnState(iteration=1, resp=MagicMock(), assistant=AssistantTurn((), ()))


def _read(wf: Workflow, state: LoopState, path: str) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        _turn(),
        "read_file",
        ReadFileResult(content="x", size=1, lines_total=1),
        {"path": path},
    )


def _edit(wf: Workflow, state: LoopState, path: str) -> None:
    wf._note_tool_effects(  # pyright: ignore[reportPrivateUsage]
        state,
        _turn(),
        "apply_edit",
        EditResult(applied=("create",), path=Path(path).name),
        {"path": path, "edits": []},
    )


def test_a_read_under_the_store_counts_the_fact(tmp_path: Path) -> None:
    wf = _wf(tmp_path)
    state = _state()
    store = memory_dir(tmp_path)
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


def test_an_edit_under_the_store_records_the_fact_name(tmp_path: Path) -> None:
    wf = _wf(tmp_path)
    state = _state()
    store = memory_dir(tmp_path)
    _edit(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "quirk.md"))
    assert state.memory.wrote == ["quirk"]
    assert state.memory.written is True
    # An index-only edit is a memory write for the nudges, but names no fact.
    state = _state()
    _edit(wf, state, str(store / "MEMORY.md"))
    assert state.memory.written is True
    assert state.memory.wrote == []


def test_a_state_dir_behind_a_symlink_still_counts(tmp_path: Path) -> None:
    """The model is told the store's unresolved path (a symlinked
    XDG_STATE_HOME); the check resolved the model's path against the
    unresolved store, so every memory edit counted as workspace work."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    wf = _wf(link)
    state = _state()
    store = memory_dir(link)
    _read(wf, state, str(store / "quirk.md"))
    _edit(wf, state, str(store / "quirk.md"))
    assert state.memory.read == {"quirk": 1}
    assert state.memory.wrote == ["quirk"]
    assert state.memory.written is True


def test_only_names_the_store_accepts_are_counted(tmp_path: Path) -> None:
    """A read of `<store>/Quirk.md` recorded a `Quirk` entry the list can never
    show: the name rule (`memory add`'s) filters the facts a call touched."""
    wf = _wf(tmp_path)
    state = _state()
    store = memory_dir(tmp_path)
    _read(wf, state, str(store / "Quirk.md"))
    _read(wf, state, str(store / "notes.txt"))
    _read(wf, state, str(store / "ok-name.md"))
    assert state.memory.read == {"ok-name": 1}


def test_no_store_means_nothing_is_counted() -> None:
    wf = _wf(None)
    state = _state()
    _read(wf, state, "/anywhere/memory/quirk.md")
    assert state.memory.read == {}


def test_the_leg_end_persists_what_it_wrote_and_read(tmp_path: Path) -> None:
    wf = _wf(tmp_path)
    state = _state()
    state.memory = MemoryState(wrote=["quirk"], read={"quirk": 2, "other": 1})
    wf._record_memory_use(state)  # pyright: ignore[reportPrivateUsage]
    use = read_use(tmp_path)
    assert use["quirk"].created_by == "run-a"
    assert use["quirk"].updated_by == "run-a"
    assert use["quirk"].reads == 2
    assert use["quirk"].read_by == "run-a"
    assert use["other"].created_by == ""
    assert use["other"].reads == 1


def test_a_leg_that_touched_nothing_writes_no_record(tmp_path: Path) -> None:
    wf = _wf(tmp_path)
    wf._record_memory_use(_state())  # pyright: ignore[reportPrivateUsage]
    assert not use_path(tmp_path).exists()


def test_a_resumed_leg_starts_its_own_count_with_the_nudge_flags_carried() -> None:
    """The nudge flags are run-lifetime (the snapshot); the touched facts are
    leg-local: a resumed leg records only what it touches itself."""
    from agent6.workflows._loop_state import restore_completion_state
    from agent6.workflows._session_state import SessionSnapshot

    snap = SessionSnapshot(
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
    restore_completion_state(state, snap)
    assert (state.memory.written, state.memory.flip_nudged) == (True, True)
    assert (state.memory.wrote, state.memory.read) == ([], {})
    # The shape itself: the snapshot carries the three flags and nothing
    # else of the memory bookkeeping, so no restore can bring a leg's
    # touched facts into the next one.
    assert {f for f in SessionSnapshot.model_fields if f.startswith("memory_")} == {
        "memory_written",
        "memory_flip_nudged",
        "memory_finish_nudged",
    }


def test_finish_records_the_use(tmp_path: Path) -> None:
    """Every end goes through `_finish`, so the record lands whichever way a
    leg ends; a write fault there must not break the end."""
    wf = _wf(tmp_path)
    state = _state()
    state.memory = MemoryState(read={"quirk": 1})
    end = End(reason="finish_session", summary="done", completed=True, verdict="passed")
    result = wf._finish(state, end, iteration=3)  # pyright: ignore[reportPrivateUsage]
    assert result.completed is True
    assert read_use(tmp_path)["quirk"].reads == 1


def test_a_record_that_cannot_be_written_logs_and_lets_the_end_stand(tmp_path: Path) -> None:
    """A read-only state dir must not turn a finished run into a crash: the
    end stands and the log names the fault."""
    import os
    import stat

    import pytest

    if os.geteuid() == 0:
        pytest.skip("root writes anywhere")
    logs: list[str] = []
    events = MagicMock()
    events.path = Path("/x/sessions/runs/run-a/logs.jsonl")
    wf = Workflow(
        chain=RunChain(Path("/tmp")),
        config=Config.model_validate({}),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=logs.append,
        state_dir=tmp_path,
        events=events,
    )
    state = _state()
    state.memory = MemoryState(read={"quirk": 1})
    tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        end = End(reason="finish_session", summary="done", completed=True, verdict="passed")
        result = wf._finish(state, end, iteration=1)  # pyright: ignore[reportPrivateUsage]
    finally:
        tmp_path.chmod(stat.S_IRWXU)
    assert result.completed is True
    assert any("memory use record failed" in line for line in logs)
    assert not use_path(tmp_path).exists()
