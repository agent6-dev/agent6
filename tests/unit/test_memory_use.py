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
