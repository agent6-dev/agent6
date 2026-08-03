# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A durable scratchpad the agent may restructure.

The memory store is append-only on purpose, so it cannot be curated: after
twenty sessions it is a log. Notes are one document the agent rewrites, which
is what keeps it readable as it grows -- and the size cap is what forces the
curation rather than letting it eat the context window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.notes import NOTES_MAX_CHARS, NotesError, notes_path, read_notes, write_notes


def test_reading_before_anything_is_written_is_empty(tmp_path: Path) -> None:
    assert read_notes(tmp_path) == ""


def test_a_write_round_trips(tmp_path: Path) -> None:
    write_notes(tmp_path, "# Plan\n\n- one\n")
    assert read_notes(tmp_path) == "# Plan\n\n- one\n"


def test_a_write_REPLACES_rather_than_appends(tmp_path: Path) -> None:
    """The whole point: the agent restructures. An append-only store cannot
    strike through a resolved item or merge two entries."""
    write_notes(tmp_path, "first\n")
    write_notes(tmp_path, "second\n")
    assert read_notes(tmp_path) == "second\n"


def test_it_lives_outside_the_workspace(tmp_path: Path) -> None:
    """Never in a diff, never committed by accident, never in the jail."""
    assert notes_path(tmp_path).parent == tmp_path
    assert notes_path(tmp_path).name == "notes.md"


def test_oversized_notes_are_refused_with_the_reason(tmp_path: Path) -> None:
    """The cap is what forces curation. Refused, not truncated: silently losing
    the tail of the agent's own notes is worse than saying no."""
    with pytest.raises(NotesError, match="prune"):
        write_notes(tmp_path, "x" * (NOTES_MAX_CHARS + 1))
    assert read_notes(tmp_path) == ""


def test_a_write_at_the_cap_is_allowed(tmp_path: Path) -> None:
    write_notes(tmp_path, "y" * NOTES_MAX_CHARS)
    assert len(read_notes(tmp_path)) == NOTES_MAX_CHARS


def test_a_hand_written_notes_file_cannot_swamp_the_prompt(tmp_path: Path) -> None:
    """The cap has to hold on the way IN as well as on the way out.

    `write_notes` refuses past it, and the block trusted that -- but the file is
    the operator's to read and edit, so an over-cap one arrives from disk having
    passed no check. Probed: a 400,000-char notes.md hand-written into the state
    dir produced a 410,307-char system prompt, verbatim, on every turn, while
    the AGENTS.md beside it clipped at 16,000 and the memories block at 12,000.

    Clipped with a pointer rather than refused: refusing here would take the
    session down over a file the agent did not write.
    """
    from agent6.workflows._prompt_blocks import notes_block

    notes_path(tmp_path).write_text("# by hand\n" + "x" * 400_000, encoding="utf-8")
    block = notes_block(read_notes(tmp_path))

    assert len(block) < NOTES_MAX_CHARS + 2000, f"the block is {len(block)} chars"
    assert "# by hand" in block, "the head of the operator's file must survive"
    assert "read_notes" in block, "the model must be told the rest exists"


def test_unreadable_notes_do_not_take_the_run_down(tmp_path: Path) -> None:
    """A hand-mangled file is the operator's business; the session continues."""
    notes_path(tmp_path).write_bytes(b"\xff\xfe not utf-8")
    assert read_notes(tmp_path) == ""


def test_the_notes_reach_a_later_session_s_prompt(tmp_path: Path) -> None:
    """The point of durability: what one session writes, the next one reads
    without being told to go looking."""
    from agent6.workflows._prompt_blocks import notes_block

    write_notes(tmp_path, "## open\n- the CSV parser still mishandles CRLF\n")
    block = notes_block(read_notes(tmp_path))

    assert "<notes>" in block
    assert "mishandles CRLF" in block
    assert "write_notes" in block, "the block must say how to change it"


def test_an_empty_scratchpad_adds_nothing_to_the_prompt(tmp_path: Path) -> None:
    """Every unused block is context the task does not get."""
    from agent6.workflows._prompt_blocks import notes_block

    assert notes_block(read_notes(tmp_path)) == ""
    assert notes_block("   \n  ") == ""


def test_the_tool_surface_exposes_both_halves() -> None:
    """A writable scratchpad the agent cannot read is a scratchpad it will
    overwrite blind."""
    from agent6.tools.schema import LOOP_EXTRA_TOOLS, ReadNotesInput, WriteNotesInput

    assert ReadNotesInput in LOOP_EXTRA_TOOLS
    assert WriteNotesInput in LOOP_EXTRA_TOOLS


def test_prompt_show_reports_the_notes_a_session_would_receive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 prompt show` exists to answer "what will a session actually
    get". A block the loop injects and this omits makes it lie by omission --
    the same defect this function's docstring records having fixed once for
    memories.
    """
    from agent6.config import Config
    from agent6.workflows import system_prompt_for

    monkeypatch.chdir(tmp_path)
    write_notes(tmp_path, "## open\n- the CSV parser still mishandles CRLF\n")

    shown = system_prompt_for(Config(), tmp_path, "run", state_dir=tmp_path)
    assert "mishandles CRLF" in shown, "prompt show under-reports the session's prompt"
