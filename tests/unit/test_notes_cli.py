# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 notes show` / `notes edit`: the operator surface over notes.md."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.notes import read_notes
from agent6.ui.cli import main


def _state(repo: Path) -> Path:
    """This repo's state dir, with cwd pointed at it."""
    state = resolved_state_dir(repo)
    state.mkdir(parents=True, exist_ok=True)
    return state


def test_notes_show_prints_the_scratchpad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    state = resolved_state_dir(tmp_path)
    state.mkdir(parents=True, exist_ok=True)
    (state / "notes.md").write_text("# Notes\n\n- the suite takes 7 minutes\n", encoding="utf-8")
    assert main(["notes", "show"]) == 0
    assert "the suite takes 7 minutes" in capsys.readouterr().out


def test_notes_show_says_when_there_are_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty scratchpad is the ordinary first-session state, not an error:
    it says so and names the writer, like every other first-contact message."""
    monkeypatch.chdir(tmp_path)
    assert main(["notes", "show"]) == 0
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "no notes yet" in out


def test_notes_edit_opens_the_file_in_the_editor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor gets the current notes and its save is what the next session
    reads. It works on a draft, not notes.md itself: the agent may write while
    the editor is open, and a whole-file save over that is a lost update."""
    monkeypatch.chdir(tmp_path)
    state = _state(tmp_path)
    (state / "notes.md").write_text("before\n", encoding="utf-8")
    seen = tmp_path / "seen.txt"
    fake = tmp_path / "fake-editor.sh"
    fake.write_text(f'#!/bin/sh\ncat "$1" > {seen}\nprintf \'after\\n\' > "$1"\n', encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(fake))
    assert main(["notes", "edit"]) == 0
    assert seen.read_text(encoding="utf-8") == "before\n", "the editor did not get the notes"
    assert read_notes(state) == "after\n", "the save did not reach notes.md"


def test_notes_edit_starts_notes_that_do_not_exist_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notes may not exist yet, and `notes edit` is how an operator starts
    them: the editor opens on an empty buffer and its save becomes notes.md.
    An editor that saved nothing used to leave an empty file behind."""
    monkeypatch.chdir(tmp_path)
    fake = tmp_path / "starts.sh"
    fake.write_text("#!/bin/sh\nprintf 'first notes\\n' > \"$1\"\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(fake))
    assert main(["notes", "edit"]) == 0
    notes = resolved_state_dir(tmp_path) / "notes.md"
    assert notes.is_file() and os.access(notes, os.R_OK)
    assert read_notes(resolved_state_dir(tmp_path)) == "first notes\n"

    monkeypatch.setenv("EDITOR", "true")  # saves nothing
    assert main(["notes", "edit"]) == 0
    assert read_notes(resolved_state_dir(tmp_path)) == "first notes\n"


def test_notes_edit_refuses_to_erase_a_write_it_did_not_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`notes edit` opened notes.md in $EDITOR without the notes lock, and the
    editor's save REPLACES the whole file. A `write_notes` landing while the
    operator had it open was erased on save (and the operator's edit erased by
    the agent's write) -- the lost update the lock exists to prevent. Blocking a
    live run for the length of an editing session is not the answer: the write
    is checked against what the editor was given.
    """
    from agent6.notes import write_notes
    from agent6.ui.cli.notes_cmds import _cmd_notes_edit  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    state = _state(tmp_path)
    write_notes(state, "the agent's first notes\n")

    # An editor that rewrites the file, while the agent writes underneath it.
    editor = tmp_path / "editor.sh"
    editor.write_text("#!/bin/sh\nprintf 'what the operator typed\\n' > \"$1\"\n", encoding="utf-8")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    def racing_write(_path: Path) -> str:
        write_notes(state, "the agent's SECOND notes\n")
        return "the agent's first notes\n"

    monkeypatch.setattr("agent6.ui.cli.notes_cmds._notes_before_editing", racing_write)
    rc = _cmd_notes_edit()

    assert rc != 0, "a lost update was reported as a clean save"
    assert read_notes(state) == "the agent's SECOND notes\n", "the agent's write was erased"
    assert "changed while" in capsys.readouterr().err.lower()


def test_notes_edit_refuses_over_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cap is the tool's contract with the model (the notes ride in every
    prompt), and editing by hand went around it: an oversized notes.md was
    injected uncapped for the rest of the run."""
    from agent6.notes import NOTES_MAX_CHARS
    from agent6.ui.cli.notes_cmds import _cmd_notes_edit  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    state = _state(tmp_path)
    editor = tmp_path / "big.sh"
    editor.write_text(
        f"#!/bin/sh\nhead -c {NOTES_MAX_CHARS + 100} /dev/zero | tr '\\0' 'x' > \"$1\"\n",
        encoding="utf-8",
    )
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    assert _cmd_notes_edit() != 0
    assert "cap" in capsys.readouterr().err.lower()
    assert len(read_notes(state)) <= NOTES_MAX_CHARS
