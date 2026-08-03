# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 notes show` / `notes edit`: the operator surface over notes.md."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.ui.cli import main


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
    """The editor is spawned on notes.md itself, so an operator edit is what
    the next session reads (the agent re-reads the file, never a copy)."""
    monkeypatch.chdir(tmp_path)
    state = resolved_state_dir(tmp_path)
    state.mkdir(parents=True, exist_ok=True)
    (state / "notes.md").write_text("before\n", encoding="utf-8")
    marker = tmp_path / "edited.txt"
    fake = tmp_path / "fake-editor.sh"
    fake.write_text(f'#!/bin/sh\necho "$1" > {marker}\n', encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(fake))
    assert main(["notes", "edit"]) == 0
    assert marker.read_text(encoding="utf-8").strip() == str(state / "notes.md")


def test_notes_edit_creates_the_file_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`plan edit` opens an existing plan; notes may not exist yet, and an
    editor opened on a missing path is how an operator starts them."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EDITOR", "true")
    assert main(["notes", "edit"]) == 0
    assert (resolved_state_dir(tmp_path) / "notes.md").is_file()
    assert os.access(resolved_state_dir(tmp_path) / "notes.md", os.R_OK)
