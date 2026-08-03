# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 notes` subcommands (show/edit) over the agent's scratchpad."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from agent6.notes import NotesError, notes_path, read_notes, replace_notes_if_unchanged
from agent6.paths import mkdir_for_real_user
from agent6.ui.cli._common import _state_dir

NO_NOTES_YET = (
    "no notes yet. The agent writes them with `write_notes` during a run;"
    " `agent6 notes edit` starts them by hand."
)


def _cmd_notes_show() -> int:
    """Print this repo's notes.md, or say there are none."""
    text = read_notes(_state_dir(Path.cwd()))
    if not text.strip():
        print(NO_NOTES_YET)
        return 0
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


def _notes_before_editing(path: Path) -> str:
    """The notes as they stand, for the editor to work from."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _cmd_notes_edit() -> int:
    """Open this repo's notes in $EDITOR (default: vi).

    The editor works on a copy, and the save lands only if the notes still read
    as they did when it opened. Holding the notes lock across an editing
    session would stall a live run for as long as a person takes; holding
    nothing made the save a whole-file replace over whatever the agent wrote
    meanwhile. The cap applies here too: an oversized notes.md rides in every
    prompt for the rest of the run.

    Operator-controlled argv (the editor name + the path), not LLM-controlled,
    so direct subprocess.run is allowed.
    """
    state = _state_dir(Path.cwd())
    mkdir_for_real_user(state)
    before = _notes_before_editing(notes_path(state))
    draft = state / "notes.editing.md"
    draft.write_text(before, encoding="utf-8")
    editor = os.environ.get("EDITOR", "vi")
    # $EDITOR may be a command with flags ("code --wait"); split it.
    argv = shlex.split(editor) or ["vi"]
    try:
        result = subprocess.run([*argv, str(draft)], check=False)
    except OSError as exc:
        print(f"ERROR: failed to spawn editor {editor!r}: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(
            f"ERROR: editor {editor!r} exited {result.returncode}; notes unchanged", file=sys.stderr
        )
        return result.returncode
    edited = draft.read_text(encoding="utf-8")
    if edited == before:
        draft.unlink(missing_ok=True)
        return 0
    try:
        wrote = replace_notes_if_unchanged(state, expected=before, content=edited)
    except NotesError as exc:
        print(f"ERROR: {exc}. Your version is kept at {draft}", file=sys.stderr)
        return 1
    if not wrote:
        print(
            f"ERROR: the notes changed while your editor was open, so this save would"
            f" erase that write. Your version is kept at {draft}",
            file=sys.stderr,
        )
        return 1
    draft.unlink(missing_ok=True)
    return 0
