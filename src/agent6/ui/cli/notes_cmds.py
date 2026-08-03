# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 notes` subcommands (show/edit) over the agent's scratchpad."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from agent6.notes import notes_path, read_notes
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


def _cmd_notes_edit() -> int:
    """Open this repo's notes.md in $EDITOR (default: vi).

    The file is created empty when absent, so starting notes by hand is one
    command. Operator-controlled argv (the editor name + the notes path), not
    LLM-controlled, so direct subprocess.run is allowed.
    """
    state = _state_dir(Path.cwd())
    path = notes_path(state)
    if not path.exists():
        mkdir_for_real_user(state)
        path.touch()
    editor = os.environ.get("EDITOR", "vi")
    # $EDITOR may be a command with flags ("code --wait"); split it.
    argv = shlex.split(editor) or ["vi"]
    try:
        result = subprocess.run([*argv, str(path)], check=False)
    except OSError as exc:
        print(f"ERROR: failed to spawn editor {editor!r}: {exc}", file=sys.stderr)
        return 1
    return result.returncode
