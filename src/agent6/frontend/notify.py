# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Best-effort desktop notification via `notify-send` (device-present channel).

Used by the CLI (`agent6 watch`) and the TUI to surface a machine's
`machine.notify`/end while an operator is at the machine. Fire-and-forget with a
FIXED argv (`notify-send <title> <body>`): the two data arguments are positional
and never reach a shell, so a model-authored notify message is inert data, not a
command. A missing `notify-send` is a silent no-op (the caller also rings the
terminal bell / uses the in-app toast). No network, no `notify-send` install is
required.
"""

from __future__ import annotations

import shutil
import subprocess


def desktop_notify(title: str, body: str = "") -> bool:
    """Fire a desktop notification if `notify-send` is on PATH. Returns True when
    it was launched, False when unavailable (so a caller can fall back to a bell)."""
    exe = shutil.which("notify-send")
    if exe is None:
        return False
    try:
        subprocess.Popen(
            [exe, title, body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return True
