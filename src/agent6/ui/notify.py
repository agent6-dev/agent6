# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Send a desktop notification through `notify-send`, best-effort.

A fixed argv with `--` before the data keeps a model-authored message inert,
even one starting with `-`. A missing `notify-send` is a silent no-op.
"""

from __future__ import annotations

import shutil
import subprocess


def desktop_notify(title: str, body: str = "") -> bool:
    """Send a desktop notification when `notify-send` is on PATH.

    Returns:
        True when launched, False when unavailable, so a caller can ring a bell instead.
    """
    exe = shutil.which("notify-send")
    if exe is None:
        return False
    try:
        subprocess.Popen(
            [exe, "--", title, body],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return True
