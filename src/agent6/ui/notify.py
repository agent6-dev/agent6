# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Send a desktop notification through `notify-send`, best-effort.

A fixed argv with `--` before the data keeps a model-authored message inert,
even one starting with `-`. The body is escaped, because a notification server reads
markup there (`<b>`, `<a href>`, `<img>`). A missing `notify-send` is a silent no-op.
"""

from __future__ import annotations

import html
import shutil
import subprocess


def desktop_notify(title: str, body: str = "") -> bool:
    """Send a desktop notification when `notify-send` is on PATH.

    Args:
        title: The notification's headline.
        body: The text under the headline, shown as text: `&`, `<` and `>` are escaped.

    Returns:
        True when launched, False when unavailable, so a caller can ring a bell instead.
    """
    exe = shutil.which("notify-send")
    if exe is None:
        return False
    try:
        subprocess.Popen(
            [exe, "--", title, html.escape(body, quote=False)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return True
