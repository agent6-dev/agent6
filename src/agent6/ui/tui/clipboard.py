# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Copy-to-clipboard primitives for the TUI, with no textual import.

Over SSH the only route to the operator's clipboard is OSC 52 through the
terminal; tmux and screen swallow a bare one unless configured, so wrapped
variants and tmux's own `set-buffer` exist beside it. `resolve_method("auto")`
picks per environment; the `copy_method` UI preference pins one.
"""

from __future__ import annotations

import base64
import os
import pathlib
import subprocess
import tempfile
from collections.abc import Callable
from typing import Literal

CopyMethod = Literal["auto", "osc52", "osc52-tmux", "osc52-screen", "tmux-buffer"]
COPY_METHODS: tuple[CopyMethod, ...] = (
    "auto",
    "osc52",
    "osc52-tmux",
    "osc52-screen",
    "tmux-buffer",
)


def osc52_sequence(text: str, *, wrap: str) -> str:
    """Return an OSC 52 clipboard-set escape.

    Args:
        text: The text to put on the clipboard.
        wrap: "tmux" or "screen" wraps it for that multiplexer's passthrough; "" is bare.

    Returns:
        The escape sequence.
    """
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    seq = f"\x1b]52;c;{b64}\x07"
    if wrap == "tmux":  # DCS passthrough with every inner ESC doubled
        return "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    if wrap == "screen":  # a long payload would need chunking
        return "\x1bP" + seq + "\x1b\\"
    return seq


def mux_passthrough(seq: str) -> str:
    """Wrap a terminal escape for the active multiplexer's passthrough, else return it bare.

    Inside tmux the outer terminal also needs `allow-passthrough on` (off by default
    since tmux 3.3).

    Args:
        seq: The escape sequence.

    Returns:
        The sequence, wrapped for tmux or screen when one is active.
    """
    if os.environ.get("TMUX"):
        return "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    if os.environ.get("STY"):
        return "\x1bP" + seq + "\x1b\\"
    return seq


def resolve_method(pref: str) -> str:
    """Return the concrete copy method for a preference.

    Args:
        pref: The `copy_method` UI preference, any string.

    Returns:
        The preference itself, or for "auto" the method chosen for this environment.
    """
    if pref != "auto":
        return pref
    if os.environ.get("TMUX"):
        return "tmux-buffer"  # tmux emits the OSC 52 itself
    if os.environ.get("STY"):
        return "osc52-screen"
    return "osc52"


def emit_clipboard(text: str, method: str, write: Callable[[str], None]) -> str:
    """Copy text to the clipboard by a concrete method.

    Args:
        text: The text to copy.
        method: A resolved method, never "auto".
        write: Emits a raw terminal escape; the driver's write.

    Returns:
        A short status naming the route taken.

    Raises:
        subprocess.CalledProcessError: `tmux set-buffer` failed.
    """
    if method == "tmux-buffer":
        subprocess.run(["tmux", "set-buffer", "-w", text], check=True)
        return "via tmux set-buffer -w"
    wrap = "tmux" if method == "osc52-tmux" else ("screen" if method == "osc52-screen" else "")
    write(osc52_sequence(text, wrap=wrap))
    return f"via OSC 52 ({wrap}-wrapped)" if wrap else "via OSC 52"


def write_transcript_file(text: str) -> pathlib.Path:
    """Return the path of a new temp file holding the text."""
    fd, name = tempfile.mkstemp(prefix="agent6-transcript-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return pathlib.Path(name)
