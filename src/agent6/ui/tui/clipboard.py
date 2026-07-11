# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Copy-to-clipboard primitives for the TUI, framework-agnostic.

Over SSH the only route to the operator's LOCAL clipboard is OSC 52 through the
terminal (a remote xclip/wl-copy would hit the server's non-existent clipboard).
byobu/tmux swallow a bare app OSC 52 unless configured, so we offer wrapped
variants and tmux's own set-buffer. ``resolve_method("auto")`` picks a sane
default per environment; the operator can pin one via the ``copy_method`` UI
preference (see ``settings``).

Pure string/subprocess helpers only: no textual import, so the conversation view
and any dashboard card can reuse them. The Textual-coupled paths (suspend to the
native terminal, pager) live in the view.
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

CopyMethod = Literal["auto", "osc52", "osc52-tmux", "tmux-buffer"]
COPY_METHODS: tuple[CopyMethod, ...] = ("auto", "osc52", "osc52-tmux", "tmux-buffer")


def osc52_sequence(text: str, *, wrap: str) -> str:
    """An OSC 52 clipboard-set escape. ``wrap="tmux"``/``"screen"`` wraps it for
    that multiplexer's passthrough so it reaches the outer terminal instead of
    being swallowed; ``wrap=""`` is the bare sequence."""
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    seq = f"\x1b]52;c;{b64}\x07"
    if wrap == "tmux":  # tmux DCS passthrough: double every inner ESC, wrap in DCS
        return "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    if wrap == "screen":  # GNU screen passthrough (long payloads would need chunking)
        return "\x1bP" + seq + "\x1b\\"
    return seq


def mux_passthrough(seq: str) -> str:
    """Wrap a terminal escape for the ACTIVE multiplexer's passthrough (tmux DCS
    with doubled inner ESCs; GNU screen DCS), else return it bare. Inside tmux
    the outer terminal additionally needs ``allow-passthrough on`` (off by
    default since tmux 3.3)."""
    if os.environ.get("TMUX"):
        return "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
    if os.environ.get("STY"):
        return "\x1bP" + seq + "\x1b\\"
    return seq


def resolve_method(pref: str) -> str:
    """The concrete method for *pref* (from the ``copy_method`` UI pref, so any
    string); ``"auto"`` chooses per environment, anything else passes through."""
    if pref != "auto":
        return pref
    if os.environ.get("TMUX"):
        return "tmux-buffer"  # tmux emits the OSC 52 itself, the most reliable path
    if os.environ.get("STY"):
        return "osc52-tmux"  # screen: wrapped OSC 52 (set-buffer is tmux-only)
    return "osc52"


def emit_clipboard(text: str, method: str, write: Callable[[str], None]) -> str:
    """Copy *text* using a concrete *method*. *write* emits a raw terminal escape
    (the caller supplies the driver write); the ``tmux-buffer`` method shells to
    ``tmux set-buffer -w`` instead. Returns a short human status."""
    if method == "tmux-buffer":
        subprocess.run(["tmux", "set-buffer", "-w", text], check=True)
        return "via tmux set-buffer -w"
    wrap = "tmux" if method == "osc52-tmux" else ("screen" if method == "osc52-screen" else "")
    write(osc52_sequence(text, wrap=wrap))
    return "via OSC 52 (tmux-wrapped)" if wrap else "via OSC 52"


def write_transcript_file(text: str) -> Path:
    """Write *text* to a temp file and return its path (for the write-to-file copy)."""
    fd, name = tempfile.mkstemp(prefix="agent6-transcript-", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return Path(name)
