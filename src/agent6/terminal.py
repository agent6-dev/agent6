# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Scrub terminal sequences for command results and operator rendering."""

from __future__ import annotations

import re

# String-carrying families drop their payloads, including an unfinished trailing payload.
_ESCAPE_PATTERN = (
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC, BEL- or ST-terminated (or cut off)
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"  # DCS / SOS / PM / APC, ST-terminated
    r"|\x1b."  # any other escape, with its final byte
)
_ESCAPE_RE = re.compile(_ESCAPE_PATTERN + r"|\x1b", re.DOTALL)
# Default-deny for rendering: a C1 byte opens the same doors 8-bit; only \n and \t stay.
_CONTROL_RE = re.compile(_ESCAPE_PATTERN + r"|[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]", re.DOTALL)
# SGR styling passes, conceal (SGR 8) excepted; styling cannot move the cursor or rewrite a line.
_TERMINAL_RE = re.compile(
    r"(?P<own>\x1b\[(?!(?:[0-9;]*;)?0*8(?:;|m))[0-9;]*m)|" + _CONTROL_RE.pattern, re.DOTALL
)


def scrub_terminal_escapes(text: str) -> str:
    """Return command text without escape sequences, preserving non-escape characters."""
    return _ESCAPE_RE.sub("", text)


def scrub_terminal_controls(text: str) -> str:
    """Return the text without terminal control sequences and stray control characters.

    Idempotent, so an accumulating stream tail re-scrubs for free.
    """
    return _CONTROL_RE.sub("", text)


def scrub_terminal_output(text: str) -> str:
    """Return the text the CLI may write to its terminal: SGR styling kept, controls dropped."""
    return _TERMINAL_RE.sub(lambda m: m.group("own") or "", text)
