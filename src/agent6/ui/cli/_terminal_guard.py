# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Scrub model-influenced text before it reaches the operator's terminal.

A terminal obeys a control sequence wherever it appears in a line: OSC 52 writes
the clipboard, a title change or a forged line follows. `guarded_terminal` wraps
stdout and stderr for the CLI's lifetime and `scrub_terminal_output` decides what
passes; the `/dev/tty` writers in `_steer` scrub each write the same way. Under
the wrapper sit the ACP and MCP stdio protocols (bytes through `sys.stdout.buffer`),
the spinners and the composer, whose erase and cursor idioms go to `raw_stream`.
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Generator, Iterable
from typing import IO, Any

from agent6.viewmodel import transcript


class ScrubbedStream:
    """A text stream whose every write is scrubbed; everything else is the wrapped stream's."""

    def __init__(self, raw: IO[str]) -> None:
        self.raw = raw

    def write(self, text: str) -> int:
        """Return what the wrapped stream wrote of the scrubbed text."""
        return self.raw.write(transcript.scrub_terminal_output(text))

    def writelines(self, lines: Iterable[str]) -> None:
        """Write each line, scrubbed."""
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


def raw_stream(stream: IO[str] | ScrubbedStream) -> IO[str]:
    """Return the stream under the scrubber, for a writer that scrubs its own text."""
    return stream.raw if isinstance(stream, ScrubbedStream) else stream


@contextlib.contextmanager
def guarded_terminal() -> Generator[None]:
    """Scrub stdout and stderr for the block, restoring the originals after.

    A redirected stdout is block-buffered, so a run's log file would stay empty until
    exit: it is line-buffered here, for the same lifetime.

    Yields:
        Nothing; the streams are wrapped for the block.
    """
    out, err = sys.stdout, sys.stderr
    if isinstance(out, io.TextIOWrapper):
        out.reconfigure(line_buffering=True)
    sys.stdout, sys.stderr = ScrubbedStream(out), ScrubbedStream(err)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err
