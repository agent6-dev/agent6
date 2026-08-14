# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The operator-error boundary.

A bad value or unreadable file from the operator raises :class:`OperatorError`;
``cli_main`` turns that into a one-line ``ERROR:`` refusal at exit 2, and
any other fault into a crash report (Ctrl-C exits 130 with a plain line;
argparse exits pass through). Subsystem error types for operator-owned
input (``ConfigError``) subclass it, so no reader needs its own except arm to
keep an operator mistake out of the crash reporter.
"""

from __future__ import annotations

from pathlib import Path


class OperatorError(Exception):
    """The operator's input or file is bad; not an agent6 defect.

    The message is the whole surface: name the flag or file and the bad value,
    and say what a valid one looks like.
    """


def read_operator_file(path: Path) -> str:
    """Read a file the operator named, refusing when it cannot be read.

    The one reader for operator-supplied files: an unreadable or undecodable
    one is a refusal naming the file, never a crash report.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OperatorError(f"could not read {path}: {exc}") from exc
