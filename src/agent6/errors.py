# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Define the operator-error boundary.

An `OperatorError` becomes a one-line `ERROR:` refusal at exit 2; any other
fault becomes a crash report. Subsystem errors for operator-owned input
subclass it.
"""

from __future__ import annotations

import pathlib


class OperatorError(Exception):
    """The operator's input or file is bad; not an agent6 defect.

    The message is the whole surface: name the flag or file and the bad value,
    and say what a valid one looks like.
    """


def read_operator_file(path: pathlib.Path) -> str:
    """Read a file the operator named, the one reader for operator-supplied files.

    Returns:
        The text.

    Raises:
        OperatorError: The file cannot be read or decoded; a refusal, never a crash.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OperatorError(f"could not read {path}: {exc}") from exc
