# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Define shared errors and the operator-error boundary.

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


class PackError(OperatorError):
    """Refuse an unsafe or conflicting state transfer."""


class GitError(Exception):
    """Generic git failure."""

    def __init__(self, message: str, *, safe_message: str | None = None) -> None:
        """Keep a diagnostic and a reason safe to record in model-visible review evidence.

        Args:
            message: The full diagnostic for the operator.
            safe_message: A reason without subprocess output, or None for a checked message.
        """
        super().__init__(message)
        self.safe_message = message if safe_message is None else safe_message


def read_operator_file(path: pathlib.Path) -> str:
    """Read a file the operator named, the one reader for operator-supplied files.

    Args:
        path: The file, which the refusal names.

    Returns:
        The text.

    Raises:
        OperatorError: The file cannot be read or decoded; a refusal, never a crash.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OperatorError(f"could not read {path}: {exc}") from exc
