# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Define the two output channels the app pipelines write through.

`ui/cli` owns the real streams (`STDIO_REPORTER`); a test or another front-end injects a
capturing pair. Each channel takes one formatted line and writes it as `print` would.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reporter:
    """Write status and results through two channels with one wording per kind.

    Attributes:
        out: Stdout, the piped result the operator captures.
        err: Stderr, for status, warnings and refusals.
        receipt: Where the cost receipt goes when a live view renders it; None means `out`.
    """

    out: Callable[[str], None]
    err: Callable[[str], None]
    receipt: Callable[[str], None] | None = None

    def cost(self, msg: str) -> None:
        """Write the cost receipt."""
        (self.receipt or self.out)(msg)

    def refuse(self, msg: str) -> None:
        """Write a refusal; the run does not start (exit 2)."""
        self.err(f"REFUSING: {msg}")

    def error(self, msg: str) -> None:
        """Write an error line."""
        self.err(f"ERROR: {msg}")

    def warn(self, msg: str) -> None:
        """Write a warning line."""
        self.err(f"[agent6] WARNING: {msg}")

    def note(self, msg: str) -> None:
        """Write a status line."""
        self.err(f"[agent6] {msg}")


def _print_out(msg: str) -> None:
    print(msg)


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


# The default the app entry points fall back to.
STDIO_REPORTER = Reporter(out=_print_out, err=_print_err)
