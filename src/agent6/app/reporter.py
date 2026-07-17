# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The two output channels the app pipelines write through instead of calling
`print` directly.

`out` is stdout (a piped result the operator captures); `err` is stderr (status,
warnings, refusals). `ui/cli` is the composition root that owns the real streams
(`STDIO_REPORTER`); a test or an alternate front-end injects a capturing pair.
Each channel takes one already-formatted line and writes it exactly as the
matching `print` would, so threading the reporter is behaviour-preserving."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reporter:
    out: Callable[[str], None]
    err: Callable[[str], None]


def _print_out(msg: str) -> None:
    print(msg)


def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


# The real-stream wiring: identical to `print(msg)` / `print(msg, file=sys.stderr)`.
# The default the app entry points fall back to and `ui/cli` relies on.
STDIO_REPORTER = Reporter(out=_print_out, err=_print_err)
