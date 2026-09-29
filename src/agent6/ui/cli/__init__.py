# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 command-line interface.

`entry.py` parses argv and dispatches to one module per command family; the
console script and the tests enter through `main` and `cli_main`.
"""

from __future__ import annotations

from agent6.ui.cli.entry import cli_main, main  # noqa: ICN003  # re-export

__all__ = ["cli_main", "main"]
