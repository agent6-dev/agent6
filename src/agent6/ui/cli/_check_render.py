# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render check output with monochrome paths and hanging annotations."""

from __future__ import annotations

import re
import shutil
import sys
import textwrap

from agent6.ui.cli import _common, _sandbox_paths


def line(prefix: str, detail: str = "", *, hang: int = 4, warning: bool = False) -> None:
    """Print a styled prefix and subdued detail, wrapping only the detail on a terminal."""
    if warning:
        prefix += _common.sgr("!", "33") + " "
    width = shutil.get_terminal_size().columns if sys.stdout.isatty() else 0
    size = len(re.sub(r"\x1b\[[0-9;]*m", "", prefix))
    if not width or size + len(detail) <= width:
        print(prefix + _common.sgr(detail, "2"))
        return
    first_word = detail.split(maxsplit=1)[0] if detail else ""
    if size + len(first_word) > width:
        print(prefix.rstrip())
        prefix, size = " " * hang, hang
    wrapped = textwrap.wrap(
        detail,
        width=max(width, hang + 1),
        initial_indent=" " * size,
        subsequent_indent=" " * hang,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if wrapped:
        print(prefix + _common.sgr(wrapped[0][size:], "2"))
        for continuation in wrapped[1:]:
            print(_common.sgr(continuation, "2"))
    else:
        print(prefix.rstrip())


def rows(entries: list[_sandbox_paths.Row]) -> None:
    """Print a fixed-width purpose column without aligning annotations to the longest path."""
    for row in entries:
        prefix = f"    {row.mode}  {_common.sgr(f'{row.tag:4}', '2')}  {row.path}"
        line(
            prefix + ("  " if row.note else ""),
            f"({row.note})" if row.note else "",
            hang=14,
            warning=row.warning,
        )


def result(status: str, name: str, detail: str) -> None:
    """Print a check result with a readable status even without colour."""
    code = {"PASS": "32", "SKIP": "2", "FAIL": "31", "WARN": "33"}.get(status, "2")
    line(f"{_common.sgr(f'[{status}]', code)} {name}: ", detail)
