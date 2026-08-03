# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every flag the docs name in backticks is a flag the CLI actually has.

`--max-input-tokens` and `--max-output-tokens` outlived the budget redesign in
docs/config.md AND in three bench scripts, which would have died on
"unrecognized arguments" at the first invocation. A doc that names a flag is a
promise; this checks it against the parser.

Backticks are the whole heuristic: the docs write every flag as code, and
scanning the file (not the line) is what catches one named on a continuation
line, which is exactly where the stale pair hid.
"""

from __future__ import annotations

import re
from argparse import ArgumentParser
from pathlib import Path

import pytest

from agent6.ui.cli.parser import build_parser

ROOT = Path(__file__).resolve().parents[2]
DOCS = [*sorted((ROOT / "docs").glob("*.md")), ROOT / "README.md"]

# Other tools' flags, named in prose about what agent6 does with them.
_NOT_OURS = {"--no-ext-diff", "--no-textconv", "--no-ff"}


def _cli_flags() -> set[str]:
    """Every long option the parser knows, at every subcommand depth."""
    found: set[str] = set()

    def walk(parser: ArgumentParser) -> None:
        for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
            found.update(o for o in action.option_strings if o.startswith("--"))
            # Subparsers hang off a dict-valued `choices`; a plain argument's
            # is a tuple of values with no parser to descend into.
            if isinstance(action.choices, dict):
                for sub in action.choices.values():
                    if isinstance(sub, ArgumentParser):
                        walk(sub)

    walk(build_parser())
    return found


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_flags_exist(doc: Path) -> None:
    named = {
        m.rstrip(".,;:)") for m in re.findall(r"`(--[a-z0-9][a-z0-9-]+)", doc.read_text("utf-8"))
    }
    missing = named - _cli_flags() - _NOT_OURS
    assert not missing, f"{doc.name} names flags the CLI does not have: {sorted(missing)}"
