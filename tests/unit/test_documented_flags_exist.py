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


def _named_flags(text: str) -> set[str]:
    """Every flag a doc attributes to agent6.

    Two rules, because the docs name flags two ways. BACKTICKED anywhere: prose
    writes them as code, and scanning the file rather than the line is what
    catches one on a continuation line. And every flag on an agent6-invoking
    LINE inside a fenced block: that is where a quickstart lives, and a broken
    flag there is the first thing a new user hits.

    Line-scoped inside blocks on purpose. A shell block often mixes tools --
    `tailscale serve --bg` sits under `agent6 web` in docs/web.md -- and
    block-scoping would attribute that to us.
    """
    named = {m.rstrip(".,;:)") for m in re.findall(r"`(--[a-z0-9][a-z0-9-]+)", text)}
    for block in re.finditer(r"```[a-z]*\n(.*?)```", text, re.S):
        for line in block.group(1).splitlines():
            if re.search(r"(^|\s)agent6\s", line):
                named |= {m.rstrip(".,;:)`") for m in re.findall(r"--[a-z0-9][a-z0-9-]+", line)}
    return named


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_flags_exist(doc: Path) -> None:
    missing = _named_flags(doc.read_text("utf-8")) - _cli_flags() - _NOT_OURS
    assert not missing, f"{doc.name} names flags the CLI does not have: {sorted(missing)}"


def _config_leaves() -> set[str]:
    """Every dotted leaf a config file can set, from the model itself.

    A name-keyed table (`providers`, `mcp.servers`) has no row of its own until
    an entry exists, so it contributes its ENTRY's fields under `<name>` --
    which is how the docs write them.
    """
    import typing

    from pydantic import BaseModel

    from agent6.config import Config

    leaves: set[str] = set()

    def sections_of(annotation: object) -> list[type[BaseModel]]:
        """Every model behind a field, through `X | None`, `Annotated[...]` and
        a discriminated union (a provider entry is one of two shapes)."""
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        found: list[type[BaseModel]] = []
        for arg in typing.get_args(annotation):
            found += sections_of(arg)
        return found

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            if typing.get_origin(field.annotation) is dict:
                for entry in sections_of(typing.get_args(field.annotation)[1]):
                    walk(entry, f"{path}.<name>.")
                continue
            nested = sections_of(field.annotation)
            for section in nested:
                walk(section, f"{path}.")
            if not nested:
                leaves.add(path)

    walk(Config, "")
    return leaves


def test_every_config_leaf_is_documented() -> None:
    """A knob nobody can find is a knob nobody set on purpose.

    docs/config.md is the operator's map of the config; a field added without a
    row is invisible until someone reads the source. The match is loose on
    purpose -- a row may name a leaf in full or from any section boundary
    (`checkpoint.message` under `[git.commit.checkpoint]`) -- because the claim
    being checked is "documented at all", and a strict one would be a test of
    the page's table style.

    The mirror direction (no row names a key that no longer exists) is NOT
    checked here: the page mixes config tables with preset, directory and
    environment tables, and several headings cover two sections at once, so
    every version of that check was a heuristic about formatting. The real fix
    is to GENERATE these tables from the model, the way docs/data-contracts.md
    is generated -- then both directions hold by construction.
    """
    text = (ROOT / "docs" / "config.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`([A-Za-z_][\w.<>-]*)`", text))
    missing = sorted(
        leaf
        for leaf in _config_leaves()
        if not any(named_key == leaf or leaf.endswith(f".{named_key}") for named_key in named)
    )
    assert not missing, f"undocumented config leaves in docs/config.md: {missing}"
