# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 assembles its system prompt in two places -- the run loop and
``system_prompt_for`` behind `agent6 prompt show` -- and they must agree.

A block wired into one and hardcoded away in the other is invisible: the
suite passes because the block itself is unit-tested, while `prompt show`
under-reports what a session receives, or the loop silently sends nothing.
This pins the seam instead of one instance of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "agent6"

# Blocks whose content is loaded per-repo. `config`, `repo` and `mode` are
# structural arguments, not content, so they are not listed.
CONTENT_KEYWORDS = {"memories", "notes", "skills"}


def _assembly_calls() -> list[tuple[Path, ast.Call]]:
    calls: list[tuple[Path, ast.Call]] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_system_prompt"
            ):
                calls.append((path, node))
    return calls


def test_every_assembly_site_feeds_every_block() -> None:
    """Both sites pass every content block, so adding a block to one and
    forgetting the other is a red test rather than a quiet omission."""
    calls = _assembly_calls()
    assert len(calls) >= 2, "expected the loop and system_prompt_for to both assemble"

    for path, call in calls:
        passed = {kw.arg for kw in call.keywords if kw.arg}
        missing = CONTENT_KEYWORDS - passed
        assert not missing, (
            f"{path.name}:{call.lineno} assembles the system prompt without {sorted(missing)};"
            " a session gets a block that `agent6 prompt show` denies, or vice versa"
        )


def test_no_assembly_site_hardcodes_a_block_away() -> None:
    """Passing a literal is the same omission with the keyword present: the
    block is loaded for one caller and constant-empty for the other."""
    for path, call in _assembly_calls():
        for kw in call.keywords:
            if kw.arg in CONTENT_KEYWORDS:
                assert not isinstance(kw.value, ast.Constant), (
                    f"{path.name}:{call.lineno} hardcodes {kw.arg}={ast.unparse(kw.value)};"
                    " load it as the other assembly site does"
                )
