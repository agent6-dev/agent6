# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pin point-form config descriptions at their source."""

from __future__ import annotations

import ast
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_field_descriptions_have_no_sentence_breaks() -> None:
    """Keep config table cells and terminal descriptions out of sentence sequences."""
    descriptions: list[tuple[str, str]] = []
    for path in sorted((_ROOT / "src" / "agent6" / "config").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "description"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                descriptions.append((f"{path.name}:{node.lineno}", node.value.value))
    assert descriptions
    for location, description in descriptions:
        plain = re.sub(r"`[^`]*`", "code", description).replace("e.g.", "for example")
        assert not re.search(r"[.!?]\s+\S", plain), (
            f"{location}: use semicolon-separated fragments: {description}"
        )
