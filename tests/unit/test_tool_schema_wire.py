# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structural pin for the LLM-facing tool schemas.

``schemas_as_provider_tools()`` emits the Anthropic-shape descriptors the model
sees for every tool in ``ALL_TOOLS``. Their STRUCTURE -- tool names, required
fields, property names and types (incl. nested ``$defs`` like ``EditPair``) --
is frozen LLM I/O: a silent drift changes what every model can call. This pins
that structure against a golden digest so a schema change is deliberate.

Description prose is EXCLUDED on purpose: it is tuned deliberately (small-model
phrasing) and would make the pin fight every wording tweak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent6.tools.schema import schemas_as_provider_tools

_GOLDEN = Path(__file__).parent / "data" / "golden_tool_schemas.json"


def _prop_type(sub: dict[str, Any]) -> str:
    """A property subschema -> a compact type descriptor (no description prose)."""
    if "type" in sub:
        t = sub["type"]
        if t == "array" and isinstance(sub.get("items"), dict):
            return f"array[{_prop_type(sub['items'])}]"
        return str(t)
    if "$ref" in sub:
        return sub["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in sub:
        return "anyOf[" + ",".join(sorted(_prop_type(s) for s in sub["anyOf"])) + "]"
    if "allOf" in sub:
        return "allOf[" + ",".join(_prop_type(s) for s in sub["allOf"]) + "]"
    return "enum" if "enum" in sub else "?"


def _props(schema: dict[str, Any]) -> dict[str, str]:
    return {k: _prop_type(v) for k, v in sorted(schema.get("properties", {}).items())}


def _digest() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in schemas_as_provider_tools():
        schema = tool["input_schema"]
        entry: dict[str, Any] = {
            "name": tool["name"],
            "required": sorted(schema.get("required", [])),
            "properties": _props(schema),
        }
        if schema.get("$defs"):
            entry["defs"] = {name: _props(d) for name, d in sorted(schema["$defs"].items())}
        out.append(entry)
    return out


def test_tool_schemas_structure_matches_golden() -> None:
    generated = json.dumps(_digest(), indent=2) + "\n"
    committed = _GOLDEN.read_text(encoding="utf-8")
    assert generated == committed, (
        "LLM-facing tool schema structure drifted; if intended, regenerate the "
        'golden: python -c "import json,tests.unit.test_tool_schema_wire as t; '
        "open(t._GOLDEN,'w').write(json.dumps(t._digest(),indent=2)+chr(10))\""
    )
