# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Check a frame against the vendored ACP schemas.

`data/acp-schema.json` is the protocol's published v1 schema and `data/acp-schema-v2.json`
its `schema-v2.0.0-beta.3`, each the file as published. A v2 union accepts any tag, so
an unchecked frame with a tag the schema names would pass whatever its shape: `v2_errors`
makes each union's open variant refuse the named tags, and refuses a key the definition
does not name.
"""

from __future__ import annotations

import functools
import json
import pathlib
from typing import Any

import jsonschema  # pyrefly: ignore [untyped-import]

_SCHEMA = pathlib.Path(__file__).parent / "data" / "acp-schema.json"
_SCHEMA_V2 = pathlib.Path(__file__).parent / "data" / "acp-schema-v2.json"
# The fields of a v2 payload that hold a tagged union's member, and the definition to check it by.
_NESTED = {
    "update": "SessionUpdate",
    "content": "ToolCallContent",
    "changes": "DiffChange",
    "patch": "DiffPatch",
    "locations": "ToolCallLocation",
    "plan": "PlanUpdateContent",
    "entries": "PlanEntry",
    "availableCommands": "AvailableCommand",
    "input": "AvailableCommandInput",
    "configOptions": "SessionConfigOption",
    "options": "SessionConfigSelectOption",
    "cost": "Cost",
    "error": "Error",
    "subject": "RequestPermissionSubject",
    "toolCall": "ToolCallUpdate",
}
# The same field names under another parent hold another definition.
_NESTED_IN = {
    ("agent_message", "content"): "ContentBlock",
    ("agent_thought", "content"): "ContentBlock",
    ("user_message", "content"): "ContentBlock",
    ("agent_message_chunk", "content"): "ContentBlock",
    ("agent_thought_chunk", "content"): "ContentBlock",
    ("user_message_chunk", "content"): "ContentBlock",
    ("content", "content"): "ContentBlock",
    ("RequestPermissionRequest", "options"): "PermissionOption",
}


@functools.cache
def _definitions() -> dict[str, Any]:
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))["$defs"]


def errors(definition: str, payload: Any) -> list[str]:
    """Return what the v1 schema's `$defs` entry rejects in the payload, empty when it fits."""
    validator = jsonschema.Draft202012Validator(
        {"$defs": _definitions(), "$ref": f"#/$defs/{definition}"}
    )
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(payload)]


@functools.cache
def _v2_definitions() -> dict[str, Any]:
    definitions = json.loads(_SCHEMA_V2.read_text(encoding="utf-8"))["$defs"]
    _close_unions(definitions)
    return definitions


def _close_unions(node: Any) -> None:
    """Make every tagged union's open variant refuse the tags its other variants name."""
    if isinstance(node, list):
        for child in node:
            _close_unions(child)
    if not isinstance(node, dict):
        return
    for child in node.values():
        _close_unions(child)
    variants = [v for v in node.get("anyOf") or () if isinstance(v, dict)]
    named: dict[str, list[Any]] = {}
    for variant in variants:
        for field, schema in (variant.get("properties") or {}).items():
            if isinstance(schema, dict) and "const" in schema:
                named.setdefault(field, []).append(schema["const"])
    for variant in variants:
        for field, schema in (variant.get("properties") or {}).items():
            if field in named and isinstance(schema, dict) and "const" not in schema:
                schema["not"] = {"enum": named[field]}


def named(definition: str) -> set[Any]:
    """Return the values a v2 open enum names: the consts of the definition's variants."""
    variants = _v2_definitions()[definition]["anyOf"]
    return {variant["const"] for variant in variants if "const" in variant}


def tags(definition: str, field: str) -> set[Any]:
    """Return the tags a v2 union names for one field of its variants."""
    variants = _v2_definitions()[definition]["anyOf"]
    return {
        variant["properties"][field]["const"]
        for variant in variants
        if "const" in variant.get("properties", {}).get(field, {})
    }


def methods() -> set[str]:
    """Return every method the v2 schema defines a message for."""
    return {entry["x-method"] for entry in _v2_definitions().values() if "x-method" in entry}


def v2_errors(definition: str, payload: Any, *, tag: str = "") -> list[str]:
    """Return what the v2 schema rejects in the payload and in the unions nested in it.

    Args:
        definition: The `$defs` entry the payload is checked by.
        payload: The payload.
        tag: The name the nested fields are looked up under; "" takes the definition's.

    Returns:
        Each rejection with its path, empty when the payload fits; a key the definition
        does not name is a rejection.
    """
    validator = jsonschema.Draft202012Validator(
        {
            "$defs": _v2_definitions(),
            "$ref": f"#/$defs/{definition}",
            "unevaluatedProperties": False,
        }
    )
    found = [
        f"{definition}{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(payload)
    ]
    if not isinstance(payload, dict):
        return found
    parent = str(payload.get("sessionUpdate") or payload.get("type") or tag or definition)
    for field, value in payload.items():
        nested = _NESTED_IN.get((parent, field)) or _NESTED.get(field)
        if nested is None or value is None:
            continue
        for member in value if isinstance(value, list) else [value]:
            if isinstance(member, dict):
                found += v2_errors(nested, member)
    return found
