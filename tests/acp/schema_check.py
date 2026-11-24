# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Check a frame against the vendored ACP schema (`data/acp-schema.json`)."""

from __future__ import annotations

import functools
import json
import pathlib
from typing import Any

import jsonschema

_SCHEMA = pathlib.Path(__file__).parent / "data" / "acp-schema.json"


@functools.cache
def _definitions() -> dict[str, Any]:
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))["$defs"]


def errors(definition: str, payload: Any) -> list[str]:
    """Return what the schema's `$defs` entry rejects in the payload, empty when it fits."""
    validator = jsonschema.Draft202012Validator(
        {"$defs": _definitions(), "$ref": f"#/$defs/{definition}"}
    )
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(payload)]
