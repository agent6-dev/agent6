#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render docs/config.md from config_template.md and the config model.

The template is the page. Where a field table belongs it carries one marker
naming the sections whose leaves fill it:

    <!-- config-table: git.commit.checkpoint git.commit.squash -->

Each row comes from the model, so a renamed field moves its row and a wrong
default is not expressible; a row that reads badly has a bad description.
Regenerate with `uv run python docs/gen_config.py`; pinned byte for byte by
tests/unit/test_config_doc.py.
"""

from __future__ import annotations

import json
import pathlib
import re
import typing
from typing import Any, cast

import pydantic
import pydantic_core

from agent6.config import layer
from agent6.config import model as config_model

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = _ROOT / "docs" / "config_template.md"
_OUT = _ROOT / "docs" / "config.md"
REGEN_CMD = "uv run python docs/gen_config.py"

_MARKER = re.compile(r"^<!-- config-table:\s*(.+?)\s*-->$")
_PRESETS_MARKER = "<!-- presets-table -->"

# Fields whose default is resolved at runtime, so only the page can say what it becomes.
_RUNTIME_DEFAULTS = {
    "context.drop_at_chars": "_adaptive_",
    "context.summarise_at_chars": "_adaptive_",
    "sandbox.memory_limit_mb": "`0` (off)",
    "providers.<name>.base_url": "per (format, deployment)",
    "providers.<name>.auth_style": "per (format, deployment)",
}


def _sections_of(annotation: object) -> list[type[pydantic.BaseModel]]:
    """Return every model class an annotation can hold, through `Annotated` and unions alike."""
    if isinstance(annotation, type):
        return [annotation] if issubclass(annotation, pydantic.BaseModel) else []
    found: list[type[pydantic.BaseModel]] = []
    for arg in typing.get_args(annotation) or ():
        found.extend(_sections_of(arg))
    return found


def leaves() -> dict[str, tuple[str, str]]:
    """Return every leaf's rendered default and description by dotted path, in model order.

    A `dict[str, Section]` field is one section spelled `<name>`, documented once.
    """
    out: dict[str, tuple[str, str]] = {}

    def walk(model: type[pydantic.BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}{name}"
            if typing.get_origin(field.annotation) is dict:
                entries = _sections_of(typing.get_args(field.annotation)[1])
                # `dict[str, Section]` is a section map; `dict[str, str]` is one leaf with a table.
                if entries:
                    for entry in entries:
                        walk(entry, f"{path}.<name>.")
                    continue
            nested = _sections_of(field.annotation)
            for section in nested:
                walk(section, f"{path}.")
            if not nested:
                out[path] = (_default_cell(path, field), field.description or "")

    walk(config_model.Config, "")
    return out


def _default_cell(path: str, field: object) -> str:
    """Return a leaf's default as the table shows it."""
    if path in _RUNTIME_DEFAULTS:
        return _RUNTIME_DEFAULTS[path]
    default = getattr(field, "default", pydantic_core.PydanticUndefined)
    factory = getattr(field, "default_factory", None)
    if default is pydantic_core.PydanticUndefined and factory is None:
        return "*(required)*"
    value = factory() if factory is not None else default
    if value is None:
        return "none"
    return f"`{json.dumps(list(value) if isinstance(value, tuple) else value)}`"


def _common_parent(sections: list[str]) -> str:
    """Return the deepest section prefix every section in a table shares.

    A table over siblings keeps enough of the path to tell them apart.
    """
    parts = [s.split(".") for s in sections]
    shared: list[str] = []
    for piece in zip(*parts, strict=False):
        if len({*piece}) != 1:
            break
        shared.append(piece[0])
    return ".".join(shared)


def render_table(sections: list[str], all_leaves: dict[str, tuple[str, str]]) -> list[str]:
    """Return one field table's lines.

    Raises:
        SystemExit: The marker matched no fields.
    """
    parent = _common_parent(sections) if len(sections) > 1 else sections[0]
    rows = ["| Field | Default | Meaning |", "|---|---|---|"]
    seen = 0
    for path, (default, description) in all_leaves.items():
        if not any(path == s or path.startswith(f"{s}.") for s in sections):
            continue
        # A nested section's leaves belong to that section's own table.
        owner = path.rsplit(".", 1)[0]
        if owner not in sections:
            continue
        key = path[len(parent) + 1 :] if parent and path.startswith(f"{parent}.") else path
        rows.append(f"| `{key}` | {default} | {description} |")
        seen += 1
    if not seen:
        raise SystemExit(f"config-table marker matched no fields: {' '.join(sections)}")
    return rows


def _flatten(prefix: str, node: dict[str, Any]) -> list[str]:
    """Return a preset's overrides as `path = value` code spans."""
    out: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_flatten(path, cast(dict[str, Any], value)))
        else:
            out.append(f"`{path} = {json.dumps(value)}`")
    return out


def render_presets_table() -> list[str]:
    """Return the built-in presets table's lines."""
    rows = ["| Preset | For | Sets |", "|---|---|---|"]
    for name, overrides in layer.BUILTIN_PRESETS.items():
        sets = ", ".join(_flatten("", overrides)) or "nothing (the defaults)"
        rows.append(f"| `{name}` | {layer.BUILTIN_PRESET_NOTES[name]} | {sets} |")
    return rows


def render(template: str) -> str:
    """Return the page rendered from the template."""
    all_leaves = leaves()
    out: list[str] = [
        "<!-- Generated from docs/config_template.md by docs/gen_config.py;"
        " edit those, then regenerate. -->",
    ]
    for line in template.splitlines():
        if line.strip() == _PRESETS_MARKER:
            out.extend(render_presets_table())
            continue
        marker = _MARKER.match(line)
        if marker is None:
            out.append(line)
            continue
        out.extend(render_table(marker.group(1).split(), all_leaves))
    return "\n".join(out) + "\n"


def main() -> None:
    """Write the page."""
    page = render(_TEMPLATE.read_text(encoding="utf-8"))
    _OUT.write_text(page, encoding="utf-8")
    print(f"wrote {_OUT.relative_to(_ROOT)} ({len(page.splitlines())} lines)")


if __name__ == "__main__":
    main()
