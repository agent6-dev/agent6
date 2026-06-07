# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Comment-preserving TOML read/write helpers for the CLI config writers."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any


def _toml_value(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _upsert_toml_table(path: Path, table: str, fields: dict[str, str | bool | None]) -> None:
    """Insert or replace a single ``[table]`` block in *path*, preserving the
    rest of the file (other tables and their comments).

    Append-only-ish: we never round-trip the whole document through a TOML
    serializer (which would drop comments); we only rewrite the target
    table's span. ``None`` field values are omitted.
    """
    block_lines = [f"[{table}]"]
    for key, val in fields.items():
        if val is None:
            continue
        block_lines.append(f"{key} = {_toml_value(val)}")
    block = "\n".join(block_lines)

    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    header = f"[{table}]"
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is None:
        prefix = text if not text or text.endswith("\n") else text + "\n"
        sep = "\n" if prefix and not prefix.endswith("\n\n") else ""
        path.write_text(prefix + sep + block + "\n", encoding="utf-8")
        return
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].lstrip().startswith("["):
            end = j
            break
    new_lines = lines[:start] + block.splitlines() + [""] + lines[end:]
    path.write_text("\n".join(new_lines).rstrip("\n") + "\n", encoding="utf-8")


def _toml_repr(value: object) -> str:
    """Serialize a scalar or list-of-scalars to its TOML literal form."""
    if isinstance(value, bool):  # bool first: it is a subclass of int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_value(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_repr(v) for v in value) + "]"
    raise ValueError(f"cannot serialize {value!r} to TOML")


def _parse_cli_value(value: str) -> object:
    """Interpret a CLI-supplied value the way TOML would.

    ``true``/``false`` become bools, numbers become int/float, quoted or
    bracketed text parses as a TOML string/array, and anything else (e.g. a
    bare enum like ``provider_only`` or a model id) is taken verbatim as a
    string. This keeps ``config set sandbox.network provider_only`` ergonomic
    while still allowing ``config set sandbox.protect_git false``.
    """
    try:
        return tomllib.loads(f"_v = {value}")["_v"]
    except tomllib.TOMLDecodeError:
        return value


def _split_dotted_key(dotted_key: str) -> tuple[str, str]:
    """Split ``sandbox.network`` into ``("sandbox", "network")``.

    Config leaves always live under a section table, so a usable key has at
    least two non-empty segments; the parent segments form the TOML table.
    """
    parts = dotted_key.split(".")
    if len(parts) < 2 or any(not p for p in parts):
        raise ValueError(
            f"config key must be a dotted leaf path like 'sandbox.network', got {dotted_key!r}"
        )
    return ".".join(parts[:-1]), parts[-1]


def _upsert_toml_leaf(path: Path, dotted_key: str, value: object) -> None:
    """Set a single ``table.leaf`` key in *path*, preserving the rest verbatim.

    Like :func:`_upsert_toml_table` this is deliberate line surgery rather than
    a full serializer round-trip, so comments and sibling keys/tables survive.
    Creates the ``[table]`` block if it is absent.
    """
    table, leaf = _split_dotted_key(dotted_key)
    new_line = f"{leaf} = {_toml_repr(value)}"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    header = f"[{table}]"
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        prefix = text if (not text or text.endswith("\n")) else text + "\n"
        sep = "\n" if prefix and not prefix.endswith("\n\n") else ""
        path.write_text(prefix + sep + header + "\n" + new_line + "\n", encoding="utf-8")
        return
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].lstrip().startswith("[")),
        len(lines),
    )
    leaf_re = re.compile(rf"^\s*{re.escape(leaf)}\s*=")
    for j in range(start + 1, end):
        if leaf_re.match(lines[j]):
            lines[j] = new_line
            path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
            return
    insert_at = end
    while insert_at - 1 > start and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines[insert_at:insert_at] = [new_line]
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def _remove_toml_leaf(path: Path, dotted_key: str) -> bool:
    """Delete a single ``table.leaf`` line from *path*. Returns True if removed."""
    table, leaf = _split_dotted_key(dotted_key)
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    header = f"[{table}]"
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        return False
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].lstrip().startswith("[")),
        len(lines),
    )
    leaf_re = re.compile(rf"^\s*{re.escape(leaf)}\s*=")
    for j in range(start + 1, end):
        if leaf_re.match(lines[j]):
            del lines[j]
            out = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
            path.write_text(out, encoding="utf-8")
            return True
    return False


def _read_toml_file(path: Path) -> dict[str, Any]:
    """Parse *path* as TOML, or return an empty dict if it does not exist."""
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_toml_leaf(data: dict[str, Any], dotted_key: str) -> object:
    """Walk *data* by the dotted key, returning the value or None if absent."""
    cur: object = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
