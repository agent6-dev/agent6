# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Comment-preserving TOML read/write surgery for config writers.

Low-level, UI-agnostic: used by the `config` CLI subcommands, by
`config_layer`'s shared edit path, and (through it) by the TUI/web config
editors, so every writer preserves comments + siblings identically."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from agent6.config.model import ConfigError
from agent6.portable import atomic_write


def _write(path: Path, text: str) -> None:
    """Publish config text via tmp+rename, matching every other agent6 state
    writer. Plain `write_text` truncated in place, so a crash mid-write left a
    truncated/empty config; the rename makes the update all-or-nothing."""
    atomic_write(path, text)


def _toml_value(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def upsert_toml_table(path: Path, table: str, fields: dict[str, str | bool | None]) -> None:
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
        _write(path, prefix + sep + block + "\n")
        return
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].lstrip().startswith("["):
            end = j
            break
    new_lines = lines[:start] + block.splitlines() + [""] + lines[end:]
    _write(path, "\n".join(new_lines).rstrip("\n") + "\n")


def format_toml_value(value: object) -> str:  # noqa: PLR0911
    """Serialize a scalar, list, or (inline-table) dict to its TOML literal form."""
    if isinstance(value, bool):  # bool first: it is a subclass of int
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _toml_value(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        # Inline table, e.g. an OpenRouter routing value:
        #   extra_body = { provider = { sort = "throughput" } }
        # Written on one line so the existing leaf-line surgery can replace it
        # wholesale (a nested `[table]` would collide with the inline parent).
        if not value:
            return "{}"
        items = ", ".join(f"{_toml_key(k)} = {format_toml_value(v)}" for k, v in value.items())
        return "{ " + items + " }"
    raise ValueError(f"cannot serialize {value!r} to TOML")


def _toml_key(key: object) -> str:
    """A TOML key: bare if it is a simple identifier, else a quoted string."""
    k = str(key)
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _toml_value(k)


def parse_cli_value(value: str) -> object:
    """Interpret a CLI-supplied value the way TOML would.

    ``true``/``false`` become bools, numbers become int/float, quoted or
    bracketed text parses as a TOML string/array, and anything else (e.g. a
    bare enum like ``provider_only`` or a model id) is taken verbatim as a
    string. This keeps ``config set sandbox.agent_network providers`` ergonomic
    while still allowing ``config set sandbox.protect_git false``.
    """
    try:
        return tomllib.loads(f"_v = {value}")["_v"]
    except tomllib.TOMLDecodeError:
        return value


def _split_dotted_key(dotted_key: str) -> tuple[str, str]:
    """Split ``sandbox.agent_network`` into ``("sandbox", "agent_network")``.

    Config leaves always live under a section table, so a usable key has at
    least two non-empty segments; the parent segments form the TOML table.
    """
    parts = dotted_key.split(".")
    if len(parts) < 2 or any(not p for p in parts):
        raise ValueError(
            f"config key must be a dotted leaf path like 'sandbox.network', got {dotted_key!r}"
        )
    return ".".join(parts[:-1]), parts[-1]


def upsert_toml_leaf(path: Path, dotted_key: str, value: object) -> None:
    """Set a single ``table.leaf`` key in *path*, preserving the rest verbatim.

    Like :func:`upsert_toml_table` this is deliberate line surgery rather than
    a full serializer round-trip, so comments and sibling keys/tables survive.
    Creates the ``[table]`` block if it is absent.
    """
    table, leaf = _split_dotted_key(dotted_key)
    new_line = f"{leaf} = {format_toml_value(value)}"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    header = f"[{table}]"
    start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        prefix = text if (not text or text.endswith("\n")) else text + "\n"
        sep = "\n" if prefix and not prefix.endswith("\n\n") else ""
        _write(path, prefix + sep + header + "\n" + new_line + "\n")
        return
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].lstrip().startswith("[")),
        len(lines),
    )
    leaf_re = re.compile(rf"^\s*{re.escape(leaf)}\s*=")
    for j in range(start + 1, end):
        if leaf_re.match(lines[j]):
            lines[j] = new_line
            _write(path, "\n".join(lines).rstrip("\n") + "\n")
            return
    insert_at = end
    while insert_at - 1 > start and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines[insert_at:insert_at] = [new_line]
    _write(path, "\n".join(lines).rstrip("\n") + "\n")


def _scan_toml_line(text: str, depth: int, triple: str | None) -> tuple[int, str | None]:
    """Advance the (bracket-depth, open-triple-quote) state across one line, so
    ``_value_line_span`` can tell where a multi-line value ends. Brackets and
    quotes inside a string, and everything after a ``#`` comment, do not count."""
    i, n = 0, len(text)
    while i < n:
        if triple is not None:
            triple, i = (None, i + 3) if text.startswith(triple, i) else (triple, i + 1)
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            triple, i = text[i : i + 3], i + 3
            continue
        ch = text[i]
        if ch in ('"', "'"):
            i += 1
            while i < n and text[i] != ch:
                i += 2 if (ch == '"' and text[i] == "\\") else 1
            i += 1
            continue
        if ch == "#":
            break  # rest of the line is a comment
        depth += (ch in "[{") - (ch in "]}")
        i += 1
    return depth, triple


def _value_line_span(lines: list[str], start: int) -> int:
    """How many lines the TOML value assigned on ``lines[start]`` spans (>=1).

    A multi-line array (``leaf = [``...``]``) or triple-quoted string occupies
    several lines; deleting only the opening line orphans the rest and leaves an
    unparseable file."""
    eq = lines[start].find("=")
    text = lines[start][eq + 1 :] if eq != -1 else lines[start]
    depth, triple = 0, None
    idx = start
    while True:
        depth, triple = _scan_toml_line(text, depth, triple)
        if triple is None and depth <= 0:
            return idx - start + 1
        idx += 1
        if idx >= len(lines):
            return idx - start  # unterminated value: consume to EOF
        text = lines[idx]


def remove_toml_leaf(path: Path, dotted_key: str) -> bool:
    """Delete a single ``table.leaf`` line from *path*. Returns True if removed.
    Removing the section's last leaf drops the now-empty ``[table]`` header too
    (a dangling header otherwise accretes across unsets); a section that still
    holds comments is kept, they are the operator's."""
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
            span = _value_line_span(lines, j)
            del lines[j : j + span]
            remaining_end = end - span  # next section header shifted up by span
            if all(not rest.strip() for rest in lines[start + 1 : remaining_end]):
                del lines[start:remaining_end]
            out = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
            _write(path, out)
            return True
    return False


def remove_toml_table(path: Path, table: str) -> bool:
    """Delete a whole ``[table]`` section (its header, body, and any ``[table.sub]``
    subtables) from *path*. Returns True if the table was present. Used by
    ``config fix`` to drop an unknown/extra top-level table (e.g. a leftover
    ``[cli]`` from a removed feature), where deleting a single leaf would leave an
    empty-but-still-invalid table behind."""
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    dropping = False
    removed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped.strip("[]").strip()
            dropping = name == table or name.startswith(f"{table}.")
            removed = removed or dropping
        if not dropping:
            kept.append(line)
    if not removed:
        return False
    out = "\n".join(kept).rstrip("\n") + "\n" if any(ln.strip() for ln in kept) else ""
    _write(path, out)
    return True


def read_toml_file(path: Path) -> dict[str, Any]:
    """Parse *path* as TOML, or return an empty dict if it does not exist.

    Wrap a parse error in ``ConfigError`` (matching ``config_layer._read_toml``)
    so the ``config ... --machine-file FILE`` commands surface a clean message
    instead of letting a raw ``TOMLDecodeError`` traceback escape -- and, for
    ``set``/``add``, so the malformed file is reported before it is rewritten.
    """
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc


def read_toml_leaf(data: dict[str, Any], dotted_key: str) -> object:
    """Walk *data* by the dotted key, returning the value or None if absent."""
    cur: object = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
