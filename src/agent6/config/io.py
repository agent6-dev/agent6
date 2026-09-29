# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Comment-preserving TOML line surgery for the config writers.

The `config` CLI, `config.write` and through it the TUI and web editors all write through
here, so every writer preserves comments and sibling keys identically.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent6.config.model import ConfigError
from agent6.errors import read_operator_file
from agent6.portable import atomic_write, locked_file, toml_basic_string


def _header_name(line: str) -> str | None:
    """Return the table name of a `[table]` header line, or None.

    The one owner of header matching: a trailing comment and interior whitespace are
    tolerated, an array-of-tables `[[x]]` is not a match.

    Args:
        line: One line of the file.

    Returns:
        The name inside the brackets, stripped; None when the line is not a header.
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return None
    end = stripped.find("]")
    if end == -1:
        return None
    trailing = stripped[end + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        return None
    return stripped[1:end].strip()


def _section_name(line: str) -> str | None:
    """Return the dotted name of a `[table]` or `[[array.of.tables]]` header line, or None.

    For dropping a whole section: a `[[table.sub]]` goes with its dropped parent, so both
    forms match here where `_header_name` rejects the second.

    Args:
        line: One line of the file.

    Returns:
        The name, stripped; None when the line is not a header of either form.
    """
    stripped = line.strip()
    if stripped.startswith("[["):
        close = stripped.find("]]")
        if close == -1:
            return None
        trailing = stripped[close + 2 :].strip()
        if trailing and not trailing.startswith("#"):
            return None
        return stripped[2:close].strip()
    return _header_name(line)


def _toml_value(value: str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return toml_basic_string(value)


def upsert_toml_table(path: Path, table: str, fields: dict[str, ConfigLeafValue]) -> None:
    """Insert or replace one `[table]` block, preserving the rest of the file.

    Only the table's span is rewritten, never a serializer round-trip. The read, surgery
    and publish run under `locked_file`, as in every writer here: two concurrent writers
    would otherwise read the same base text and the second publish drop the first's update.

    Args:
        path: The config file, created when absent.
        table: The table name.
        fields: The leaves to write; a None value is omitted.

    Raises:
        ConfigError: A value has no TOML form.
    """
    block_lines = [f"[{table}]"]
    for key, val in fields.items():
        if val is None:
            continue
        block_lines.append(f"{key} = {format_toml_value(val)}")
    block = "\n".join(block_lines)

    with locked_file(path):
        text = read_operator_file(path) if path.is_file() else ""
        lines = text.splitlines()
        start = _header_line(lines, table)
        if start is None:
            prefix = text if not text or text.endswith("\n") else text + "\n"
            sep = "\n" if prefix and not prefix.endswith("\n\n") else ""
            atomic_write(path, prefix + sep + block + "\n")
            return
        end = _region_end(lines, start + 1)
        new_lines = lines[:start] + block.splitlines() + [""] + lines[end:]
        atomic_write(path, "\n".join(new_lines).rstrip("\n") + "\n")


# Matches what `format_toml_value` serializes; None omits the leaf.
ConfigLeafValue = str | bool | int | float | Sequence[str] | None


def format_toml_value(value: object) -> str:  # noqa: PLR0911
    """Serialize a scalar, a list or an inline-table dict to its TOML literal.

    Args:
        value: The value.

    Returns:
        The TOML text.

    Raises:
        ConfigError: The value has no TOML form (a CLI value parsed as a TOML date lands
            here, so the refusal carries to the boundary rather than the crash reporter).
    """
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
        # One line, so the leaf surgery replaces it whole; a nested `[table]` would collide.
        if not value:
            return "{}"
        items = ", ".join(f"{toml_key(k)} = {format_toml_value(v)}" for k, v in value.items())
        return "{ " + items + " }"
    raise ConfigError(f"cannot serialize {value!r} to TOML")


def toml_key(key: object) -> str:
    """Return the key as TOML: bare when it is a simple identifier, else quoted.

    Args:
        key: The key.

    Returns:
        The TOML key text.
    """
    k = str(key)
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _toml_value(k)


def parse_cli_value(value: str) -> object:
    """Interpret a CLI value the way TOML would.

    `true`, numbers, quoted text and bracketed arrays parse as TOML; anything else (a bare
    enum value, a model id) is the string itself.

    Args:
        value: The text as typed.

    Returns:
        The parsed value, or the text verbatim.
    """
    try:
        return tomllib.loads(f"_v = {value}")["_v"]
    except tomllib.TOMLDecodeError:
        return value


def _split_dotted_key(dotted_key: str) -> tuple[str, str]:
    """Split a dotted key into its table and leaf.

    Args:
        dotted_key: The key, `sandbox.network`.

    Returns:
        `(table, leaf)`; a single-segment key gives table `""`, the bare top region.

    Raises:
        ConfigError: A segment is empty.
    """
    parts = dotted_key.split(".")
    if any(not p for p in parts):
        raise ConfigError(
            f"config key must be a dotted leaf path like 'sandbox.network', got {dotted_key!r}"
        )
    return ".".join(parts[:-1]), parts[-1]


def upsert_toml_leaf(path: Path, dotted_key: str, value: object) -> None:
    """Set one `table.leaf` key, preserving the rest of the file verbatim.

    The `[table]` block is created when absent. TOML forbids a bare top-level key and a
    same-named `[table]` coexisting, so a write replaces the conflicting other shape.

    Args:
        path: The config file, created when absent.
        dotted_key: The key.
        value: The value, serialized by `format_toml_value`.

    Raises:
        ConfigError: The key is malformed, its ancestor is not a plain `[table]`, or the
            value has no TOML form.
    """
    table, leaf = _split_dotted_key(dotted_key)
    new_line = f"{leaf} = {format_toml_value(value)}"
    with locked_file(path):
        text = read_operator_file(path) if path.is_file() else ""
        lines = text.splitlines()
        if table:
            # Raised here so every writer hits it; the header the surgery would emit collides
            # with the headerless ancestor, and _drop_top_region_key would then delete it whole.
            if owner := undeclared_table_ancestor(path, dotted_key):
                raise ConfigError(
                    f"{dotted_key} lives inside {owner}, which is not a plain [table]"
                    " (a value, an inline table, a dotted key, or an array-of-tables),"
                    f" so it cannot be set on its own. Set {owner} as a whole, or edit"
                    f" {path} by hand."
                )
            lines = _drop_top_region_key(lines, table.split(".", 1)[0])
            # An existing `[table.leaf]` block plus the inline value declares the key twice.
            lines, _ = _drop_table_lines(lines, dotted_key)
            start = _header_line(lines, table)
            if start is None:
                text = "\n".join(lines) + "\n" if lines else ""
                sep = "\n" if text and not text.endswith("\n\n") else ""
                atomic_write(path, text + sep + f"[{table}]" + "\n" + new_line + "\n")
                return
            region = start + 1
        else:
            lines, _ = _drop_table_lines(lines, leaf)
            region = 0  # top-level key: the bare region before any [table] header
        end = _region_end(lines, region)
        j = _find_leaf_line(lines, region, end, leaf)
        if j is not None:
            # The whole span: rewriting only a multi-line value's opening line orphans the rest.
            span = _value_line_span(lines, j)
            replacement = new_line
            if span == 1 and (comment := _line_comment(lines[j])):
                replacement = f"{new_line}  {comment}"
            lines[j : j + span] = [replacement]
            atomic_write(path, "\n".join(lines).rstrip("\n") + "\n")
            return
        insert_at = end
        while insert_at - 1 >= region and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        # A top-level key flush against the first [table] header reads as that table's member.
        flush_against_header = insert_at < len(lines) and lines[insert_at].lstrip().startswith("[")
        gap = [""] if not table and flush_against_header else []
        lines[insert_at:insert_at] = [new_line, *gap]
        atomic_write(path, "\n".join(lines).rstrip("\n") + "\n")


def _drop_table_lines(lines: list[str], table: str) -> tuple[list[str], bool]:
    """Drop the `[table]` section with its `[table.sub]` subtables.

    Args:
        lines: The file's lines.
        table: The table name.

    Returns:
        The remaining lines, and whether anything was dropped.
    """
    kept: list[str] = []
    dropping = False
    removed = False
    j = 0
    while j < len(lines):
        line = lines[j]
        if line.strip().startswith("["):
            # _section_name, so a `[[table.sub]]` is dropped with its parent.
            name = _section_name(line)
            dropping = name is not None and (name == table or name.startswith(f"{table}."))
            removed = removed or dropping
            span = 1
        else:
            # A multi-line value is jumped whole, or an interior `[` line flips `dropping`.
            span = _value_line_span(lines, j) if _ASSIGN_RE.match(line) else 1
        if not dropping:
            kept.extend(lines[j : j + span])
        j += span
    return kept, removed


# A line that opens a `leaf = value` assignment; _value_line_span measures the value's lines.
_ASSIGN_RE = re.compile(r"^\s*[^#\s=\[][^=]*=")


def _region_end(lines: list[str], region: int) -> int:
    """Return the index of the first header line at or after the region start.

    The one owner of where a table's body ends. Every multi-line value is jumped whole: a
    triple-quoted value with a line starting `[` would otherwise end the region early and
    land an insert inside the operator's string.

    Args:
        lines: The file's lines.
        region: The index the body starts at.

    Returns:
        The header's index, or the line count when no header follows.
    """
    j = region
    while j < len(lines):
        if lines[j].lstrip().startswith("["):
            return j
        # _value_line_span is >= 1, so j advances.
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return len(lines)


def _find_leaf_line(lines: list[str], region: int, end: int, leaf: str) -> int | None:
    """Return the index of the line assigning the leaf within `[region, end)`, or None.

    The quoted spelling (`"protect_git" = true`) names the same leaf and matches too;
    unmatched, the surgery would append a duplicate key. Multi-line value interiors are
    skipped as in `_region_end`.

    Args:
        lines: The file's lines.
        region: The first index to scan.
        end: The index to stop before.
        leaf: The leaf name.

    Returns:
        The line's index, or None.
    """
    leaf_re = re.compile(rf"^\s*(\"|')?{re.escape(leaf)}(\"|')?\s*=")
    j = region
    while j < end:
        if leaf_re.match(lines[j]):
            return j
        # _value_line_span is >= 1, so this always advances.
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return None


def _iter_headers(lines: list[str]) -> list[tuple[int, str]]:
    """Return `(index, name)` for each `[table]` header.

    The one owner every header lookup uses; multi-line value interiors are skipped so a
    header-looking line inside a string is never taken for one.

    Args:
        lines: The file's lines.

    Returns:
        The headers in file order.
    """
    out: list[tuple[int, str]] = []
    j = 0
    while j < len(lines):
        name = _header_name(lines[j])
        if name is not None:
            out.append((j, name))
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return out


def _header_line(lines: list[str], table: str) -> int | None:
    """Return the index of the `[table]` header line, or None.

    Args:
        lines: The file's lines.
        table: The table name.

    Returns:
        The index, or None when the table has no header.
    """
    return next((i for i, name in _iter_headers(lines) if name == table), None)


def _drop_top_region_key(lines: list[str], key: str) -> list[str]:
    """Drop a bare top-level `key = ...` with its whole value.

    A same-named key inside a table is someone else's and stays.

    Args:
        lines: The file's lines.
        key: The top-level key.

    Returns:
        The remaining lines.
    """
    end = _region_end(lines, 0)
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    j = 0
    while j < end:
        if key_re.match(lines[j]):
            return lines[:j] + lines[j + _value_line_span(lines, j) :]
        # A `key = ...`-looking line inside an earlier triple-quoted value is skipped.
        j += _value_line_span(lines, j) if _ASSIGN_RE.match(lines[j]) else 1
    return lines


def _scan_toml_line(text: str, depth: int, triple: str | None) -> tuple[int, str | None]:
    """Advance the bracket-depth and open-triple-quote state across one line.

    Brackets and quotes inside a string, and everything after a `#` comment, do not count.

    Args:
        text: The line.
        depth: The bracket depth before the line.
        triple: The open triple-quote delimiter before the line, or None.

    Returns:
        The depth and open delimiter after the line.
    """
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


def _line_comment(line: str) -> str:
    """Return the trailing `# comment` of one TOML line, or "".

    Args:
        line: The line.

    Returns:
        The comment from its `#`, right-stripped; "" when the line has none (a `#` inside a
        string is not a comment).
    """
    i, n, triple = 0, len(line), None
    while i < n:
        if triple is not None:
            triple, i = (None, i + 3) if line.startswith(triple, i) else (triple, i + 1)
            continue
        if line.startswith('"""', i) or line.startswith("'''", i):
            triple, i = line[i : i + 3], i + 3
            continue
        ch = line[i]
        if ch in ('"', "'"):
            i += 1
            while i < n and line[i] != ch:
                i += 2 if (ch == '"' and line[i] == "\\") else 1
            i += 1
            continue
        if ch == "#":
            return line[i:].rstrip()
        i += 1
    return ""


def _value_line_span(lines: list[str], start: int) -> int:
    """Return how many lines the value assigned on the start line spans.

    Args:
        lines: The file's lines.
        start: The index of the assignment's opening line.

    Returns:
        The span, at least 1; an unterminated value spans to the end of the file.
    """
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
    """Delete one `table.leaf` assignment.

    Removing a section's last leaf drops the empty `[table]` header too; a section that
    still holds comments is kept, they are the operator's.

    Args:
        path: The config file.
        dotted_key: The key.

    Returns:
        True when a line was removed.

    Raises:
        ConfigError: The key is malformed, or its ancestor is not a plain `[table]`.
    """
    table, leaf = _split_dotted_key(dotted_key)
    with locked_file(path):
        if not path.is_file():
            return False
        # Without this a leaf inside an inline table reads "not found" while `config get` shows it.
        if table and (owner := undeclared_table_ancestor(path, dotted_key)):
            raise ConfigError(
                f"{dotted_key} lives inside {owner}, which is not a plain [table]"
                " (an inline table, a dotted key, or an array-of-tables), so it"
                f" cannot be unset on its own. Set {owner} as a whole, or edit {path}"
                " by hand."
            )
        lines = read_operator_file(path).splitlines()
        if table:
            start = _header_line(lines, table)
            if start is None:
                return False
            region = start + 1
        else:
            start = None  # top-level key: no header line to clean up after
            region = 0
        end = _region_end(lines, region)
        j = _find_leaf_line(lines, region, end, leaf)
        if j is not None:
            span = _value_line_span(lines, j)
            del lines[j : j + span]
            if start is not None:
                remaining_end = end - span  # next section header shifted up by span
                if all(not rest.strip() for rest in lines[start + 1 : remaining_end]):
                    del lines[start:remaining_end]
            out = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
            atomic_write(path, out)
            return True
        return False


def remove_toml_table(path: Path, table: str) -> bool:
    """Delete a whole `[table]` section with its `[table.sub]` subtables.

    `config fix` drops an unknown top-level table this way, where deleting one leaf would
    leave an empty but still invalid table behind.

    Args:
        path: The config file.
        table: The table name.

    Returns:
        True when the table was present.
    """
    with locked_file(path):
        if not path.is_file():
            return False
        lines = read_operator_file(path).splitlines()
        kept, removed = _drop_table_lines(lines, table)
        if not removed:
            return False
        out = "\n".join(kept).rstrip("\n") + "\n" if any(ln.strip() for ln in kept) else ""
        atomic_write(path, out)
        return True


def read_toml_file(path: Path) -> dict[str, Any]:
    """Parse a TOML file.

    Args:
        path: The file.

    Returns:
        The parsed table; an empty dict when the file does not exist.

    Raises:
        ConfigError: The file is not valid TOML or cannot be read, so `set` and `add` report
            a malformed file before rewriting it.
    """
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot be read: {exc}") from exc


def undeclared_table_ancestor(path: Path, dotted_key: str) -> str | None:
    """Return the outermost ancestor of the key that the leaf surgery cannot write under.

    Such an ancestor is a plain value, an inline table, a dotted key or an array-of-tables:
    the surgery knows only `[table]` headers, and the one it would emit declares the
    ancestor twice.

    Args:
        path: The config file.
        dotted_key: The key.

    Returns:
        The ancestor's dotted name, or None when every ancestor is a `[table]` or absent.
    """
    if not path.is_file():
        return None
    text = read_operator_file(path)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None  # a file that does not parse is refused earlier, with its own message
    headers = [name for _, name in _iter_headers(text.splitlines())]
    parts = dotted_key.split(".")
    for i in range(1, len(parts)):  # proper ancestors, outermost first
        prefix = ".".join(parts[:i])
        val = read_toml_leaf(data, prefix)
        if val is None:
            continue  # absent: the surgery declares the [table] itself
        if isinstance(val, list):
            return prefix  # an array-of-tables: a leaf can't be set on it
        if not isinstance(val, dict):
            # A scalar where a table belongs; `_drop_top_region_key` replaces a bare top-level one.
            if "." in prefix:
                return prefix
            continue
        if any(h == prefix or h.startswith(f"{prefix}.") for h in headers):
            continue  # a real [table] header declares it
        return prefix
    return None


def read_toml_leaf(data: dict[str, Any], dotted_key: str) -> object:
    """Walk the parsed data by a dotted key.

    Args:
        data: The parsed TOML.
        dotted_key: The key.

    Returns:
        The value, or None when any segment is absent.
    """
    cur: object = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
