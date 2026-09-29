# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tree-sitter navigation handlers: outline, find_definition and find_references.

The lazily built `SymbolIndex` lives on `ToolDispatcher`, shared with the edit tools' change
notification; each handler takes the dispatcher's ensure callable and invokes it only after the
arguments and path validate, so a rejected call never scans the index.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent6.tools._path_safety import Workspace
from agent6.tools.errors import ToolError
from agent6.tools.index import SymbolIndex
from agent6.tools.results import DefinitionsResult, OutlineResult, ReferencesResult
from agent6.tools.schema import (
    FindDefinitionInput,
    FindReferencesInput,
    OutlineInput,
)

INDEX_RESULT_CAP = 500


def outline(
    ws: Workspace, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> OutlineResult:
    """List the symbols of one file, capped at `INDEX_RESULT_CAP`.

    Args:
        ws: The workspace the path resolves in.
        ensure_index: Builds or returns the symbol index.
        raw: The tool call's arguments.

    Returns:
        The file's symbols in order, and whether the list was cut.

    Raises:
        ToolError: The path is not a file, has no parser, or lies outside the indexed workspace.
    """
    args = OutlineInput.model_validate(raw)
    sp = ws.resolve_read(args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    index = ensure_index()
    # An empty outline over a file full of symbols misleads the model: name the cause instead.
    if index.language_of(sp.abs_path) is None:
        raise ToolError(f"outline: no parser for {sp.abs_path.suffix or 'a file without a suffix'}")
    if not index.indexes(sp.abs_path):
        raise ToolError(f"outline: {args.path} is outside the indexed workspace")
    syms = index.outline(sp.abs_path)
    out = [{"name": s.name, "kind": s.kind, "line": s.line, "col": s.col} for s in syms]
    truncated = len(out) > INDEX_RESULT_CAP
    return OutlineResult(symbols=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_definition(
    ws: Workspace, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> DefinitionsResult:
    """Find where a symbol is defined, capped at `INDEX_RESULT_CAP`.

    Args:
        ws: The workspace paths are reported relative to.
        ensure_index: Builds or returns the symbol index.
        raw: The tool call's arguments.

    Returns:
        The definitions inside the workspace, and whether the list was cut.
    """
    args = FindDefinitionInput.model_validate(raw)
    defs = ensure_index().find_definition(args.symbol)
    out: list[dict[str, Any]] = []
    for s in defs:
        try:
            rel = s.path.relative_to(ws.root)
        except ValueError:
            continue
        out.append({"name": s.name, "kind": s.kind, "path": str(rel), "line": s.line, "col": s.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return DefinitionsResult(definitions=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_references(
    ws: Workspace, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> ReferencesResult:
    """Find where a symbol is referenced, capped at `INDEX_RESULT_CAP`.

    Args:
        ws: The workspace paths are reported relative to.
        ensure_index: Builds or returns the symbol index.
        raw: The tool call's arguments.

    Returns:
        The references inside the workspace, and whether the list was cut.
    """
    args = FindReferencesInput.model_validate(raw)
    refs = ensure_index().find_references(args.symbol)
    out: list[dict[str, Any]] = []
    for r in refs:
        try:
            rel = r.path.relative_to(ws.root)
        except ValueError:
            continue
        out.append({"name": r.name, "path": str(rel), "line": r.line, "col": r.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return ReferencesResult(references=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)
