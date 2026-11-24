# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tree-sitter navigation handlers: outline, find_definition and find_references.

The lazily built `SymbolIndex` lives on `ToolDispatcher`, shared with the edit tools' change
notification; handlers request it only after the arguments and path validate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent6.tools import fs_tools, results, schema, tool_errors

if TYPE_CHECKING:
    from agent6.tools import dispatch

INDEX_RESULT_CAP = 500


def outline(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.OutlineResult:
    """List the symbols of one file, capped at `INDEX_RESULT_CAP`.

    Args:
        ctx: The workspace and its lazy symbol index.
        raw: The tool call's arguments.

    Returns:
        The file's symbols in order, and whether the list was cut.

    Raises:
        ToolError: The path is not a file, has no parser, or lies outside the indexed workspace.
    """
    args = schema.OutlineInput.model_validate(raw)
    sp = fs_tools.resolve_read_path(ctx.workspace, args.path, "file")
    index = ctx.symbol_index()
    # An empty outline over a file full of symbols misleads the model: name the cause instead.
    if index.language_of(sp.abs_path) is None:
        raise tool_errors.ToolError(
            f"outline: no parser for {sp.abs_path.suffix or 'a file without a suffix'}"
        )
    if not index.indexes(sp.abs_path):
        raise tool_errors.ToolError(f"outline: {args.path} is outside the indexed workspace")
    syms = index.outline(sp.abs_path)
    out = [{"name": s.name, "kind": s.kind, "line": s.line, "col": s.col} for s in syms]
    truncated = len(out) > INDEX_RESULT_CAP
    return results.OutlineResult(symbols=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_definition(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.DefinitionsResult:
    """Find where a symbol is defined, capped at `INDEX_RESULT_CAP`.

    Args:
        ctx: The workspace and its lazy symbol index.
        raw: The tool call's arguments.

    Returns:
        The definitions inside the workspace, and whether the list was cut.
    """
    args = schema.FindDefinitionInput.model_validate(raw)
    defs = ctx.symbol_index().find_definition(args.symbol)
    out: list[dict[str, Any]] = []
    for s in defs:
        try:
            rel = s.path.relative_to(ctx.workspace.root)
        except ValueError:
            continue
        out.append({"name": s.name, "kind": s.kind, "path": str(rel), "line": s.line, "col": s.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return results.DefinitionsResult(definitions=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_references(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.ReferencesResult:
    """Find where a symbol is referenced, capped at `INDEX_RESULT_CAP`.

    Args:
        ctx: The workspace and its lazy symbol index.
        raw: The tool call's arguments.

    Returns:
        The references inside the workspace, and whether the list was cut.
    """
    args = schema.FindReferencesInput.model_validate(raw)
    refs = ctx.symbol_index().find_references(args.symbol)
    out: list[dict[str, Any]] = []
    for r in refs:
        try:
            rel = r.path.relative_to(ctx.workspace.root)
        except ValueError:
            continue
        out.append({"name": r.name, "path": str(rel), "line": r.line, "col": r.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return results.ReferencesResult(references=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)
