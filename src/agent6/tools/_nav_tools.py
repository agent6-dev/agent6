# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tree-sitter / LSP navigation handlers: outline, find_definition,
find_references, find_definition_lsp, find_references_lsp.

The lazy-built ``SymbolIndex`` / ``LspClient`` singletons stay on
``ToolDispatcher`` (shared with the non-tool passthroughs ``hot_symbols`` /
``file_outlines`` and with ``apply_edit``/``apply_patch``'s change
notification); these functions take the dispatcher's ensure callable and
invoke it only after argument/path validation, exactly where the original
handlers did (an LSP spawn or index scan never happens for a rejected call).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent6.tools._path_safety import resolve_in_root
from agent6.tools.errors import ToolError
from agent6.tools.index import SymbolIndex
from agent6.tools.lsp import LspClient, LspError
from agent6.tools.results import DefinitionsResult, OutlineResult, ReferencesResult
from agent6.tools.schema import (
    FindDefinitionInput,
    FindDefinitionLspInput,
    FindReferencesInput,
    FindReferencesLspInput,
    OutlineInput,
)

INDEX_RESULT_CAP = 500


def outline(
    root: Path, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> OutlineResult:
    args = OutlineInput.model_validate(raw)
    sp = resolve_in_root(root, args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    syms = ensure_index().outline(sp.abs_path)
    out = [{"name": s.name, "kind": s.kind, "line": s.line, "col": s.col} for s in syms]
    truncated = len(out) > INDEX_RESULT_CAP
    return OutlineResult(symbols=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_definition(
    root: Path, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> DefinitionsResult:
    args = FindDefinitionInput.model_validate(raw)
    defs = ensure_index().find_definition(args.name)
    out: list[dict[str, Any]] = []
    for s in defs:
        try:
            rel = s.path.relative_to(root)
        except ValueError:
            continue
        out.append({"name": s.name, "kind": s.kind, "path": str(rel), "line": s.line, "col": s.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return DefinitionsResult(definitions=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_references(
    root: Path, ensure_index: Callable[[], SymbolIndex], raw: dict[str, Any]
) -> ReferencesResult:
    args = FindReferencesInput.model_validate(raw)
    refs = ensure_index().find_references(args.name)
    out: list[dict[str, Any]] = []
    for r in refs:
        try:
            rel = r.path.relative_to(root)
        except ValueError:
            continue
        out.append({"name": r.name, "path": str(rel), "line": r.line, "col": r.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return ReferencesResult(references=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_definition_lsp(
    root: Path, ensure_lsp: Callable[[], LspClient], raw: dict[str, Any]
) -> DefinitionsResult:
    args = FindDefinitionLspInput.model_validate(raw)
    sp = resolve_in_root(root, args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    try:
        locs = ensure_lsp().find_definition(sp.abs_path, args.symbol)
    except LspError as exc:
        raise ToolError(str(exc)) from exc
    out: list[dict[str, Any]] = []
    for loc in locs:
        try:
            rel = loc.path.resolve().relative_to(root)
        except ValueError:
            continue
        out.append({"path": str(rel), "line": loc.line, "col": loc.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return DefinitionsResult(definitions=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)


def find_references_lsp(
    root: Path, ensure_lsp: Callable[[], LspClient], raw: dict[str, Any]
) -> ReferencesResult:
    args = FindReferencesLspInput.model_validate(raw)
    sp = resolve_in_root(root, args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    try:
        locs = ensure_lsp().find_references(sp.abs_path, args.symbol)
    except LspError as exc:
        raise ToolError(str(exc)) from exc
    out: list[dict[str, Any]] = []
    for loc in locs:
        try:
            rel = loc.path.resolve().relative_to(root)
        except ValueError:
            continue
        out.append({"path": str(rel), "line": loc.line, "col": loc.col})
    truncated = len(out) > INDEX_RESULT_CAP
    return ReferencesResult(references=tuple(out[:INDEX_RESULT_CAP]), truncated=truncated)
