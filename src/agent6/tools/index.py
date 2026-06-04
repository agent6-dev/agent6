# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tree-sitter symbol index for the LLM-visible navigation tools.

Provides a `SymbolIndex` over a project root that supports three operations:

    outline(path)            -> top-level symbol declarations in one file
    find_definition(name)    -> every declaration of `name` across the project
    find_references(name)    -> every identifier occurrence of `name` (incl. def)

The index is built lazily on the first query, then updated incrementally when
the caller marks files changed via `mark_changed(path)` / `mark_deleted(path)`.
Re-parses happen in batch on the next query, so a worker can call `apply_edit`
many times and pay the parse cost only when it next asks for symbol info.

Language support is intentionally limited to a small audited set:

    .py            -> python
    .rs            -> rust
    .ts, .tsx      -> typescript

Other extensions are silently ignored. Adding a language is mechanical:
extend `_LANG_TABLE` with the tree-sitter language name and a definitions
query in the grammar's syntax. References use a generic identifier query
per language; cross-file reference *resolution* (which `foo` is the same
symbol?) is out of scope and requires a real LSP — for our purposes,
identifier-level grep filtered through tree-sitter is vastly better than
plain text grep because it never matches inside strings or comments.

This module is NOT exposed via `ToolDispatcher` here; that wiring lives
in `tools/dispatch.py` + `tools/schema.py` and requires a security review
note when added.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tree_sitter import Parser, Query, QueryCursor
from tree_sitter_language_pack import get_language


class IndexError(RuntimeError):
    """Raised on unrecoverable index errors. Currently unused — individual
    file failures are absorbed silently so one bad file does not poison
    the whole index."""


@dataclass(frozen=True, slots=True)
class Symbol:
    """A definition site. `path` is absolute; `line`/`col` are 0-indexed."""

    name: str
    kind: str  # 'function' | 'class' | 'method' | 'struct' | 'enum' | ...
    path: Path
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class Reference:
    """An identifier occurrence. `path` is absolute; `line`/`col` are 0-indexed.

    Includes the definition site itself. Callers that want call-sites-only
    should subtract the result of `find_definition(name)`.
    """

    name: str
    path: Path
    line: int
    col: int


# ---------------------------------------------------------------------------
# Per-language queries
# ---------------------------------------------------------------------------

_PYTHON_DEFS: Final = """
(function_definition name: (identifier) @function)
(class_definition name: (identifier) @class)
"""

_RUST_DEFS: Final = """
(function_item name: (identifier) @function)
(struct_item name: (type_identifier) @struct)
(enum_item name: (type_identifier) @enum)
(trait_item name: (type_identifier) @trait)
(type_item name: (type_identifier) @type)
(mod_item name: (identifier) @module)
(const_item name: (identifier) @const)
(static_item name: (identifier) @const)
(macro_definition name: (identifier) @macro)
"""

_TS_DEFS: Final = """
(function_declaration name: (identifier) @function)
(class_declaration name: (type_identifier) @class)
(method_definition name: (property_identifier) @method)
(interface_declaration name: (type_identifier) @interface)
(type_alias_declaration name: (type_identifier) @type)
(enum_declaration name: (identifier) @enum)
"""

# Per-language identifier query for references. Different grammars surface
# names under different node types (rust splits `identifier` vs
# `type_identifier`; ts adds `property_identifier`).
_REF_QUERIES: Final[dict[str, str]] = {
    "python": "(identifier) @id",
    "rust": "[(identifier) (type_identifier)] @id",
    "typescript": "[(identifier) (type_identifier) (property_identifier)] @id",
    "tsx": "[(identifier) (type_identifier) (property_identifier)] @id",
}

# suffix -> (tree-sitter language name, definitions query)
_LANG_TABLE: Final[dict[str, tuple[str, str]]] = {
    ".py": ("python", _PYTHON_DEFS),
    ".rs": ("rust", _RUST_DEFS),
    ".ts": ("typescript", _TS_DEFS),
    ".tsx": ("tsx", _TS_DEFS),
}

# Directories never indexed. Hard-coded; we are not parsing .gitignore here.
_DEFAULT_EXCLUDES: Final[tuple[str, ...]] = (
    ".git",
    ".agent6",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


class SymbolIndex:
    """Lazy, incrementally-updated tree-sitter symbol index for a project root."""

    def __init__(
        self,
        root: Path,
        *,
        excludes: tuple[str, ...] = _DEFAULT_EXCLUDES,
    ) -> None:
        self._root = root.resolve()
        self._excludes = excludes
        # path -> per-file caches. Absolute, resolved paths.
        self._symbols: dict[Path, list[Symbol]] = {}
        self._refs: dict[Path, list[Reference]] = {}
        self._scanned = False
        self._dirty: set[Path] = set()
        # lang_name -> (parser, def_query, ref_query). Built on first use.
        self._parsers: dict[str, tuple[Parser, Query, Query]] = {}

    # ------------------------------------------------------------------
    # Dirty-tracking surface for the dispatcher to call after apply_edit
    # ------------------------------------------------------------------

    def mark_changed(self, path: Path) -> None:
        """Record that `path` was created or modified; re-parsed on next query."""
        self._dirty.add(path.resolve())

    def mark_deleted(self, path: Path) -> None:
        """Drop a path from the index immediately."""
        p = path.resolve()
        self._symbols.pop(p, None)
        self._refs.pop(p, None)
        self._dirty.discard(p)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def outline(self, path: Path) -> list[Symbol]:
        """Top-level + nested definitions in one file, in source order."""
        self._ensure_fresh()
        p = path.resolve()
        if p not in self._symbols and p.is_file():
            # On-demand parse for a file we hadn't seen at scan time.
            self._reparse(p)
        out = list(self._symbols.get(p, []))
        out.sort(key=lambda s: (s.line, s.col))
        return out

    def find_definition(self, name: str) -> list[Symbol]:
        """All definition sites of `name` across the project, in path order."""
        self._ensure_fresh()
        out: list[Symbol] = []
        for syms in self._symbols.values():
            for s in syms:
                if s.name == name:
                    out.append(s)
        out.sort(key=lambda s: (str(s.path), s.line, s.col))
        return out

    def find_references(self, name: str) -> list[Reference]:
        """All identifier occurrences of `name` (incl. defs), in path order."""
        self._ensure_fresh()
        out: list[Reference] = []
        for refs in self._refs.values():
            for r in refs:
                if r.name == name:
                    out.append(r)
        out.sort(key=lambda r: (str(r.path), r.line, r.col))
        return out

    def hot_symbols(
        self,
        *,
        max_symbols: int = 20,
        min_files_referenced: int = 2,
    ) -> list[tuple[str, str, str, int, int]]:
        """Top symbols by cross-file reference count.

        Returns a list of (name, kind, def_path, def_line, files_referenced)
        tuples, sorted by `files_referenced` descending. Only symbols whose
        identifier appears in at least `min_files_referenced` distinct
        files are included - this filters out file-local helpers and
        surfaces only symbols whose rename/signature change would touch
        multiple files. Definition site is taken as the first symbol's
        path:line; ambiguous names with multiple definitions return the
        alphabetically-first def.

        Cheap planner prior: knowing that "build_kernel" is referenced
        across 4 files lets the planner enumerate those files in
        relevant_paths up-front, the same payoff shape as 's
        co-change pairs but driven by static analysis instead of git
        history. Works on fresh repos (no history needed).
        """
        self._ensure_fresh()
        files_per_name: defaultdict[str, set[Path]] = defaultdict(set)
        ref_count: Counter[str] = Counter()
        for refs in self._refs.values():
            for r in refs:
                files_per_name[r.name].add(r.path)
                ref_count[r.name] += 1
        defs_by_name: dict[str, list[Symbol]] = defaultdict(list)
        for syms in self._symbols.values():
            for s in syms:
                defs_by_name[s.name].append(s)
        qualifying: list[tuple[str, str, str, int, int]] = []
        for name, files in files_per_name.items():
            n_files = len(files)
            if n_files < min_files_referenced:
                continue
            defs = defs_by_name.get(name) or []
            # Some references have no def in the index (e.g. stdlib /
            # third-party names); skip those - the planner can't action
            # them.
            if not defs:
                continue
            d = sorted(defs, key=lambda s: (str(s.path), s.line))[0]
            try:
                rel = d.path.resolve().relative_to(self._root.resolve())
                rel_str = str(rel)
            except ValueError:
                rel_str = str(d.path)
            qualifying.append((name, d.kind, rel_str, d.line, n_files))
        qualifying.sort(key=lambda t: (-t[4], t[0]))
        return qualifying[:max_symbols]

    def file_outlines(self) -> dict[Path, list[Symbol]]:
        """Per-file top-level symbol lists across the whole index.

        Returns a fresh dict mapping absolute file path -> in-source-order
        list of Symbol records. Used by the system-prompt repo map to
        give the agent a one-line-per-symbol outline of the codebase
        without round-tripping ``outline`` for every file.
        """
        self._ensure_fresh()
        out: dict[Path, list[Symbol]] = {}
        for path, syms in self._symbols.items():
            out[path] = sorted(syms, key=lambda s: (s.line, s.col))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fresh(self) -> None:
        if not self._scanned:
            self._scan_all()
            self._scanned = True
        if not self._dirty:
            return
        for p in list(self._dirty):
            self._reparse(p)
        self._dirty.clear()

    def _scan_all(self) -> None:
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if self._is_excluded(path):
                continue
            if self._lang_for(path) is None:
                continue
            self._reparse(path)

    def _reparse(self, path: Path) -> None:
        p = path.resolve()
        if not p.is_file() or self._is_excluded(p):
            self._symbols.pop(p, None)
            self._refs.pop(p, None)
            return
        lang_name = self._lang_for(p)
        if lang_name is None:
            return
        bits = self._parser_for(lang_name)
        if bits is None:
            return
        parser, def_query, ref_query = bits
        try:
            src = p.read_bytes()
        except OSError:
            return
        try:
            tree = parser.parse(src)
        except Exception:  # tree-sitter errors are opaque; absorb per-file failures
            return
        root = tree.root_node
        syms: list[Symbol] = []
        for kind, nodes in QueryCursor(def_query).captures(root).items():
            for n in nodes:
                try:
                    name = src[n.start_byte : n.end_byte].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                syms.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        path=p,
                        line=n.start_point[0],
                        col=n.start_point[1],
                    )
                )
        refs: list[Reference] = []
        for _, nodes in QueryCursor(ref_query).captures(root).items():
            for n in nodes:
                try:
                    name = src[n.start_byte : n.end_byte].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                refs.append(
                    Reference(
                        name=name,
                        path=p,
                        line=n.start_point[0],
                        col=n.start_point[1],
                    )
                )
        self._symbols[p] = syms
        self._refs[p] = refs

    def _is_excluded(self, p: Path) -> bool:
        # Compare against parts of the path relative to root so an excluded
        # dirname embedded in an outside ancestor doesn't matter.
        try:
            rel = p.relative_to(self._root)
        except ValueError:
            return True
        return any(part in self._excludes for part in rel.parts)

    def _lang_for(self, path: Path) -> str | None:
        info = _LANG_TABLE.get(path.suffix)
        return info[0] if info else None

    def _parser_for(self, lang_name: str) -> tuple[Parser, Query, Query] | None:
        cached = self._parsers.get(lang_name)
        if cached is not None:
            return cached
        # Find the def query for this language (linear scan; tiny table).
        def_src: str | None = None
        for _, (n, q) in _LANG_TABLE.items():
            if n == lang_name:
                def_src = q
                break
        if def_src is None:
            return None
        try:
            lang = get_language(lang_name)  # pyright: ignore[reportArgumentType]
        except Exception:  # unknown lang name -> skip
            return None
        parser = Parser(lang)
        def_query = Query(lang, def_src)
        ref_query = Query(lang, _REF_QUERIES.get(lang_name, "(identifier) @id"))
        self._parsers[lang_name] = (parser, def_query, ref_query)
        return self._parsers[lang_name]


__all__ = [
    "IndexError",
    "Reference",
    "Symbol",
    "SymbolIndex",
]
