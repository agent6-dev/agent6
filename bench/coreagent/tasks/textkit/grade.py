#!/usr/bin/env python3
"""Authoritative hidden grader for textkit. Not shipped into the agent's repo.

Usage: python3 grade.py <worktree-dir>
Imports <worktree-dir>/textkit.py, runs a thorough per-component battery, and
prints one JSON line: cases_passed/cases_total (fine score) and per-component
pass (coarse "did it forget a component" signal). A component counts as passed
only if every one of its cases passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any


def _load(worktree: str) -> Any:
    spec = importlib.util.spec_from_file_location("textkit", f"{worktree}/textkit.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load textkit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case(fn: Callable[..., Any], args: tuple[Any, ...], expected: Any) -> bool:
    try:
        return fn(*args) == expected
    except Exception:
        return False


def grade(worktree: str) -> dict[str, Any]:
    try:
        m = _load(worktree)
    except Exception as exc:
        return {
            "task": "textkit",
            "import_error": str(exc)[:200],
            "cases_passed": 0,
            "cases_total": 1,
            "score": 0.0,
            "components": {},
            "components_passed": 0,
            "components_total": 5,
        }

    components: dict[str, list[tuple[tuple[Any, ...], Any]]] = {
        "normalize_whitespace": [
            (("  a\t b\n\nc  ",), "a b c"),
            (("   \n\t ",), ""),
            (("",), ""),
            (("single",), "single"),
            (("a  b   c",), "a b c"),
        ],
        "word_count": [
            (("  one  two\tthree ",), 3),
            (("  \n ",), 0),
            (("",), 0),
            (("a",), 1),
            (("a b c d e",), 5),
        ],
        "most_common_words": [
            (("the cat, the dog. The bird?", 2), [("the", 3), ("bird", 1)]),
            (("a a b", 0), []),
            (("a a b", -1), []),
            (("Apple apple BANANA banana banana", 2), [("banana", 3), ("apple", 2)]),
            # tie broken alphabetically ascending
            (("zebra apple mango", 2), [("apple", 1), ("mango", 1)]),
            (("...!", 3), []),  # all punctuation -> empty tokens dropped
            (("one two", 5), [("one", 1), ("two", 1)]),  # fewer than n
        ],
        "wrap_text": [
            (("the quick brown fox", 9), ["the quick", "brown fox"]),
            (("   ", 5), []),
            (("", 5), []),
            # word longer than width gets its own line
            (("a supercalifragilistic b", 5), ["a", "supercalifragilistic", "b"]),
            (("aa bb cc", 2), ["aa", "bb", "cc"]),
            (("one two three", 100), ["one two three"]),
        ],
        "to_snake_case": [
            (("fooBarBaz",), "foo_bar_baz"),
            (("HTTPServer",), "http_server"),
            (("getHTTPResponseCode",), "get_http_response_code"),
            (("foo-bar baz",), "foo_bar_baz"),
            (("already_snake",), "already_snake"),
            (("PascalCase",), "pascal_case"),
            (("simple",), "simple"),
        ],
    }

    results: dict[str, dict[str, int]] = {}
    cases_passed = 0
    cases_total = 0
    components_passed = 0
    for name, cases in components.items():
        fn = getattr(m, name, None)
        p = 0
        for args, expected in cases:
            cases_total += 1
            if callable(fn) and _case(fn, args, expected):
                p += 1
                cases_passed += 1
        results[name] = {"passed": p, "total": len(cases)}
        if p == len(cases):
            components_passed += 1

    return {
        "task": "textkit",
        "cases_passed": cases_passed,
        "cases_total": cases_total,
        "score": round(cases_passed / cases_total, 4) if cases_total else 0.0,
        "components": results,
        "components_passed": components_passed,
        "components_total": len(components),
    }


if __name__ == "__main__":
    wt = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(grade(wt)))
