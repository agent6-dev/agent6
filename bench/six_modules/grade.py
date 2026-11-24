"""Check the twelve extensions on a submission in the current directory."""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
from typing import Any

CASES = {
    "pairs": [("swap", ([7, 3],), (3, 7), ([],)), ("span", ([7, 3],), 4, ([1],))],
    "limits": [("clamp", (8, -1, 3), 3, (0, 3, 2)), ("contains", (3, -1, 3), True, (1, 5, 4))],
    "labels": [("slug", (" MiX\t Ed ",), "mix-ed", (" ",)), ("initials", ("a  bee",), "AB", ("",))],
    "batches": [
        ("groups", ([1, 2, 3], 2), [[1, 2], [3]], ([], 0)),
        ("needed", ([4, 5, 6], 2), 2, ([], -1)),
    ],
    "clockface": [("parts", (1502,), (25, 2), (-1,)), ("display", (1502,), "25:02", (-1,))],
    "rangeset": [("inclusive", (-1, 1), [-1, 0, 1], (1, 0)), ("width", (3, 3), 1, (4, 3))],
}


def check(
    module: str, name: str, args: tuple[Any, ...], expected: Any, invalid: tuple[Any, ...]
) -> bool:
    try:
        fn = getattr(importlib.import_module(module), name)
        if fn(*args) != expected:
            return False
        try:
            fn(*invalid)
        except ValueError:
            return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path.cwd()))
    print(
        json.dumps(
            {module: all(check(module, *case) for case in cases) for module, cases in CASES.items()}
        )
    )
