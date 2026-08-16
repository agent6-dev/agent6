# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only checker for the code-fixer machine.

Imports the ``stats`` module and checks ``median`` against a few cases. Prints
one JSON line ``{"passed": bool, "summary": str}`` and always exits 0, so the
machine's tool state routes on the captured ``passed`` flag, not on the process
exit code.

``--ref REF`` reads ``stats.py`` from a git ref instead of the working tree: a
``mode="run"`` agent state works a clone and lands its commits on
``agent6/machine-<id>``, leaving the checkout untouched, so the machine's check
state reads that branch. With no ``--ref`` it checks the working tree, which is
what the agent itself runs inside its clone.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CASES: list[tuple[list[float], float]] = [
    ([3, 1, 2], 2.0),
    ([1, 2, 3, 4], 2.5),
    ([5], 5.0),
    ([4, 1], 2.5),
    ([10, 2, 8, 4, 6], 6.0),
]


def _emit(passed: bool, summary: str) -> int:
    print(json.dumps({"passed": passed, "summary": summary}))
    return 0


def _stage_ref(ref: str) -> str:
    """Stage ``stats.py`` as it exists on *ref* and put it on `sys.path`.

    Returns "" on success, else the reason. It stages into
    `$AGENT6_MACHINE_DATA_DIR`, granted read-write in every machine tool jail,
    so this works at every isolation level.
    """
    proc = subprocess.run(
        ["git", "show", f"{ref}:stats.py"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return f"could not read stats.py from {ref}: {proc.stderr.strip() or proc.returncode}"
    base = Path(os.environ.get("AGENT6_MACHINE_DATA_DIR") or tempfile.mkdtemp()) / "ref-stats"
    base.mkdir(parents=True, exist_ok=True)
    (base / "stats.py").write_text(proc.stdout, encoding="utf-8")
    # Successive iterations rewrite that file: a cached .pyc could shadow it.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(base))
    return ""


def _median_under_check(argv: list[str]) -> tuple[Any, str]:
    """The `median` to check, or (None, reason)."""
    if argv[:1] == ["--ref"] and len(argv) == 2:
        staged = _stage_ref(argv[1])
        if staged:
            return None, staged
    elif argv:
        return None, f"usage: verify.py [--ref REF]; got {argv}"
    else:
        sys.path.insert(0, str(Path.cwd()))
    try:
        module = importlib.import_module("stats")
    except Exception as exc:  # noqa: BLE001 -- the code under check may raise anything
        return None, f"could not import stats: {exc}"
    median: Any = getattr(module, "median", None)
    if not callable(median):
        return None, "stats.median is missing or not callable"
    return median, ""


def main() -> int:
    median, reason = _median_under_check(sys.argv[1:])
    if reason:
        return _emit(False, reason)
    for xs, want in CASES:
        try:
            got = median(list(xs))
        except Exception as exc:  # noqa: BLE001 -- the code under check may raise anything
            return _emit(False, f"median({xs}) raised {exc!r}")
        if got != want:
            return _emit(False, f"median({xs}) = {got!r}, want {want!r}")
    return _emit(True, f"all {len(CASES)} median cases pass")


if __name__ == "__main__":
    raise SystemExit(main())
