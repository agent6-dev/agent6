#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Composite quality scorer for agent6 bench tasks.

Per-task Q in [0..1], a weighted sum of deterministic components:

  verify          1 if `python3 -m unittest -v` passes, else 0. Gate component.
  test_integrity  0 if the agent modified a frozen test file (per TASK.md
                  rule "Do not modify ..."), else 1.
  diff_size       penalises bloat. score = clamp((REF / max(REF, lines)) where
                  REF is a reference patch-size budget read from REFERENCE.md
                  or defaults to 30 lines. (Smaller patch -> closer to 1.)
  lint_clean     1 if `ruff check` on touched .py files emits zero diagnostics,
                  else 1 - min(1, n_diagnostics / 10).
  hidden_tests   if HIDDEN_TESTS/ exists in the task dir, runs that test
                  module against the worktree and scores pass_rate. Else 1.0.

Composite is gated: if verify=0, the report still emits the breakdown but
caps the composite at 0.30 (so a half-correct patch can't outscore a true
PASS).

Reads from <task_dir>/result.json (written by run_bench.sh) and writes
<task_dir>/quality.json. Prints a one-line summary on stdout.

Usage:
  python3 bench/quality.py <task_dir> [<task_dir> ...]
  python3 bench/quality.py --bench-root /tmp/agent6-bench
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QualityReport:
    task: str
    verify: float
    test_integrity: float
    diff_size: float
    lint_clean: float
    hidden_tests: float
    composite: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "components": {
                "verify": self.verify,
                "test_integrity": self.test_integrity,
                "diff_size": self.diff_size,
                "lint_clean": self.lint_clean,
                "hidden_tests": self.hidden_tests,
            },
            "composite": self.composite,
            "notes": list(self.notes),
        }


WEIGHTS = {
    "verify": 0.40,
    "test_integrity": 0.15,
    "diff_size": 0.10,
    "lint_clean": 0.15,
    "hidden_tests": 0.20,
}


def _git(task_dir: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(task_dir), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


def _touched_py_files(task_dir: Path) -> list[Path]:
    # Diff against the initial commit on master. Includes uncommitted
    # worktree changes (claude-code edits the tree but does not commit).
    txt = _git(task_dir, "diff", "--name-only", "master")
    return [task_dir / line.strip() for line in txt.splitlines() if line.strip().endswith(".py")]


def _diff_added_lines(task_dir: Path) -> int:
    txt = _git(task_dir, "diff", "--shortstat", "master")
    # e.g. " 2 files changed, 14 insertions(+), 3 deletions(-)"
    added = 0
    for token in txt.split(","):
        token = token.strip()
        if token.endswith("insertion(+)") or token.endswith("insertions(+)"):
            try:
                added = int(token.split()[0])
            except (ValueError, IndexError):
                pass
    return added


def _test_files(task_dir: Path) -> list[Path]:
    return [
        p
        for p in task_dir.iterdir()
        if p.is_file() and p.name.startswith("test_") and p.name.endswith(".py")
    ]


def _frozen_in_task(task_dir: Path) -> list[str]:
    task_md = task_dir / "TASK.md"
    if not task_md.is_file():
        return []
    text = task_md.read_text(encoding="utf-8", errors="replace")
    # Heuristic: any test_*.py mentioned alongside "Do not modify".
    frozen: list[str] = []
    for test_file in _test_files(task_dir):
        if test_file.name in text and "do not modify" in text.lower():
            frozen.append(test_file.name)
    return frozen


def _modified(task_dir: Path, name: str) -> bool:
    return name in _git(task_dir, "diff", "--name-only", "master").splitlines()


def _ruff_diagnostics(files: list[Path]) -> int:
    if not files:
        return 0
    out = subprocess.run(
        ["uv", "run", "ruff", "check", "--output-format=concise", *[str(f) for f in files]],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    # Each diagnostic is one line; the final "Found N errors" line is parseable.
    for line in reversed(out.stdout.splitlines()):
        line = line.strip()
        if line.startswith("Found ") and "error" in line:
            try:
                return int(line.split()[1])
            except (ValueError, IndexError):
                pass
        if line == "All checks passed!":
            return 0
    # Fallback: count non-empty lines.
    return sum(1 for line in out.stdout.splitlines() if line.strip())


def _run_hidden_tests(task_dir: Path) -> tuple[float, str]:
    """Run any task_dir/HIDDEN_TESTS/test_*.py against the post-edit worktree.

    Returns (pass_rate, note). Pass rate is in [0..1]. If no hidden tests
    exist, returns (1.0, "n/a").
    """
    hidden_dir = task_dir / "HIDDEN_TESTS"
    if not hidden_dir.is_dir():
        return 1.0, "n/a"
    # Copy hidden tests next to the source temporarily, then unittest discover.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # mirror the worktree (excluding .git, HIDDEN_TESTS, .agent6).
        # Skip sockets and other non-regular files (curator IPC sockets
        # may live under .agent6/runs/.../curator.sock; they can't be
        # copied by shutil.copytree).
        def _ignore(_src: str, names: list[str]) -> list[str]:
            skip: list[str] = []
            for n in names:
                full = Path(_src) / n
                try:
                    st = full.lstat()
                except OSError:
                    skip.append(n)
                    continue
                import stat as _stat

                if _stat.S_ISSOCK(st.st_mode) or _stat.S_ISFIFO(st.st_mode):
                    skip.append(n)
            return skip

        for p in task_dir.iterdir():
            if p.name in {".git", "HIDDEN_TESTS", ".agent6"}:
                continue
            dst = tmp_path / p.name
            if p.is_dir():
                shutil.copytree(p, dst, ignore=_ignore)
            else:
                shutil.copy2(p, dst)
        # Copy hidden tests into root.
        for tfile in hidden_dir.glob("test_*.py"):
            shutil.copy2(tfile, tmp_path / tfile.name)
        out = subprocess.run(
            ["python3", "-m", "unittest", "-v"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(tmp_path),
        )
        # unittest writes summary to stderr: "Ran N tests" and "FAILED (failures=X, errors=Y)" or "OK"
        ran = 0
        failed = 0
        for line in out.stderr.splitlines():
            line = line.strip()
            if line.startswith("Ran ") and "test" in line:
                try:
                    ran = int(line.split()[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("FAILED"):
                # parse failures=X, errors=Y
                for piece in line.replace("(", " ").replace(")", " ").split(","):
                    piece = piece.strip()
                    if piece.startswith("failures="):
                        failed += int(piece.split("=", 1)[1])
                    elif piece.startswith("errors="):
                        failed += int(piece.split("=", 1)[1])
        if ran == 0:
            return 1.0, "no hidden tests ran"
        rate = max(0.0, (ran - failed) / ran)
        return rate, f"{ran - failed}/{ran} hidden tests pass"


def score_task(task_dir: Path) -> QualityReport:
    name = task_dir.name
    notes: list[str] = []
    result_path = task_dir / "result.json"
    verify = 0.0
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            verify = 1.0 if data.get("verify_pass") else 0.0
        except (OSError, json.JSONDecodeError):
            notes.append("result.json unreadable")

    # test_integrity
    frozen = _frozen_in_task(task_dir)
    integrity = 1.0
    for f in frozen:
        if _modified(task_dir, f):
            integrity = 0.0
            notes.append(f"modified frozen test: {f}")

    # diff_size
    added = _diff_added_lines(task_dir)
    ref_path = task_dir / "REFERENCE.md"
    ref_budget = 30
    if ref_path.is_file():
        for line in ref_path.read_text().splitlines():
            if line.startswith("budget_lines:"):
                try:
                    ref_budget = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    if added == 0:
        diff_score = 0.0 if verify < 0.5 else 1.0  # no edits + verify pass = trivial
    else:
        diff_score = min(1.0, ref_budget / max(ref_budget, added))

    # lint_clean
    touched = _touched_py_files(task_dir)
    diags = _ruff_diagnostics(touched)
    lint_score = 1.0 if diags == 0 else max(0.0, 1.0 - min(1.0, diags / 10.0))
    if diags:
        notes.append(f"ruff: {diags} diagnostics on {len(touched)} touched files")

    # hidden tests
    hidden, hidden_note = _run_hidden_tests(task_dir)
    if hidden_note != "n/a":
        notes.append(hidden_note)

    composite = (
        WEIGHTS["verify"] * verify
        + WEIGHTS["test_integrity"] * integrity
        + WEIGHTS["diff_size"] * diff_score
        + WEIGHTS["lint_clean"] * lint_score
        + WEIGHTS["hidden_tests"] * hidden
    )
    if verify < 0.5:
        composite = min(composite, 0.30)
        notes.append("verify failed: composite capped at 0.30")

    return QualityReport(
        task=name,
        verify=verify,
        test_integrity=integrity,
        diff_size=diff_score,
        lint_clean=lint_score,
        hidden_tests=hidden,
        composite=round(composite, 4),
        notes=tuple(notes),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dirs", nargs="*", type=Path)
    ap.add_argument("--bench-root", type=Path, default=None)
    args = ap.parse_args()

    dirs: list[Path] = list(args.task_dirs)
    if args.bench_root is not None:
        dirs.extend(sorted(p for p in args.bench_root.iterdir() if p.is_dir() and p.name != "logs"))
    if not dirs:
        print("no task dirs given (use --bench-root or pass paths)", file=sys.stderr)
        return 2

    reports: list[QualityReport] = []
    for d in dirs:
        if not d.is_dir():
            print(f"skip {d}: not a directory", file=sys.stderr)
            continue
        rep = score_task(d)
        (d / "quality.json").write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        reports.append(rep)
        print(
            f"{rep.task:30s} Q={rep.composite:.3f}  "
            f"verify={rep.verify:.0f} integ={rep.test_integrity:.0f} "
            f"diff={rep.diff_size:.2f} lint={rep.lint_clean:.2f} hid={rep.hidden_tests:.2f}"
        )

    if reports:
        avg = sum(r.composite for r in reports) / len(reports)
        pass_count = sum(1 for r in reports if r.verify >= 0.5)
        print(f"--- mean Q={avg:.3f} over {len(reports)} tasks, verify {pass_count}/{len(reports)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
