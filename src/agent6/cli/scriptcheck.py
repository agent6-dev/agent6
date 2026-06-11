# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validate the helper scripts `machine create` generates so a committed bundle
is production-ready: lint-clean, typed, and proven to *simulate* offline.

Two layers, matching their risk:

* :func:`lint_and_typecheck`, STATIC analysis only (ruff + ty read the files,
  they never run them), so it shells out directly with a fixed argv on a private
  temp copy. ``ruff --isolated`` ignores any repo config (the scratch bundle
  lives under the user's repo, whose ruleset must not bleed in); ty checks the
  real scripts only, mock-heavy ``*_test.py`` files trip ty on
  ``unittest.mock`` internals, so they are gated by *execution* instead.
* :func:`run_offline_tests`, EXECUTES each ``*_test.py``. Because that runs
  model-authored code, it goes through :func:`run_in_jail` (no network, the same
  confinement a tool state gets), never a bare subprocess.

A missing ruff/ty is skipped silently (a stripped install still produces a
bundle). An unavailable jail is different: it surfaces a diagnostic rather than
silently dropping the offline-test gate, except on profile ``none``, where
there is no jail to run model-authored code in and execution is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from agent6.sandbox import run_in_jail
from agent6.sandbox.jail import JailUnavailableError
from agent6.types import JailPolicy, SandboxProfile

__all__ = ["available_tools", "lint_and_typecheck", "run_offline_tests"]

_TEST_SUFFIX = "_test.py"
_MAX_DIAG_LINES = 30


def _resolve_tool(name: str) -> list[str] | None:
    """Locate a bundled dev tool (``ruff`` / ``ty``) as an argv prefix.

    Prefer the console script installed next to the running interpreter (the
    runtime dependency), then anything on ``PATH``, then a self-contained
    ``uvx <name>``. ``None`` if the tool can't be found at all (skip it)."""
    local = Path(sys.executable).parent / name
    if local.is_file():
        return [str(local)]
    on_path = shutil.which(name)
    if on_path:
        return [on_path]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, name]
    return None


def available_tools() -> list[str]:
    """Which of ruff/ty resolve in this environment (for a 'skipped' note)."""
    return [name for name in ("ruff", "ty") if _resolve_tool(name) is not None]


def _trim(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= _MAX_DIAG_LINES:
        return text.strip()
    kept = lines[:_MAX_DIAG_LINES]
    return "\n".join(kept).strip() + f"\n... ({len(lines) - _MAX_DIAG_LINES} more lines)"


def _run_static(argv: list[str], cwd: Path, label: str) -> str | None:
    """Run a static checker; return a problem string on failure, else None."""
    # Fixed argv (an operator-installed tool + flags); the only LLM-derived input
    # is the *files* it statically reads, it never executes them. See AGENTS.md.
    try:
        res = subprocess.run(
            argv, capture_output=True, text=True, timeout=180, cwd=cwd, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{label} could not run ({exc})"
    if res.returncode == 0:
        return None
    out = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
    return f"{label} found problems:\n{_trim(out)}"


def lint_and_typecheck(scripts_dir: Path) -> list[str]:
    """Lint (ruff) and type-check (ty) the bundle's Python scripts, no execution.

    Works on a private temp copy of *scripts_dir* so neither tool picks up the
    user's repo config. Returns human-readable problems (empty = clean / tools
    absent). ``*_test.py`` files are linted but not type-checked."""
    if not scripts_dir.is_dir() or not any(scripts_dir.rglob("*.py")):
        return []
    problems: list[str] = []
    # The temp copy exists for ty: it has no config-isolation flag and walks up
    # from the checked files to the nearest pyproject.toml, which would be the
    # user's repo (the scratch bundle lives under .agent6/). ruff is isolated by
    # its --isolated flag; it just shares the copy.
    work = Path(tempfile.mkdtemp(prefix="agent6-scriptcheck-"))
    try:
        dst = work / "scripts"
        shutil.copytree(scripts_dir, dst)
        if ruff := _resolve_tool("ruff"):
            problem = _run_static(
                [*ruff, "check", "--isolated", "--output-format", "concise", str(dst)],
                work,
                "ruff (lint)",
            )
            if problem:
                problems.append(problem)
        else:
            print("note: ruff not installed; script lint skipped", file=sys.stderr)
        if ty := _resolve_tool("ty"):
            real = [str(p) for p in sorted(dst.rglob("*.py")) if not p.name.endswith(_TEST_SUFFIX)]
            if real:
                problem = _run_static([*ty, "check", *real], work, "ty (type check)")
                if problem:
                    problems.append(problem)
        else:
            print("note: ty not installed; script type check skipped", file=sys.stderr)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return problems


def run_offline_tests(
    bundle_dir: Path, profile: SandboxProfile, *, timeout_s: float = 30.0
) -> list[str]:
    """Execute every ``scripts/**/*_test.py`` in a no-network jail (the bundle's
    offline simulation). Returns failures (empty = all green / nothing to run).

    Skipped on profile ``none`` (no jail to confine model-authored code in),
    the static checks still apply. Each test gets a fresh writable
    ``$AGENT6_MACHINE_DATA_DIR`` so record-style scripts can be exercised."""
    scripts_dir = bundle_dir / "scripts"
    if not scripts_dir.is_dir():
        return []
    tests = sorted(scripts_dir.rglob(f"*{_TEST_SUFFIX}"))
    if not tests:
        return []
    if profile == "none":
        # No jail to confine model-authored code in. Skipping is the only safe
        # option, but say so: a silent pass here looks like "tests ran green".
        print(
            f"note: no sandbox on this host; {len(tests)} offline script test(s) NOT run",
            file=sys.stderr,
        )
        return []
    data_dir = bundle_dir / ".scriptcheck_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    try:
        for test in tests:
            rel = test.relative_to(bundle_dir).as_posix()
            policy = JailPolicy(
                cwd=bundle_dir,
                argv=("python3", rel),
                profile=profile,
                env=(("AGENT6_MACHINE_DATA_DIR", ".scriptcheck_data"),),
                allow_network=False,
                extra_rw_paths=(data_dir,),
                timeout_s=timeout_s,
            )
            try:
                res = run_in_jail(policy)
            except JailUnavailableError as exc:
                # The jail is a prerequisite for ANY test here, so fail fast on
                # the first unavailability rather than repeating it per test.
                return [
                    f"could not run offline tests in a jail ({exc}); static checks still applied"
                ]
            if res.returncode != 0:
                detail = (res.stderr or res.stdout or "").strip()
                # Tracebacks name the absolute bundle dir. Relativize so the
                # diagnostic (which is fed back into the authoring prompt and
                # journaled) stays short and free of host paths.
                detail = detail.replace(str(bundle_dir.resolve()) + "/", "").replace(
                    str(bundle_dir) + "/", ""
                )
                problems.append(
                    f"offline test {rel} failed (exit {res.returncode}):\n{_trim(detail)}"
                )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return problems
