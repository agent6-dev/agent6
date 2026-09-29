# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validate the scripts `machine create` generates: lint-clean, typed, and simulated offline.

Two layers, matching their risk. `lint_and_typecheck` is static analysis (ruff and ty read
the files, never run them), so it shells out with a fixed argv; ruff runs on the real files
under its own config discovery, ty checks a private temp copy. `run_offline_tests` executes
each `*_test.py`, model-authored code, so it goes through `run_in_jail` with no network.

A missing ruff or ty is skipped silently, so a stripped install still produces a bundle. An
unavailable jail surfaces a diagnostic, except on isolation `none`, where execution is skipped.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

from agent6 import kinds
from agent6.sandbox import jail, run_in_jail

__all__ = ["OfflineTestOutcome", "available_tools", "lint_and_typecheck", "run_offline_tests"]

_TEST_SUFFIX = "_test.py"
_MAX_DIAG_LINES = 30


def _resolve_tool(name: str) -> list[str] | None:
    """Return the argv prefix that runs a dev tool (`ruff`, `ty`), or None when it is absent.

    The console script beside the running interpreter wins, then `PATH`, then `uvx <name>`.
    """
    local = pathlib.Path(sys.executable).parent / name
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
    """Return which of ruff and ty resolve in this environment."""
    return [name for name in ("ruff", "ty") if _resolve_tool(name) is not None]


def _trim(text: str) -> str:
    """Return the text cut to the diagnostic line cap, with a count of what was dropped."""
    lines = text.splitlines()
    if len(lines) <= _MAX_DIAG_LINES:
        return text.strip()
    kept = lines[:_MAX_DIAG_LINES]
    return "\n".join(kept).strip() + f"\n... ({len(lines) - _MAX_DIAG_LINES} more lines)"


def _run_static(
    argv: list[str], cwd: pathlib.Path, label: str, *, strip: pathlib.Path | None = None
) -> str | None:
    """Run a static checker and return its problems, or None when it passed.

    Args:
        argv: The checker's fixed argv; the model-authored files are only read, never run.
        cwd: Where the checker runs.
        label: The checker's name in the problem text.
        strip: The path prefix diagnostics lose so they read as bundle paths; defaults to cwd.

    Returns:
        The problem text, or None on a clean pass.
    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        res = subprocess.run(
            argv, capture_output=True, text=True, timeout=180, cwd=cwd, check=False, env=env
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{label} could not run ({exc})"
    if res.returncode == 0:
        return None
    out = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
    base = strip or cwd
    out = out.replace(str(base.resolve()) + "/", "").replace(str(base) + "/", "")
    return f"{label} found problems:\n{_trim(out)}"


def _nearest_ruff_config(start: pathlib.Path) -> pathlib.Path | None:
    """Return the ruff config governing a path, walking up as ruff's own discovery does.

    For the one caller whose files live outside the tree their config governs: the scratch
    bundle of `machine create`.
    """
    base = start.resolve()
    for directory in (base, *base.parents):
        for name in (".ruff.toml", "ruff.toml", "pyproject.toml"):
            candidate = directory / name
            if not candidate.is_file():
                continue
            if name == "pyproject.toml":
                try:
                    data = tomllib.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, tomllib.TOMLDecodeError):
                    continue
                if "ruff" not in data.get("tool", {}):
                    continue
            return candidate
    return None


def _ruff_invocation(
    ruff: list[str], scripts_dir: pathlib.Path, ruff_config_from: pathlib.Path | None, *, fix: bool
) -> tuple[list[str], pathlib.Path]:
    """Return ruff's argv and cwd.

    Native discovery runs from the bundle dir. A config resolved from the publish destination
    runs from that config's own directory on the absolute scripts path: `--config` anchors a
    relative pattern to the cwd, where discovery anchors it to the config's directory.

    Args:
        ruff: The argv prefix that runs ruff.
        scripts_dir: The bundle's scripts directory.
        ruff_config_from: Where to resolve the config from, else native discovery.
        fix: Apply ruff's safe fixes.

    Returns:
        The argv and the directory to run it from.
    """
    argv = [*ruff, "check", "--no-cache", "--output-format", "concise"]
    cwd, target = scripts_dir.resolve().parent, scripts_dir.name
    if ruff_config_from is not None:
        found = _nearest_ruff_config(ruff_config_from)
        if found is None:
            argv.append("--isolated")
        else:
            argv += ["--config", str(found)]
            cwd, target = found.parent, str(scripts_dir.resolve())
    if fix:
        argv.append("--fix")
    return [*argv, target], cwd


def lint_and_typecheck(
    scripts_dir: pathlib.Path, *, fix: bool = False, ruff_config_from: pathlib.Path | None = None
) -> list[str]:
    """Lint and type-check the bundle's Python scripts without running them.

    `*_test.py` files are linted, not type-checked. `--no-cache` keeps the operator-facing
    verbs write-free.

    Args:
        scripts_dir: The bundle's scripts directory.
        fix: Apply ruff's safe fixes in place; only `machine create` does, on its own bundle.
        ruff_config_from: Resolve the ruff config from the publish destination, so the draft
            gate agrees with the `machine check` the published bundle faces.

    Returns:
        The problems found; empty when clean or the tools are absent.
    """
    if not scripts_dir.is_dir() or not any(scripts_dir.rglob("*.py")):
        return []
    problems: list[str] = []
    if ruff := _resolve_tool("ruff"):
        argv, cwd = _ruff_invocation(ruff, scripts_dir, ruff_config_from, fix=fix)
        problem = _run_static(argv, cwd, "ruff (lint)", strip=scripts_dir.resolve().parent)
        if problem:
            problems.append(problem)
    else:
        print("note: ruff not installed; script lint skipped", file=sys.stderr)
    if ty := _resolve_tool("ty"):
        real = sorted(p for p in scripts_dir.rglob("*.py") if not p.name.endswith(_TEST_SUFFIX))
        if real:
            # ty walks up to the nearest pyproject.toml and has no isolation flag: a temp copy.
            work = pathlib.Path(tempfile.mkdtemp(prefix="agent6-scriptcheck-"))
            try:
                dst = work / "scripts"
                shutil.copytree(scripts_dir, dst, symlinks=True)
                copies = [str(dst / p.relative_to(scripts_dir)) for p in real]
                problem = _run_static([*ty, "check", *copies], work, "ty (type check)")
                if problem:
                    problems.append(problem)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    else:
        print("note: ty not installed; script type check skipped", file=sys.stderr)
    return problems


@dataclasses.dataclass(frozen=True, slots=True)
class OfflineTestOutcome:
    """Record the offline tests' failures and what could not run.

    A skip rides to the caller's verdict surface: buried in stderr it reads as tests green.

    Attributes:
        problems: One entry per failed test.
        skipped: How many tests could not run.
        skip_reason: Why they could not.
    """

    problems: tuple[str, ...] = ()
    skipped: int = 0
    skip_reason: str = ""


def run_offline_tests(
    bundle_dir: pathlib.Path, isolation: kinds.IsolationLevel, *, timeout_s: float = 30.0
) -> OfflineTestOutcome:
    """Execute every `scripts/**/*_test.py` in a no-network jail.

    Only the strict isolation has the network namespace that enforces the no-network contract
    on model-authored code; on `none` and `hardened` the tests count as skipped. Each test gets
    a fresh writable `$AGENT6_MACHINE_DATA_DIR`, under the default `JailPolicy` memory cap.

    Args:
        bundle_dir: The bundle holding `scripts/`.
        isolation: The isolation level the run resolved.
        timeout_s: The bound on each test.

    Returns:
        The failures and what could not run.
    """
    scripts_dir = bundle_dir / "scripts"
    if not scripts_dir.is_dir():
        return OfflineTestOutcome()
    if not sorted(scripts_dir.rglob(f"*{_TEST_SUFFIX}")):
        return OfflineTestOutcome()
    # A temp copy: the jail masks the state dir the real bundle lives under.
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="agent6-scripttest-"))
    try:
        bundle_copy = workdir / "bundle"
        # A drafting workspace is a git repo; `.git` holds every draft and the tests need none.
        shutil.copytree(
            bundle_dir, bundle_copy, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )
        return _run_offline_tests_in(bundle_copy, isolation, timeout_s=timeout_s)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_offline_tests_in(
    bundle_dir: pathlib.Path, isolation: kinds.IsolationLevel, *, timeout_s: float
) -> OfflineTestOutcome:
    """Return the outcome of the tests of a bundle copy, one jail each."""
    scripts_dir = bundle_dir / "scripts"
    tests = sorted(scripts_dir.rglob(f"*{_TEST_SUFFIX}"))
    if isolation != "strict":
        # hardened has no network namespace: model-authored code would reach the host network.
        reason = "no sandbox" if isolation == "none" else "no network isolation (hardened)"
        return OfflineTestOutcome(skipped=len(tests), skip_reason=reason)
    data_dir = bundle_dir / ".scriptcheck_data"
    problems: list[str] = []
    try:
        for test in tests:
            # Fresh per test: a record-style script's state must not reach the next test.
            shutil.rmtree(data_dir, ignore_errors=True)
            data_dir.mkdir(parents=True)
            rel = test.relative_to(bundle_dir).as_posix()
            policy = kinds.JailPolicy(
                cwd=bundle_dir,
                argv=("python3", rel),
                isolation=isolation,
                env=(
                    ("AGENT6_MACHINE_DATA_DIR", ".scriptcheck_data"),
                    ("PYTHONDONTWRITEBYTECODE", "1"),
                ),
                network="none",
                extra_rw_paths=(data_dir,),
                timeout_s=timeout_s,
            )
            try:
                res = run_in_jail(policy)
            except jail.JailUnavailableError as exc:
                return OfflineTestOutcome(
                    problems=(
                        f"could not run offline tests in a jail ({exc});"
                        " static checks still applied",
                    )
                )
            if res.returncode != 0:
                detail = (res.stderr or res.stdout or "").strip()
                # The diagnostic feeds the authoring prompt and the journal: no host paths.
                detail = detail.replace(str(bundle_dir.resolve()) + "/", "").replace(
                    str(bundle_dir) + "/", ""
                )
                problems.append(
                    f"offline test {rel} failed (exit {res.returncode}):\n{_trim(detail)}"
                )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    return OfflineTestOutcome(problems=tuple(problems))
