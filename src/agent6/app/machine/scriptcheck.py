# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Check a machine bundle's scripts: lint, types, and offline mock tests.

Every child runs through `run_in_jail` with no network. `lint_and_typecheck` runs ruff under
its own config discovery and pyrefly under an empty config and the `strict` preset.
`run_offline_tests` executes each `*_test.py`, model-authored code.

Only `full` `sandbox.isolation` cuts a child off the network: under `partial` and `none`
nothing runs and the caller is told. A missing ruff or pyrefly is skipped with a note, and a
jail that cannot start is a problem.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import sys
import tempfile
import tomllib

from agent6 import kinds
from agent6.sandbox import jail, run_in_jail

__all__ = ["OfflineTestOutcome", "available_tools", "lint_and_typecheck", "run_offline_tests"]

_TEST_SUFFIX = "_test.py"
_MAX_DIAG_LINES = 30


def _resolve_tool(name: str) -> pathlib.Path | None:
    """Return a checker's binary (`ruff`, `pyrefly`), or None when it is absent.

    `uvx` is no source: it downloads the tool, and the jail has no network.

    Args:
        name: The tool's command name.

    Returns:
        The real path of the console script beside the running interpreter, else of the one
        on `PATH`, else None.
    """
    local = pathlib.Path(sys.executable).parent / name
    found = str(local) if local.is_file() else shutil.which(name)
    return pathlib.Path(found).resolve() if found else None


def _offline_gap(isolation: kinds.IsolationLevel) -> str:
    """Return why the isolation level cannot run a check off the network, or "" when it can."""
    if isolation == "full":
        return ""
    # Under `partial`, the check would reach the host network.
    return "no sandbox" if isolation == "none" else "needs sandbox.isolation = 'full'"


def available_tools() -> list[str]:
    """Return which of ruff and pyrefly resolve in this environment."""
    return [name for name in ("ruff", "pyrefly") if _resolve_tool(name) is not None]


def _trim(text: str) -> str:
    """Return the text cut to the diagnostic line cap, with a count of what was dropped."""
    lines = text.splitlines()
    if len(lines) <= _MAX_DIAG_LINES:
        return text.strip()
    kept = lines[:_MAX_DIAG_LINES]
    return "\n".join(kept).strip() + f"\n... ({len(lines) - _MAX_DIAG_LINES} more lines)"


def _run_static(
    argv: list[str],
    cwd: pathlib.Path,
    label: str,
    *,
    strip: pathlib.Path,
    read: tuple[pathlib.Path, ...] = (),
    write: tuple[pathlib.Path, ...] = (),
) -> str | None:
    """Run a checker in a no-network jail and return its problems, or None when it passed.

    The jail shows the checker its own binary, its working directory read-only, and the
    paths named here.

    Args:
        argv: The checker's argv; its first element is the binary's real path.
        cwd: Where the checker runs.
        label: The checker's name in the problem text.
        strip: The path prefix diagnostics lose so they read as bundle paths.
        read: What the checker reads outside its working directory.
        write: What the checker may rewrite outside its working directory.

    Returns:
        The problem text, or None on a clean pass.
    """
    policy = kinds.JailPolicy(
        cwd=cwd,
        argv=tuple(argv),
        network="none",
        extra_ro_paths=(pathlib.Path(argv[0]), *read),
        extra_rw_paths=write,
        extra_protect_paths=(cwd,),
        timeout_s=180,
    )
    try:
        res = run_in_jail(policy)
    except jail.JailUnavailableError as exc:
        return f"{label} could not run in a jail ({exc})"
    if res.returncode == 0:
        return None
    out = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
    if res.timed_out or res.exec_failed:
        return f"{label} could not run ({'timed out' if res.timed_out else out})"
    out = out.replace(str(strip.resolve()) + "/", "").replace(str(strip) + "/", "")
    return f"{label} found problems:\n{_trim(out)}"


def _nearest_ruff_config(start: pathlib.Path) -> pathlib.Path | None:
    """Return the ruff config governing a path, walking up as ruff's own discovery does.

    Args:
        start: The directory the walk starts at.

    Returns:
        The nearest `.ruff.toml`, `ruff.toml`, or `pyproject.toml` with a `[tool.ruff]`
        table, else None.
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


def _lint(
    ruff: pathlib.Path, base: pathlib.Path, ruff_config_from: pathlib.Path | None, *, fix: bool
) -> str | None:
    """Run ruff on the scripts and return its problems, or None when it passed.

    ruff runs from an empty directory, on the absolute scripts path. `--config` anchors a
    relative pattern to the cwd, and there a pattern of the destination's config matches no
    draft file. The jail's cwd is read-only, so the scripts are a mount of their own, which
    `fix` makes writable.

    Args:
        ruff: ruff's binary.
        base: The bundle's scripts directory, resolved.
        ruff_config_from: Where to resolve the config from, else native discovery.
        fix: Apply ruff's safe fixes.

    Returns:
        The problem text, or None on a clean pass.
    """
    argv = [str(ruff), "check", "--no-cache", "--output-format", "concise"]
    # The jail shows ruff no directory above the scripts, so the governing config is granted.
    config = _nearest_ruff_config(ruff_config_from or base)
    if ruff_config_from is not None:
        argv += ["--isolated"] if config is None else ["--config", str(config)]
    if fix:
        argv.append("--fix")
    governing = () if config is None else (config,)
    with tempfile.TemporaryDirectory(prefix="agent6-scriptcheck-") as empty:
        return _run_static(
            [*argv, str(base)],
            pathlib.Path(empty).resolve(),
            "ruff (lint)",
            strip=base.parent,
            read=governing if fix else (base, *governing),
            write=(base,) if fix else (),
        )


def lint_and_typecheck(
    scripts_dir: pathlib.Path,
    isolation: kinds.IsolationLevel,
    *,
    fix: bool = False,
    ruff_config_from: pathlib.Path | None = None,
) -> list[str]:
    """Lint and type-check the bundle's Python scripts in a no-network jail.

    `*_test.py` files are linted, not type-checked. The bundle is read-only to both checkers,
    except the scripts to ruff under `fix`.

    Args:
        scripts_dir: The bundle's scripts directory.
        isolation: The isolation level the run resolved.
        fix: Apply ruff's safe fixes in place; only `machine create` does, on its own bundle.
        ruff_config_from: Resolve the ruff config from the publish destination, so the draft
            gate agrees with the `machine check` the published bundle faces.

    Returns:
        The problems found; empty when clean, or when a note says a check did not run.
    """
    if not scripts_dir.is_dir() or not any(scripts_dir.rglob("*.py")):
        return []
    if gap := _offline_gap(isolation):
        print(f"note: script lint and type check not run ({gap})", file=sys.stderr)
        return []
    base = scripts_dir.resolve()
    problems: list[str] = []
    if ruff := _resolve_tool("ruff"):
        problem = _lint(ruff, base, ruff_config_from, fix=fix)
        if problem:
            problems.append(problem)
    else:
        print("note: ruff not installed; script lint skipped", file=sys.stderr)
    if pyrefly := _resolve_tool("pyrefly"):
        real = sorted(
            base / p.relative_to(scripts_dir)
            for p in scripts_dir.rglob("*.py")
            if not p.name.endswith(_TEST_SUFFIX)
        )
        if real:
            # No config file applies; unconfigured, the `basic` preset reports no type error.
            argv = [str(pyrefly), "check", "--config", os.devnull, "--preset", "strict"]
            # Imports resolve against the `python3` that runs the scripts, on the jail's PATH.
            argv += ["--python-interpreter-path", "python3"]
            # A script's own directory is on its import path when it runs.
            for directory in sorted({p.parent for p in real}):
                argv += ["--search-path", str(directory)]
            argv += ["--output-format", "min-text", "--summary=none", *map(str, real)]
            problem = _run_static(argv, base.parent, "pyrefly (type check)", strip=base.parent)
            if problem:
                problems.append(problem)
    else:
        print("note: pyrefly not installed; script type check skipped", file=sys.stderr)
    return problems


@dataclasses.dataclass(frozen=True, slots=True)
class OfflineTestOutcome:
    """Record the offline tests' failures and what could not run.

    A skip rides to the caller's verdict surface: buried in stderr it reads as tests green.

    Attributes:
        problems: One entry per failed test.
        ran: How many script files executed, not individual test cases.
        skipped: How many tests could not run.
        skip_reason: Why they could not.
    """

    problems: tuple[str, ...] = ()
    ran: int = 0
    skipped: int = 0
    skip_reason: str = ""


def run_offline_tests(
    bundle_dir: pathlib.Path, isolation: kinds.IsolationLevel, *, timeout_s: float = 30.0
) -> OfflineTestOutcome:
    """Execute every `scripts/**/*_test.py` in a no-network jail.

    Only `full` `sandbox.isolation` can cut the tests off the network, so under `none` and
    `partial` they count as skipped. Each test gets a fresh writable `$AGENT6_MACHINE_DATA_DIR`,
    under the default `JailPolicy` memory cap.

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
    if gap := _offline_gap(isolation):
        return OfflineTestOutcome(skipped=len(tests), skip_reason=gap)
    data_dir = bundle_dir / ".scriptcheck_data"
    problems: list[str] = []
    ran = 0
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
                    ran=ran,
                    problems=(f"could not run offline tests in a jail ({exc})",),
                )
            ran += 1
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
    return OfflineTestOutcome(problems=tuple(problems), ran=ran)
