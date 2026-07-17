# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `machine create`/`check`/`test` script validation (cli/scriptcheck)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.machine import _scriptcheck as scriptcheck
from agent6.types import CommandResult, JailPolicy

_CLEAN = "import json\n\n\ndef f(x: int) -> str:\n    return json.dumps({'v': x})\n"


def _write(scripts_dir: Path, name: str, body: str) -> None:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / name).write_text(body, encoding="utf-8")


def _need(tool: str) -> None:
    if tool not in scriptcheck.available_tools():
        pytest.skip(f"{tool} not installed in this environment")


# --- static: ruff + ty ------------------------------------------------------


def test_lint_typecheck_clean(tmp_path: Path) -> None:
    _need("ruff")
    _need("ty")
    _write(tmp_path / "scripts", "ok.py", _CLEAN)
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []


def test_static_checks_disable_python_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "ok.py", _CLEAN)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    seen_env: dict[str, str] = {}

    def _resolve_tool(name: str) -> list[str] | None:
        return ["fake-tool"] if name == "ruff" else None

    def _run(
        _argv: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        cwd: Path,
        check: bool,
        env: dict[str, str],
    ) -> object:
        del capture_output, text, timeout, cwd, check
        seen_env.update(env)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(scriptcheck, "_resolve_tool", _resolve_tool)
    monkeypatch.setattr(scriptcheck.subprocess, "run", _run)

    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []
    assert seen_env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_lint_catches_undefined_name(tmp_path: Path) -> None:
    _need("ruff")
    _write(tmp_path / "scripts", "bad.py", "print(undefined_name)\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert any("ruff" in p for p in problems)


def test_typecheck_catches_type_error(tmp_path: Path) -> None:
    _need("ty")
    _write(tmp_path / "scripts", "bad.py", "def f(x: str) -> int:\n    return x + 1\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert any("ty" in p for p in problems)


def test_typecheck_skips_test_files(tmp_path: Path) -> None:
    """ty is NOT run on *_test.py (mock internals trip it); ruff still is."""
    _need("ty")
    # A type error that only ty would catch, in a *_test.py file -> not flagged.
    _write(tmp_path / "scripts", "x_test.py", "def f(a: str) -> int:\n    return a + 1\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert not any("ty" in p for p in problems)


def test_no_python_scripts_is_clean(tmp_path: Path) -> None:
    _write(tmp_path / "scripts", "run.sh", "#!/bin/sh\necho hi\n")
    assert scriptcheck.lint_and_typecheck(tmp_path / "scripts") == []


def test_missing_scripts_dir_is_clean(tmp_path: Path) -> None:
    assert scriptcheck.lint_and_typecheck(tmp_path / "nope") == []


# --- dynamic: offline test execution (jail patched, no fork) ----------------


def _fake_jail(returncode: int, stderr: str = "") -> object:
    def run(_policy: object) -> CommandResult:
        return CommandResult(
            argv=("python3", "scripts/thing_test.py"),
            returncode=returncode,
            stdout="",
            stderr=stderr,
            duration_s=0.0,
        )

    return run


def test_offline_tests_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    monkeypatch.setattr(scriptcheck, "run_in_jail", _fake_jail(0))
    assert scriptcheck.run_offline_tests(tmp_path, "hardened") == []


def test_offline_tests_disable_python_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    seen: list[JailPolicy] = []

    def _run(policy: JailPolicy) -> CommandResult:
        seen.append(policy)
        return CommandResult(
            argv=policy.argv,
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )

    monkeypatch.setattr(scriptcheck, "run_in_jail", _run)

    assert scriptcheck.run_offline_tests(tmp_path, "hardened") == []
    assert ("PYTHONDONTWRITEBYTECODE", "1") in seen[0].env


def test_offline_tests_fail_surfaces_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "raise SystemExit(1)\n")
    monkeypatch.setattr(scriptcheck, "run_in_jail", _fake_jail(1, "AssertionError: boom"))
    problems = scriptcheck.run_offline_tests(tmp_path, "hardened")
    assert len(problems) == 1
    assert "thing_test.py" in problems[0]
    assert "boom" in problems[0]


def test_offline_tests_relativize_bundle_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tracebacks from the jailed test name the absolute bundle dir; the
    # diagnostic is fed back into the authoring prompt, so host paths get
    # stripped down to bundle-relative ones.
    _write(tmp_path / "scripts", "thing_test.py", "raise SystemExit(1)\n")
    stderr = f'File "{tmp_path}/scripts/thing_test.py", line 1\nNameError: x'
    monkeypatch.setattr(scriptcheck, "run_in_jail", _fake_jail(1, stderr))
    problems = scriptcheck.run_offline_tests(tmp_path, "hardened")
    assert str(tmp_path) not in problems[0]
    assert 'File "scripts/thing_test.py"' in problems[0]


def test_offline_tests_skipped_on_none_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")
    called = False

    def _boom(_policy: object) -> CommandResult:  # pragma: no cover - must not run
        nonlocal called
        called = True
        return CommandResult(argv=(), returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(scriptcheck, "run_in_jail", _boom)
    assert scriptcheck.run_offline_tests(tmp_path, "none") == []
    assert not called


def test_offline_tests_no_test_files(tmp_path: Path) -> None:
    _write(tmp_path / "scripts", "real.py", "print('hi')\n")
    assert scriptcheck.run_offline_tests(tmp_path, "hardened") == []


def test_offline_tests_jail_unavailable_surfaces_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.sandbox.jail import JailUnavailableError

    _write(tmp_path / "scripts", "thing_test.py", "print('ok')\n")

    def _raise(_policy: object) -> CommandResult:
        raise JailUnavailableError("no namespaces")

    monkeypatch.setattr(scriptcheck, "run_in_jail", _raise)
    problems = scriptcheck.run_offline_tests(tmp_path, "strict")
    assert len(problems) == 1
    assert "could not run offline tests" in problems[0]


def test_static_diagnostics_relativize_temp_paths(tmp_path: Path) -> None:
    # ruff diagnostics used to name the private temp copy; they now read as
    # bundle-relative paths, like the offline-test diagnostics.
    _need("ruff")
    _write(tmp_path / "scripts", "bad.py", "print(undefined_name)\n")
    problems = scriptcheck.lint_and_typecheck(tmp_path / "scripts")
    assert problems
    assert "agent6-scriptcheck-" not in problems[0]
    assert "scripts/bad.py" in problems[0]


def test_offline_tests_get_a_fresh_data_dir_per_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The docstring promise: state one test's script leaves in
    # $AGENT6_MACHINE_DATA_DIR must not leak into the next test.
    _write(tmp_path / "scripts", "a_test.py", "pass\n")
    _write(tmp_path / "scripts", "b_test.py", "pass\n")

    def _run(policy: JailPolicy) -> CommandResult:
        data = Path(policy.extra_rw_paths[0])
        marker = data / "marker"
        rc = 1 if marker.exists() else 0  # a leaked marker fails the later test
        marker.write_text("x", encoding="utf-8")
        return CommandResult(
            argv=policy.argv, returncode=rc, stdout="", stderr="leaked marker", duration_s=0.0
        )

    monkeypatch.setattr(scriptcheck, "run_in_jail", _run)
    assert scriptcheck.run_offline_tests(tmp_path, "hardened") == []
    assert not (tmp_path / ".scriptcheck_data").exists()  # still cleaned up after
