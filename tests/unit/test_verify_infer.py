# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for verify_command inference (AGENTS.md -> repo signals -> LLM)."""

from __future__ import annotations

from pathlib import Path

from agent6.verify_infer import (
    gather_repo_manifests,
    infer_verify_command,
    parse_llm_verify,
    verify_from_agents_md,
    verify_from_repo_signals,
)

_AGENTS_PIPELINE = """\
# AGENTS.md

## Verify command

Also what `verify_command` should be in this repo's agent6 config:

```bash
uv run ruff check && uv run ruff format --check && \\
  uv run pyright && uv run pytest
```
"""

_AGENTS_SIMPLE = """\
## Verify command

```bash
# EDIT: replace with your actual verify pipeline.
.venv/bin/python -m pytest -x
```
"""


def test_agents_md_pipeline_wraps_in_sh_c() -> None:
    argv = verify_from_agents_md(_AGENTS_PIPELINE)
    assert argv is not None
    assert argv[0] == "sh" and argv[1] == "-c"
    # backslash-continuation joined, both halves of the && chain present.
    assert "ruff check" in argv[2] and "pytest" in argv[2]
    assert "&&" in argv[2]


def test_agents_md_simple_command_tokenises_and_skips_comments() -> None:
    argv = verify_from_agents_md(_AGENTS_SIMPLE)
    assert argv == (".venv/bin/python", "-m", "pytest", "-x")


def test_agents_md_inline_marker() -> None:
    assert verify_from_agents_md("Verify: pytest -q") == ("pytest", "-q")
    assert verify_from_agents_md("Test: make check") == ("make", "check")


def test_agents_md_none_when_absent() -> None:
    assert verify_from_agents_md("# Readme\n\nNo verify here.") is None
    assert verify_from_agents_md("") is None


def test_repo_signal_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    assert verify_from_repo_signals(tmp_path) == (("npm", "test", "--silent"), "package.json")


def test_repo_signal_makefile_target(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("build:\n\tcc x.c\ntest:\n\t./run\n", encoding="utf-8")
    assert verify_from_repo_signals(tmp_path) == (("make", "test"), "Makefile:test")


def test_repo_signal_pyproject(tmp_path: Path) -> None:
    # No .venv present (e.g. a container or system-python checkout) -> fall back
    # to python3 on PATH, NOT the missing .venv/bin/python that would break verify.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert verify_from_repo_signals(tmp_path) == (
        ("python3", "-m", "pytest", "-q"),
        "pyproject",
    )


def test_repo_signal_pyproject_prefers_existing_venv(tmp_path: Path) -> None:
    # When a project .venv/bin/python exists, prefer it (jail-visible convention).
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    venv_py = tmp_path / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("", encoding="utf-8")
    assert verify_from_repo_signals(tmp_path) == (
        (".venv/bin/python", "-m", "pytest", "-q"),
        "pyproject",
    )


def test_repo_signal_cargo_and_go(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert verify_from_repo_signals(tmp_path) == (("cargo", "test", "--quiet"), "Cargo.toml")
    (tmp_path / "Cargo.toml").unlink()
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    assert verify_from_repo_signals(tmp_path) == (("go", "test", "./..."), "go.mod")


def test_repo_signal_none(tmp_path: Path) -> None:
    assert verify_from_repo_signals(tmp_path) is None


def test_parse_llm_verify() -> None:
    assert parse_llm_verify('["pytest","-q"]') == ("pytest", "-q")
    assert parse_llm_verify('here you go:\n```json\n["cargo","test"]\n```') == ("cargo", "test")
    assert parse_llm_verify("[]") is None
    assert parse_llm_verify("I cannot tell") is None
    assert parse_llm_verify("[1, 2]") is None
    assert parse_llm_verify('["", "x"]') is None


def test_infer_layering_prefers_agents_md(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    got = infer_verify_command(tmp_path, "Verify: pytest -q")
    assert got is not None and got.source == "agents_md" and got.argv == ("pytest", "-q")


def test_infer_falls_back_to_signals_then_llm(tmp_path: Path) -> None:
    # No AGENTS.md hint, no manifests -> only the LLM tier can answer.
    calls: list[str] = []

    def fake_llm(ctx: str) -> str:
        calls.append(ctx)
        return '["make","verify"]'

    got = infer_verify_command(tmp_path, "", llm_call=fake_llm)
    assert got is not None and got.source == "llm" and got.argv == ("make", "verify")
    assert calls, "llm tier should have been consulted"

    # A repo signal short-circuits before the LLM.
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    calls.clear()
    got2 = infer_verify_command(tmp_path, "", llm_call=fake_llm)
    assert got2 is not None and got2.source == "go.mod"
    assert not calls, "repo signal should short-circuit the LLM"


def test_infer_llm_failure_is_safe(tmp_path: Path) -> None:
    def boom(_ctx: str) -> str:
        raise RuntimeError("provider down")

    assert infer_verify_command(tmp_path, "", llm_call=boom) is None


def test_gather_repo_manifests_clips_and_includes(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    ctx = gather_repo_manifests(tmp_path, "Verify: pytest", cap=10)
    assert "pyproject.toml" in ctx and "AGENTS.md" in ctx and "top-level" in ctx
