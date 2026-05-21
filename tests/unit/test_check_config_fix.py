# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for `agent6.config_fix` and the `check-config --fix` flow."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config
from agent6.config_fix import (
    Fix,
    FixKind,
    apply_fixes,
    format_value,
    propose_fixes,
    starter_recommendations,
)
from agent6.init import _STARTER_TOML  # pyright: ignore[reportPrivateUsage]

# ---------------------------------------------------------------------------
# Starter-template ↔ recommendations parity
# ---------------------------------------------------------------------------


def _all_required_paths_from_schema() -> set[str]:
    """Walk `Config` fields and return every `<section>.<field>` path
    that is required (no default, simple section), expressed in
    starter-template flat form (e.g. `"providers.anthropic.api_key_env"`,
    `"models.planner.provider"`).

    The starter template uses a single representative provider
    (`anthropic`) and the five fixed roles (planner / worker / critic /
    reviewer / summarizer), so dict-valued sections like `providers`
    and `models` enumerate exactly those keys.
    """
    parsed = tomllib.loads(_STARTER_TOML)
    paths: set[str] = set()

    def walk(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in node.items():
            new_prefix = (*prefix, key)
            if isinstance(value, dict):
                walk(value, new_prefix)
            else:
                paths.add(".".join(new_prefix))

    walk(parsed, ())
    return paths


def test_starter_recommendations_parses_template() -> None:
    recs = starter_recommendations()
    # Every section we expect is present.
    assert "agent6" in recs
    assert "providers.anthropic" in recs
    assert "models.planner" in recs
    assert "models.summarizer" in recs
    assert "sandbox" in recs
    assert "git" in recs
    assert "workflow" in recs
    assert "budget" in recs

    # Spot-check values are typed correctly.
    assert recs["agent6"]["config_version"] == 1
    assert recs["sandbox"]["profile"] == "auto"
    assert recs["git"]["require_clean_worktree"] is True
    assert recs["budget"]["max_input_tokens"] == 2_000_000
    assert recs["workflow"]["verify_command"] == ["uv", "run", "pytest", "-x"]


def test_starter_template_validates_against_config() -> None:
    """The starter template must be a complete, valid agent6.toml on its own.

    This is the parity test: if a field is added to `Config` without
    landing in `_STARTER_TOML`, this asserts before `--fix` ever ships
    a stale recommendation.
    """
    raw = tomllib.loads(_STARTER_TOML)
    Config.model_validate(raw)


def test_all_starter_paths_appear_in_recommendations() -> None:
    """Every leaf path in the starter is reachable through `starter_recommendations()`."""
    expected = _all_required_paths_from_schema()
    recs = starter_recommendations()
    actual = {f"{section}.{key}" for section, fields in recs.items() for key in fields}
    assert expected == actual


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "true"),
        (False, "false"),
        (1, "1"),
        (2_000_000, "2000000"),
        ("auto", '"auto"'),
        ('with "quote"', '"with \\"quote\\""'),
        ([], "[]"),
        (["uv", "run", "pytest", "-x"], '["uv", "run", "pytest", "-x"]'),
    ],
)
def test_format_value(value: Any, expected: str) -> None:
    assert format_value(value) == expected


def test_format_value_rejects_unknown_types() -> None:
    with pytest.raises(TypeError):
        format_value(1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helpers for proposal tests
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


_MINIMAL_VALID = _STARTER_TOML


# ---------------------------------------------------------------------------
# Proposal
# ---------------------------------------------------------------------------


def test_propose_fixes_on_valid_config_returns_empty(tmp_path: Path) -> None:
    cfg = _write(tmp_path, "agent6.toml", _MINIMAL_VALID)
    result = propose_fixes(cfg)
    assert result.fixes == ()
    assert result.remaining_errors == ()


def test_propose_fixes_missing_whole_section(tmp_path: Path) -> None:
    # Build a config missing the [budget] section entirely.
    text = _MINIMAL_VALID.replace(
        "[budget]\n"
        "# Hard stop. The run is resumable from the persistent task graph.\n"
        "max_input_tokens = 2000000\n"
        "max_output_tokens = 200000\n",
        "",
    )
    assert "[budget]" not in text
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    assert result.remaining_errors == ()
    assert len(result.fixes) == 1
    fix = result.fixes[0]
    assert fix.kind is FixKind.NEW_SECTION
    assert fix.section == "budget"
    assert "max_input_tokens = 2000000" in fix.lines
    assert "max_output_tokens = 200000" in fix.lines


def test_propose_fixes_missing_field_in_present_section(tmp_path: Path) -> None:
    # Drop `auto_stash` from [git]; section header remains.
    text = _MINIMAL_VALID.replace("auto_stash = false\n", "")
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    assert result.remaining_errors == ()
    assert len(result.fixes) == 1
    fix = result.fixes[0]
    assert fix.kind is FixKind.INSERT_FIELD
    assert fix.section == "git"
    assert fix.key == "auto_stash"
    assert fix.lines == ("auto_stash = false",)


def test_propose_fixes_mixed_sections_and_fields(tmp_path: Path) -> None:
    # Strip [budget] entirely AND drop a single field from [sandbox].
    text = _MINIMAL_VALID.replace(
        "[budget]\n"
        "# Hard stop. The run is resumable from the persistent task graph.\n"
        "max_input_tokens = 2000000\n"
        "max_output_tokens = 200000\n",
        "",
    ).replace('run_commands = "ask"\n', "")
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    assert result.remaining_errors == ()
    kinds = {f.kind for f in result.fixes}
    assert kinds == {FixKind.NEW_SECTION, FixKind.INSERT_FIELD}
    sections = {f.section for f in result.fixes}
    assert sections == {"budget", "sandbox"}


def test_propose_fixes_passes_through_non_missing_errors(tmp_path: Path) -> None:
    # Wrong type for sandbox.profile is a type error, not a missing one.
    text = _MINIMAL_VALID.replace('profile = "auto"', 'profile = "bogus"')
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    assert result.fixes == ()
    assert any("sandbox.profile" in line for line in result.remaining_errors)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_apply_fixes_inserts_field_after_header(tmp_path: Path) -> None:
    text = _MINIMAL_VALID.replace("auto_stash = false\n", "")
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    apply_fixes(cfg, result.fixes)
    # File now validates.
    Config.model_validate(tomllib.loads(cfg.read_text(encoding="utf-8")))
    # And the new line sits directly after the [git] header.
    lines = cfg.read_text(encoding="utf-8").splitlines()
    git_index = lines.index("[git]")
    assert lines[git_index + 1] == "auto_stash = false"


def test_apply_fixes_appends_whole_section(tmp_path: Path) -> None:
    text = _MINIMAL_VALID.replace(
        "[budget]\n"
        "# Hard stop. The run is resumable from the persistent task graph.\n"
        "max_input_tokens = 2000000\n"
        "max_output_tokens = 200000\n",
        "",
    )
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    apply_fixes(cfg, result.fixes)
    # File validates and contains the appended [budget] block.
    final = cfg.read_text(encoding="utf-8")
    Config.model_validate(tomllib.loads(final))
    assert "[budget]" in final
    assert "max_input_tokens = 2000000" in final


def test_apply_fixes_preserves_existing_comments(tmp_path: Path) -> None:
    # User has a custom comment above the [git] section; field is missing.
    text = _MINIMAL_VALID.replace(
        "[git]\n",
        "# my custom comment\n[git]\n",
    ).replace("auto_stash = false\n", "")
    cfg = _write(tmp_path, "agent6.toml", text)
    result = propose_fixes(cfg)
    apply_fixes(cfg, result.fixes)
    final = cfg.read_text(encoding="utf-8")
    assert "# my custom comment" in final
    Config.model_validate(tomllib.loads(final))


def test_apply_fixes_idempotent_after_validation(tmp_path: Path) -> None:
    # Once applied, a second propose returns no fixes.
    text = _MINIMAL_VALID.replace("auto_stash = false\n", "")
    cfg = _write(tmp_path, "agent6.toml", text)
    apply_fixes(cfg, propose_fixes(cfg).fixes)
    again = propose_fixes(cfg)
    assert again.fixes == ()
    assert again.remaining_errors == ()


# ---------------------------------------------------------------------------
# Fix preview rendering
# ---------------------------------------------------------------------------


def test_fix_render_preview_new_section() -> None:
    fix = Fix(
        kind=FixKind.NEW_SECTION,
        section="budget",
        key=None,
        lines=("max_input_tokens = 2000000", "max_output_tokens = 200000"),
        description="x",
    )
    assert (
        fix.render_preview() == "[budget]\nmax_input_tokens = 2000000\nmax_output_tokens = 200000"
    )


def test_fix_render_preview_insert_field() -> None:
    fix = Fix(
        kind=FixKind.INSERT_FIELD,
        section="git",
        key="auto_stash",
        lines=("auto_stash = false",),
        description="x",
    )
    assert fix.render_preview() == "auto_stash = false"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_check_config_fix_assume_yes_repairs_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.cli import _cmd_check_config  # pyright: ignore[reportPrivateUsage]

    text = _MINIMAL_VALID.replace("auto_stash = false\n", "")
    cfg = _write(tmp_path, "agent6.toml", text)
    rc = _cmd_check_config(cfg, fix=True, assume_yes=True)
    assert rc == 0
    Config.model_validate(tomllib.loads(cfg.read_text(encoding="utf-8")))
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "auto_stash = false" in combined
    assert "now validates cleanly" in combined


def test_cli_check_config_fix_interactive_decline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent6.cli import _cmd_check_config  # pyright: ignore[reportPrivateUsage]

    text = _MINIMAL_VALID.replace("auto_stash = false\n", "")
    cfg = _write(tmp_path, "agent6.toml", text)
    original = cfg.read_text(encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    rc = _cmd_check_config(cfg, fix=True, assume_yes=False)
    assert rc == 2
    assert cfg.read_text(encoding="utf-8") == original
    combined = capsys.readouterr().out
    assert "aborted, no changes written" in combined
