# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the LSP client + dispatcher ``*_lsp`` tools.

The smoke tests that actually spawn ``ty`` are skipped when the binary
is not available so the suite stays portable. Pure-logic tests
(fail-open, symbol-position resolution, parsing) always run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent6.config import Config, load_config
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.lsp import (
    LspClient,
    LspError,
    LspLocation,
    _find_ty_argv,  # pyright: ignore[reportPrivateUsage]
    _path_to_uri,  # pyright: ignore[reportPrivateUsage]
    _symbol_position,  # pyright: ignore[reportPrivateUsage]
    _uri_to_path,  # pyright: ignore[reportPrivateUsage]
)

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
profile = "auto"
agent_network = "open"
run_commands = "no"
protect_git = true
protect_agent6 = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _config(tmp_path: Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


_HAS_TY = shutil.which("ty") is not None
_HAS_UVX = shutil.which("uvx") is not None
_HAS_LSP = _HAS_TY or _HAS_UVX

# --- pure helpers -----------------------------------------------------


def test_symbol_position_first_occurrence() -> None:
    text = "x = 1\ndef foo(): return foo\n"
    assert _symbol_position(text, "foo") == (1, 4)


def test_symbol_position_word_boundary() -> None:
    # `foobar` must not match a search for `foo`.
    text = "foobar = 1\nfoo = 2\n"
    assert _symbol_position(text, "foo") == (1, 0)


def test_symbol_position_missing() -> None:
    assert _symbol_position("nothing here\n", "absent") is None


def test_path_to_uri_and_back(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("", encoding="utf-8")
    uri = _path_to_uri(p)
    assert uri.startswith("file://")
    assert _uri_to_path(uri) == p


# --- fail-open --------------------------------------------------------


def test_lsp_client_fail_open_when_ty_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent6.tools.lsp.shutil.which", lambda _name: None)  # type: ignore[misc]
    client = LspClient(tmp_path)
    with pytest.raises(LspError, match="LSP unavailable"):
        client.start()


def test_find_ty_argv_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent6.tools.lsp.shutil.which", lambda _name: None)  # type: ignore[misc]
    assert _find_ty_argv() is None


def test_find_ty_argv_prefers_ty_over_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    def which(name: str) -> str | None:
        return "/fake/ty" if name == "ty" else "/fake/uvx"

    monkeypatch.setattr("agent6.tools.lsp.shutil.which", which)
    argv = _find_ty_argv()
    assert argv is not None
    assert argv[0] == "/fake/ty"
    assert argv[1] == "server"


def test_dispatch_find_definition_lsp_no_server_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent6.tools.lsp.shutil.which", lambda _name: None)  # type: ignore[misc]
    cfg = _config(tmp_path)
    (tmp_path / "x.py").write_text("def foo(): pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="LSP unavailable"):
        d.dispatch("find_definition_lsp", {"path": "x.py", "symbol": "foo"})


def test_dispatch_find_definition_lsp_rejects_nonfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent6.tools.lsp.shutil.which", lambda _name: None)  # type: ignore[misc]
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Not a file"):
        d.dispatch("find_definition_lsp", {"path": "nope.py", "symbol": "foo"})


# --- end-to-end (skipped if ty not available) -------------------------


@pytest.mark.skipif(not _HAS_LSP, reason="requires `ty` or `uvx` on PATH")
def test_lsp_find_definition_e2e(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text("def helper():\n    return 1\n\nx = helper()\n", encoding="utf-8")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    try:
        result = d.dispatch(
            "find_definition_lsp",
            {"path": "mod.py", "symbol": "helper"},
        )
    finally:
        d.close()
    assert "definitions" in result
    defs = result["definitions"]
    assert len(defs) >= 1
    assert defs[0]["path"] == "mod.py"
    assert defs[0]["line"] == 1


@pytest.mark.skipif(not _HAS_LSP, reason="requires `ty` or `uvx` on PATH")
def test_lsp_find_references_e2e(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text("def helper():\n    return 1\n\nx = helper()\ny = helper()\n", encoding="utf-8")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    try:
        result = d.dispatch(
            "find_references_lsp",
            {"path": "mod.py", "symbol": "helper"},
        )
    finally:
        d.close()
    refs = result["references"]
    # ty returns the definition + both call sites (3 total).
    assert len(refs) >= 2
    lines = sorted(r["line"] for r in refs)
    assert 1 in lines  # definition site


@pytest.mark.skipif(not _HAS_LSP, reason="requires `ty` or `uvx` on PATH")
def test_lsp_missing_symbol_raises(tmp_path: Path) -> None:
    src = tmp_path / "mod.py"
    src.write_text("x = 1\n", encoding="utf-8")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    try:
        with pytest.raises(ToolError, match="not found"):
            d.dispatch(
                "find_definition_lsp",
                {"path": "mod.py", "symbol": "missing"},
            )
    finally:
        d.close()


# --- LspLocation dataclass --------------------------------------------


def test_lsp_location_is_frozen() -> None:
    loc = LspLocation(path=Path("x.py"), line=1, col=2)
    with pytest.raises(AttributeError):
        loc.line = 99  # type: ignore[misc]
