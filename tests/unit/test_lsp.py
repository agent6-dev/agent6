# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the LSP client + dispatcher ``*_lsp`` tools.

The smoke tests that actually spawn ``ty`` are skipped when the binary
is not available so the suite stays portable. Pure-logic tests
(fail-open, symbol-position resolution, parsing) always run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import cast

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
api_format = "anthropic"
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
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
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


def test_symbol_position_prefers_code_over_comment_and_import() -> None:
    # First textual hits are a comment + an import; the code use on line 3 is
    # the right anchor for an LSP definition/reference request.
    text = "# foo is great\nfrom mod import foo\nfoo()\n"
    assert _symbol_position(text, "foo") == (2, 0)


def test_symbol_position_skips_inline_comment_match() -> None:
    text = "y = 1  # foo here\nfoo = 2\n"
    assert _symbol_position(text, "foo") == (1, 0)


def test_symbol_position_falls_back_to_comment_only_match() -> None:
    # If the symbol appears ONLY in a comment, still anchor there (better than
    # None) rather than silently failing.
    assert _symbol_position("# mentions foo only\n", "foo") == (0, 11)


def test_path_to_uri_and_back(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("", encoding="utf-8")
    uri = _path_to_uri(p)
    assert uri.startswith("file://")
    assert _uri_to_path(uri) == p


def test_uri_round_trips_path_with_space() -> None:
    # A workspace path with a space encodes to %20 in the URI (LSP spec). Without
    # percent-decoding, _uri_to_path returned a literal "%20" path that failed
    # relative_to(root) and the location was silently dropped.
    p = Path("/tmp/my project/mod.py")
    uri = _path_to_uri(p)
    assert "%20" in uri  # the space is percent-encoded on the wire
    assert _uri_to_path(uri) == p  # and decodes back to the real path
    # A server that emits a pre-encoded file:// URI also decodes correctly.
    assert _uri_to_path("file:///tmp/my%20project/mod.py") == p


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


def test_query_non_utf8_file_raises_clean_lsp_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-UTF-8 source raises UnicodeDecodeError -- a ValueError the
    OSError-only read guard missed, leaking an opaque codec error through the
    generic dispatch handler. It must become the subsystem LspError with the
    same "not UTF-8 text" wording read_file uses for the identical file."""
    bad = tmp_path / "latin.py"
    bad.write_bytes(b"x = 'caf\xe9'\n")
    client = LspClient(tmp_path)

    def fake_start(self: LspClient) -> None:
        self._proc = cast("subprocess.Popen[bytes]", object())  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(LspClient, "start", fake_start)
    with pytest.raises(LspError, match="not UTF-8"):
        client.find_definition(bad, "x")


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
        ).to_wire()
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
        ).to_wire()
    finally:
        d.close()
    refs = result["references"]
    # ty returns the definition + both call sites (3 total).
    assert len(refs) >= 2
    lines = sorted(r["line"] for r in refs)
    assert 1 in lines  # definition site


@pytest.mark.skipif(not _HAS_LSP, reason="requires `ty` or `uvx` on PATH")
def test_lsp_respawns_after_server_dies(tmp_path: Path) -> None:
    """If the ty server dies mid-run, the next query must respawn it. _query calls
    start() before every request; pre-fix start() returned early because _proc was
    still a (dead) object, so the LSP tools stayed broken for the rest of the run."""
    src = tmp_path / "mod.py"
    src.write_text("def helper():\n    return 1\n\nx = helper()\n", encoding="utf-8")
    client = LspClient(tmp_path)
    try:
        first = client.find_definition(src, "helper")
        assert len(first) >= 1
        # Simulate a crash: kill the server out from under the client.
        assert client._proc is not None  # pyright: ignore[reportPrivateUsage]
        client._proc.kill()  # pyright: ignore[reportPrivateUsage]
        client._proc.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]
        # The next query must transparently respawn and still work.
        again = client.find_definition(src, "helper")
        assert len(again) >= 1
        assert again[0].line == 1
    finally:
        client.close()


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


def _hung_server_argv() -> list[str]:
    # Sends a Content-Length header then never the body -> a blocking read in
    # the old code hung forever; the reader-thread + deadline must time out.
    import sys

    return [
        sys.executable,
        "-c",
        "import sys,time;sys.stdout.buffer.write(b'Content-Length: 100\\r\\n\\r\\n');"
        "sys.stdout.buffer.flush();time.sleep(30)",
    ]


def test_lsp_request_times_out_on_hung_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import time

    from agent6.tools import lsp as lspmod

    monkeypatch.setattr(lspmod, "_find_ty_argv", _hung_server_argv)
    client = LspClient(tmp_path)
    client._START_TIMEOUT_S = 0.6  # pyright: ignore[reportPrivateUsage]
    started = time.monotonic()
    with pytest.raises(LspError, match="timed out"):
        client.start()  # initialize never gets its response
    assert time.monotonic() - started < 5.0  # deadline enforced, did not hang
    client.close()


# --- lsp_tools_useful gating ------------------------------------------------


def test_lsp_tools_useful_false_without_ty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent6.tools import lsp as lspmod
    from agent6.tools.lsp import lsp_tools_useful

    monkeypatch.setattr(lspmod, "_find_ty_argv", lambda: None)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert lsp_tools_useful(tmp_path) is False  # no ty/uvx -> dead tools


def test_lsp_tools_useful_true_for_python_with_ty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.tools import lsp as lspmod
    from agent6.tools.lsp import lsp_tools_useful

    monkeypatch.setattr(lspmod, "_find_ty_argv", lambda: ["ty", "server"])
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert lsp_tools_useful(tmp_path) is True


def test_lsp_tools_useful_false_for_nonpython_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.tools import lsp as lspmod
    from agent6.tools.lsp import lsp_tools_useful

    monkeypatch.setattr(lspmod, "_find_ty_argv", lambda: ["ty", "server"])
    (tmp_path / "main.go").write_text("package main\n", encoding="utf-8")
    assert lsp_tools_useful(tmp_path) is False  # ty present but no Python


def test_lsp_tools_hidden_from_available_when_not_useful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr("agent6.tools.dispatch.lsp_tools_useful", lambda _root: False)  # type: ignore[misc]
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = set(d.available_tool_names())
    assert "find_definition_lsp" not in names
    assert "find_references_lsp" not in names
    # The tree-sitter symbol tools are never gated this way.
    assert "find_definition" in names

    monkeypatch.setattr("agent6.tools.dispatch.lsp_tools_useful", lambda _root: True)  # type: ignore[misc]
    d2 = ToolDispatcher(root=tmp_path, config=cfg)
    assert {"find_definition_lsp", "find_references_lsp"} <= set(d2.available_tool_names())
