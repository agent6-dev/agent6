# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""In-house LSP JSON-RPC client.

Spawns `pyright-langserver --stdio`, speaks the subset of the LSP we need
(initialize, initialized, textDocument/didOpen, textDocument/definition,
textDocument/references, shutdown, exit), and exposes two repo-level helpers:

    definition(path, line, character) -> list[Location]
    references(path, line, character, include_decl=False) -> list[Location]

Plus a regex-based `outline(path)` fallback that returns top-level `def` /
`class` lines for Python files — used by the worker for cheap symbol context
without taking a tree-sitter runtime dep.

This module is NOT yet exposed as an LLM-visible tool. Exposing it via
`agent6.tools.dispatch.ToolDispatcher` requires an `agent6-security-review:`
trailer per AGENTS.md, and a careful audit of the path inputs.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

_HEADER_TERMINATOR = b"\r\n\r\n"
_CONTENT_LENGTH = b"Content-Length:"


class LspError(RuntimeError):
    """Raised when the language server returns an error or misbehaves."""


@dataclass(frozen=True, slots=True)
class Location:
    """A 0-indexed LSP location, with absolute filesystem path."""

    path: Path
    line: int
    character: int
    end_line: int
    end_character: int


def _path_to_uri(path: Path) -> str:
    # LSP wants file:// URIs; pathlib's as_uri is fine on POSIX.
    return path.resolve().as_uri()


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise LspError(f"non-file URI from server: {uri!r}")
    return Path(unquote(parsed.path))


def _location_from_lsp(obj: dict[str, Any]) -> Location:
    rng = obj["range"]
    start = rng["start"]
    end = rng["end"]
    return Location(
        path=_uri_to_path(obj["uri"]),
        line=int(start["line"]),
        character=int(start["character"]),
        end_line=int(end["line"]),
        end_character=int(end["character"]),
    )


class LspClient:
    """Minimal LSP client over stdio. Single-threaded request/response."""

    def __init__(self, proc: subprocess.Popen[bytes], root: Path) -> None:
        self._proc = proc
        self._root = root.resolve()
        self._next_id = 1
        self._lock = threading.Lock()
        self._opened: set[Path] = set()
        # Drain stderr in a background thread so the server can't block on it.
        self._stderr_buf: list[bytes] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="lsp-stderr"
        )
        self._stderr_thread.start()

    @classmethod
    def start(cls, root: Path, server_argv: list[str] | None = None) -> LspClient:
        argv = server_argv if server_argv is not None else ["pyright-langserver", "--stdio"]
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(root),
        )
        client = cls(proc, root)
        client._initialize()
        return client

    # ------------------------------------------------------------------
    # Wire format
    # ------------------------------------------------------------------

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_buf.append(line)

    def _write_message(self, payload: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)
        self._proc.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        assert self._proc.stdout is not None
        # Parse LSP headers (terminated by \r\n\r\n).
        header_bytes = bytearray()
        while not header_bytes.endswith(_HEADER_TERMINATOR):
            byte = self._proc.stdout.read(1)
            if not byte:
                raise LspError("language server closed stdout while reading headers")
            header_bytes.extend(byte)
        content_length: int | None = None
        for raw_line in header_bytes.split(b"\r\n"):
            if raw_line.lower().startswith(_CONTENT_LENGTH.lower()):
                content_length = int(raw_line.split(b":", 1)[1].strip())
        if content_length is None:
            raise LspError("missing Content-Length header from language server")
        body = self._proc.stdout.read(content_length)
        if len(body) != content_length:
            raise LspError("short read on language server body")
        return json.loads(body.decode("utf-8"))

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._write_message(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            # Drain notifications until we see our response.
            while True:
                msg = self._read_message()
                if "id" in msg and msg.get("id") == request_id:
                    if "error" in msg:
                        raise LspError(f"{method}: {msg['error']}")
                    return msg.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    # ------------------------------------------------------------------
    # LSP lifecycle
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        self._request(
            "initialize",
            {
                "processId": None,
                "rootUri": _path_to_uri(self._root),
                "capabilities": {},
                "workspaceFolders": [
                    {"uri": _path_to_uri(self._root), "name": self._root.name},
                ],
            },
        )
        self._notify("initialized", {})

    def shutdown(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._request("shutdown", {})
            self._notify("exit", {})
        except (LspError, OSError, BrokenPipeError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)

    # ------------------------------------------------------------------
    # Document state
    # ------------------------------------------------------------------

    def _ensure_open(self, path: Path) -> None:
        if path in self._opened:
            return
        text = path.read_text(encoding="utf-8")
        language_id = "python" if path.suffix == ".py" else "plaintext"
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": _path_to_uri(path),
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )
        self._opened.add(path)

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def definition(self, path: Path, line: int, character: int) -> list[Location]:
        path = path.resolve()
        self._ensure_open(path)
        result = self._request(
            "textDocument/definition",
            {
                "textDocument": {"uri": _path_to_uri(path)},
                "position": {"line": line, "character": character},
            },
        )
        return _normalize_locations(result)

    def references(
        self, path: Path, line: int, character: int, *, include_decl: bool = False
    ) -> list[Location]:
        path = path.resolve()
        self._ensure_open(path)
        result = self._request(
            "textDocument/references",
            {
                "textDocument": {"uri": _path_to_uri(path)},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": include_decl},
            },
        )
        return _normalize_locations(result)


def _normalize_locations(result: Any) -> list[Location]:
    if result is None:
        return []
    if isinstance(result, dict):
        return [_location_from_lsp(result)]
    if isinstance(result, list):
        out: list[Location] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            # LocationLink has targetUri+targetRange; Location has uri+range.
            if "targetUri" in item and "targetRange" in item:
                out.append(
                    _location_from_lsp({"uri": item["targetUri"], "range": item["targetRange"]})
                )
            elif "uri" in item and "range" in item:
                out.append(_location_from_lsp(item))
        return out
    return []


# ---------------------------------------------------------------------------
# Regex-based outline fallback (no tree-sitter dep)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """A top-level symbol declaration in a Python file. 0-indexed line."""

    kind: str  # "def" or "class"
    name: str
    line: int


_PY_TOP_LEVEL = re.compile(r"^(?P<kind>def|class|async\s+def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")


def outline(path: Path) -> list[OutlineEntry]:
    """Return top-level (column-0) `def`/`class` declarations in a Python file.

    Cheap, dependency-free symbol outline. Skips nested defs and decorators.
    For non-Python files returns []. Caller is responsible for the path being
    inside the repo.
    """
    if path.suffix != ".py":
        return []
    out: list[OutlineEntry] = []
    text = path.read_text(encoding="utf-8")
    for idx, line in enumerate(text.splitlines()):
        # Only column-0 declarations count as top-level.
        if not line or line[0] in " \t":
            continue
        match = _PY_TOP_LEVEL.match(line)
        if match is None:
            continue
        kind_raw = match.group("kind")
        kind = "def" if kind_raw.endswith("def") else "class"
        out.append(OutlineEntry(kind=kind, name=match.group("name"), line=idx))
    return out


# Re-export for convenience.
__all__ = [
    "Location",
    "LspClient",
    "LspError",
    "OutlineEntry",
    "outline",
]


# Silence "unused" lint on quote (kept for callers that build URIs manually).
_ = quote
