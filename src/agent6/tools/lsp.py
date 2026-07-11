# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Minimal stdio LSP client for Astral's ``ty`` server.

Powers the ``find_definition_lsp`` / ``find_references_lsp`` tools.
Trust posture: the subprocess runs in the agent process, **not** in
the jail. Argv is constant (``ty server`` or ``uvx ty server``); no
LLM-controlled arguments reach the spawn. The JSON-RPC stream is
constructed entirely from validated tool input (a path that's already
been ``_resolve_in_root``-checked and a symbol name we look up
verbatim). Same trust boundary as ``tools/index.py``.

The client is intentionally small: synchronous request/response,
single in-flight request at a time. The dispatcher serialises all
tool calls, so we never need a request queue.

Fail-open: if ``ty`` is not on PATH and ``uvx`` is also unavailable,
:meth:`LspClient.start` raises :class:`LspError`. The dispatcher
catches that and translates to a ``ToolError`` telling the agent to
fall back to the tree-sitter ``find_definition`` / ``find_references``
tools.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit
from urllib.request import pathname2url, url2pathname


class LspError(Exception):
    """LSP server is unavailable or a request failed."""


@dataclass(frozen=True, slots=True)
class LspLocation:
    """One result of a definition/references query (1-based line/col)."""

    path: Path
    line: int
    col: int


def _find_ty_argv() -> list[str] | None:
    """Locate the ``ty`` LSP binary. Returns argv or None.

    Prefers a direct ``ty`` on PATH; falls back to ``uvx ty`` if ``uvx``
    is available (uvx handles installation transparently on first use).
    """
    ty = shutil.which("ty")
    if ty is not None:
        return [ty, "server"]
    uvx = shutil.which("uvx")
    if uvx is not None:
        return [uvx, "ty", "server"]
    return None


def lsp_tools_useful(root: Path) -> bool:
    """True when the ``find_*_lsp`` tools can actually do something for *root*.

    The ``ty`` server is Python-only, so the tools are dead weight on a non-Python
    repo or when neither ``ty`` nor ``uvx`` is on PATH. Gating their exposure on
    this keeps two near-duplicate symbol tools (and their schema tokens) out of
    the model's tool list where they would only ever error. Cheap, shallow
    checks: the common project markers, then a top-level / src-level ``*.py``.
    """
    if _find_ty_argv() is None:
        return False
    if any((root / m).is_file() for m in ("pyproject.toml", "setup.py", "setup.cfg")):
        return True
    return any(next(root.glob(pat), None) is not None for pat in ("*.py", "*/*.py", "src/*/*.py"))


def _symbol_position(text: str, symbol: str) -> tuple[int, int] | None:
    """Return (line, character) of the best whole-word occurrence to anchor on.

    Both are 0-based (LSP convention). The LSP server only resolves
    definition/references for an identifier *use* in code, so a match in a
    comment, an import, or an inline `# ...` comment is a poor anchor. Prefer
    the first match in a code line; fall back to the very first match (better
    to anchor somewhere than return None). The ty server is Python-only, so the
    Python comment/import heuristics are appropriate. Returns None only when the
    symbol does not appear as an identifier at all.
    """
    pat = re.compile(rf"\b{re.escape(symbol)}\b")
    fallback: tuple[int, int] | None = None
    for line_idx, line in enumerate(text.splitlines()):
        m = pat.search(line)
        if m is None:
            continue
        if fallback is None:
            fallback = (line_idx, m.start())
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith(("import ", "from ")):
            continue  # whole-line comment or import: not a definition/use site
        hash_idx = line.find("#")
        if hash_idx != -1 and m.start() > hash_idx:
            continue  # match sits in an inline comment
        return line_idx, m.start()
    return fallback


def _uri_to_path(uri: str) -> Path:
    """Convert ``file://...`` URI to a Path. Best-effort, stdlib-only.

    Percent-decodes the path so a server-returned URI for a workspace path
    containing a space or other special char (LSP encodes these as ``%XX``)
    round-trips to the real filesystem path instead of a literal ``%20`` that
    would fail ``relative_to(root)`` and be silently dropped.
    """
    if uri.startswith("file://"):
        return Path(url2pathname(urlsplit(uri).path))
    return Path(uri)


def _path_to_uri(path: Path) -> str:
    return "file://" + pathname2url(str(path))


class LspClient:
    """Single-language (Python via ``ty``) stdio LSP client.

    Not thread-safe across requests; callers must serialise.
    """

    _REQUEST_TIMEOUT_S = 15.0
    _START_TIMEOUT_S = 10.0

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._proc: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._open_versions: dict[str, int] = {}
        self._lock = threading.Lock()
        # A background reader drains stdout into `_responses` (keyed by request
        # id) and notifies `_resp_cond`. `_send_request` then waits on the
        # condition with a wall-clock deadline -- so a server that accepts a
        # request and then HANGS (never sends the body) trips the timeout
        # instead of blocking the agent forever in a synchronous read.
        self._responses: dict[int, dict[str, Any]] = {}
        self._resp_cond = threading.Condition()
        self._reader: threading.Thread | None = None
        self._reader_stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle

    def start(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                return  # already running
            # The server died (crashed or was killed). _query calls start()
            # before every request, so tear the corpse down and respawn instead
            # of leaving the LSP tools broken for the rest of the run.
            self.close()
        argv = _find_ty_argv()
        if argv is None:
            raise LspError(
                "LSP unavailable: `ty` is not on PATH and `uvx` is also"
                " not available. Use `find_definition` instead."
            )
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self._root),
                # Its own session: a terminal Ctrl-C signals the foreground
                # process group, and a language server that dies with it breaks
                # the symbol tools for the rest of the run.
                start_new_session=True,
            )
        except OSError as exc:
            raise LspError(f"LSP unavailable: failed to spawn {argv[0]}: {exc}") from exc
        self._reader_stop.clear()
        self._responses.clear()
        self._reader = threading.Thread(
            target=self._reader_loop, args=(self._proc.stdout,), daemon=True
        )
        self._reader.start()
        try:
            self._initialize()
        except LspError:
            self.close()
            raise

    def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._reader_stop.set()
        with contextlib.suppress(Exception):
            self._send_notification("exit", None)
        self._proc = None
        with contextlib.suppress(OSError):
            if proc.stdin is not None:
                proc.stdin.close()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2.0)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
        # Wake any in-flight _send_request and join the reader (stdout has EOF'd
        # now that the process is gone).
        with self._resp_cond:
            self._resp_cond.notify_all()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        self._open_versions.clear()

    # ------------------------------------------------------------------
    # Public queries

    def find_definition(self, path: Path, symbol: str) -> list[LspLocation]:
        return self._query(path, symbol, "textDocument/definition", extra_params={})

    def find_references(self, path: Path, symbol: str) -> list[LspLocation]:
        return self._query(
            path,
            symbol,
            "textDocument/references",
            extra_params={"context": {"includeDeclaration": True}},
        )

    # ------------------------------------------------------------------
    # Internals

    def _query(
        self,
        path: Path,
        symbol: str,
        method: str,
        *,
        extra_params: dict[str, Any],
    ) -> list[LspLocation]:
        with self._lock:
            self.start()
            assert self._proc is not None  # for type-checker
            abs_path = path.resolve()
            try:
                text = abs_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise LspError(f"cannot read {path}: {exc}") from exc
            pos = _symbol_position(text, symbol)
            if pos is None:
                raise LspError(f"symbol {symbol!r} not found in {path}")
            line, char = pos
            uri = _path_to_uri(abs_path)
            self._ensure_open(uri, text)
            params: dict[str, Any] = {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": char},
                **extra_params,
            }
            result = self._send_request(method, params)
        return self._parse_locations(result)

    def _initialize(self) -> None:
        params = {
            "processId": None,
            "rootUri": _path_to_uri(self._root),
            "capabilities": {
                "textDocument": {
                    "definition": {"linkSupport": False},
                    "references": {},
                    "synchronization": {"didSave": False, "willSave": False},
                },
            },
            "workspaceFolders": [
                {"uri": _path_to_uri(self._root), "name": self._root.name},
            ],
        }
        self._send_request("initialize", params, timeout_s=self._START_TIMEOUT_S)
        self._send_notification("initialized", {})

    def _ensure_open(self, uri: str, text: str) -> None:
        if uri in self._open_versions:
            # Send didChange with the current text so the server stays
            # in sync if the worktree mutated since the last query.
            version = self._open_versions[uri] + 1
            self._open_versions[uri] = version
            self._send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                },
            )
            return
        self._open_versions[uri] = 1
        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": text,
                }
            },
        )

    def _parse_locations(self, result: Any) -> list[LspLocation]:
        # LSP returns Location | Location[] | LocationLink[] | null.
        if result is None:
            return []
        if isinstance(result, dict):
            result = [result]
        if not isinstance(result, list):
            return []
        out: list[LspLocation] = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            # LocationLink uses targetUri/targetSelectionRange; Location
            # uses uri/range. Handle both.
            uri = entry.get("uri") or entry.get("targetUri")
            rng = entry.get("range") or entry.get("targetSelectionRange")
            if not isinstance(uri, str) or not isinstance(rng, dict):
                continue
            start = rng.get("start")
            if not isinstance(start, dict):
                continue
            line = start.get("line")
            char = start.get("character")
            if not isinstance(line, int) or not isinstance(char, int):
                continue
            out.append(LspLocation(path=_uri_to_path(uri), line=line + 1, col=char + 1))
        return out

    # ------------------------------------------------------------------
    # JSON-RPC framing

    def _send_request(
        self,
        method: str,
        params: Any,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        req_id = self._next_id
        self._next_id += 1
        deadline_s = timeout_s if timeout_s is not None else self._REQUEST_TIMEOUT_S
        self._write_message({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        # Wait for the reader thread to surface our response, enforcing the
        # wall-clock deadline via the condition (so a hung server can't block us
        # forever -- the blocking read lives in the reader thread, not here).
        deadline = time.monotonic() + deadline_s
        with self._resp_cond:
            while req_id not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LspError(f"LSP request {method!r} timed out after {deadline_s}s")
                if self._reader is None or not self._reader.is_alive():
                    raise LspError(f"LSP server closed stdout during {method!r}")
                self._resp_cond.wait(timeout=remaining)
            reply = self._responses.pop(req_id)
        if "error" in reply:
            raise LspError(f"LSP {method!r} error: {reply['error']}")
        return reply.get("result")

    def _reader_loop(self, stdout: IO[bytes]) -> None:
        """Drain stdout into `_responses`. Runs in a daemon thread so the
        synchronous framing reads can block without stalling the caller."""
        while not self._reader_stop.is_set():
            try:
                msg = self._read_message_blocking(stdout)
            except LspError:
                break  # framing error -> stop reading
            if msg is None:
                break  # EOF
            rid = msg.get("id")
            # Only responses to OUR requests (an id plus result/error, no
            # method). Server-initiated requests/notifications are discarded.
            if isinstance(rid, int) and "method" not in msg:
                with self._resp_cond:
                    self._responses[rid] = msg
                    self._resp_cond.notify_all()
        with self._resp_cond:  # wake any waiter so it can fail fast on EOF
            self._resp_cond.notify_all()

    def _send_notification(self, method: str, params: Any) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write_message(msg)

    def _write_message(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LspError("LSP server not running")
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        try:
            self._proc.stdin.write(header + body)
            self._proc.stdin.flush()
        except OSError as exc:
            raise LspError(f"LSP write failed: {exc}") from exc

    def _read_message_blocking(self, stdout: IO[bytes]) -> dict[str, Any] | None:
        content_length: int | None = None
        while True:
            line = stdout.readline()
            if not line:
                return None
            stripped = line.strip()
            if not stripped:
                break
            header = stripped.decode("ascii", errors="replace")
            name, _, value = header.partition(":")
            if name.strip().lower() == "content-length":
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise LspError(f"LSP bad Content-Length: {value!r}") from exc
        if content_length is None:
            raise LspError("LSP message missing Content-Length header")
        body = stdout.read(content_length)
        if len(body) != content_length:
            raise LspError(f"LSP short read: wanted {content_length} bytes, got {len(body)}")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LspError(f"LSP bad JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LspError(f"LSP message is not an object: {type(parsed).__name__}")
        return parsed
