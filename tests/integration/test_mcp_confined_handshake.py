# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A CONFINED MCP server, end to end: spawn, handshake, call a tool, close.

The seam nothing else covers. `test_mcp_client.py` handshakes an unconfined
server; `test_mcp_network.py` asserts the argv the confinement builds without
running it; `tests/security/test_mcp_network_confinement.py` runs the shim
alone and inspects its namespaces. Whether a server actually WORKS through
confinement -- reads what it was granted, is denied what it was not, and keeps
its JSON-RPC pipe alive across the confinement boundary -- was untested.

Pinned here because the confinement mechanism is being reworked: these
assertions are about the contract (a granted path is readable, an ungranted
one is not, tools still answer), never the mechanism, so they hold across the
change and fail loudly if it breaks the pipe.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.policy import jail_policy
from agent6.types import JailPolicy, NetworkMode

pytestmark = pytest.mark.needs_namespaces


def _landlock_available() -> bool:
    from agent6.sandbox.landlock import landlock_abi

    try:
        return landlock_abi() >= 1
    except Exception:
        return False


def _reader_server_argv(probe: Path) -> tuple[str, ...]:
    """A minimal MCP server exposing one tool: read the file it is asked for,
    and report what happened. Enough to prove the pipe survives confinement
    AND to observe the filesystem boundary from inside the server."""
    script = textwrap.dedent(
        """
        import json, sys
        def reply(i, result):
            sys.stdout.write(json.dumps({"jsonrpc":"2.0","id":i,"result":result}) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method, mid = msg.get("method"), msg.get("id")
            if method == "initialize":
                reply(mid, {"protocolVersion": "2024-11-05", "capabilities": {},
                            "serverInfo": {"name": "reader", "version": "1"}})
            elif method == "tools/list":
                reply(mid, {"tools": [{"name": "cat", "description": "read a file",
                                       "inputSchema": {"type": "object",
                                                       "properties": {"path": {"type": "string"}},
                                                       "required": ["path"]}}]})
            elif method == "tools/call":
                path = msg["params"]["arguments"]["path"]
                try:
                    text = open(path).read()
                except OSError as exc:
                    text = f"DENIED:{type(exc).__name__}"
                reply(mid, {"content": [{"type": "text", "text": text}]})
            elif mid is not None:
                reply(mid, {})
        """
    )
    _ = probe
    # The SYSTEM python, not sys.executable: a jailed command's binary has to
    # exist inside the assembled root, and a venv interpreter in some other
    # checkout does not. Real servers are `npx`/`node`/`python3` for the same
    # reason -- found on the jail's PATH, or granted explicitly.
    return ("/usr/bin/python3", "-c", script)


def _call_cat(mgr: MCPManager, path: Path) -> str:
    return str(mgr.call("mcp__reader__cat", {"path": str(path)}))


def _policy(
    argv: tuple[str, ...], cwd: Path, *, read: tuple[Path, ...] = (), net: NetworkMode = "none"
) -> JailPolicy:
    """A server policy exactly as production builds it: the same sandbox a
    jailed command gets, plus this server's additive grants. Nothing here
    names an interpreter -- that is the point of the shared base."""
    return jail_policy(
        cwd,
        Config(),
        "strict",
        argv,
        extra_ro_paths=read,
        network=net,
    )


@pytest.fixture
def granted(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(workspace, granted file, ungranted file).

    The ungranted file lives OUTSIDE the workspace on purpose. An earlier
    version put it in a sibling directory under the workspace and passed --
    but only because the workspace was remapped to /workspace back then, so
    the host path did not resolve. The file was reachable the whole time, at
    a different spelling. A boundary test must not be able to pass because of
    a path alias."""
    ws = tmp_path / "ws"
    ws.mkdir()
    ok = tmp_path / "granted"
    ok.mkdir()
    (ok / "note.txt").write_text("VISIBLE\n", encoding="utf-8")
    secret = tmp_path / "ungranted"
    secret.mkdir()
    (secret / "secret.txt").write_text("HIDDEN\n", encoding="utf-8")
    return ws, ok / "note.txt", secret / "secret.txt"


def test_a_confined_server_handshakes_serves_and_respects_its_grants(
    granted: tuple[Path, Path, Path],
) -> None:
    """The contract, whatever applies it: the JSON-RPC pipe survives the
    confinement boundary (initialize + tools/list + tools/call all answer),
    a granted path reads, and an ungranted one does not."""
    if not _landlock_available():
        pytest.skip("no Landlock on this kernel")
    ws, visible, hidden = granted
    # The interpreter and its stdlib have to be readable or the server cannot
    # start at all -- the reason read_paths is required for a filesystem block.
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="reader",
                command=_reader_server_argv(visible),
                startup_timeout_s=20.0,
                call_timeout_s=20.0,
                policy=_policy(_reader_server_argv(visible), ws, read=(visible.parent,)),
            )
        ]
    )
    try:
        assert [d.qualified_name for d in mgr.descriptors()] == ["mcp__reader__cat"]
        assert "VISIBLE" in _call_cat(mgr, visible)
        assert "DENIED" in _call_cat(mgr, hidden)
    finally:
        mgr.close()


def test_closing_the_manager_leaves_no_confined_server_running(
    granted: tuple[Path, Path, Path],
) -> None:
    """A confinement wrapper adds a process between agent6 and the server, so
    the teardown has to reach through it: a leaked server holds the pipe and
    outlives the run."""
    if not _landlock_available():
        pytest.skip("no Landlock on this kernel")
    ws, visible, _hidden = granted
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="reader",
                command=_reader_server_argv(visible),
                startup_timeout_s=20.0,
                call_timeout_s=20.0,
                policy=_policy(_reader_server_argv(visible), ws),
            )
        ]
    )
    assert mgr.descriptors()
    mgr.close()
    survivors = subprocess.run(
        ["pgrep", "-af", "reader"], capture_output=True, text=True, check=False
    ).stdout
    assert "protocolVersion" not in survivors
