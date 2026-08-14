# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check` must not mutate the repo through MCP startup.

The doctor started each configured MCP server in the repo cwd with the ordinary
writable workspace, so a server that wrote a file during its `initialize` (a
cache, a log) left it in the repo while `check mcp` reported PASS -- a diagnostic
whose docstring promises it "never writes to the repo". The checkable servers
now start in a throwaway directory, and a server that would need write access to
start (unconfined, or a write grant) is reported and left unstarted.

Forced to `none` isolation so the pin is portable and hits the most exposed
case: with no jail, only WHERE the process runs (its cwd) keeps a startup write
off the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.ui.cli import check_cmds

# A well-behaved (not hostile) stdio MCP server that drops a file in its cwd on
# `initialize` -- a cache, say -- then answers initialize + tools/list.
_WRITER_SERVER = (
    "import json,sys,pathlib\n"
    "pathlib.Path('wrote-during-check.txt').write_text('x', encoding='utf-8')\n"
    "def w(o):\n"
    "    sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
    "for line in sys.stdin:\n"
    "    m=json.loads(line); mid=m.get('id')\n"
    "    if m.get('method')=='initialize':\n"
    "        w({'jsonrpc':'2.0','id':mid,'result':{'protocolVersion':'2024-11-05',"
    "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
    "    elif m.get('method')=='tools/list':\n"
    "        w({'jsonrpc':'2.0','id':mid,'result':{'tools':[{'name':'ping','inputSchema':{}}]}})\n"
)


def _check_mcp(
    servers: dict[str, object], repo: Path, monkeypatch: pytest.MonkeyPatch
) -> list[check_cmds._DoctorCheck]:  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("AGENT6_DANGEROUSLY_DISABLE_SANDBOX", "1")  # force none: portable
    monkeypatch.chdir(repo)
    cfg = Config.model_validate({"mcp": {"enabled": True, "servers": servers}})
    return check_cmds._doctor_check_mcp(cfg)  # pyright: ignore[reportPrivateUsage]


def test_a_startup_write_lands_off_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checks = _check_mcp(
        {"notes": {"command": [sys.executable, "-c", _WRITER_SERVER]}}, repo, monkeypatch
    )
    # The write went to the throwaway cwd, not the repo...
    assert list(repo.iterdir()) == [], "check wrote into the repo through MCP startup"
    # ...and the server was still verified, so the read-only check is not theatre.
    assert [(c.name, c.status) for c in checks] == [("mcp.notes", "PASS")]
    assert "1 tool" in checks[0].detail


def test_a_server_it_cannot_confine_read_only_is_reported_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checks = _check_mcp(
        {
            "unconf": {
                "command": [sys.executable, "-c", _WRITER_SERVER],
                "sandbox": {"unconfined": True},
            },
            "writer": {
                "command": [sys.executable, "-c", _WRITER_SERVER],
                "sandbox": {"write_paths": ["/srv/data"]},
            },
        },
        repo,
        monkeypatch,
    )
    by = {c.name: c for c in checks}
    # Neither is verified as PASS, and each says why it was left unstarted.
    assert by["mcp.unconf"].status == "INFO" and "unconfined" in by["mcp.unconf"].detail
    assert by["mcp.writer"].status == "INFO" and "write" in by["mcp.writer"].detail
    # Refused means not spawned, so neither server's startup write happened anywhere.
    assert list(repo.iterdir()) == []
