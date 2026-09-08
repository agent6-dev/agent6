# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 check mcp` starts each server as a run does.

A probe in a throwaway directory failed every server whose script lives in
the workspace ("can't open file ... No such file or directory", even by
absolute path under strict), while `mcp connect` and a real run started it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app._setup import MCPServerSpec
from agent6.config import Config
from agent6.sandbox.jail import JailUnavailableError, SessionNetwork
from agent6.tools.mcp_client import MCPToolDescriptor
from agent6.ui.cli import check_cmds

# The interpreter a jailed probe can reach: the run's sandbox grants /usr,
# not the venv.
_JAIL_PYTHON = "/usr/bin/python3"

_SERVER = (
    "import json,sys\n"
    "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
    "for line in sys.stdin:\n"
    "    m=json.loads(line)\n"
    "    if m.get('method')=='initialize':\n"
    "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
    "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
    "    elif m.get('method')=='tools/list':\n"
    "        w({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[{'name':'ping',"
    "'inputSchema':{'type':'object'}}]}})\n"
)


def test_an_http_only_mcp_check_does_not_create_a_command_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checking an HTTP client needs no namespace for command tools the check never starts."""

    class _Manager:
        def __init__(self) -> None:
            self.networks = {"remote": "host"}
            self.failures: tuple[()] = ()

        def descriptors(self) -> list[MCPToolDescriptor]:
            return [MCPToolDescriptor("remote", "ping", "", {})]

        def close(self) -> None:
            return None

    def _strict(_requested: str, _env: object) -> str:
        return "strict"

    def _start(
        _specs: list[MCPServerSpec], *, session_net: SessionNetwork | None = None
    ) -> _Manager:
        assert session_net is None
        return _Manager()

    def _unexpected_network() -> SessionNetwork:
        raise AssertionError("HTTP-only check created a command session network")

    monkeypatch.setattr(check_cmds, "detect_env", object)
    monkeypatch.setattr(check_cmds, "resolve_isolation", _strict)
    monkeypatch.setattr(check_cmds.MCPManager, "start", staticmethod(_start))
    monkeypatch.setattr(check_cmds.SessionNetwork, "open", staticmethod(_unexpected_network))
    cfg = Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"remote": {"url": "https://mcp.example"}}}}
    )

    checks = check_cmds._doctor_check_mcp(cfg)  # pyright: ignore[reportPrivateUsage]

    assert [(check.name, check.status) for check in checks] == [("mcp.remote", "PASS")]


def test_mcp_check_reports_a_session_network_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network holder failure is a failed preflight row, not a traceback."""

    def _strict(_requested: str, _env: object) -> str:
        return "strict"

    def _refuse_network() -> SessionNetwork:
        raise JailUnavailableError("the session network could not be created: denied")

    monkeypatch.setattr(check_cmds, "detect_env", object)
    monkeypatch.setattr(check_cmds, "resolve_isolation", _strict)
    monkeypatch.setattr(check_cmds.SessionNetwork, "open", staticmethod(_refuse_network))
    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {
                    "notes": {
                        "command": ["notes-mcp"],
                        "sandbox": {"network": "session"},
                    }
                },
            }
        }
    )

    checks = check_cmds._doctor_check_mcp(cfg)  # pyright: ignore[reportPrivateUsage]

    assert checks[-1].name == "mcp"
    assert checks[-1].status == "FAIL"
    assert "session network could not be created" in checks[-1].detail


@pytest.mark.needs_namespaces
def test_a_server_script_inside_the_workspace_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The workspace root of the check is the repository, as a run's is: a
    relative script path resolves there on every isolation level."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "server.py").write_text(_SERVER, encoding="utf-8")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    cfg = Config.model_validate(
        {"mcp": {"enabled": True, "servers": {"inrepo": {"command": [_JAIL_PYTHON, "server.py"]}}}}
    )

    checks = check_cmds._doctor_check_mcp(cfg)  # pyright: ignore[reportPrivateUsage]

    assert [(c.name, c.status) for c in checks] == [("mcp.inrepo", "PASS")], checks
    assert "1 tool," in checks[0].detail
    assert "inrepo: 1 tool," in capsys.readouterr().out
