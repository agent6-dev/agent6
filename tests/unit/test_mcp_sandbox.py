# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem confinement for a SPAWNED MCP server.

An MCP server is third-party code running as the operator, with their whole
filesystem. Naming paths opts into a Landlock domain the server and everything
it spawns inherit.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

import pytest

from agent6.app.reporter import Reporter
from agent6.config import Config
from agent6.tools.mcp_client import _confined_argv  # pyright: ignore[reportPrivateUsage]

_SHIM = [sys.executable, "-m", "agent6.sandbox.exec_confined"]


def test_a_server_without_a_block_is_spawned_unchanged() -> None:
    """Absent means unconfined: agent6 cannot know what a given server needs,
    and a guess that breaks it is worse than none."""
    assert _confined_argv(("npx", "server"), None) == ("npx", "server")


def test_a_confined_server_is_wrapped_in_the_shim() -> None:
    """A shim, not `preexec_fn`: Landlock is restrict-self-then-exec and
    inherited across the exec, and preexec_fn is unsafe in a threaded process
    -- which the MCP client is."""
    argv = _confined_argv(("npx", "server"), (("/usr", "/etc"), ("/tmp",), False))
    assert list(argv) == [
        *_SHIM,
        "--read",
        "/usr",
        "--read",
        "/etc",
        "--write",
        "/tmp",
        "--",
        "npx",
        "server",
    ]


def test_require_is_passed_through() -> None:
    argv = _confined_argv(("x",), ((), ("/tmp",), True))
    assert "--require" in argv


@pytest.mark.needs_namespaces
def test_the_shim_really_confines_what_it_execs(tmp_path: Path) -> None:
    """End to end: the domain is applied before the exec, so the server it
    becomes inherits it."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "key").write_text("sk-DECOY\n", encoding="utf-8")

    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "p = Path(sys.argv[1])\n"
        "print(p.read_text().strip() if p.exists() else 'unreadable')\n"
    )
    res = subprocess.run(
        [
            *_SHIM,
            "--read",
            "/usr",
            "--read",
            "/lib",
            "--read",
            "/lib64",
            "--read",
            str(allowed),
            "--write",
            str(allowed),
            "--",
            sys.executable,
            "-c",
            probe,
            str(secret / "key"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "sk-DECOY" not in res.stdout, "the shim did not confine what it exec'd"


def test_a_block_with_no_read_paths_is_refused() -> None:
    """Landlock grants READ and EXECUTE together, so a server with no read path
    cannot reach its own interpreter and dies on startup with an import error
    that says nothing about the sandbox."""
    with pytest.raises(ValueError, match="read_paths is required"):
        Config.model_validate(
            {"mcp": {"enabled": True, "servers": {"s": {"command": ["x"], "sandbox": {}}}}}
        )


@pytest.mark.parametrize("path", ["rel/path", "./x", "x"])
def test_a_relative_sandbox_path_is_refused(path: str) -> None:
    """It would resolve against whatever directory agent6 started in."""
    with pytest.raises(ValueError, match="must be absolute"):
        Config.model_validate(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {"s": {"command": ["x"], "sandbox": {"read_paths": [path]}}},
                }
            }
        )


def test_a_connected_server_cannot_carry_a_sandbox_block() -> None:
    """It is the operator's own process; agent6 never starts it, so there is
    nothing here to confine."""
    with pytest.raises(ValueError, match="SPAWNS"):
        Config.model_validate(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {"s": {"url": "https://h/mcp", "sandbox": {"read_paths": ["/usr"]}}},
                }
            }
        )


def test_the_shim_reaches_the_spawn_from_config(tmp_path: Path) -> None:
    """The whole path: config block -> spec -> the argv actually spawned."""
    from agent6.app._setup import start_mcp_manager_if_enabled

    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {
                    "s": {
                        "command": [sys.executable, "-c", "pass"],
                        "sandbox": {
                            "read_paths": ["/usr", str(tmp_path)],
                            "write_paths": [str(tmp_path)],
                        },
                    }
                },
            }
        }
    )
    spawned: list[list[str]] = []

    class _Proc:
        pid = 4242
        stdin = None
        stdout = None

        def poll(self) -> int:
            return 0

    def _capture(argv: list[str], **_kw: object) -> _Proc:
        spawned.append(list(argv))
        raise OSError("stop here; the argv is what this test is about")

    import agent6.tools.mcp_client as client

    original = client.subprocess.Popen
    client.subprocess.Popen = _capture  # pyright: ignore[reportAttributeAccessIssue]
    try:
        start_mcp_manager_if_enabled(cfg)
    finally:
        client.subprocess.Popen = original  # pyright: ignore[reportAttributeAccessIssue]

    # The Landlock probe shells out to ldconfig on import, so pick the spawn
    # this test is about rather than assuming it is the first.
    servers = [argv for argv in spawned if argv[:3] == _SHIM]
    assert servers, f"the server was not spawned through the shim: {spawned}"
    assert "--read" in servers[0] and "/usr" in servers[0]
    assert servers[0][-2:] == [sys.executable, "-c"] or "--" in servers[0]


def test_a_confined_server_loses_the_session_bus() -> None:
    """PROVED escape: Landlock gates filesystem paths, not `connect()` to a
    unix socket. A server denied /etc/passwd directly could reach the session
    bus, ask the UNCONFINED `systemd --user` to read it, and have the result
    written outside its write set. Any reachable unconfined spawner is a way
    out, so a confined child does not get the addresses that reach one."""
    from agent6.child_env import curated_env

    confined = curated_env(desktop=False)
    for var in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY"):
        assert var not in confined, f"{var} reaches a process that is not confined"
    assert "PATH" in curated_env(desktop=False), "it still has to be able to run"


def test_a_notify_hook_keeps_the_desktop_it_needs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split is per-child, not a blanket removal: `notify-send` talks to
    the session bus, and a hook is the operator's own command."""
    from agent6.child_env import curated_env

    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    assert "DBUS_SESSION_BUS_ADDRESS" in curated_env()


def test_only_a_confined_spawn_drops_the_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfined MCP server is no more bounded than the operator's shell,
    so taking its desktop away would break things for no gain."""
    import agent6.tools.mcp_client as client

    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    seen: list[dict[str, str]] = []

    class _Proc:
        pid = 1
        stdin = None
        stdout = None

        def poll(self) -> int:
            return 0

    def _capture(_argv: object, **kw: object) -> _Proc:
        env = kw["env"]
        assert isinstance(env, dict)
        seen.append(dict(env))
        raise OSError("stop here")

    original = client.subprocess.Popen
    client.subprocess.Popen = _capture  # pyright: ignore[reportAttributeAccessIssue]
    try:
        for confine in (None, (("/usr",), (), False)):
            srv = client._MCPServer(  # pyright: ignore[reportPrivateUsage]
                name="s",
                command=("x",),
                startup_timeout_s=1.0,
                call_timeout_s=1.0,
                confine=confine,
            )
            with contextlib.suppress(client.MCPError):
                srv.start()
    finally:
        client.subprocess.Popen = original  # pyright: ignore[reportAttributeAccessIssue]

    assert "DBUS_SESSION_BUS_ADDRESS" in seen[0], "an unconfined server keeps it"
    assert "DBUS_SESSION_BUS_ADDRESS" not in seen[1], "a confined one must not"


def test_a_kernel_without_landlock_is_reported_where_it_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shim's stderr goes to /dev/null (a chatty server would spam every
    run), so its degrade warning was discarded -- and the next line said the
    server started, which reads as success."""
    import agent6.app._setup as setup

    monkeypatch.setattr(setup, "landlock_abi", lambda: 0)
    said: list[str] = []
    cfg = Config.model_validate(
        {
            "mcp": {
                "enabled": True,
                "servers": {
                    "soft": {"command": ["x"], "sandbox": {"read_paths": ["/usr"]}},
                    "hard": {
                        "command": ["x"],
                        "sandbox": {"read_paths": ["/usr"], "require": True},
                    },
                },
            }
        }
    )
    setup._warn_unconfinable(  # pyright: ignore[reportPrivateUsage]
        cfg, reporter=Reporter(out=said.append, err=said.append)
    )
    assert any("'soft'" in line and "full filesystem" in line for line in said)
    assert not any("'hard'" in line for line in said), "it refuses; that is not a degrade"


def test_a_server_description_cannot_repaint_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The server chose this text. It is printed, so it must not carry ESC."""
    from agent6.ui.cli.mcp_connect import cmd_mcp_connect

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "cfg"))
    hostile = "\\u001b[2J\\u001b[1;31mWRITTEN TO CONFIG"
    script = (
        "import json,sys\n"
        "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    m=json.loads(line)\n"
        "    if m.get('method')=='initialize':\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
        "'capabilities':{},'serverInfo':{'name':'t','version':'1'}}})\n"
        "    elif m.get('method')=='tools/list':\n"
        f"        w({{'jsonrpc':'2.0','id':m['id'],'result':{{'tools':[{{'name':'t',"
        f"'description':\"{hostile}\",'inputSchema':{{}}}}]}}}})\n"
    )
    assert (
        cmd_mcp_connect(
            "h",
            command=[sys.executable, "-c", script],
            url="",
            token_env="",
            pass_env=[],
            to_repo=False,
        )
        == 0
    )
    assert "\x1b" not in capsys.readouterr().out
