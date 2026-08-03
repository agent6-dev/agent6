# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Filesystem confinement for a SPAWNED MCP server.

An MCP server is third-party code running as the operator, with their whole
filesystem. Naming paths opts into a Landlock domain the server and everything
it spawns inherit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_a_block_that_confines_nothing_is_refused() -> None:
    """It would read as protection while granting everything."""
    with pytest.raises(ValueError, match="confines nothing"):
        Config.model_validate(
            {"mcp": {"enabled": True, "servers": {"s": {"command": ["x"], "sandbox": {}}}}}
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
                        "sandbox": {"read_paths": ["/usr"], "write_paths": [str(tmp_path)]},
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

    assert spawned, "nothing was spawned"
    assert spawned[0][:3] == _SHIM
    assert "--read" in spawned[0] and "/usr" in spawned[0]
    assert json.dumps(spawned[0]).count("--") >= 1
