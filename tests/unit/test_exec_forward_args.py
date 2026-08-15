# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`exec` and `forward` command grammars.

The optional session positional used to eat the first command word
(`agent6 exec -- echo hi` treated `echo` as the session), `forward 8000`
read the port as a session id, and dispatch stripped EVERY literal `--`
from the command, corrupting valid argv like `git log -- path`. The
contract now: only the FIRST `--` separates an optional session from the
command, the command rides verbatim, and a bare number to `forward` is a
port of the newest session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.sessions.layout import SessionLayout
from agent6.ui import cli


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def _resolve(target: str) -> SessionLayout:
        return SessionLayout(state_dir=tmp_path, session_id=target or "newest-run")

    def _exec(layout: SessionLayout, cfg: Any, cwd: Path, argv: tuple[str, ...]) -> int:
        calls.update(target=layout.session_id, argv=argv)
        return 0

    def _forward(layout: SessionLayout, port: int, local_port: int) -> int:
        calls.update(target=layout.session_id, port=port)
        return 0

    def _effective(*args: Any, **kwargs: Any) -> Any:
        return type("E", (), {"config": None})()

    monkeypatch.setattr(cli, "_resolve_target", _resolve)
    monkeypatch.setattr(cli, "exec_in_session", _exec)
    monkeypatch.setattr(cli, "forward", _forward)
    monkeypatch.setattr(cli, "load_effective", _effective)
    return calls


def test_exec_command_after_separator_runs_in_the_newest_session(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "--", "echo", "hi"]) == 0
    assert seen == {"target": "newest-run", "argv": ("echo", "hi")}


def test_exec_names_a_session_before_the_separator(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "brave-otter", "--", "echo", "hi"]) == 0
    assert seen == {"target": "brave-otter", "argv": ("echo", "hi")}


def test_exec_keeps_a_later_separator_in_the_command(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "brave-otter", "--", "git", "log", "--", "p"]) == 0
    assert seen["argv"] == ("git", "log", "--", "p")


def test_exec_without_separator_is_all_command(seen: dict[str, Any]) -> None:
    assert cli.main(["exec", "ls", "-la"]) == 0
    assert seen == {"target": "newest-run", "argv": ("ls", "-la")}


def test_exec_refuses_two_tokens_before_the_separator(
    seen: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["exec", "a", "b", "--", "cmd"]) == 2
    assert "at most one session id" in capsys.readouterr().err
    assert seen == {}


def test_forward_bare_number_is_a_port_of_the_newest_session(seen: dict[str, Any]) -> None:
    assert cli.main(["forward", "8000"]) == 0
    assert seen == {"target": "newest-run", "port": 8000}


def test_forward_session_and_port(seen: dict[str, Any]) -> None:
    assert cli.main(["forward", "brave-otter", "8000"]) == 0
    assert seen == {"target": "brave-otter", "port": 8000}


def test_exec_refuses_a_session_network_nobody_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[sandbox].network = "session"` + an ended run used to reach
    `os.open("/proc/None/ns/user")` -- an unexpected traceback instead of a
    refusal naming the situation. exec joins a LIVE session's network only.
    The isolation seam is pinned to strict so the policy derives "session"
    on every host this suite runs on."""
    from agent6.config import Config
    from agent6.ui.cli import net_cmds

    def _strict(req: str, env: Any) -> str:
        return "strict"

    monkeypatch.setattr(net_cmds, "resolve_isolation", _strict)
    (tmp_path / "run").mkdir()
    layout = SessionLayout(state_dir=tmp_path, session_id="run")
    cfg = Config.model_validate({"sandbox": {"network": "session"}})
    rc = net_cmds.exec_in_session(layout, cfg, tmp_path, ("true",))
    err = capsys.readouterr().err
    assert rc == 2
    assert "no live session network" in err
