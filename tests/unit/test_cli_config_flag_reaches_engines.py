# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The top-level --config reaches every command's config load.

The planner, compare, exec, sessions merge and machine create thread the explicit path into
load_effective.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from agent6 import git_ops, paths
from agent6.config import layer
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import sessions_cmds, sessions_merge


def test_merge_planner_passes_the_explicit_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    seen: list[pathlib.Path | None] = []

    def fake_load(cwd: pathlib.Path, explicit: pathlib.Path | None) -> Any:
        seen.append(explicit)
        raise sessions_merge.ConfigError("stop here")

    monkeypatch.setattr(layer, "load_effective", fake_load)

    # A resolvable path flows through unchanged when the planner reaches the load.
    class _Layout:
        session_dir = tmp_path / "sess"

    layout = _Layout()

    def _dead(d: pathlib.Path) -> bool:
        return False

    monkeypatch.setattr(ipc, "worker_is_alive", _dead)

    class _Manifest:
        session_id = "sid"
        base_branch = "main"
        base_sha = "0" * 40
        run_branch = "agent6/x"

    def _resolved(cwd: pathlib.Path, sid: str) -> Any:
        return (layout, _Manifest())

    def _exists(cwd: pathlib.Path, b: str) -> bool:
        return True

    monkeypatch.setattr(sessions_cmds, "_resolve_session_manifest", _resolved)
    monkeypatch.setattr(git_ops, "branch_exists", _exists)
    explicit = tmp_path / "special.toml"
    rc = sessions_merge._plan_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, "sid", None, None, config_path=explicit
    )
    assert rc == 2  # the stubbed ConfigError surfaced as the exit path
    assert seen == [explicit]


def test_acp_run_bridge_passes_the_explicit_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The ACP run bridge loads the explicit config path, so its model.

    Budget and run_commands apply.
    """
    from agent6.config import ConfigError as _ConfigError
    from agent6.ui.acp import runner as acp_runner
    from agent6.ui.acp import session as acp_session

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    seen: list[pathlib.Path | None] = []

    def fake_load(cwd: pathlib.Path, explicit: pathlib.Path | None = None, **_kw: Any) -> Any:
        seen.append(explicit)
        raise _ConfigError("stop here")

    monkeypatch.setattr(acp_runner, "load_session_config", fake_load)

    class _Server:
        def __init__(self) -> None:
            self.notes: list[dict[str, Any]] = []

        def notify_raw(self, obj: dict[str, Any]) -> None:
            self.notes.append(obj)

    server = _Server()
    cfg_path = tmp_path / "overlay.toml"
    bridge = acp_runner.RunBridge(server=server, config_path=cfg_path)  # type: ignore[arg-type]
    session = acp_session.Session(acp_id="t", cwd=tmp_path)
    assert bridge.run(session, "task") == "refusal"
    assert seen == [cfg_path]


def test_hub_spawns_stamp_the_explicit_config_into_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A hub started with --config F stamps F into the argv of everything it spawns."""
    from agent6.ui import spawn

    argvs: list[list[str]] = []

    def fake_spawn(
        argv: list[str], cwd: pathlib.Path, **_kw: Any
    ) -> tuple[pathlib.Path | None, str]:
        argvs.append(argv)
        return None, "stubbed"

    cfg = tmp_path / "overlay.toml"
    monkeypatch.setattr(spawn, "spawn_and_locate", fake_spawn)
    spawn.spawn_new_work(tmp_path, "run", "t", config_path=cfg)
    assert len(argvs) == 1
    for argv in argvs:
        flag = argv.index("--config")
        assert argv[flag + 1] == str(cfg)
        assert flag < argv.index("run"), "--config is a top-level flag: before the subcommand"


def test_detached_resume_reapplies_the_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    import agent6.ui.spawn as spawn_mod

    seen: list[list[str]] = []

    class _Proc:
        pid = 4242

        def poll(self) -> int | None:
            return None

    def fake_popen(argv: list[str], **_kw: Any) -> Any:
        seen.append(list(argv))
        return _Proc()

    monkeypatch.setattr(spawn_mod.subprocess, "Popen", fake_popen)

    def _no_sweep(_pid: int) -> None:
        pass

    monkeypatch.setattr(spawn_mod, "keep_out_of_the_sweep", _no_sweep)
    run = sessions_layout.bucket_dir(paths.state_dir(tmp_path), "runs") / "run-1"
    run.mkdir(parents=True)
    (run / "worker.pid").write_text("4242", encoding="utf-8")  # the fake child owns the run
    cfg = tmp_path / "overlay.toml"
    assert spawn_mod.spawn_detached_resume(tmp_path, "run-1", config_path=cfg) == ""
    (argv,) = seen
    assert argv[1:3] == ["--config", str(cfg)]
    assert "resume" in argv
