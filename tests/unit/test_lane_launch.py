# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The lane launch file: private, validated, and the only carrier of a lane's task."""

from __future__ import annotations

import os
import pathlib
import stat
import sys
from typing import Any

import pytest

from agent6 import lane_records, paths
from agent6.app import lane_launch, workdirs
from agent6.config import Config, ConfigError
from agent6.ui import spawn


def _launch(tmp_path: pathlib.Path, **over: Any) -> lane_launch.LaneLaunch:
    fields: dict[str, Any] = {
        "workdir": tmp_path / "lane-1",
        "session_id": "fan-l1",
        "task": "--allow-root pwn\n$(rm -rf /)\n/parallel 3 -- x",
        "pins": ("never touch schema files",),
        "max_usd": 2.5,
        "auto_approve": True,
        "away": "deny",
        "config": tmp_path / "launch" / "lane-1-config.toml",
        "notes": ("models.worker.temperature: selected None; TOML key omitted; reload uses 0.0.",),
        "coordinator": "co",
        "group": "fan",
        "lane": 1,
        "gate_source": "inferred",
        "operator_task": "pwn",
    }
    fields.update(over)
    return lane_launch.LaneLaunch(**fields)


def test_write_and_read_round_trip_a_private_file(tmp_path: pathlib.Path) -> None:
    """The file lands `0600` in a `0700` dir and reads back as written, every field intact."""
    directory = tmp_path / "launch"
    launch = _launch(tmp_path)
    path = lane_launch.write(directory, launch)
    assert path == directory / "lane-1.json"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert lane_launch.read_file(path) == launch
    # A rewrite replaces the file in place and keeps its mode.
    lane_launch.write(directory, launch.model_copy(update={"task": "again"}))
    assert lane_launch.read_file(path).task == "again"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_read_follows_the_environment_and_is_none_outside_a_lane(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(lane_launch.ENV, raising=False)
    assert lane_launch.read() is None
    path = lane_launch.write(tmp_path / "launch", _launch(tmp_path))
    monkeypatch.setenv(lane_launch.ENV, str(path))
    assert lane_launch.read() is not None
    monkeypatch.setenv(lane_launch.ENV, str(tmp_path / "launch" / "missing.json"))
    with pytest.raises(ConfigError, match="lane launch"):
        lane_launch.read()


@pytest.mark.parametrize(
    "break_it",
    ["symlink", "readable", "dir_writable", "dir_symlink", "malformed", "not_json", "directory"],
)
def test_read_refuses_an_exposed_or_malformed_file(tmp_path: pathlib.Path, break_it: str) -> None:
    """A planted link, a readable mode, a shared dir or bad content is refused, naming why."""
    directory = tmp_path / "launch"
    path = lane_launch.write(directory, _launch(tmp_path))
    why = "lane launch"
    if break_it == "symlink":
        real = tmp_path / "elsewhere.json"
        real.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.unlink()
        path.symlink_to(real)
        why = "symlink"
    elif break_it == "readable":
        path.chmod(0o644)
        why = "lets others read it"
    elif break_it == "dir_writable":
        directory.chmod(0o777)
        why = "lets others write it"
    elif break_it == "dir_symlink":
        moved = tmp_path / "moved"
        directory.rename(moved)
        directory.symlink_to(moved)
        why = "symlink"
    elif break_it == "malformed":
        lane_launch.write_private(path, '{"lane": 0}')
        why = "validation error"
    elif break_it == "not_json":
        lane_launch.write_private(path, "not json")
    else:
        path.unlink()
        path.mkdir()
        why = "not a regular file"
    with pytest.raises(ConfigError, match=why):
        lane_launch.read_file(path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root owns everything")
def test_read_refuses_a_file_owned_by_another_user(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lane_launch.write(tmp_path / "launch", _launch(tmp_path))
    real_fstat = os.fstat

    def foreign(fd: int) -> os.stat_result:
        st = real_fstat(fd)
        return os.stat_result((*st[:4], st.st_uid + 1, *st[5:]))

    monkeypatch.setattr(os, "fstat", foreign)
    with pytest.raises(ConfigError, match="owned by uid"):
        lane_launch.read_file(path)


def test_private_dir_accepts_only_this_users_unshared_dir(tmp_path: pathlib.Path) -> None:
    made = tmp_path / "a" / "b"
    lane_launch.private_dir(made)
    assert stat.S_IMODE(made.stat().st_mode) == 0o700
    lane_launch.private_dir(made)  # idempotent
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)
    with pytest.raises(ConfigError, match="lets others write it"):
        lane_launch.private_dir(shared)
    link = tmp_path / "link"
    link.symlink_to(made)
    with pytest.raises(ConfigError, match="symlink"):
        lane_launch.private_dir(link)


def test_the_bootstrap_argv_is_fixed_and_safe_path(tmp_path: pathlib.Path) -> None:
    """Nothing model-written reaches the host argv: the interpreter, `-P`, `-E` and the module."""
    argv = lane_launch.bootstrap_argv()
    assert argv == [sys.executable, "-P", "-E", "-m", "agent6.ui.cli.lane_agent"]
    assert spawn.subcommand_label(argv) == "lane agent"


def test_write_lane_config_lands_in_the_private_dir(tmp_path: pathlib.Path) -> None:
    spec = lane_records.LaneSpec(
        lane=2, session_id="fan-l2", workdir=tmp_path / "lane-2", settings=None
    )
    private = tmp_path / "state" / "launch" / "fan"
    written = workdirs.write_lane_config(Config(), spec, explicit_leaves=frozenset(), into=private)
    assert written.path == private / "lane-2-config.toml"
    assert stat.S_IMODE(written.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert lane_launch.read_private(written.path) == written.path.read_text(encoding="utf-8")


def test_launch_dir_is_under_the_origin_state(tmp_path: pathlib.Path) -> None:
    """The launch files live under the state base every jail hides, never beside the clones."""
    origin_state = paths.state_dir(tmp_path / "repo")
    private = lane_launch.launch_dir(origin_state, "co-p1")
    assert private == origin_state / "launch" / "co-p1"
    assert private.is_relative_to(paths.state_base())


def test_spawn_lane_returns_the_pid_with_a_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that exits before reporting, or never reports, is still known by its pid."""

    class _Exited:
        pid = 4242

        def poll(self) -> int | None:
            return 2

    def fake_popen(argv: list[str], **kw: Any) -> _Exited:
        return _Exited()

    monkeypatch.setattr(spawn.subprocess, "Popen", fake_popen)
    found, pid, err = spawn.spawn_lane(
        lane_launch.bootstrap_argv(), tmp_path, before=set(), list_dirs=list, env={}
    )
    assert (found, pid) == (None, 4242)
    assert "agent6 lane agent exited 2" in err

    def failing_popen(argv: list[str], **kw: Any) -> _Exited:
        raise OSError("no such interpreter")

    monkeypatch.setattr(spawn.subprocess, "Popen", failing_popen)
    found, pid, err = spawn.spawn_lane(
        lane_launch.bootstrap_argv(), tmp_path, before=set(), list_dirs=list, env={}
    )
    assert (found, pid) == (None, 0)
    assert "failed to start agent6 lane agent" in err
