# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit regressions for background command lifecycle bookkeeping."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator
from typing import Any, cast

import pytest

from agent6 import kinds
from agent6.sandbox import jail
from agent6.tools import background


class _Job:
    def __init__(self, *, stop_error: str = "") -> None:
        self.stopped = False
        self.running = True
        self.stop_error = stop_error

    def status(self) -> jail.BackgroundStatus:
        return jail.BackgroundStatus(
            running=self.running, returncode=None if self.running else -9, error=""
        )

    def stop(self) -> str:
        self.stopped = True
        if not self.stop_error:
            self.running = False
        return self.stop_error


class _Session:
    def __init__(self, survivors: frozenset[int] = frozenset()) -> None:
        self.stopped: list[int] = []
        self.survivors = survivors

    def open_job(self, _pid: int, _before: kinds.ChildSnapshot) -> None:
        pass

    def status_background(self, _pid: int) -> jail.BackgroundStatus:
        return jail.BackgroundStatus(running=True, returncode=None, error="")

    def stop_background(self, pid: int) -> jail.Stopped:
        self.stopped.append(pid)
        return jail.Stopped(returncode=-9, survivors=self.survivors)

    def sweep_for(self, _pid: int, _before: kinds.ChildSnapshot) -> frozenset[int]:
        return frozenset()


def _fail_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    real_write = pathlib.Path.write_text

    def write_text(self: pathlib.Path, *args: Any, **kwargs: Any) -> int:
        if self.name == "meta.json":
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", write_text)


def test_start_stops_a_command_when_its_metadata_cannot_be_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed roster write must not leave a started command unreachable."""
    job = _Job()

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    _fail_metadata(monkeypatch)
    shells = background.BackgroundShells(tmp_path / "shells")

    with pytest.raises(background.BackgroundError, match="could not record"):
        shells.start(("sleep", "60"), lambda _a, _rw: cast(kinds.JailPolicy, object()))

    assert job.stopped
    assert shells.roster() == []


def test_a_command_is_still_reachable_when_registration_and_its_stop_fail(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed cleanup must stay in memory so teardown can retry it."""
    job = _Job(stop_error="pid 42 survived SIGKILL")

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    _fail_metadata(monkeypatch)
    shells = background.BackgroundShells(tmp_path / "shells")

    with pytest.raises(background.BackgroundError, match="stopping it failed"):
        shells.start(("sleep", "60"), lambda _a, _rw: cast(kinds.JailPolicy, object()))

    assert [view.state for view in shells.roster()] == ["stop failed"]
    job.stop_error = ""
    assert shells.stop("bg1").state == "stopped"


def test_stop_all_closes_every_log_descriptor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run teardown releases the raw descriptors held for safe output reads."""
    job = _Job()

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("sleep", "60"), lambda _a, _rw: cast(kinds.JailPolicy, object()))
    log_fd = shells._get(view.id).log_fd  # pyright: ignore[reportPrivateUsage]
    os.fstat(log_fd)

    shells.stop_all()

    with pytest.raises(OSError):
        os.fstat(log_fd)
    assert shells.stop_all() == []


def test_read_names_the_size_when_the_byte_cap_cuts_the_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capped result says how large the complete background output was."""
    job = _Job()

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("sleep", "60"), lambda _a, _rw: cast(kinds.JailPolicy, object()))
    log = tmp_path / "shells" / "logs" / view.id / "out.log"
    log.write_bytes(b"x" * (background._TAIL_BYTES + 1000))  # pyright: ignore[reportPrivateUsage]

    _view, output = shells.read(view.id, tail_lines=200)

    total = background._TAIL_BYTES + 1000  # pyright: ignore[reportPrivateUsage]
    assert f"{total} bytes total" in output


def test_read_drops_the_line_the_byte_cap_cut_through(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read drops the line the byte cap cut through; a cut on a line boundary keeps the line."""
    job = _Job()

    def start_in_jail(*_args: object, **_kwargs: object) -> _Job:
        return job

    monkeypatch.setattr(background, "start_in_jail", start_in_jail)
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(("sleep", "60"), lambda _a, _rw: cast(kinds.JailPolicy, object()))
    log = tmp_path / "shells" / "logs" / view.id / "out.log"
    cap = background._TAIL_BYTES  # pyright: ignore[reportPrivateUsage]

    log.write_bytes(b"a" * (cap + 100) + b"\nlast\n")
    _view, output = shells.read(view.id, tail_lines=200)
    assert output.splitlines()[1:] == ["last"]

    log.write_bytes(b"a" * 99 + b"\n" + b"b" * (cap - 1) + b"\n")
    _view, output = shells.read(view.id, tail_lines=200)
    assert output.splitlines()[1:] == ["b" * (cap - 1)]


def test_the_disk_roster_skips_metadata_that_is_not_an_object(tmp_path: pathlib.Path) -> None:
    """One malformed shell record must not break every roster surface."""
    shell = tmp_path / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text("[]", encoding="utf-8")

    assert background.roster_from_dir(tmp_path / "shells") == []


def test_the_disk_roster_tolerates_its_root_disappearing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent session cleanup between the existence check and listing is harmless."""
    root = tmp_path / "shells"
    root.mkdir()
    real_iterdir = pathlib.Path.iterdir

    def iterdir(self: pathlib.Path) -> Iterator[pathlib.Path]:
        if self == root:
            raise FileNotFoundError(root)
        return real_iterdir(self)

    monkeypatch.setattr(pathlib.Path, "iterdir", iterdir)

    assert background.roster_from_dir(root) == []


def test_adopt_stops_a_command_when_its_metadata_cannot_be_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handed-back command is live before its roster write, like a fresh start."""
    log = tmp_path / "handoff.log"
    log.write_text("", encoding="utf-8")
    session = _Session()
    _fail_metadata(monkeypatch)
    shells = background.BackgroundShells(tmp_path / "shells")
    handoff = kinds.BackgroundHandoff(
        argv=("sleep", "60"),
        pid=42,
        log=str(log),
        stdout="",
        stderr="",
        duration_s=900.0,
        before=kinds.ChildSnapshot(1, frozenset()),
    )

    with pytest.raises(background.BackgroundError, match="could not record"):
        shells.adopt(handoff, session=cast(Any, session))

    assert session.stopped == [42]
    assert shells.roster() == []


def test_adopt_retains_a_command_when_its_log_and_cleanup_fail(tmp_path: pathlib.Path) -> None:
    """A failed cleanup remains reachable for a later stop and teardown retry."""
    session = _Session(frozenset({777}))
    shells = background.BackgroundShells(tmp_path / "shells")
    handoff = kinds.BackgroundHandoff(
        argv=("sleep", "60"),
        pid=42,
        log=str(tmp_path / "missing.log"),
        stdout="",
        stderr="",
        duration_s=900.0,
        before=kinds.ChildSnapshot(1, frozenset()),
    )

    with pytest.raises(background.BackgroundError, match="stopping it failed"):
        shells.adopt(handoff, session=cast(Any, session))

    assert [view.state for view in shells.roster()] == ["stop failed"]
    session.survivors = frozenset()
    shells.stop("bg1")
    assert session.stopped == [42, 42]
