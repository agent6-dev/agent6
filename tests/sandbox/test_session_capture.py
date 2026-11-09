from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from collections.abc import Iterator
from unittest import mock

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_jobs, jail_session
from agent6.tools import background


@pytest.fixture(params=["none", "partial", "full"])
def session(
    tmp_path: pathlib.Path, request: pytest.FixtureRequest
) -> Iterator[jail_session.JailSession]:
    policy = kinds.JailPolicy(cwd=tmp_path, argv=(), isolation=request.param)
    owner = jail_session.JailSession.open(policy)
    try:
        yield owner
    finally:
        owner.close()


def handback(session: jail_session.JailSession) -> kinds.BackgroundHandoff:
    result = session.run(
        ("/bin/sh", "-c", "printf 'before\\n'; sleep 30"), checkin_s=0.05, timeout_s=0.3
    )
    assert isinstance(result, kinds.BackgroundHandoff), result
    assert "log" not in {field.name for field in dataclasses.fields(result)}
    return result


def test_handback_adopts_without_a_target_path(
    session: jail_session.JailSession, tmp_path: pathlib.Path
) -> None:
    """Adoption uses an owned handle, not a path visible on the host."""
    shells = background.BackgroundShells(tmp_path / "shells")
    try:
        view = shells.adopt(handback(session), session=session)
        assert "before" in shells.read(view.id, tail_lines=10)[1]
    finally:
        shells.stop_all()


def test_explicit_background_uses_original_argv(
    session: jail_session.JailSession, tmp_path: pathlib.Path
) -> None:
    """The launcher captures output without a command-side redirection shell."""
    shells = background.BackgroundShells(tmp_path / "shells")
    argv = ("/bin/sh", "-c", "printf 'direct\\n'; sleep 30")
    real_start = jail_session.JailSession.start_background
    started: list[tuple[str, ...]] = []

    def start(self: jail_session.JailSession, command: tuple[str, ...], **kwargs: object) -> str:
        started.append(command)
        return real_start(self, command)

    try:
        with mock.patch.object(jail_session.JailSession, "start_background", start):
            view = shells.start(
                argv, lambda a, rw: kinds.JailPolicy(cwd=tmp_path, argv=a), session=session
            )
        assert started == [argv]
        assert "direct" in shells.read(view.id, tail_lines=10, wait_s=0.1)[1]
    finally:
        shells.stop_all()


def test_offset_reads_are_bounded_and_keep_raw_bytes(
    session: jail_session.JailSession,
) -> None:
    """Offset reads preserve non-UTF8 bytes and reject unowned handles and invalid bounds."""
    handle = session.start_background(
        ("/usr/bin/python3", "-c", "import os; os.write(1, bytes(range(256))*600)")
    )
    deadline = time.monotonic() + 5
    while session.status_background(handle).running:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    session.stop_background(handle)
    read = session.read_background(handle, offset=0, limit=65536)
    assert read.data == bytes(range(256)) * 256
    assert read.next_offset == 65536 and read.size == 256 * 600
    tail = session.read_background(handle, offset=read.next_offset, limit=65536)
    assert tail.data == (bytes(range(256)) * 600)[65536:131072]
    for offset, limit in [(-1, 1), (0, 0), (0, 65537)]:
        with pytest.raises(jail.JailUnavailableError, match="read"):
            session.read_background(handle, offset=offset, limit=limit)
    with pytest.raises(jail.JailUnavailableError, match="unknown job"):
        session.read_background("not-owned", offset=0, limit=1)
    for offset, limit in [(0, 0), (0, 65537), (256 * 600 + 1, 1)]:
        answer = session._request(  # pyright: ignore[reportPrivateUsage]
            {"kind": "read", "handle": handle, "offset": offset, "limit": limit}
        )
        assert "error" in answer and "data" not in answer
    assert session.status_background(handle).returncode == 0


def test_partial_host_copy_retries_without_skipping_bytes(
    session: jail_session.JailSession, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial host write resumes at the last copied byte, without duplication."""
    handoff = handback(session)
    job = jail_jobs.SessionJob(session, handoff.handle, tmp_path)
    fd = os.open(tmp_path / "copy", os.O_CREAT | os.O_RDWR, 0o600)
    real_write = os.pwrite
    calls = 0

    def partial(fd: int, data: bytes, offset: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[:2], offset)
        raise OSError("disk full")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "pwrite", partial)
            with pytest.raises(OSError, match="disk full"):
                job.copy_output(fd)
        job.copy_output(fd)
        assert os.pread(fd, 100, 0) == b"before\n"
    finally:
        os.close(fd)
        job.stop()


def wait_file(path: pathlib.Path) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, f"command never wrote {path.name}"
        time.sleep(0.01)


_LATE_OUTPUT = (
    "printf 'before\\n'; while [ ! -f release ]; do sleep 0.01; done; "
    "printf 'after\\n'; touch emitted; sleep 30"
)


@pytest.mark.parametrize("replacement", ["symlink", "file"])
def test_host_log_replacement_never_redirects_session_copies(
    session: jail_session.JailSession, tmp_path: pathlib.Path, replacement: str
) -> None:
    """A replaced log path supplies neither output nor a write target."""
    shells = background.BackgroundShells(tmp_path / "shells")
    handoff = session.run(("/bin/sh", "-c", _LATE_OUTPUT), checkin_s=0.05, timeout_s=1)
    assert isinstance(handoff, kinds.BackgroundHandoff)
    view = shells.adopt(handoff, session=session)
    log = shells.log_root / view.id / "out.log"
    fd = shells._get(view.id).log_fd  # pyright: ignore[reportPrivateUsage]
    secret = tmp_path / "secret"
    secret.write_text("DECOY")
    log.unlink()
    if replacement == "symlink":
        log.symlink_to(secret)
    else:
        log.write_text("DECOY")
    try:
        (tmp_path / "release").touch()
        wait_file(tmp_path / "emitted")
        session.stop_background(handoff.handle)
        assert shells.read(view.id, tail_lines=10)[1] == "before\nafter"
        assert os.pread(fd, 100, 0) == b"before\nafter\n"
        assert log.read_text() == "DECOY"
        assert secret.read_text() == "DECOY"
    finally:
        shells.stop_all()


@pytest.mark.parametrize("isolation", ["none", "partial", "full"])
def test_teardown_copies_output_before_private_session_cleanup(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """Teardown copies output emitted after registration before closing its private session."""
    work = tmp_path / "work"
    work.mkdir()
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(
        ("/bin/sh", "-c", _LATE_OUTPUT),
        lambda a, rw: kinds.JailPolicy(cwd=work, argv=a, isolation=isolation),
    )
    shell = shells._get(view.id)  # pyright: ignore[reportPrivateUsage]
    assert shell.job._owns_session  # pyright: ignore[reportPrivateUsage]
    owner = shell.job._session  # pyright: ignore[reportPrivateUsage]
    fd = shell.log_fd
    try:
        assert owner._proc.poll() is None  # pyright: ignore[reportPrivateUsage]
        (work / "release").touch()
        wait_file(work / "emitted")
        # No read or settle after releasing the command: teardown must copy the late bytes.
        shells.stop_all()
        assert owner._proc.poll() == 0  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(OSError):
            os.fstat(fd)
        assert (shells.log_root / view.id / "out.log").read_bytes() == b"before\nafter\n"
    finally:
        shells.stop_all()


def test_host_record_failure_does_not_skip_job_cleanup(
    session: jail_session.JailSession, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed retry of a terminal record cannot bypass the stop request."""
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.start(
        ("/bin/sh", "-c", "sleep 30 & echo child-started; exit 0"),
        lambda a, rw: kinds.JailPolicy(cwd=tmp_path, argv=a),
        session=session,
    )
    shell = shells._get(view.id)  # pyright: ignore[reportPrivateUsage]
    deadline = time.monotonic() + 5
    while shell.job.status().running:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    stopped: list[str] = []
    real_stop = jail_session.JailSession.stop_background
    real_write = pathlib.Path.write_text

    def stop(self: jail_session.JailSession, handle: str) -> jail_session.Stopped:
        stopped.append(handle)
        return real_stop(self, handle)

    def write(self: pathlib.Path, data: str, **kwargs: object) -> int:
        if self.name == "result.tmp":
            raise OSError("disk full")
        return real_write(self, data)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(jail_session.JailSession, "stop_background", stop)
            patch.setattr(pathlib.Path, "write_text", write)
            with pytest.raises(OSError, match="disk full"):
                shells.stop(view.id)
            assert stopped, "host persistence bypassed target cleanup"
    finally:
        shells.stop_all()


def test_loss_between_status_and_read_is_not_reported_running(
    session: jail_session.JailSession, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed read can observe channel loss after the earlier status response."""
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.adopt(handback(session), session=session)
    real_read = jail_session.JailSession.read_background

    def lose(
        self: jail_session.JailSession, handle: str, *, offset: int, limit: int
    ) -> jail_session.BackgroundRead:
        self._proc.kill()  # pyright: ignore[reportPrivateUsage]
        self._proc.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]
        return real_read(self, handle, offset=offset, limit=limit)

    monkeypatch.setattr(jail_session.JailSession, "read_background", lose)
    try:
        after, output = shells.read(view.id, tail_lines=10)
        assert after.state != "running"
        assert "output incomplete" in output
    finally:
        shells.stop_all()


def test_lost_outcome_and_copied_output_survive_replay(
    session: jail_session.JailSession, tmp_path: pathlib.Path
) -> None:
    """Losing the supervisor preserves copied bytes and records an unknown outcome."""
    shells = background.BackgroundShells(tmp_path / "shells")
    view = shells.adopt(handback(session), session=session)
    assert "before" in shells.read(view.id, tail_lines=10)[1]
    session._proc.kill()  # pyright: ignore[reportPrivateUsage]
    session._proc.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]
    shells.settle()
    shells.stop_all()
    replay = background.roster_from_dir(tmp_path / "shells")
    assert len(replay) == 1 and "outcome unknown" in replay[0]
    assert (shells.log_root / view.id / "out.log").read_bytes() == b"before\n"
