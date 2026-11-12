from __future__ import annotations

import contextlib
import json
import os
import pathlib
import tempfile
import time

import pytest

from agent6 import config, kinds, paths
from agent6.app.session import runtime as session_runtime
from agent6.sessions import event_log, layout
from agent6.tools import background, dispatch


@pytest.mark.parametrize("isolation", ["none", "partial", "full"])
@pytest.mark.parametrize("failure", ["copy", "outcome"])
def test_registration_rollback_retains_private_job_until_persistence_succeeds(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    isolation: kinds.IsolationLevel,
    failure: str,
) -> None:
    """Failed rollback keeps its session and log reachable for teardown to retry."""
    work = tmp_path / "work"
    work.mkdir()
    shells = background.BackgroundShells(tmp_path / "shells")
    registered: list[background._Shell] = []  # pyright: ignore[reportPrivateUsage]
    real_register = background.BackgroundShells._register  # pyright: ignore[reportPrivateUsage]
    real_write = pathlib.Path.write_text
    real_pwrite = os.pwrite
    writes = 0

    def register(
        self: background.BackgroundShells,
        shell: background._Shell,  # pyright: ignore[reportPrivateUsage]
    ) -> background.ShellView:
        registered.append(shell)
        return real_register(self, shell)

    def write(self: pathlib.Path, data: str, **kwargs: object) -> int:
        if self.name == "meta.json":
            deadline = time.monotonic() + 5
            while not (work / "ready").exists():
                assert time.monotonic() < deadline, "command never emitted its output"
                time.sleep(0.01)
            raise OSError("metadata full")
        if self.name == "result.tmp" and failure == "outcome":
            raise OSError("outcome full")
        return real_write(self, data)

    def partial(fd: int, data: bytes, offset: int) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_pwrite(fd, data[:2], offset)
        raise OSError("copy full")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(background.BackgroundShells, "_register", register)
            patch.setattr(pathlib.Path, "write_text", write)
            if failure == "copy":
                patch.setattr(os, "pwrite", partial)
            with pytest.raises((background.BackgroundError, OSError)) as raised:
                shells.start(
                    ("/bin/sh", "-c", "printf 'before\\n'; touch ready; sleep 30"),
                    lambda a, rw: kinds.JailPolicy(cwd=work, argv=a, isolation=isolation),
                )
            assert len(registered) == 1
            shell = registered[0]
            assert shells._shells.get(shell.id) is shell, (  # pyright: ignore[reportPrivateUsage]
                "rollback lost ownership before persistence completed"
            )
            assert isinstance(raised.value, background.BackgroundError)
            assert "metadata full" in str(raised.value) and f"{failure} full" in str(raised.value)
            fd = shell.log_fd
            owner = shell.job._session  # pyright: ignore[reportPrivateUsage]
            assert owner._proc.poll() is None  # pyright: ignore[reportPrivateUsage]
            os.fstat(fd)
            if failure == "copy":
                assert os.pread(fd, 100, 0) == b"be"
            with pytest.raises(background.BackgroundError, match="could not persist"):
                shells.stop_all()
            assert shell.log_fd == fd
            assert owner._proc.poll() is None  # pyright: ignore[reportPrivateUsage]

        shells.stop_all()
        assert owner._proc.poll() == 0  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(OSError):
            os.fstat(fd)
        assert (shells.log_root / shell.id / "out.log").read_bytes() == b"before\n"
        assert (shell.dir / "result.json").is_file()
        assert shells.stop_all() == []
    finally:
        # A failed assertion must still release unregistered resources.
        for shell in registered:
            shell.job.stop()
            shell.job.close()
            if shell.log_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(shell.log_fd)


@pytest.mark.parametrize("detached", [False, True], ids=["handed_back", "detached"])
@pytest.mark.parametrize("isolation", ["none", "partial", "full"])
def test_dispatcher_close_retains_capture_until_copy_succeeds(
    monkeypatch: pytest.MonkeyPatch, isolation: kinds.IsolationLevel, detached: bool
) -> None:
    """A failed close keeps the session and its capture; settlement is unknown until the retry."""
    _assert_dispatcher_close_retries(monkeypatch, isolation, detached=detached)


def _assert_dispatcher_close_retries(
    monkeypatch: pytest.MonkeyPatch, isolation: kinds.IsolationLevel, *, detached: bool
) -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(dir=repo, prefix=".rollback-records-") as scratch:
        root = pathlib.Path(scratch)
        work = root / "work"
        work.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
        saved = layout.SessionLayout(paths.state_dir(work), "s1")
        cfg = config.Config.model_validate(
            {
                "sandbox": {
                    "run_commands": "allow",
                    "network": "session" if isolation == "full" else "host",
                },
                "harness": {"command_checkin_s": 0.5},
            }
        )
        tools = dispatch.ToolDispatcher(
            root=work,
            config=cfg,
            isolation=isolation,
            session_dir=saved.session_dir,
            events=event_log.EventSink(saved.logs_path),
            command_access=session_runtime.prepare_local_execution(cfg, work, isolation),
        )
        real_write = os.pwrite
        writes = 0

        def partial(fd: int, data: bytes, offset: int) -> int:
            nonlocal writes
            writes += 1
            if writes == 1:
                return real_write(fd, data[:2], offset)
            raise OSError("copy full")

        try:
            started = tools.dispatch(
                "run_command",
                {
                    "argv": [
                        "/bin/sh",
                        "-c",
                        "printf 'before\\n'; touch ready; "
                        "while [ ! -f release ]; do sleep 0.01; done; "
                        "printf 'after\\n'; touch emitted; sleep 30",
                    ],
                    "background": detached,
                },
            ).to_wire()
            assert started["background_id"] == "bg1" and started["still_running"]
            shells = tools._shells  # pyright: ignore[reportPrivateUsage]
            access = tools._command_access  # pyright: ignore[reportPrivateUsage]
            assert shells is not None and access is not None
            shell = shells._get("bg1")  # pyright: ignore[reportPrivateUsage]
            owner = shell.job._session  # pyright: ignore[reportPrivateUsage]
            handle = shell.job._handle  # pyright: ignore[reportPrivateUsage]
            assert not shell.job._owns_session  # pyright: ignore[reportPrivateUsage]
            assert access.session is owner
            temp, net = access.temp, access.network
            if detached:
                deadline = time.monotonic() + 5
                while not (work / "ready").exists():
                    assert time.monotonic() < deadline, "command never emitted its output"
                    time.sleep(0.01)
                tools.dispatch("read_background", {"id": "bg1", "wait_s": 0})
            fd = shell.log_fd
            assert os.pread(fd, 100, 0) == b"before\n"
            holder = net._holder if net is not None else None  # pyright: ignore[reportPrivateUsage]
            assert isolation != "full" or holder is not None
            assert isolation == "full" or temp is not None
            assert holder is None or holder.poll() is None
            (work / "release").touch()
            deadline = time.monotonic() + 5
            while not (work / "emitted").exists():
                assert time.monotonic() < deadline, "command never emitted its late output"
                time.sleep(0.01)

            # No read or settle after release: close must persist the late bytes.
            with monkeypatch.context() as patch:
                patch.setattr(os, "pwrite", partial)
                for attempt in range(1, 3):
                    with pytest.raises(background.BackgroundError, match="copy full") as raised:
                        tools.close()
                    assert owner._proc.poll() is None  # pyright: ignore[reportPrivateUsage]
                    assert shells._get("bg1") is shell  # pyright: ignore[reportPrivateUsage]
                    assert shell.log_fd == fd and os.pread(fd, 100, 0) == b"before\naf"
                    assert not tools._closed  # pyright: ignore[reportPrivateUsage]
                    assert access.session is owner
                    assert access.temp is temp and access.network is net
                    assert holder is None or holder.poll() is None
                    assert temp is None or pathlib.Path(temp.name).is_dir()
                    status = owner.status_background(handle)
                    assert not status.running and status.returncode is not None
                    assert not status.error
                    assert owner.read_background(handle, offset=0, limit=100).data == (
                        b"before\nafter\n"
                    )
                    rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
                    jobs = [row for row in rows if row["type"] == "execution.jobs"]
                    assert len(jobs) == attempt
                    assert jobs[-1]["outcome"] == "unknown"
                    assert str(raised.value) in jobs[-1]["reason"]
                    degraded = [row for row in rows if row["type"] == "jail.degraded"]
                    assert str(raised.value) in degraded[-1]["detail"]

            tools.close()
            assert tools._closed  # pyright: ignore[reportPrivateUsage]
            assert owner._close_error == ""  # pyright: ignore[reportPrivateUsage]
            assert owner._proc.poll() == 0  # pyright: ignore[reportPrivateUsage]
            assert shell.log_fd == -1
            with pytest.raises(OSError):
                os.fstat(fd)
            assert (shells.log_root / "bg1/out.log").read_bytes() == b"before\nafter\n"
            result = json.loads((shell.dir / "result.json").read_text())
            assert result["running"] is False and result["returncode"] is not None
            assert not result["error"] and not result["capture_error"]
            assert holder is None or holder.poll() is not None
            assert temp is None or not pathlib.Path(temp.name).exists()
            journal = saved.logs_path.read_bytes()
            rows = [json.loads(line) for line in journal.splitlines()]
            (target,) = [row for row in rows if row["type"] == "execution.target"]
            jobs = [row for row in rows if row["type"] == "execution.jobs"]
            assert [row["outcome"] for row in jobs] == [
                "unknown",
                "unknown",
                "settled",
            ]
            assert jobs[-1]["reason"] == ""
            environments = [row for row in rows if row["type"] == "execution.environment"]
            assert [row["outcome"] for row in environments] == ["not_applicable"] * 3
            for row in (*jobs, *environments):
                assert row["execution_id"] == target["execution_id"] == tools.execution_id
                assert row["target_id"] == target["target_id"] == tools.target_id
            tools.close()
            assert saved.logs_path.read_bytes() == journal
        finally:
            tools.close()
