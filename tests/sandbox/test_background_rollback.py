from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile
import time

import pytest

from agent6 import config, kinds, paths
from agent6.app import session_runtime
from agent6.sessions import layout
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


@pytest.mark.parametrize("isolation", ["none", "partial", "full"])
def test_dispatcher_close_retains_shared_capture_until_copy_succeeds(
    monkeypatch: pytest.MonkeyPatch, isolation: kinds.IsolationLevel
) -> None:
    """Dispatcher close retains capture and retry handles after a partial host write."""
    repo = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(dir=repo, prefix=".rollback-records-") as scratch:
        root = pathlib.Path(scratch)
        work = root / "work"
        work.mkdir()
        monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
        session_dir = layout.SessionLayout(paths.state_dir(work), "s1").session_dir
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
            session_dir=session_dir,
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
                        "printf 'before\\n'; while [ ! -f release ]; do sleep 0.01; done; "
                        "printf 'after\\n'; touch emitted; sleep 30",
                    ]
                },
            ).to_wire()
            assert started["background_id"] == "bg1" and started["still_running"]
            shells = tools._shells  # pyright: ignore[reportPrivateUsage]
            access = tools._command_access  # pyright: ignore[reportPrivateUsage]
            assert access is not None
            owner = access.session
            assert shells is not None and owner is not None
            shell = shells._get("bg1")  # pyright: ignore[reportPrivateUsage]
            assert not shell.job._owns_session  # pyright: ignore[reportPrivateUsage]
            fd = shell.log_fd
            assert os.pread(fd, 100, 0) == b"before\n"
            temp = access.temp  # pyright: ignore[reportPrivateUsage]
            net = access.network  # pyright: ignore[reportPrivateUsage]
            holder = net._holder if net is not None else None  # pyright: ignore[reportPrivateUsage]
            if isolation == "full":
                assert holder is not None and holder.poll() is None
            else:
                assert temp is not None
            (work / "release").touch()
            deadline = time.monotonic() + 5
            while not (work / "emitted").exists():
                assert time.monotonic() < deadline, "command never emitted its late output"
                time.sleep(0.01)

            # No read or settle after release: close must persist the late bytes.
            with monkeypatch.context() as patch:
                patch.setattr(os, "pwrite", partial)
                with pytest.raises(background.BackgroundError, match="copy full"):
                    tools.close()
                assert access.session is owner  # pyright: ignore[reportPrivateUsage]
                assert owner._proc.poll() is None  # pyright: ignore[reportPrivateUsage]
                assert shells._get("bg1") is shell  # pyright: ignore[reportPrivateUsage]
                assert shell.log_fd == fd and os.pread(fd, 100, 0) == b"before\naf"
                assert access.temp is temp  # pyright: ignore[reportPrivateUsage]
                assert access.network is net  # pyright: ignore[reportPrivateUsage]
                if holder is not None:
                    assert holder.poll() is None
                if temp is not None:
                    assert pathlib.Path(temp.name).is_dir()

            tools.close()
            assert owner._close_error is not None  # pyright: ignore[reportPrivateUsage]
            assert owner._proc.poll() == 0  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(OSError):
                os.fstat(fd)
            assert (shells.log_root / "bg1/out.log").read_bytes() == b"before\nafter\n"
            assert access.temp is None or not pathlib.Path(access.temp.name).exists()  # pyright: ignore[reportPrivateUsage]
            if holder is not None:
                assert holder.poll() is not None
            if temp is not None:
                assert not pathlib.Path(temp.name).exists()
            tools.close()
        finally:
            tools.close()
            if tools._command_access is not None:  # pyright: ignore[reportPrivateUsage]
                net = tools._command_access.network  # pyright: ignore[reportPrivateUsage]
                if net is not None:
                    net.close()
