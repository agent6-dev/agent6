from __future__ import annotations

import json
import os
import pathlib
import signal
import tempfile
import time

import pytest

from agent6 import config, kinds, paths
from agent6.sandbox import jail_session
from agent6.sessions import layout
from agent6.tools import background, dispatch


@pytest.mark.parametrize("isolation", ["none", "partial", "full"])
def test_loss_after_confirmed_stop_preserves_exit_and_records_missing_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, isolation: kinds.IsolationLevel
) -> None:
    """Teardown persists a lost final copy even when stop and exit are already cached."""
    work = tmp_path / "work"
    work.mkdir()
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=work, argv=(), isolation=isolation)
    )
    shells = background.BackgroundShells(tmp_path / "state")
    try:
        view = shells.start(
            (
                "/bin/sh",
                "-c",
                "printf 'before\\n'; while [ ! -f release ]; do sleep 0.01; done; "
                "printf 'unread\\n'",
            ),
            lambda a, rw: kinds.JailPolicy(cwd=work, argv=a, isolation=isolation),
            session=session,
        )
        deadline = time.monotonic() + 5
        while "before" not in shells.read(view.id, tail_lines=10)[1]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        shell = shells._get(view.id)  # pyright: ignore[reportPrivateUsage]
        handle = shell.job._handle  # pyright: ignore[reportPrivateUsage]
        (work / "release").touch()
        while session.status_background(handle).running:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        real_read = jail_session.JailSession.read_background

        def lose(
            self: jail_session.JailSession, handle: str, *, offset: int, limit: int
        ) -> jail_session.BackgroundRead:
            assert shell.job._stopped  # pyright: ignore[reportPrivateUsage]
            assert shell.job.status().returncode == 0
            self._proc.kill()  # pyright: ignore[reportPrivateUsage]
            self._proc.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]
            return real_read(self, handle, offset=offset, limit=limit)

        with monkeypatch.context() as patch:
            patch.setattr(jail_session.JailSession, "read_background", lose)
            shells.stop_all()
        record = json.loads((shell.dir / "result.json").read_text())
        assert record["returncode"] == 0 and record["error"] == ""
        assert "jail session protocol failed" in record["capture_error"]
        live = shells.roster()[0]
        assert live.returncode == 0 and live.detail == record["capture_error"]
        assert (shells.log_root / view.id / "out.log").read_bytes() == b"before\n"
        assert shell.job.stop() == ""
        shells.stop_all()
        assert json.loads((shell.dir / "result.json").read_text()) == record
        replay = background.roster_from_dir(tmp_path / "state")
        assert "exited 0" in replay[0] and record["capture_error"] in replay[0]
    finally:
        shells.stop_all()
        session.close()


@pytest.mark.parametrize("isolation", ["none", "partial"])
def test_job_supervisor_loss_marks_capture_incomplete_after_observed_exit(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A retained exit code cannot hide loss of the job's capture worker."""
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation=isolation)
    )
    shells = background.BackgroundShells(tmp_path / "state")
    try:
        command = (
            "import os, pathlib, time\n"
            "pathlib.Path('worker').write_text(str(os.getppid()))\n"
            "print('before', flush=True)\n"
            "if os.fork(): os._exit(0)\n"
            "while not pathlib.Path('release').exists(): time.sleep(0.01)\n"
            "print('unread', flush=True)\n"
            "time.sleep(30)\n"
        )
        view = shells.start(
            ("/usr/bin/python3", "-c", command),
            lambda a, rw: kinds.JailPolicy(cwd=tmp_path, argv=a, isolation=isolation),
            session=session,
        )
        deadline = time.monotonic() + 5
        while True:
            view, text = shells.read(view.id, tail_lines=10)
            if view.returncode == 0 and "before" in text:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        worker = int((tmp_path / "worker").read_text())
        assert (
            f"PPid:\t{session._proc.pid}\n"
            in pathlib.Path(  # pyright: ignore[reportPrivateUsage]
                f"/proc/{worker}/status"
            ).read_text()
        )
        os.kill(worker, signal.SIGKILL)
        view, text = shells.read(view.id, tail_lines=10)
        assert view.returncode == 0 and "before" in text
        assert "output incomplete" in text and "job outcome or cleanup unknown" in view.detail
        record = json.loads((tmp_path / "state" / view.id / "result.json").read_text())
        assert "job outcome or cleanup unknown" in record["capture_error"]
    finally:
        shells.stop_all()
        session.close()


@pytest.mark.parametrize("isolation", ["partial", "full"])
def test_dispatcher_adoption_does_not_grant_host_records(
    monkeypatch: pytest.MonkeyPatch, isolation: kinds.IsolationLevel
) -> None:
    """Confined commands can use the workspace but cannot read or overwrite adopted records."""
    # Partial isolation grants the host /tmp, so host records must live outside it.
    repo = pathlib.Path(__file__).resolve().parents[2]
    if repo.is_relative_to("/tmp"):
        pytest.skip("requires storage outside partial isolation's shared /tmp")
    with tempfile.TemporaryDirectory(dir=repo, prefix=".capture-records-") as scratch:
        tmp_path = pathlib.Path(scratch)
        work = tmp_path / "work"
        work.mkdir()
        control = work / "control"
        control.write_text("workspace")
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        session_dir = layout.SessionLayout(paths.state_dir(work), "s1").session_dir
        cfg = config.Config.model_validate(
            {"sandbox": {"run_commands": "allow"}, "harness": {"command_checkin_s": 0.5}}
        )
        tools = dispatch.ToolDispatcher(
            root=work,
            config=cfg,
            isolation=isolation,
            session_dir=session_dir,
            use_jail_session=True,
        )
        try:
            started = tools.dispatch(
                "run_command", {"argv": ["/bin/sh", "-c", "echo adopted; sleep 30"]}
            ).to_wire()
            assert started["background_id"] == "bg1" and started["still_running"]
            tools.dispatch("stop_background", {"id": "bg1"})
            root = session_dir / background.SHELLS_DIR
            records = [root / "logs/bg1/out.log", root / "bg1/meta.json", root / "bg1/result.json"]
            saved = [p.read_bytes() for p in records]
            assert b"adopted" in saved[0]
            probe = (
                "import json, sys; answers=[]\n"
                "for path in sys.argv[1:]:\n"
                " for mode in ('rb', 'ab'):\n"
                "  try:\n"
                "   with open(path, mode) as f:\n"
                "    f.read() if mode == 'rb' else f.write(b'intruder')\n"
                "   allowed=True\n"
                "  except OSError: allowed=False\n"
                "  answers.append(allowed)\n"
                "print(json.dumps(answers))\n"
            )
            result = tools.dispatch(
                "run_command",
                {"argv": ["/usr/bin/python3", "-c", probe, str(control), *map(str, records)]},
            ).to_wire()
            assert result["returncode"] == 0
            assert json.loads(str(result["stdout"])) == [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
            assert [p.read_bytes() for p in records] == saved
        finally:
            tools.close()
