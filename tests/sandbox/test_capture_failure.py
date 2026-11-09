from __future__ import annotations

import json
import os
import pathlib
import signal
import time

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_session
from agent6.tools import background


@pytest.mark.parametrize(
    ("isolation", "lose_worker"),
    [("none", False), ("partial", False), ("full", False), ("none", True), ("partial", True)],
)
@pytest.mark.parametrize("observe_exit", [False, True])
def test_launcher_capture_failure_is_visible_before_stop(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    isolation: kinds.IsolationLevel,
    lose_worker: bool,
    observe_exit: bool,
) -> None:
    """A full target capture keeps its partial bytes and reports loss before cleanup."""
    binary = jail.require_jail_binary()
    wrapper = tmp_path / "limited-launcher"
    wrapper.write_text(
        "#!/usr/bin/python3\nimport os, resource, signal, sys\n"
        "signal.signal(signal.SIGXFSZ, signal.SIG_IGN)\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))\n"
        f"os.execv({str(binary)!r}, [{str(binary)!r}, *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(wrapper))
    work = tmp_path / "work"
    work.mkdir()
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=work, argv=(), isolation=isolation)
    )
    shells = background.BackgroundShells(tmp_path / "shells")
    try:
        view = shells.start(
            (
                "/usr/bin/python3",
                "-c",
                "import os, pathlib, time; os.write(1, b'x' * 8192); "
                "pathlib.Path('emitted').touch(); "
                "exec(\"while not pathlib.Path('release').exists(): time.sleep(0.01)\")",
            ),
            lambda a, rw: kinds.JailPolicy(cwd=work, argv=a, isolation=isolation),
            session=session,
        )
        deadline = time.monotonic() + 5
        while not (work / "emitted").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        while True:
            running, output = shells.read(view.id, tail_lines=10)
            if "output incomplete" in output or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert (shells.log_root / view.id / "out.log").read_bytes() == b"x" * 1024
        assert running.state == "running" and running.returncode is None
        assert "output incomplete" in output and "File too large" in output
        shell = shells._get(view.id)  # pyright: ignore[reportPrivateUsage]
        status = session.status_background(shell.job._handle)  # pyright: ignore[reportPrivateUsage]
        assert status.running and "File too large" in status.capture_error
        if observe_exit:
            (work / "release").touch()
            ended, output = shells.read(view.id, tail_lines=10, wait_s=5)
            assert ended.state == "exited" and ended.returncode == 0
            assert "output incomplete" in output and "File too large" in output
            assert "File too large" in ended.detail
            replay = background.roster_from_dir(tmp_path / "shells")
            assert "exited 0" in replay[0] and "File too large" in replay[0]
        if lose_worker:
            pid = session._proc.pid  # pyright: ignore[reportPrivateUsage]
            worker = int(pathlib.Path(f"/proc/{pid}/task/{pid}/children").read_text())
            os.kill(worker, signal.SIGKILL)
            reason = "job outcome or cleanup unknown"
        else:
            if observe_exit:
                stopped = shells.stop(view.id)
                assert stopped.returncode == 0 and stopped.state != "stop failed"
            session._proc.kill()  # pyright: ignore[reportPrivateUsage]
            session._proc.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]
            reason = "jail session protocol failed"
        lost, output = shells.read(view.id, tail_lines=10)
        code = 0 if observe_exit else None
        assert lost.returncode == code and "output incomplete" in output
        replay = background.roster_from_dir(tmp_path / "shells")[0]
        assert ("exited 0" if observe_exit else "outcome unknown") in replay
        for detail in (lost.detail, output, replay):
            assert "File too large" in detail and reason in detail
        record = json.loads((shell.dir / "result.json").read_text())
        assert "File too large" in record["capture_error"]
        assert reason in record["error"] or reason in record["capture_error"]
        shells.stop_all()
        after = json.loads((shell.dir / "result.json").read_text())
        assert after["returncode"] == code and "File too large" in after["capture_error"]
        assert reason in after["error"] or reason in after["capture_error"]
        if observe_exit:
            assert after["capture_error"] == record["capture_error"]
        stopped = shells.roster()[0]
        replay = background.roster_from_dir(tmp_path / "shells")[0]
        for detail in (stopped.detail, replay):
            assert "File too large" in detail and reason in detail
    finally:
        shells.stop_all()
        session.close()
