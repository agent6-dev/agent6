from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_session
from agent6.sessions import shells as session_shells
from agent6.tools import background


@pytest.mark.parametrize(
    ("changed", "padding"),
    [
        pytest.param({"offset": 1}, 0, id="wrong-start"),
        pytest.param({"next_offset": 0}, 0, id="wrong-next"),
        pytest.param({"size": 0}, 0, id="beyond-size"),
        pytest.param({"data": [], "next_offset": 0}, 0, id="no-progress"),
        pytest.param({"data": [-1]}, 0, id="negative-byte"),
        pytest.param({"data": [256]}, 0, id="large-byte"),
        pytest.param({"data": [True]}, 0, id="boolean-byte"),
        pytest.param({"data": [1.5]}, 0, id="fractional-byte"),
        pytest.param({"data": [65, 66], "next_offset": 2, "size": 2}, 0, id="beyond-limit"),
        pytest.param({}, 400000, id="oversized-frame"),
    ],
)
def test_invalid_read_response_abandons_the_peer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
    padding: int,
) -> None:
    """Read validation rejects hostile offsets, bytes and frames and reaps the peer."""
    answer = {
        "id": 1,
        "instance": "a" * 32,
        "offset": 0,
        "next_offset": 1,
        "size": 1,
        "data": [65],
        "capture_error": "",
        **changed,
    }
    peer = tmp_path / "peer"
    peer.write_text(
        "#!/usr/bin/python3\nimport json, signal, sys\nsignal.alarm(10)\n"
        "sys.stdin.readline()\n"
        "print(json.dumps({'ready': True, 'protocol': 4, 'instance': 'a'*32}), flush=True)\n"
        "request = json.loads(sys.stdin.readline())\nassert request['kind'] == 'read'\n"
        f"sys.stdout.write(json.dumps({answer!r}) + ' ' * {padding} + '\\n')\n"
        "sys.stdout.flush()\nsys.stdin.read()\n"
    )
    peer.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(peer))
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    try:
        reason = r"frame.*exceeds" if padding else "invalid background read"
        with pytest.raises(jail.JailUnavailableError, match=reason):
            session.read_background("owned", offset=0, limit=1)
        proc = session._proc  # pyright: ignore[reportPrivateUsage]
        assert proc.poll() is not None
        assert proc.stdin is not None and proc.stdin.closed
        assert proc.stdout is not None and proc.stdout.closed
    finally:
        session.close()


def test_invalid_read_keeps_its_reason_after_status_and_stop_fail(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost read keeps its first diagnosis through live reads, stop and replay."""
    peer = tmp_path / "peer"
    peer.write_text(
        "#!/usr/bin/python3\nimport json, signal, sys\nsignal.alarm(10)\n"
        "sys.stdin.readline()\n"
        "print(json.dumps({'ready': True, 'protocol': 4, 'instance': 'a'*32}), flush=True)\n"
        "reads = 0\n"
        "for line in sys.stdin:\n"
        " request = json.loads(line)\n"
        " if request['kind'] == 'background':\n"
        "  answer = {'handle': 'owned'}\n"
        " elif request['kind'] == 'status':\n"
        "  answer = {'running': True, 'returncode': None, 'error': '', 'capture_error': ''}\n"
        " elif request['kind'] == 'read':\n"
        "  reads += 1\n"
        "  data = list(b'kept\\n') if reads == 1 else []\n"
        "  offset = request['offset']\n"
        "  answer = {'offset': offset if reads == 1 else offset + 1,\n"
        "   'next_offset': offset + len(data), 'size': 5, 'data': data, 'capture_error': ''}\n"
        " else: raise AssertionError(request)\n"
        " print(json.dumps({'id': request['id'], 'instance': 'a'*32, **answer}), flush=True)\n"
    )
    peer.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(peer))
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation="none")
    )
    shells = background.BackgroundShells(tmp_path / "state")
    try:
        view = shells.start(
            ("producer",),
            lambda argv, rw: kinds.JailPolicy(cwd=tmp_path, argv=argv, isolation="none"),
            session=session,
        )
        assert view.state == "running"
        for stop in (False, True):
            if stop:
                shells.stop(view.id)
            view, text = shells.read(view.id, tail_lines=10)
            assert view.returncode is None
            assert "invalid background read" in view.detail
            assert "output incomplete" in text and "invalid background read" in text
            assert "kept" in text
            record = json.loads((tmp_path / "state" / view.id / "result.json").read_text())
            assert "invalid background read" in record["error"]
            assert "invalid background read" in record["capture_error"]
            assert (
                "invalid background read" in session_shells.roster_from_dir(tmp_path / "state")[0]
            )
        assert session._proc.poll() is not None  # pyright: ignore[reportPrivateUsage]
    finally:
        shells.stop_all()
        session.close()
