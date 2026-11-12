from __future__ import annotations

import json
import os
import pathlib

import pytest

from agent6 import paths
from agent6.sessions import manifest
from agent6.ui.cli import common
from agent6.viewmodel import listing


@pytest.mark.parametrize("operation", ["newest", "ordering", "holding"])
def test_ordering_reads_tails_not_whole_journals(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Selecting or ordering 100 sessions must not replay their 100+ MiB of journal history."""
    counters = pathlib.Path("/proc/self/io")
    if not counters.exists():
        pytest.skip("read-byte accounting requires Linux /proc/self/io")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    state = paths.state_dir(repo)
    body = (
        json.dumps({"ts": 1000, "type": "model.text_delta", "text": "x" * 64}) + "\n"
    ).encode() * 10_000
    sessions: list[pathlib.Path] = []
    for i in range(100):
        bucket = ("runs", "asks", "plans")[i % 3]
        session = state / "sessions" / bucket / f"s{i:04}"
        session.mkdir(parents=True)
        (session / "manifest.json").write_text(
            manifest.SessionManifest(session_id=session.name).model_dump_json()
        )
        log = session / "logs.jsonl"
        log.write_bytes(
            body + (json.dumps({"type": "session.stop", "ts": 1800000000 + i}) + "\n").encode()
        )
        os.utime(log, (2000000000 - i, 2000000000 - i))
        sessions.append(session)
    assert len(body) * len(sessions) > 100 * 1024 * 1024

    before = int(counters.read_text().split("rchar: ")[1].splitlines()[0])
    if operation == "ordering":
        assert listing.session_dirs(state) == list(reversed(sessions))
    else:
        found = (
            common.resolve_or_newest_layout(repo, "")
            if operation == "newest"
            else common.newest_layout_holding(repo, "logs.jsonl")
        )
        assert found is not None and found.session_dir == sessions[-1]
    consumed = int(counters.read_text().split("rchar: ")[1].splitlines()[0]) - before
    assert consumed < 8 * 1024 * 1024, f"session ordering read {consumed:,} bytes"


@pytest.mark.parametrize(
    ("journal", "expected"),
    [
        (b'{"ts": 2000}\n{"ts": 1000}\n', 1000.0),
        (b'{"ts": "2020-01-02T04:04:05+01:00"}\n', 1577934245.0),
        (b'{"ts": 2000}\n{"ts": 0}', 0.0),
        (b'{"ts": 1000}\n\n[2000]\nnull\n{"ts": true}\n{"ts": "bad"}\n{}\n', 1000.0),
        (b'{"ts": 1000}\n{"ts": 2000, "text": "torn\xe2\x82', 1000.0),
        (b'{"ts": 1000, "text": "bad utf8: \xff"}\n', 1000.0),
        (b'{"ts": 1000}\r\n{"ts": 2000}\r\n', 2000.0),
        (b'{"ts": 1000}\n' + b'{"text": "' + b"x" * 150_000 + b'", "ts": 2000}\n', 2000.0),
        (b'{"ts": 1000}\n' + b'{"ts": 2000, "text": "' + b"x" * 150_000, 1000.0),
        (b'{"ts": 1000}\n' + b'{"text": "' + "\u2603".encode() * 50_000 + b'"}\n', 1000.0),
    ],
    ids=[
        "last-not-maximum",
        "iso-offset",
        "zero-without-newline",
        "unusable-timestamps",
        "torn-utf8",
        "replacement-decoding",
        "crlf",
        "large-last-event",
        "large-torn-tail",
        "large-untimed-unicode-event",
    ],
)
def test_activity_keeps_the_last_readable_recorded_time(
    tmp_path: pathlib.Path, journal: bytes, expected: float
) -> None:
    """Tail lookup agrees with the full scan, including large, untimed and torn final lines."""
    from agent6.viewmodel.session import log as session_log

    log = tmp_path / "logs.jsonl"
    log.write_bytes(journal)
    os.utime(log, (9000, 9000))
    assert listing.session_mtime(tmp_path) == expected
    assert session_log.scan_session_log(log).last_ep == expected


@pytest.mark.parametrize("journal", [None, b"", b"not json\n{}\n"])
def test_activity_without_timestamps_uses_manifest_start(
    tmp_path: pathlib.Path, journal: bytes | None
) -> None:
    """A missing, empty or untimed journal keeps the recorded start, not filesystem times."""
    (tmp_path / "manifest.json").write_text(
        manifest.SessionManifest(start_ts="2020-01-02T03:04:05Z").model_dump_json()
    )
    if journal is not None:
        (tmp_path / "logs.jsonl").write_bytes(journal)
    assert listing.session_mtime(tmp_path) == 1577934245.0


def test_activity_follows_appends_and_replacement(tmp_path: pathlib.Path) -> None:
    """Neither an unchanged mtime nor a session.stop hides later recorded activity."""
    log = tmp_path / "logs.jsonl"
    log.write_text('{"type": "session.stop", "ts": 1000}\n')
    os.utime(log, (9000, 9000))
    assert listing.session_mtime(tmp_path) == 1000
    with log.open("a") as stream:
        stream.write('{"type": "model.text_delta", "ts": 2000}\n')
    os.utime(log, (9000, 9000))
    assert listing.session_mtime(tmp_path) == 2000
    log.write_text('{"ts": 3000}\n')
    os.utime(log, (9000, 9000))
    assert listing.session_mtime(tmp_path) == 3000
