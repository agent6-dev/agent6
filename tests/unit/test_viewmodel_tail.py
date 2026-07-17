# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the stdlib JSONL tail-follower."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from agent6.viewmodel.tail import LogTail, tail_events


def test_tail_yields_existing_lines_in_non_follow_mode(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "run.start"}) + "\n" + json.dumps({"type": "run.end"}) + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["run.start", "run.end"]


def test_tail_yields_final_line_without_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(json.dumps({"type": "run.start"}) + "\n" + json.dumps({"type": "run.end"}))
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["run.start", "run.end"]


def test_tail_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "a"}) + "\n" + "{not json\n" + json.dumps({"type": "b"}) + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["a", "b"]


def test_tail_skips_non_dict_json(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text("[]\n" + json.dumps({"type": "x"}) + "\n", encoding="utf-8")
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["x"]


def test_tail_returns_when_file_missing_and_not_follow(tmp_path: Path) -> None:
    out = list(tail_events(tmp_path / "missing", follow=False))
    assert out == []


def test_tail_stops_at_run_end_when_requested(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "a"}) + "\n" + json.dumps({"type": "run.end"}) + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False, stop_when_finished=True))
    assert [e["type"] for e in out] == ["a", "run.end"]


def test_tail_does_not_stop_at_a_run_end_a_resume_superseded(tmp_path: Path) -> None:
    # A resume appends events AFTER run.end (no second end yet while it runs).
    # A watcher opening the log mid-resume must stream past the stale end, not
    # close on it and show the run as finished while it is live again.
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "run.start"})
        + "\n"
        + json.dumps({"type": "run.end", "reason": "steer_abort"})
        + "\n"
        + json.dumps({"type": "loop.resume.start"})
        + "\n"
        + json.dumps({"type": "role.call"})
        + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False, stop_when_finished=True))
    assert [e["type"] for e in out] == ["run.start", "run.end", "loop.resume.start", "role.call"]


def _torn_utf8_line() -> tuple[bytes, bytes]:
    """A JSON line split in the middle of a multibyte UTF-8 sequence (the first
    byte of the é lands in the first chunk)."""
    full = json.dumps({"type": "role.text_delta", "text": "café"}, ensure_ascii=False).encode()
    cut = full.rindex(b"\xc3\xa9") + 1
    return full[:cut], full[cut:]


def test_tail_survives_torn_utf8_tail(tmp_path: Path) -> None:
    # Writers flush >8KB lines in multiple syscalls, so a poll can hit EOF in
    # the middle of a multibyte UTF-8 sequence. The complete lines must come
    # through and the torn tail must not raise UnicodeDecodeError.
    p = tmp_path / "logs.jsonl"
    head, _rest = _torn_utf8_line()
    p.write_bytes(json.dumps({"type": "run.start"}).encode() + b"\n" + head)
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["run.start"]


def test_tail_completes_torn_utf8_line_across_polls(tmp_path: Path) -> None:
    # Follow mode: the torn byte tail stays pending and yields once the rest of
    # the line (including the newline) arrives.
    p = tmp_path / "logs.jsonl"
    head, rest = _torn_utf8_line()
    p.write_bytes(json.dumps({"type": "first"}).encode() + b"\n" + head)

    def writer() -> None:
        time.sleep(0.3)
        with p.open("ab") as fh:
            fh.write(rest + b"\n")
            fh.write(json.dumps({"type": "run.end"}).encode() + b"\n")
            fh.flush()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    out = list(tail_events(p, follow=True, poll_s=0.05, stop_when_finished=True))
    t.join(timeout=2)
    assert [e["type"] for e in out] == ["first", "role.text_delta", "run.end"]
    assert out[1]["text"] == "café"


def test_tail_follows_appended_lines(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(json.dumps({"type": "first"}) + "\n", encoding="utf-8")

    def writer() -> None:
        time.sleep(0.3)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "second"}) + "\n")
            fh.flush()
        time.sleep(0.2)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "run.end"}) + "\n")
            fh.flush()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    out = list(tail_events(p, follow=True, poll_s=0.05, stop_when_finished=True))
    t.join(timeout=2)
    assert [e["type"] for e in out] == ["first", "second", "run.end"]


def test_logtail_reads_only_new_events_incrementally(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(json.dumps({"type": "a"}) + "\n", encoding="utf-8")
    tail = LogTail(p)
    assert [e["type"] for e in tail.read()] == ["a"]
    assert tail.read() == []  # nothing new
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "b"}) + "\n")
    assert [e["type"] for e in tail.read()] == ["b"]  # only the appended event


def test_logtail_holds_a_partial_line_until_its_newline(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_bytes(b'{"type": "a"}\n{"type": "b"')  # second line has no newline yet
    tail = LogTail(p)
    assert [e["type"] for e in tail.read()] == ["a"]  # the torn line is withheld
    with p.open("a", encoding="utf-8") as fh:
        fh.write("}\n")
    assert [e["type"] for e in tail.read()] == ["b"]  # completed on the next read


def test_logtail_missing_file_is_empty(tmp_path: Path) -> None:
    assert LogTail(tmp_path / "nope.jsonl").read() == []


def test_stop_when_finished_stops_at_a_lone_run_end(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        '{"type": "run.start"}\n'
        '{"type": "tool.call", "name": "x"}\n'
        '{"type": "run.end", "reason": "finish_run"}\n',
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=True, stop_when_finished=True))
    assert [e["type"] for e in out] == ["run.start", "tool.call", "run.end"]


def test_stop_when_finished_follows_through_a_resumed_run(tmp_path: Path) -> None:
    # A stop then resume shares one log: two run.end events. stop_when_finished must
    # follow past the intermediate one (steer_abort) to the final one, not halt at
    # the stop -- else a watcher of a resumed run wrongly shows "stopped".
    p = tmp_path / "logs.jsonl"
    p.write_text(
        '{"type": "run.start"}\n'
        '{"type": "run.end", "reason": "steer_abort"}\n'
        '{"type": "role.call"}\n'
        '{"type": "run.end", "reason": "finish_run"}\n',
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=True, stop_when_finished=True))
    assert [e.get("reason") for e in out if e["type"] == "run.end"] == ["steer_abort", "finish_run"]
    assert out[-1]["reason"] == "finish_run"  # stopped at the final end, not the stop
