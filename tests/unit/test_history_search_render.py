# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 history search` renders readable windowed hits, not raw JSON lines."""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import shutil
import tracemalloc

import pytest

from agent6 import paths
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import common, history_cmds


@pytest.mark.parametrize("query", ["NEEDLE", "absent"])
def test_search_read_error_keeps_hits_and_reports_incomplete(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    query: str,
) -> None:
    """A disappearing search target must not discard other sessions' matches."""
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is required")
    monkeypatch.chdir(tmp_path)
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    state_dir = paths.state_dir(tmp_path)
    assert state_dir.is_relative_to(state_home)
    runs = state_dir / sessions_layout.SESSIONS_ROOT / "runs"
    saved = runs / "saved-record"
    saved.mkdir(parents=True)
    (saved / "logs.jsonl").write_text(
        json.dumps({"type": "session.stop", "summary": "NEEDLE survived"}) + "\n",
        encoding="utf-8",
    )
    vanished = runs / "vanished-record"
    vanished.mkdir()
    discover = common.all_session_dirs

    def discover_then_remove(cwd: pathlib.Path) -> list[pathlib.Path]:
        targets = discover(cwd)
        vanished.rmdir()
        return targets

    monkeypatch.setattr(common, "all_session_dirs", discover_then_remove)
    assert history_cmds._cmd_history_search(query, fixed=True, session_id="") == 2
    output = capsys.readouterr()
    if query == "NEEDLE":
        assert "NEEDLE survived" in output.out
        assert "1 matching line in 1 session" in output.out
    else:
        assert output.out == ""
    assert "no matches" not in output.out
    assert "search incomplete: some files were not read." in output.err
    assert str(vanished) in output.err
    assert "No such file or directory" in output.err


@pytest.mark.parametrize("wire", ["anthropic", "openai", "responses"])
@pytest.mark.parametrize("fixed", [False, True])
def test_search_excludes_prompt_sections_but_keeps_conversation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    wire: str,
    fixed: bool,
) -> None:
    """Provider prompts and skill indexes do not crowd out conversation search hits."""
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is required")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(tmp_path), session_id="search")
    layout.ensure()
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "operator NEEDLE"},
        {"role": "assistant", "content": "model NEEDLE"},
        {"role": "tool", "content": "result NEEDLE"},
    ]
    prompt = "system NEEDLE <skills>skillindex NEEDLE</skills>"
    body: dict[str, object] = {"tools": [{"description": "declaration NEEDLE"}]}
    if wire == "openai":
        messages.insert(0, {"role": "system", "content": prompt})
        messages.insert(1, {"role": "developer", "content": "developer NEEDLE"})
        body["messages"] = messages
    elif wire == "anthropic":
        messages[-1] = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call1", "content": "result NEEDLE"}
            ],
        }
        body.update(system=[{"type": "text", "text": prompt}], messages=messages)
    else:
        messages[-1] = {
            "type": "function_call_output",
            "call_id": "call1",
            "output": "result NEEDLE",
        }
        body.update(instructions=prompt, input=messages)
    call = {"name": "run_command", "arguments": '{"instructions": "toolcall NEEDLE"}'}
    responses = {
        "anthropic": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call2", "name": "run_command", "input": call}],
        },
        "openai": {
            "choices": [{"message": {"role": "assistant", "tool_calls": [{"function": call}]}}]
        },
        "responses": {"output": [{"type": "function_call", "call_id": "call2", **call}]},
    }
    record = {"request": {"body": body}, "response": {"body": responses[wire]}}
    (layout.transcripts_dir / "0001.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    assert history_cmds._cmd_history_search("NEEDLE", fixed=fixed, session_id="") == 0
    out = capsys.readouterr().out
    for word in ("operator", "model", "result", "toolcall"):
        assert f"{word} NEEDLE" in out
    for word in ("system", "skillindex", "developer", "declaration"):
        assert f"{word} NEEDLE" not in out
    assert history_cmds._cmd_history_search("skillindex", fixed=fixed, session_id="") == 1
    assert "no matches" in capsys.readouterr().out


def test_search_skips_the_harness_copies_of_the_conversation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The prompt persisted in the snapshot and the checkpoints is not a conversation hit."""
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is required")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(tmp_path), session_id="copies")
    layout.ensure()
    copy = json.dumps({"system": "snapshot NEEDLE", "messages": []})
    (layout.session_dir / "loop_state.json").write_text(copy, encoding="utf-8")
    (layout.session_dir / "turn_in_flight.json").write_text(copy, encoding="utf-8")
    (layout.session_dir / "checkpoints").mkdir(exist_ok=True)
    (layout.session_dir / "checkpoints" / "0001.json").write_text(copy, encoding="utf-8")
    (layout.session_dir / "objects").mkdir(exist_ok=True)
    (layout.session_dir / "objects" / "ab12.json").write_text(copy, encoding="utf-8")
    (layout.session_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "task": "journal NEEDLE"}) + "\n", encoding="utf-8"
    )
    assert history_cmds._cmd_history_search("NEEDLE", fixed=True, session_id="") == 0
    out = capsys.readouterr().out
    assert "journal NEEDLE" in out
    assert "snapshot NEEDLE" not in out


def test_search_reports_unreadable_transcripts_without_dropping_other_hits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed transcript cannot silently turn a partial search into a complete one."""
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is required")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(tmp_path), session_id="search")
    layout.ensure()
    (layout.transcripts_dir / "0001.json").write_text('{"request":', encoding="utf-8")
    layout.logs_path.write_text(json.dumps({"text": "operator NEEDLE"}), encoding="utf-8")
    assert history_cmds._cmd_history_search("NEEDLE", fixed=True, session_id="") == 2
    out = capsys.readouterr()
    assert "operator NEEDLE" in out.out
    assert "search incomplete" in out.err and "0001.json" in out.err


@pytest.mark.parametrize("large_match", [False, True])
def test_search_memory_is_bounded_by_one_transcript(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    large_match: bool,
) -> None:
    """A corpus larger than the memory ceiling streams through ripgrep without accumulating."""
    if shutil.which("rg") is None:
        pytest.skip("ripgrep is required")
    monkeypatch.chdir(tmp_path)
    layout = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "stream-search")
    layout.ensure()
    layout.logs_path.write_text('{"type": "session.start"}\n')
    for index in range(40):
        record = {
            "request": {
                "body": {
                    "messages": [
                        {
                            "role": "user",
                            "content": (f"needle {index}: " if large_match else "")
                            + "x" * 1_000_000,
                        }
                    ]
                }
            },
            "response": {"body": {"text": f"needle {index}"}},
        }
        (layout.transcripts_dir / f"{index:04d}.json").write_text(json.dumps(record))
        del record
    tracemalloc.start()
    try:
        assert (
            history_cmds._cmd_history_search("needle", fixed=True, session_id=layout.session_id)
            == 0
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert "needle 39" in capsys.readouterr().out
    assert peak < 28_000_000, f"search retained {peak:,} bytes for a 40 MB corpus"
    with capsys.disabled():
        print(f"40 MB search peak (large_match={large_match}): {peak:,} bytes")


def _rg_match(path: str, line: str, col: int) -> str:
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
                "submatches": [{"start": col, "end": col + 1}],
            },
        }
    )


def test_window_clips_a_huge_line_around_the_match() -> None:
    line = "x" * 500 + "NEEDLE" + "y" * 500
    out = history_cmds._window(line, 500)
    assert "NEEDLE" in out
    assert out.startswith("…") and out.endswith("…")
    assert len(out) < 200  # not the whole 1000+ char line


def test_window_collapses_json_escaped_newlines() -> None:
    assert history_cmds._window("the\\nfirst message here", 0) == "the first message here"


def test_window_decodes_backslashes_not_bare_backslash_space() -> None:
    # A double-encoded newline decodes `\\\\` first, so no "\ " artifact is left.
    assert "\\ " not in history_cmds._window("cmd = tail \\\\nlog NEEDLE", 0)
    # An escaped backslash renders as one backslash and an escaped quote as a quote.
    assert (
        history_cmds._window('path C:\\\\Users and \\"quoted\\" NEEDLE', 0)
        == 'path C:\\Users and "quoted" NEEDLE'
    )


def test_run_id_from_path_finds_the_run_dir_child() -> None:
    assert (
        history_cmds._session_id_from_path(
            pathlib.Path("/s/sessions/runs/deep-poppy-AB/logs.jsonl")
        )
        == "deep-poppy-AB"
    )
    assert (
        history_cmds._session_id_from_path(
            pathlib.Path("/s/sessions/asks/quiet-fox-CD/transcripts/0003.json")
        )
        == "quiet-fox-CD"
    )
    # A state-base ancestor sharing a bucket name must not shadow the real bucket.
    assert (
        history_cmds._session_id_from_path(
            pathlib.Path("/mnt/runs/state/agent6/repo-x/sessions/runs/deep-poppy-AB/logs.jsonl")
        )
        == "deep-poppy-AB"
    )
    assert (
        history_cmds._session_id_from_path(
            pathlib.Path("/mnt/asks/state/agent6/repo-x/sessions/asks/quiet-fox-CD/logs.jsonl")
        )
        == "quiet-fox-CD"
    )
    # A valid explicit id may itself equal a bucket name.
    assert (
        history_cmds._session_id_from_path(pathlib.Path("/s/sessions/runs/runs/logs.jsonl"))
        == "runs"
    )


def test_parse_extracts_event_type_and_time_for_logs_jsonl() -> None:
    event = {"ts": "2026-07-12T09:15:30.1Z", "type": "tool.call", "name": "grep"}
    out = history_cmds._parse_rg_matches(
        _rg_match("/s/sessions/runs/r1/logs.jsonl", json.dumps(event), 40)
    )
    assert len(out) == 1
    assert out[0].session_id == "r1"
    assert out[0].kind == "tool.call"
    assert out[0].when == "09:15:30"


def test_a_valid_json_non_event_in_the_journal_degrades_to_a_file_hit() -> None:
    line = '["NEEDLE", "a partial write"]'
    out = history_cmds._parse_rg_matches(_rg_match("/s/sessions/runs/r1/logs.jsonl", line, 2))
    assert len(out) == 1
    assert (out[0].when, out[0].kind) == ("", "logs.jsonl")


def test_summary_counts_matching_lines_not_submatches(capsys: pytest.CaptureFixture[str]) -> None:
    line = "NEEDLE then NEEDLE"
    rec = json.loads(_rg_match("/s/sessions/runs/r1/notes.md", line, 0))
    rec["data"]["submatches"].append({"start": 12, "end": 18})
    hits = history_cmds._parse_rg_matches(json.dumps(rec))
    history_cmds._render_history_hits(hits, pathlib.Path("/s/sessions"))
    assert "1 matching line in 1 session" in capsys.readouterr().out


def test_search_summary_calls_an_ask_a_session(capsys: pytest.CaptureFixture[str]) -> None:
    hits = history_cmds._parse_rg_matches(
        _rg_match(
            "/s/sessions/asks/ask-one/logs.jsonl",
            json.dumps({"type": "session.start", "user_task": "find NEEDLE"}),
            52,
        )
    )

    history_cmds._render_history_hits(hits, pathlib.Path("/s/sessions"))

    summary = capsys.readouterr().out.splitlines()[-1]
    assert summary == "1 matching line in 1 session"
    assert "run" not in summary


def test_transcripts_share_one_label(capsys: pytest.CaptureFixture[str]) -> None:
    # The same snippet across cumulative snapshots collapses to one (xN) "transcript" line.
    lines = "\n".join(
        _rg_match(f"/s/sessions/runs/r1/transcripts/000{i}.json", '  "text": "hello NEEDLE",', 12)
        for i in (3, 5, 7)
    )
    hits = history_cmds._parse_rg_matches(lines)
    assert all(h.kind == "transcript" for h in hits)
    history_cmds._render_history_hits(hits, pathlib.Path("/s/runs"))
    out = capsys.readouterr().out
    assert "(x3)" in out  # three identical snapshot hits collapsed
    assert out.count("hello NEEDLE") == 1


def test_event_snippet_windows_inside_the_matched_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A match inside an event's string field snippets the field's prose, not raw JSON.
    event = json.dumps(
        {
            "ts": "2026-07-12T09:15:30.1Z",
            "type": "role.thinking_delta",
            "text": "I improved the NEEDLE of one bullet in README.md",
        }
    )
    hits = history_cmds._parse_rg_matches(
        _rg_match("/s/sessions/runs/r1/logs.jsonl", event, event.index("NEEDLE"))
    )
    assert hits[0].snippet == "I improved the NEEDLE of one bullet in README.md"
    history_cmds._render_history_hits(hits, pathlib.Path("/s/runs"))
    assert '"type"' not in capsys.readouterr().out


def test_one_task_in_many_encodings_collapses_to_the_readable_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One task string stored many ways prints one line per run with a count, keyed by the value.
    task = "Improve the wording of the end banner"
    lines: list[str] = []
    event = json.dumps({"ts": "2026-07-12T07:36:51.1Z", "type": "session.start", "user_task": task})
    lines.append(_rg_match("/s/sessions/runs/r1/logs.jsonl", event, event.index("Improve")))
    manifest = json.dumps({"version": 2, "user_task": task})
    lines.append(
        _rg_match("/s/sessions/runs/r1/manifest.json", manifest, manifest.index("Improve"))
    )
    body = f'        "text": "TASK:\\n{task}",'
    lines.append(
        _rg_match("/s/sessions/runs/r1/transcripts/0003.json", body, body.index("Improve"))
    )

    hits = history_cmds._parse_rg_matches("\n".join(lines))
    assert len({h.key for h in hits}) == 1  # every encoding shares the content key
    history_cmds._render_history_hits(hits, pathlib.Path("/s/runs"))
    out = capsys.readouterr().out
    assert out.count("Improve the wording") == 1  # one representative line
    assert "(x3)" in out  # all three encodings counted
    assert "manifest.json" not in out  # internals lose
    assert "session.start" in out  # the timestamped event wins


def _rg_match_bytes(path: str, line: str, needle: str) -> str:
    """An rg-faithful match record: submatch offsets in BYTES, as rg reports."""
    b = line.encode("utf-8")
    start = b.index(needle.encode("utf-8"))
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": path},
                "lines": {"text": line},
                "submatches": [{"start": start, "end": start + len(needle.encode("utf-8"))}],
            },
        }
    )


def test_byte_offsets_convert_to_characters_on_non_ascii_lines() -> None:
    # rg reports byte offsets; curly quotes earlier in the line broke character slicing.
    event = json.dumps(
        {
            "ts": "2026-07-12T09:15:30.1Z",
            "type": "role.text_delta",
            "text": "café ’naïve’ résumé… NEEDLE found mid prose",  # noqa: RUF001
        },
        ensure_ascii=False,
    )
    hits = history_cmds._parse_rg_matches(
        _rg_match_bytes("/s/sessions/runs/r1/logs.jsonl", event, "NEEDLE")
    )
    assert "NEEDLE found mid prose" in hits[0].snippet
    assert (
        hits[0].key
        == hashlib.sha256("café naïve résumé needle found mid prose".encode()).hexdigest()
    )


def test_ascii_escaped_and_raw_utf8_encodings_share_one_key() -> None:
    # logs.jsonl is raw UTF-8, manifests are ascii-escaped; the \uXXXX decode unifies them.
    task = "Improve the résumé wording NEEDLE of the banner"
    raw_event = json.dumps(
        {"ts": "2026-07-12T07:36:51.1Z", "type": "session.start", "user_task": task},
        ensure_ascii=False,
    )
    escaped_manifest = json.dumps({"version": 2, "user_task": task})  # ascii-escaped
    assert "\\u00e9" in escaped_manifest  # the divergence under test
    hits = history_cmds._parse_rg_matches(
        _rg_match_bytes("/s/sessions/runs/r1/logs.jsonl", raw_event, "NEEDLE")
        + "\n"
        + _rg_match_bytes("/s/sessions/runs/r1/manifest.json", escaped_manifest, "NEEDLE")
    )
    assert len(hits) == 2
    assert hits[0].key == hits[1].key


def test_distinct_sentences_ending_with_the_query_stay_distinct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A match at end-of-string has no following context; keying on the bare word merged statements.
    e1 = json.dumps(
        {
            "ts": "2026-07-12T08:00:00.1Z",
            "type": "role.text_delta",
            "text": "I deleted the broken NEEDLE",
        }
    )
    e2 = json.dumps(
        {
            "ts": "2026-07-12T09:30:00.1Z",
            "type": "role.text_delta",
            "text": "then I rewrote a completely new NEEDLE",
        }
    )
    hits = history_cmds._parse_rg_matches(
        _rg_match_bytes("/s/sessions/runs/r1/logs.jsonl", e1, "NEEDLE")
        + "\n"
        + _rg_match_bytes("/s/sessions/runs/r1/logs.jsonl", e2, "NEEDLE")
    )
    history_cmds._render_history_hits(hits, pathlib.Path("/s/runs"))
    out = capsys.readouterr().out
    assert "deleted the broken NEEDLE" in out
    assert "rewrote a completely new NEEDLE" in out
    assert "(x2)" not in out


def test_distinct_sentences_with_the_same_match_suffix_stay_distinct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Content dedupe includes the prose before the match, not only the query and its suffix."""
    lines = []
    for when, text in (
        ("08:00:00", "I deleted the NEEDLE from the parser"),
        ("09:30:00", "I restored the NEEDLE from the parser"),
    ):
        event = json.dumps({"ts": f"2026-07-12T{when}.1Z", "type": "role.text_delta", "text": text})
        lines.append(_rg_match_bytes("/s/sessions/runs/r1/logs.jsonl", event, "NEEDLE"))
    hits = history_cmds._parse_rg_matches("\n".join(lines))
    history_cmds._render_history_hits(hits, pathlib.Path("/s/sessions"))
    out = capsys.readouterr().out
    assert "deleted the NEEDLE" in out and "restored the NEEDLE" in out
    assert "(x2)" not in out


def test_deeply_nested_json_line_degrades_instead_of_crashing() -> None:
    depth = 100_000
    line = '{"a":' * depth + "1" + "}" * depth
    hits = history_cmds._parse_rg_matches(
        _rg_match_bytes("/s/sessions/runs/r1/logs.jsonl", line, '"a"')
    )
    assert len(hits) == 1  # fell back to the raw window, no RecursionError


def test_string_walk_survives_nesting_the_parser_accepted() -> None:
    """The walk over a parsed JSON line is iterative, so deep nesting cannot crash it.

    json.loads accepts nesting up to a stack ceiling that varies by interpreter and arch.
    """
    deep: dict[str, object] = {"leaf": "NEEDLE"}
    for _ in range(100_000):
        deep = {"a": deep}
    assert list(history_cmds._strings_in(deep)) == ["NEEDLE"]


def test_a_line_that_is_not_utf8_still_parses_from_its_bytes() -> None:
    """Rg's base64 `bytes` line decodes with U+FFFD, keeping the hit's kind and snippet."""
    event = (
        b'{"type": "tool.call", "ts": "2026-09-02T09:15:30+00:00", "name": "run_command",'
        b' "args_preview": {"argv": ["grep", "caf\xe9 NEEDLE"]}}'
    )
    start = event.index(b"NEEDLE")
    rec = {
        "type": "match",
        "data": {
            "path": {"text": "/s/sessions/runs/r1/logs.jsonl"},
            "lines": {"bytes": base64.b64encode(event).decode("ascii")},
            "line_number": 1,
            "absolute_offset": 0,
            "submatches": [{"match": {"text": "NEEDLE"}, "start": start, "end": start + 6}],
        },
    }
    out = history_cmds._parse_rg_matches(json.dumps(rec))
    assert len(out) == 1
    assert (out[0].session_id, out[0].kind) == ("r1", "tool.call")
    assert "NEEDLE" in out[0].snippet and "caf\ufffd" in out[0].snippet


def test_byte_offsets_map_onto_the_decoded_line_past_a_byte_that_is_not_utf8() -> None:
    """The base64 branch's match window counts the characters the byte prefix decodes to.

    Re-encoding U+FFFD grows every non-UTF-8 byte to three and slides the window early.
    """
    line = b"caf\xe9 NEEDLE and \xe2\x80\x9cmore\xe2\x80\x9d"
    start, end = history_cmds._char_span(line, line.index(b"NEEDLE"), line.index(b"NEEDLE") + 6)
    assert line.decode("utf-8", "replace")[start:end] == "NEEDLE"
    start, end = history_cmds._char_span(line, line.index(b"more"), line.index(b"more") + 4)
    assert line.decode("utf-8", "replace")[start:end] == "more"


def test_a_scoped_search_names_the_session_it_searched(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`history search TERM --session ID` searches one session dir; an empty result names it."""
    import shutil

    from agent6 import paths
    from agent6.sessions import layout
    from agent6.ui.cli import main

    if shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    monkeypatch.chdir(tmp_path)
    runs = layout.bucket_dir(paths.state_dir(tmp_path), "runs")
    for sid, text in (("quiet-run-AAAAAA", "nothing here"), ("other-run-BBBBBB", "UNIQUETERM123")):
        (runs / sid).mkdir(parents=True)
        (runs / sid / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "user_task": text}) + "\n", encoding="utf-8"
        )
    assert main(["history", "search", "UNIQUETERM123", "--session", "quiet-run-AAAAAA"]) == 1
    out = capsys.readouterr().out
    assert "no matches under" in out and "quiet-run-AAAAAA" in out


@pytest.mark.parametrize("verb", ["graph", "transcript"])
def test_cross_bucket_inspection_errors_call_an_ask_a_session(
    verb: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent6 import paths
    from agent6.sessions import layout
    from agent6.ui.cli import main

    monkeypatch.chdir(tmp_path)
    session = layout.bucket_dir(paths.state_dir(tmp_path), "asks") / "ask-one-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "ask"}) + "\n", encoding="utf-8"
    )

    assert main(["sessions", verb, "ask-one-AAAAAA"]) == 2
    err = capsys.readouterr().err
    assert "session ask-one-AAAAAA has no" in err
    assert "run ask-one-AAAAAA" not in err


def test_an_unscoped_no_match_names_the_sessions_root_it_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `history search` reads `<state>/sessions`, and an empty result names that."""
    import shutil

    from agent6 import paths
    from agent6.context import memory
    from agent6.sessions import layout
    from agent6.ui.cli import main

    if shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
    run_dir = layout.bucket_dir(state, "runs") / "quiet-run-AAAAAA"
    run_dir.mkdir(parents=True)
    (run_dir / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "nothing here"}) + "\n",
        encoding="utf-8",
    )
    memory.add(state, "rounding-rule", "UNIQUETERM123 is banker's rounding.")
    assert main(["history", "search", "UNIQUETERM123"]) == 1
    assert f"no matches under {state / 'sessions'}." in capsys.readouterr().out
