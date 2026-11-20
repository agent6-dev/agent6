# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A session can read this project's other sessions."""

from __future__ import annotations

import json
import os
import pathlib
import tracemalloc
from collections.abc import Callable
from typing import Any

import pytest

from agent6.sessions import event_log, journal
from agent6.sessions import layout as sessions_layout
from agent6.tools import result_format, results, schema, sessions


def _session(
    state: pathlib.Path, bucket: str, sid: str, mode: str, task: str, turns: list[str]
) -> pathlib.Path:
    d = state / "sessions" / bucket / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "mode": mode,
                "session_id": sid,
                "user_task": task,
                "start_ts": "2026-07-31T01:00:00",
            }
        ),
        encoding="utf-8",
    )
    lines = [json.dumps({"type": "session.start", "user_task": task})]
    lines += [json.dumps({"type": "model.result", "text": t}) for t in turns]
    (d / "logs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_the_roster_spans_every_bucket(tmp_path: pathlib.Path) -> None:
    """The roster spans every bucket.

    A run, a plan and an ask are all sessions; a roster of runs alone hides exactly the quick ask
    the model wants to pick up.
    """
    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask", "how do I convert h264", ["use ffmpeg"])
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "add a flag", ["done"])
    modes = {b.id: b.mode for b in sessions.session_briefs(tmp_path)}
    assert modes == {"quiet-fox-AAAAAA": "ask", "brave-elk-BBBBBB": "run"}


def test_the_conversation_reads_oldest_first(tmp_path: pathlib.Path) -> None:
    d = _session(
        tmp_path,
        "asks",
        "quiet-fox-AAAAAA",
        "ask",
        "how do I convert h264",
        ["use ffmpeg", "with libx265"],
    )
    layout = sessions_layout.session_layout(tmp_path, "quiet-fox-AAAAAA")
    assert layout is not None and layout.session_dir == d
    text = sessions.conversation(layout, max_chars=10_000)
    assert text.index("how do I convert") < text.index("use ffmpeg") < text.index("libx265")
    assert text.startswith("user: how do I convert")


@pytest.mark.parametrize(("code", "ending"), [(None, "handed off as bg1"), (7, "bg1, exit 7")])
def test_a_handoff_only_reports_a_recorded_exit(
    tmp_path: pathlib.Path, code: int | None, ending: str
) -> None:
    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="proof-p1-l1")
    shell = layout.session_dir / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(json.dumps({"command": "make check"}))
    if code is not None:
        (shell / "result.json").write_text(json.dumps({"returncode": code}))
    rows = [
        {
            "type": "tool.call",
            "name": "run_command",
            "call_id": 1,
            "args": {"argv": ["make", "check"]},
        },
        {"type": "tool.result", "name": "run_command", "call_id": 1, "background_id": "bg1"},
    ]
    layout.logs_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert (
        sessions.conversation(layout, max_chars=300) == f"[tool] run_command make check ({ending})"
    )


def test_truncation_keeps_the_tail(tmp_path: pathlib.Path) -> None:
    """Truncation keeps the tail.

    A later session usually wants what the earlier one concluded; the head is the task the roster
    already carries.
    """
    _session(tmp_path, "runs", "long-BBBBBB", "run", "t", ["x" * 400, "THE ANSWER"])
    layout = sessions_layout.session_layout(tmp_path, "long-BBBBBB")
    assert layout is not None
    text = sessions.conversation(layout, max_chars=200)
    assert "THE ANSWER" in text
    assert "earlier characters elided" in text


def test_a_query_finds_a_session_by_its_content(tmp_path: pathlib.Path) -> None:
    """A query finds a session by its content.

    An id is useless to a model that does not know it; content and recency are how a session is
    found.
    """
    _session(
        tmp_path, "asks", "quiet-fox-AAAAAA", "ask", "video question", ["use ffmpeg -c:v libx265"]
    )
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "add a flag", ["unrelated"])
    assert [b.id for b in sessions.roster(tmp_path, "libx265").briefs] == ["quiet-fox-AAAAAA"]
    assert [b.id for b in sessions.roster(tmp_path, "add a flag").briefs] == ["brave-elk-BBBBBB"]
    assert sessions.roster(tmp_path, "nothing here").briefs == ()


def test_a_side_calls_answer_is_not_read_as_the_assistants_own_words(
    tmp_path: pathlib.Path,
) -> None:
    """A side call's answer is not read as the assistant's own words.

    Side calls emit `model.result` too; only the worker's answer belongs in the assistant's reply.
    """
    d = _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "t", [])
    lines = (d / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps({"type": "model.result", "role": "worker", "text": "use ffmpeg"}))
    side = ("review:security", "verify_inferer", "squash", "summariser")
    for role in side:
        lines.append(json.dumps({"type": "model.result", "role": role, "text": f"FROM {role}"}))
    (d / "logs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    layout = sessions_layout.session_layout(tmp_path, "brave-elk-BBBBBB")
    assert layout is not None
    text = sessions.conversation(layout, max_chars=10_000)
    assert "use ffmpeg" in text
    assert [role for role in side if f"FROM {role}" in text] == []


def test_a_role_result_with_no_role_reads_as_the_sessions_own_words(tmp_path: pathlib.Path) -> None:
    """Older journals carry no `role` field on `model.result`; that absence is not a side role."""
    d = _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "t", [])
    lines = (d / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps({"type": "model.result", "text": "use ffmpeg"}))
    lines.append(json.dumps({"type": "model.result", "role": "", "text": "with libx265"}))
    (d / "logs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    layout = sessions_layout.session_layout(tmp_path, "brave-elk-BBBBBB")
    assert layout is not None
    text = sessions.conversation(layout, max_chars=10_000)
    assert "use ffmpeg" in text
    assert "with libx265" in text


def test_a_torn_journal_line_does_not_break_the_read(tmp_path: pathlib.Path) -> None:
    """A live session's last line can be half-written."""
    d = _session(tmp_path, "runs", "live-BBBBBB", "run", "t", ["first"])
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "role.res')
    layout = sessions_layout.session_layout(tmp_path, "live-BBBBBB")
    assert layout is not None
    assert "first" in sessions.conversation(layout, max_chars=10_000)


def test_a_session_with_no_conversation_says_so(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "sessions" / "runs" / "empty-BBBBBB"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "run"}), encoding="utf-8")
    layout = sessions_layout.session_layout(tmp_path, "empty-BBBBBB")
    assert layout is not None
    assert "no readable journal" in sessions.conversation(layout, max_chars=100)


@pytest.mark.parametrize("escape", ["../../etc", "..", "/etc/passwd", "a/../../b"])
def test_no_path_from_the_model_reaches_the_filesystem(tmp_path: pathlib.Path, escape: str) -> None:
    """The model names a session by id, never a path.

    Resolution matches real directory names in the project's buckets, so traversal cannot resolve.
    """
    _session(tmp_path, "runs", "brave-elk-BBBBBB", "run", "t", ["x"])
    assert sessions_layout.session_layout(tmp_path, escape) is None


def test_the_tool_returns_the_roster_and_refuses_an_unknown_id(tmp_path: pathlib.Path) -> None:
    """An id lookup returns just the requested conversation or a refusal."""
    from agent6.config import Config
    from agent6.tools import dispatch, tool_errors

    _session(tmp_path, "asks", "quiet-fox-AAAAAA", "ask", "video question", ["use ffmpeg"])
    d = dispatch.ToolDispatcher(root=tmp_path, config=Config(), state_dir=tmp_path)
    out = d.dispatch("read_session", {}).to_wire()
    assert out["sessions"] == ["[quiet-fox-AAAAAA] ask · 2026-07-31T01:00: video question"]
    assert "conversation" not in out
    got = d.dispatch("read_session", {"id": "quiet-fox-AAAAAA"}).to_wire()
    assert "use ffmpeg" in got["conversation"]
    assert "sessions" not in got
    with pytest.raises(tool_errors.ToolError, match="no session"):
        d.dispatch("read_session", {"id": "nope"})


def _reader(state: pathlib.Path, rows: list[dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    from agent6.config import Config
    from agent6.tools import dispatch

    session = _session(state, "runs", "calm-owl-AAAAAA", "run", "the task", [])
    (session / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    d = dispatch.ToolDispatcher(root=state, config=Config(), state_dir=state)

    def read(**filters: Any) -> dict[str, Any]:
        return d.dispatch("read_session", {"id": session.name, **filters}).to_wire()

    return read


def test_filtered_entries_keep_their_journal_number_and_time(tmp_path: pathlib.Path) -> None:
    read = _reader(
        tmp_path,
        [
            {
                "ts": "2026-07-31T12:31:07+00:00",
                "type": "session.start",
                "task_source": "operator",
                "user_task": "Keep [a].",
            },
            {"type": "model.text_delta", "text": "not a second copy"},
            {"type": "model.result", "role": "summariser", "text": "side call"},
            {"ts": "2026-07-31T12:32:18+00:00", "type": "model.result", "text": "Check [a]."},
            {"ts": "2026-07-31T12:31:59+00:00", "type": "steer.injected", "text": "Use [A]."},
            {"type": "steer.parent", "text": "Also [a]."},
        ],
    )
    assert read(query="[a]", kinds=["operator"], limit=1) == {
        "conversation": "#1 12:31 operator: Keep [a].",
        "next_start": 3,
    }
    assert read(start=3, query="[a]", kinds=["operator"]) == {
        "conversation": "#3 12:31 operator: Use [A]."
    }
    text = read()["conversation"]
    assert text.splitlines() == [
        "#1 12:31 operator: Keep [a].",
        "#2 12:32 model: Check [a].",
        "#3 12:31 operator: Use [A].",
        "#4 --:-- model: parent: Also [a].",
    ]
    assert "no matching entries" in read(query=".*")["conversation"]


def test_tool_entries_show_bounded_arguments_and_structured_outcomes(
    tmp_path: pathlib.Path,
) -> None:
    read = _reader(
        tmp_path,
        [
            {
                "type": "tool.call",
                "name": "read_file",
                "args_preview": {"path": "x" * 230, "start_line": 8, "api_key": "private key"},
                "args": {"path": "not the preview"},
            },
            {"type": "steer.injected", "text": "Arrived before the result."},
            {
                "type": "tool.result",
                "name": "read_file",
                "ok": True,
                "summary": "private descriptive text",
                "lines_total": 23,
                "head_tail": "private file content",
                "stdout_tail": "private output",
            },
        ],
    )
    session = tmp_path / "sessions" / "runs" / "calm-owl-AAAAAA"
    (session / "transcripts").mkdir()
    (session / "transcripts" / "1.json").write_text('{"text": "private provider request"}')
    assert read(kinds=["tool"])["conversation"].splitlines() == [
        f"#1 --:-- tool: read_file path={'x' * 199}… start_line=8",
        "#3 --:-- tool: read_file result: 23 lines",
    ]
    assert "no matching entries" in read(query="private")["conversation"]
    assert not sessions.roster(tmp_path, "private").briefs


def test_fetch_previews_share_only_the_host_port_and_path(tmp_path: pathlib.Path) -> None:
    """A rejected call reaches the journal before URL validation."""
    from agent6.config import Config
    from agent6.tools import dispatch, tool_errors

    d = _session(tmp_path, "runs", "calm-owl-AAAAAA", "run", "fetch check", [])
    dispatcher = dispatch.ToolDispatcher(
        root=tmp_path,
        config=Config(),
        state_dir=tmp_path,
        events=event_log.EventSink(d / "logs.jsonl"),
    )
    for url in (
        "https://marker-a:marker-b@[2001:db8::1]:8443/guide?label=marker-c#marker-d",
        "https://example.invalid:marker-port/guide",
        # The recorded preview ends in the userinfo, before the authority is known.
        f"https://marker-{'a' * 220}@example.invalid/guide",
    ):
        with pytest.raises(tool_errors.ToolError):
            dispatcher.dispatch("fetch", {"url": url})

    page = dispatcher.dispatch("read_session", {"id": d.name, "kinds": ["tool"]}).to_wire()
    assert "fetch url=https://[2001:db8::1]:8443/guide" in page["conversation"]
    assert "marker-" not in page["conversation"]
    assert page["conversation"].count("url=") == 1
    assert page["conversation"].count("fetch result: refused") == 3
    assert not sessions.roster(tmp_path, "marker-").briefs


@pytest.mark.parametrize("name", ["mcp__demo__lookup", "unregistered_tool"])
def test_opaque_tools_disclose_only_name_and_disposition(tmp_path: pathlib.Path, name: str) -> None:
    read = _reader(
        tmp_path,
        [
            {"type": "tool.call", "name": name, "args_preview": {"path": "marker-input"}},
            {
                "type": "tool.result",
                "name": name,
                "ok": True,
                "summary": "marker-text",
                "exit_code": 7,
            },
            {"type": "tool.result", "name": name, "ok": False, "summary": "marker-error"},
        ],
    )
    assert read()["conversation"].splitlines() == [
        f"#1 --:-- tool: {name}",
        f"#2 --:-- tool: {name} result: ok",
        f"#3 --:-- tool: {name} result: refused",
    ]
    assert not sessions.roster(tmp_path, "marker-").briefs


def test_only_complete_known_summaries_are_shared_for_builtin_results(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        ("read_file", "128 bytes (truncated)"),
        ("fetch", "404 · 21 bytes"),
        ("read_file", "128 bytes (truncated) marker-suffix"),
        ("ask_user", "unanswered"),
    ]
    read = _reader(
        tmp_path,
        [{"type": "tool.result", "name": n, "ok": True, "summary": s} for n, s in rows],
    )
    assert read()["conversation"].splitlines() == [
        "#1 --:-- tool: read_file result: 128 bytes (truncated)",
        "#2 --:-- tool: fetch result: 404 · 21 bytes",
        "#3 --:-- tool: read_file result: ok",
        "#4 --:-- tool: ask_user result: unanswered",
    ]


def test_a_command_handoff_is_not_a_finished_command(tmp_path: pathlib.Path) -> None:
    result = results.ExecResult(None, "", "", 1.0, False, background_id="marker-job")
    read = _reader(
        tmp_path,
        [
            {
                "type": "tool.result",
                "name": "run_command",
                "ok": True,
                "summary": result.summary(),
                **result_format.tool_result_fields("run_command", result),
            }
        ],
    )
    assert read()["conversation"] == "#1 --:-- tool: run_command result: handed off"
    assert not sessions.roster(tmp_path, "marker-job").briefs


def test_a_handler_return_does_not_mean_its_work_succeeded(tmp_path: pathlib.Path) -> None:
    outcomes = [
        ("parallel", results.StartedResult((), "marker-note")),
        ("run_command", results.ExecResult(None, "", "", 1.0, False, stopped_by_operator=True)),
        ("run_command", results.ExecResult(None, "", "marker-error", 0.0, True)),
        ("run_verify_command", results.ExecResult(124, "", "", 2.0, False, timed_out=True)),
    ]
    read = _reader(
        tmp_path,
        [
            {
                "type": "tool.result",
                "name": name,
                "ok": True,
                "summary": result.summary(),
                **result_format.tool_result_fields(name, result),
            }
            for name, result in outcomes
        ],
    )
    assert read()["conversation"].splitlines() == [
        "#1 --:-- tool: parallel result: nothing started",
        "#2 --:-- tool: run_command result: stopped by operator",
        "#3 --:-- tool: run_command result: could not start",
        "#4 --:-- tool: run_verify_command result: exit 124 (timed out)",
    ]
    assert not sessions.roster(tmp_path, "marker-").briefs


def test_clipped_edits_do_not_hide_the_target_and_preview_flag(tmp_path: pathlib.Path) -> None:
    args = result_format.truncate_args(
        {
            "path": "sample.txt",
            "edits": [
                {"old_string": f"before {i}", "new_string": "marker-edit"} for i in range(11)
            ],
            "preview": True,
        }
    )
    read = _reader(tmp_path, [{"type": "tool.call", "name": "apply_edit", "args_preview": args}])
    assert read()["conversation"] == "#1 --:-- tool: apply_edit path=sample.txt preview=true"


def test_commands_share_the_program_without_opaque_operands(tmp_path: pathlib.Path) -> None:
    argv = ["/usr/bin/demo", "https://example.invalid/?label=marker-url", '--data={"a":"marker"}']
    read = _reader(
        tmp_path,
        [
            {"type": "tool.call", "name": "run_command", "args_preview": {"argv": argv}},
            {"type": "verify.start", "call_id": 3, "command": argv},
            {"type": "metric.start", "command": argv},
        ],
    )
    assert read()["conversation"].splitlines() == [
        "#1 --:-- tool: run_command program=demo",
        "#2 --:-- tool: run_verify_command program=demo",
        "#3 --:-- tool: run_metric_command program=demo",
    ]
    assert not sessions.roster(tmp_path, "marker").briefs


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("run_command", {"argv": '["marker-arg"]'}),
        ("run_command", {"argv": ["marker-arg", {"unexpected": "object"}]}),
        ("read_file", {"path": ["marker-path"]}),
        ("read_file", {"start_line": '{"field":"marker-number"}'}),
        ("ask_user", {"questions": "[{'question': 'marker-question'}]"}),
        ("ask_user", {"questions": [{"question": "marker-question", "options": [["nested"]]}]}),
        ("ask_user", {"questions": [{"question": "Which?", "options": ['{"field":"marker"}']}]}),
        ("apply_edit", {"edits": ["{'new_string': 'marker-edit'}"]}),
    ],
)
def test_a_malformed_argument_is_omitted_as_a_unit(
    tmp_path: pathlib.Path, name: str, args: dict[str, object]
) -> None:
    read = _reader(tmp_path, [{"type": "tool.call", "name": name, "args_preview": args}])
    assert read()["conversation"] == f"#1 --:-- tool: {name}"


@pytest.mark.parametrize(
    ("name", "args", "public"),
    [
        (
            "ask_user",
            {
                "questions": [
                    {"question": "Which?", "options": [json.dumps(json.dumps({"label": "marker"}))]}
                ],
                "wait": False,
            },
            "wait=false",
        ),
        (
            "read_file",
            {"path": "https://example.invalid/?edition=marker#part", "limit": 2},
            "limit=2",
        ),
        (
            "add_task",
            {"relevant_paths": ["src", "https://example.invalid/?marker"], "after": "1"},
            "after=1",
        ),
    ],
)
def test_only_plain_argument_labels_can_reach_a_page_or_roster_match(
    tmp_path: pathlib.Path, name: str, args: dict[str, object], public: str
) -> None:
    read = _reader(tmp_path, [{"type": "tool.call", "name": name, "args_preview": args}])
    assert read()["conversation"] == f"#1 --:-- tool: {name} {public}"
    assert not sessions.roster(tmp_path, "marker").briefs


def test_search_does_not_page_through_its_own_calls_and_results(tmp_path: pathlib.Path) -> None:
    """Dispatch journals the lookup in the same file before the reader opens it."""
    from agent6.config import Config
    from agent6.tools import dispatch

    d = _session(tmp_path, "runs", "calm-owl-AAAAAA", "run", "remember the label", ["label-42"])
    dispatcher = dispatch.ToolDispatcher(
        root=tmp_path,
        config=Config(),
        state_dir=tmp_path,
        events=event_log.EventSink(d / "logs.jsonl"),
    )
    for _ in range(2):
        page = dispatcher.dispatch("read_session", {"id": d.name, "query": "label-42", "limit": 1})
        assert page.to_wire() == {"conversation": "#2 --:-- model: label-42"}
    for query in ("no-such-label", "1 session", "read_session"):
        assert dispatcher.dispatch("read_session", {"query": query}).to_wire() == {"sessions": []}
        page = dispatcher.dispatch("read_session", {"id": d.name, "query": query})
        assert page.to_wire() == {"conversation": "(no matching entries)"}


def test_nested_arguments_keep_declared_fields_without_opaque_data(tmp_path: pathlib.Path) -> None:
    read = _reader(
        tmp_path,
        [
            {
                "type": "tool.call",
                "name": "ask_user",
                "args_preview": {
                    "questions": [{"question": "Which?", "options": ["blue"], "token": "private"}]
                },
            },
            {
                "type": "tool.call",
                "name": "finish_session",
                "args_preview": {"result": {"token": "private"}},
            },
            {
                "type": "tool.call",
                "name": "mcp__service__get",
                "args_preview": {"token": "private"},
            },
        ],
    )
    text = read()["conversation"]
    assert "Which?" in text and "blue" in text and "private" not in text


def test_a_journal_symlink_cannot_substitute_a_transcript(tmp_path: pathlib.Path) -> None:
    read = _reader(tmp_path, [])
    session = tmp_path / "sessions" / "runs" / "calm-owl-AAAAAA"
    (session / "transcripts").mkdir()
    transcript = session / "transcripts" / "1.json"
    transcript.write_text(json.dumps({"type": "model.result", "text": "private request"}))
    (session / "logs.jsonl").unlink()
    (session / "logs.jsonl").symlink_to(transcript)
    text = read()["conversation"]
    assert "no readable journal" in text and "private request" not in text
    assert not sessions.roster(tmp_path, "private request").briefs


def test_a_reader_does_not_need_linux_path_descriptors(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = _reader(tmp_path, [{"type": "model.result", "text": "portable lookup"}])
    monkeypatch.delattr(os, "O_PATH", raising=False)
    assert read()["conversation"] == "#1 --:-- model: portable lookup"
    assert [b.id for b in sessions.roster(tmp_path, "portable lookup").briefs] == [
        "calm-owl-AAAAAA"
    ]


@pytest.mark.parametrize(
    "component", ["sessions", "sessions/runs", "sessions/runs/calm-owl-AAAAAA"]
)
def test_a_parent_directory_link_is_not_a_session_source(
    tmp_path: pathlib.Path, component: str
) -> None:
    read = _reader(tmp_path, [{"type": "model.result", "text": "other tree"}])
    path = tmp_path / component
    path.rename(tmp_path / "elsewhere")
    path.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    from agent6.tools import tool_errors

    with pytest.raises(tool_errors.ToolError, match="no session"):
        read()
    assert not sessions.roster(tmp_path, "").briefs


def test_a_manifest_link_is_not_a_roster_brief(tmp_path: pathlib.Path) -> None:
    d = _session(tmp_path, "runs", "calm-owl-AAAAAA", "run", "marker-task", [])
    path = d / "manifest.json"
    path.rename(tmp_path / "elsewhere.json")
    path.symlink_to(tmp_path / "elsewhere.json")
    assert not sessions.roster(tmp_path, "marker-task").briefs


def test_max_chars_keeps_the_entry_header_and_the_next_page(tmp_path: pathlib.Path) -> None:
    read = _reader(
        tmp_path,
        [{"type": "model.result", "text": text} for text in ("first", "x" * 600, "last")],
    )
    assert read(max_chars=500) == {"conversation": "#1 --:-- model: first", "next_start": 2}
    page = read(start=2, max_chars=500)
    assert page["conversation"].startswith("#2 --:-- model: ")
    assert len(page["conversation"]) == 500 and page["conversation"].endswith("…")
    assert page["next_start"] == 3
    assert read(start=3) == {"conversation": "#3 --:-- model: last"}


def test_the_tool_is_unwired_without_a_project_state_dir(tmp_path: pathlib.Path) -> None:
    from agent6.config import Config
    from agent6.tools import dispatch, tool_errors

    d = dispatch.ToolDispatcher(root=tmp_path, config=Config())
    with pytest.raises(tool_errors.ToolError, match="state dir"):
        d.dispatch("read_session", {})


def test_a_project_with_many_sessions_does_not_flood_the_context(tmp_path: pathlib.Path) -> None:
    """The roster stays bounded: 2000 sessions rendered about 70k tokens without the cap."""
    for i in range(schema.ROSTER_MAX + 25):
        _session(tmp_path, "runs", f"s{i:04d}-AAAAAA", "run", "t", [])
    got = sessions.roster(tmp_path, "")
    assert len(got.briefs) == schema.ROSTER_MAX
    assert got.more
    assert "narrow with `query`" in got.lines()[-1], "a silent cut reads as the whole project"


def test_a_query_matching_everything_is_capped_too(tmp_path: pathlib.Path) -> None:
    for i in range(schema.ROSTER_MAX + 25):
        _session(tmp_path, "runs", f"s{i:04d}-AAAAAA", "run", "refactor the parser", [])
    got = sessions.roster(tmp_path, "parser")
    assert len(got.briefs) == schema.ROSTER_MAX and got.more


def test_a_query_reads_journals_without_holding_them_in_memory(tmp_path: pathlib.Path) -> None:
    """Search reaches a late entry without slurping the journal or a single oversized row."""
    big = _session(tmp_path, "runs", "big-AAAAAA", "run", "t", []) / "logs.jsonl"
    with big.open("w") as fh:
        fh.write(json.dumps({"type": "model.text_delta", "text": "y" * (12 << 20)}) + "\n")
        row = json.dumps({"type": "model.result", "text": "x" * 4096}) + "\n"
        for _ in range(1024):
            fh.write(row)
        fh.write(json.dumps({"type": "model.result", "text": "nEeDlE"}) + "\n")
    size = big.stat().st_size
    peak = _peak_bytes_reading(lambda: sessions.roster(tmp_path, "needle"))
    assert [b.id for b in sessions.roster(tmp_path, "needle").briefs] == ["big-AAAAAA"]
    assert peak < size // 4, f"held {peak} bytes of a {size}-byte journal"


def test_a_skipped_instruction_marks_the_scan_incomplete_even_without_a_match(
    tmp_path: pathlib.Path,
) -> None:
    read = _reader(
        tmp_path,
        [
            {"type": "model.result", "text": "x" * 411},
            {"type": "model.result", "text": "second " + "z" * 30},
            {"type": "session.start", "user_task": "marker-opening " + "y" * (1 << 20)},
            {"type": "model.result", "text": "last"},
        ],
    )
    page = read(max_chars=500)
    assert "omitted" in page["conversation"] and "#1 --:-- model:" in page["conversation"]
    assert len(page["conversation"]) <= 500 and page["next_start"] == 2
    assert "second" not in page["conversation"]
    rest = read(start=2)["conversation"]
    assert "#2 --:-- model: second" in rest and "#3 --:-- model: last" in rest
    miss = read(query="marker-opening", kinds=["operator"])["conversation"]
    assert "omitted" in miss and "1 MiB" in miss and "marker-opening" not in miss
    roster = sessions.roster(tmp_path, "marker-opening")
    assert not roster.briefs and any("omitted" in line for line in roster.lines())


def _peak_bytes_reading(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_reader_sees_what_the_assistant_said_in_a_real_journal(tmp_path: pathlib.Path) -> None:
    """Written by the real emitter, not by hand.

    The prose reached the journal only as `model.text_delta`, which is emitted only when streaming
    is on, so a headless run (CI, a redirected stdout, every spawned ask) recorded no assistant
    text, and this tool returned the task and a list of tool names. Every fixture that hand-wrote
    `{"type": "model.result", "text": ...}` passed against a shape the engine never emitted.
    """
    from unittest import mock

    from agent6 import budget
    from agent6.app import role_providers
    from agent6.providers import provider_types

    d = tmp_path / "sessions" / "asks" / "quiet-fox-AAAAAA"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": "ask", "user_task": "how do I convert h264"}),
        encoding="utf-8",
    )
    events = event_log.EventSink(d / "logs.jsonl")
    events.emit(
        journal.SessionStart(
            user_task="how do I convert h264", mode="ask", execution_id="execution"
        )
    )
    inner = mock.MagicMock()
    inner.call.return_value = provider_types.ProviderResponse(
        text="use ffmpeg -c:v libx265",
        tool_uses=(),
        refused={},
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw={},
    )
    role_providers.InstrumentedProvider(
        inner=inner,
        role="worker",
        model="m",
        provider_name="p",
        events=events,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
    ).call(system="s", messages=[{"role": "user", "content": "q"}], tools=[], max_tokens=64)

    layout = sessions_layout.session_layout(tmp_path, "quiet-fox-AAAAAA")
    assert layout is not None
    text = sessions.conversation(layout, max_chars=10_000)
    assert "use ffmpeg -c:v libx265" in text, "the answer the other session reached is missing"
    assert "how do I convert h264" in text


def test_dispatch_task_has_the_author_who_requested_it(tmp_path: pathlib.Path) -> None:
    """A child task from a model is not an operator instruction."""
    from agent6 import config, paths
    from agent6.tools import dispatch

    state = paths.state_dir(tmp_path)
    child = state / "sessions" / "runs" / "child"
    child.mkdir(parents=True)
    rows = [
        {"type": "session.start", "task_source": source, "user_task": f"{source} task"}
        for source in ("operator", "model", "panel")
    ]
    (child / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    reader = dispatch.ToolDispatcher(root=tmp_path, state_dir=state, config=config.Config())
    for kind in ("operator", "model"):
        page = reader.dispatch("read_session", {"id": child.name, "kinds": [kind]}).to_wire()
        assert f"{kind} task" in page["conversation"]
        other = "operator" if kind == "model" else "model"
        assert f"{other} task" not in page["conversation"]
        assert "panel task" not in page["conversation"]
