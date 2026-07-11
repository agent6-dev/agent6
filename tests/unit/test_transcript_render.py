# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the transcript -> conversation renderer (both provider wire shapes)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent6.ui.viewmodel.transcript_render import (
    fold_conversation,
    load_transcripts,
    render_markdown,
)

_OPENAI = [
    {
        "seq": 1,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM PROMPT"},
                    {"role": "user", "content": "do X"},
                ]
            }
        },
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "working on it",
                            "reasoning_content": "let me think",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        },
    },
    {
        "seq": 2,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM PROMPT"},
                    {"role": "user", "content": "do X"},
                    {
                        "role": "assistant",
                        "content": "working on it",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "FULL FILE CONTENTS", "tool_call_id": "c1"},
                ]
            }
        },
        "response": {
            "body": {"choices": [{"message": {"role": "assistant", "content": "all done"}}]}
        },
    },
]

_ANTHROPIC = [
    {
        "seq": 1,
        "request": {
            "body": {
                "system": "SYSTEM PROMPT",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "do X"}]},
                ],
            }
        },
        "response": {
            "body": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think"},
                    {"type": "text", "text": "working on it"},
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                ],
            }
        },
    },
    {
        "seq": 2,
        "request": {
            "body": {
                "system": "SYSTEM PROMPT",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "do X"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "u1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "u1",
                                "content": "FULL FILE CONTENTS",
                            }
                        ],
                    },
                ],
            }
        },
        "response": {
            "body": {"role": "assistant", "content": [{"type": "text", "text": "all done"}]}
        },
    },
]


@pytest.mark.parametrize("transcripts", [_OPENAI, _ANTHROPIC], ids=["openai", "anthropic"])
def test_fold_and_render_both_shapes(transcripts: list[dict[str, Any]]) -> None:
    turns = fold_conversation(transcripts)
    roles = [t.role for t in turns]
    # system, user, assistant(seq1 w/ tool_call), tool result, assistant(seq2)
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    a1 = turns[2]
    assert a1.text == "working on it" and a1.thinking == "let me think"
    assert a1.tool_calls and a1.tool_calls[0][0] == "read_file"
    assert "a.py" in a1.tool_calls[0][1]
    tool = turns[3]
    assert tool.tool_name == "read_file"  # resolved from the call id
    assert tool.text == "FULL FILE CONTENTS"  # full result, not a summary
    assert turns[4].text == "all done"

    md = render_markdown(turns, run_id="r1", show_thinking=True)
    assert "SYSTEM PROMPT" in md and "do X" in md and "all done" in md
    assert "-> read_file(" in md and "FULL FILE CONTENTS" in md
    assert "let me think" in md  # thinking shown


def test_render_flags_hide_thinking_and_tools() -> None:
    turns = fold_conversation(_OPENAI)
    md = render_markdown(turns, run_id="r1", show_thinking=False, tools="none")
    assert "let me think" not in md
    assert "-> read_file" not in md and "FULL FILE CONTENTS" not in md
    # calls-only keeps the call line but drops the result
    md2 = render_markdown(turns, run_id="r1", tools="calls")
    assert "-> read_file(" in md2 and "FULL FILE CONTENTS" not in md2


def test_compaction_restart_shows_marker() -> None:
    # seq 3's request is SHORTER than the prior history -> a summarise/restart.
    transcripts = [
        *_OPENAI,
        {
            "seq": 3,
            "request": {
                "body": {
                    "messages": [
                        {"role": "system", "content": "SYSTEM PROMPT"},
                        {"role": "user", "content": "<summary of earlier work>"},
                    ]
                }
            },
            "response": {
                "body": {"choices": [{"message": {"role": "assistant", "content": "resumed"}}]}
            },
        },
    ]
    turns = fold_conversation(transcripts)
    assert any(t.role == "marker" for t in turns)
    assert turns[-1].text == "resumed"


def test_load_transcripts_sorted_by_seq(tmp_path: Path) -> None:
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / "20260101T2-000002.json").write_text(json.dumps({"seq": 2, "x": "b"}), encoding="utf-8")
    (d / "20260101T1-000001.json").write_text(json.dumps({"seq": 1, "x": "a"}), encoding="utf-8")
    (d / "bad.json").write_text("{not json", encoding="utf-8")  # skipped, not fatal
    loaded = load_transcripts(d)
    assert [t["seq"] for t in loaded] == [1, 2]
    assert load_transcripts(tmp_path / "nope") == []


def test_cmd_history_transcript_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 runs transcript <run>` resolves the run, folds its transcripts,
    and prints the conversation (full tool I/O), with --json as the raw escape."""
    from agent6.config.layer import resolved_state_dir
    from agent6.ui.cli.history_cmds import (
        _cmd_history_transcript,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "st"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    tdir = resolved_state_dir(repo) / "runs" / "my-run" / "transcripts"
    tdir.mkdir(parents=True)
    (tdir / "20260101-000001.json").write_text(json.dumps(_OPENAI[0]), encoding="utf-8")
    (tdir / "20260101-000002.json").write_text(json.dumps(_OPENAI[1]), encoding="utf-8")

    rc = _cmd_history_transcript("my-run", as_json=False, no_thinking=False, tools="both", seq="")
    assert rc == 0
    out = capsys.readouterr().out
    assert "Transcript: my-run" in out and "-> read_file(" in out and "FULL FILE CONTENTS" in out

    rc_json = _cmd_history_transcript(
        "my-run", as_json=True, no_thinking=False, tools="both", seq="2"
    )
    assert rc_json == 0
    data = json.loads(capsys.readouterr().out)
    assert [t["seq"] for t in data] == [2]  # --seq windowed the raw transcripts

    assert (
        _cmd_history_transcript("nope", as_json=False, no_thinking=False, tools="both", seq="") == 2
    )


def test_cmd_history_transcript_latest_uses_log_activity_not_dir_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.config.layer import resolved_state_dir
    from agent6.ui.cli.history_cmds import (
        _cmd_history_transcript,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "st"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = resolved_state_dir(repo) / "runs"
    for name in ("older-run", "newer-run"):
        tdir = runs / name / "transcripts"
        tdir.mkdir(parents=True)
        (tdir / "20260101-000001.json").write_text(json.dumps(_OPENAI[0]), encoding="utf-8")
        (runs / name / "logs.jsonl").write_text('{"type":"run.start"}\n', encoding="utf-8")
    os.utime(runs / "older-run" / "logs.jsonl", (100, 100))
    os.utime(runs / "newer-run" / "logs.jsonl", (1000, 1000))
    (runs / "older-run" / "frontend.pid").write_text("12345", encoding="utf-8")

    assert _cmd_history_transcript("", as_json=True, no_thinking=False, tools="both", seq="") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)[0]["seq"] == 1
    assert "newer-run" in captured.err
