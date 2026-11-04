from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import paths
from agent6.providers import provider_types
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import history_transcripts, sessions_transcript
from agent6.viewmodel import transcript_render


@pytest.mark.parametrize("seat", ["worker", "review:security"])
def test_sessions_transcript_labels_an_unfinalized_record(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seat: str,
) -> None:
    """A pre-send record left by a killed process reads as incomplete, including side calls."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="unfinished"
    )
    layout.ensure()
    layout.logs_path.write_text('{"type": "session.start"}\n')
    provider_types.TranscriptSink(layout.transcripts_dir).for_seat(seat).open(
        request_headers={},
        request_body={"messages": [{"role": "user", "content": "pending request"}]},
    )
    assert (
        sessions_transcript.cmd_transcript(
            "unfinished", as_json=False, no_thinking=False, tools="both", seq="1"
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "call 1: incomplete" in output
    assert "completed" not in output
    assert (
        sessions_transcript.cmd_transcript(
            "unfinished", as_json=True, no_thinking=False, tools="both", seq="1"
        )
        == 0
    )
    (record,) = json.loads(capsys.readouterr().out)
    assert record["outcome"] == "incomplete"


@pytest.mark.parametrize("outcome", ["incomplete", "failed", "interrupted", "aborted"])
def test_fold_marks_every_attempt_that_did_not_complete(outcome: str) -> None:
    """The HTTP status cannot stand in for the attempt's outcome; a completed one is quiet."""
    record = {
        "seq": 7,
        "outcome": outcome,
        "request": {"body": {"messages": []}},
        "response": {"status": 200, "body": {}},
    }
    folded = transcript_render.fold_conversation([record])
    assert f"call 7: {outcome}" in transcript_render.render_markdown(folded, session_id="s")
    assert all(message.seq == 7 for message in folded)
    assert transcript_render.fold_conversation([{**record, "outcome": "completed"}]) == []


def test_legacy_record_does_not_invent_an_outcome() -> None:
    """The reader cannot infer a terminal outcome from an older record's HTTP status."""
    folded = transcript_render.fold_conversation(
        [{"seq": 1, "request": {"body": {"messages": []}}, "response": {"status": 200, "body": {}}}]
    )
    assert "completed" not in transcript_render.render_markdown(folded, session_id="s")


@pytest.mark.parametrize("error", [None, "connection dropped"])
def test_history_search_accepts_incomplete_and_error_records(
    tmp_path: pathlib.Path, error: str | None
) -> None:
    """An empty or raw error response does not make a valid attempt unreadable to search."""
    attempt = provider_types.TranscriptSink(tmp_path / "transcripts").open(
        request_headers={},
        request_body={"messages": [{"role": "user", "content": "pending request"}]},
    )
    if error:
        with pytest.raises(provider_types.ProviderError), attempt.capture():
            raise provider_types.ProviderError(error)
    (search,) = history_transcripts.search_text([tmp_path])
    assert search.error == ""
    assert "pending request" in search.text
    assert f'"outcome": "{"failed" if error else "incomplete"}"' in search.text
    if error:
        assert error in search.text
