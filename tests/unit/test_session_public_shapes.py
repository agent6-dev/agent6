from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest

from agent6 import kinds, paths
from agent6.sessions import layout as sessions_layout
from agent6.ui import cli
from agent6.viewmodel import format, listing
from agent6.viewmodel.session import state
from agent6.viewmodel.session import view as state_view


@pytest.mark.parametrize("surface", ["listing", "status", "live"])
def test_stop_reason_on_public_views(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    surface: str,
) -> None:
    """All three views retain the stop reason and clear it on a later execution."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(tmp_path), session_id="shape")
    layout.ensure()
    layout.manifest_path.write_text(json.dumps({"version": 3, "mode": "run"}))
    journal = [
        {"type": "session.start", "mode": "run", "execution_id": "first"},
        {"type": "session.stop", "reason": "steer_abort", "execution_id": "first"},
    ]
    for resumed in (False, True):
        if resumed:
            journal.append({"type": "session.resume", "execution_id": "second"})
        layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in journal))
        if surface == "listing":
            payload = listing.summary_row(listing.summarize_session_dir(layout.session_dir))
        elif surface == "status":
            assert cli.main(["sessions", "show", "shape", "--json"]) == 0
            payload = json.loads(capsys.readouterr().out)
        else:
            folded = state.fold_session(journal)
            assert folded.finished is not resumed
            payload = state_view.session_state_as_dict(folded)
        expected = (None if surface == "status" else "") if resumed else "steer_abort"
        assert "stop_reason" in payload
        assert payload["stop_reason"] == expected


@pytest.mark.parametrize("surface", ["listing", "status", "live"])
@pytest.mark.parametrize("word", ["passed", "failed", "unverified", "not_applicable", None])
def test_public_verification_word(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    surface: str,
    word: kinds.Verification | None,
) -> None:
    """Every public view preserves all four final verification words and unknown null."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(tmp_path), session_id="shape")
    layout.ensure()
    layout.manifest_path.write_text(json.dumps({"version": 3, "mode": "run"}))
    journal = [
        {"type": "session.start", "mode": "run", "execution_id": "execution"},
        {"type": "session.stop", "reason": "finish_session", "verification": word},
    ]
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in journal))
    if surface == "listing":
        payload = listing.summary_row(listing.summarize_session_dir(layout.session_dir))
    elif surface == "status":
        assert cli.main(["sessions", "show", "shape", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
    else:
        payload = state_view.session_state_as_dict(state.fold_session(journal))
    assert "verification" in payload
    assert payload["verification"] == word
    if surface == "live":
        resumed = state.apply_event(state.fold_session(journal), {"type": "session.resume"})
        assert state_view.session_state_as_dict(resumed)["verification"] is None


def test_listing_status_detail_names_the_status_explanation(tmp_path: pathlib.Path) -> None:
    """The listing uses status_detail for the explanation, separate from the stop reason."""
    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.stop", "reason": "provider_error"}) + "\n"
    )
    row = listing.summary_row(listing.summarize_session_dir(tmp_path))
    assert "status_detail" in row
    assert row["status_detail"] == "provider_error"
    assert row["stop_reason"] == "provider_error"


def test_listing_separates_session_and_group_activity(tmp_path: pathlib.Path) -> None:
    """A group's latest child activity never overwrites its own session activity."""
    summary = listing.summarize_session_dir(tmp_path)
    own = dataclasses.replace(summary, session_id="group", last_activity=1000.0)
    child = dataclasses.replace(summary, session_id="child", last_activity=2000.0)
    row = listing.row_json(listing.ListingRow(own, (listing.ListingRow(child),)), winners=set())
    assert row["last_activity"] == 1000.0
    assert row.get("group_last_activity") == 2000.0
    assert row["when"] == format.format_when(2000.0)
    child_row = listing.summary_row(child)
    assert child_row["last_activity"] == 2000.0
    assert "group_last_activity" not in child_row
    assert row["children"][0]["last_activity"] == 2000.0


@pytest.mark.parametrize("family", ["approval", "question"])
def test_live_prompt_lists_keep_answered_prompts(family: str) -> None:
    """The prompt lists include answered and outstanding prompts without duplicate lists."""
    journal = [
        {"type": "session.start", "mode": "run"},
        {"type": f"{family}.prompt", "id": "answered", "prompt": "First?", "questions": []},
        {"type": f"{family}.answer", "id": "answered", "answer": "yes", "answers": ["yes"]},
        {"type": f"{family}.prompt", "id": "open", "prompt": "Second?", "questions": []},
    ]
    payload = state_view.session_state_as_dict(state.fold_session(journal))
    name = "approvals" if family == "approval" else "questions"
    assert name in payload
    assert [(p["id"], p["answered"]) for p in payload[name]] == [
        ("answered", True),
        ("open", False),
    ]
