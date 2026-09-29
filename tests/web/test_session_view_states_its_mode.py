# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web session view says what kind of session it is showing.

The page opens for any session, but its snapshot carried no mode, so the
details panel was headed a hard-coded "Run" and the composer said "continue the
run" over a plan or an ask. The heading is exactly where the mode belongs, and
it was stating the opposite.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6.sessions import layout
from agent6.ui.web import page
from agent6.viewmodel import session_snapshot


def _session(state: pathlib.Path, bucket: str, session_id: str, mode: str) -> pathlib.Path:
    session = layout.bucket_dir(state, bucket) / session_id
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": session_id, "mode": mode, "user_task": "t"}),
        encoding="utf-8",
    )
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": mode, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    return session


@pytest.mark.parametrize(("bucket", "mode"), [("runs", "run"), ("plans", "plan"), ("asks", "ask")])
def test_the_snapshot_carries_the_mode(tmp_path: pathlib.Path, bucket: str, mode: str) -> None:
    session = _session(tmp_path, bucket, "brave-oak-AAAAAA", mode)
    assert session_snapshot(session)["mode"] == mode


def test_the_page_heads_the_panel_with_the_mode_not_a_fixed_word() -> None:
    """`paintRun` writes the snapshot's mode into the heading.

    A hard-coded 'Run' is right one time in three.
    """
    client = page.CLIENT_JS
    assert "cards._head_title.textContent = s.mode" in client


def test_the_session_view_is_the_one_conversation_page() -> None:
    """The session view is the conversation page; there is no second route with its own handler."""
    assert "renderConversation" not in page.CLIENT_JS
    assert "parts[0] === 'conversation'" not in page.CLIENT_JS


def test_the_session_view_paints_the_prompts_it_claims_to_answer() -> None:
    """`paintRun` paints the run's prompts, since opening the stream claims the run as a front-end.

    A run blocked on an approval would otherwise wait on a page that showed nothing.
    """
    start = page.CLIENT_JS.index("function paintRun(")
    body = page.CLIENT_JS[start : page.CLIENT_JS.index("function renderDiff(", start)]
    assert "paintPrompts(cards, isDead ? {} : s)" in body


def test_the_run_crumb_carries_the_state_word() -> None:
    """The state crumb sits in the fixed header on every widget page.

    A phone shows one widget at a time and opens on the conversation, so a run waiting on an
    approval, or dead, said neither on the page it opened.
    """
    client = page.CLIENT_JS
    assert "setCrumb(runState(s) + ' · ' + cards._crumb)" in client
    # One owner for the word: the state row reads the same helper.
    assert "add('state', runState(s))" in client


def test_live_and_empty_conversation_notes_use_the_server_state_words() -> None:
    """The client cannot rename waiting/dead states already worded by the viewmodel."""
    assert "function runState(s) { return s.status_label || ''; }" in page.CLIENT_JS
    assert "el('div', 'muted', '· ' + runState(s))" in page.CLIENT_JS


def test_the_composer_does_not_flatten_an_outcome_to_finished() -> None:
    """The canonical outcome stays in the header; the composer names only its action."""
    assert "This session finished" not in page.CLIENT_JS


def test_the_run_card_shows_the_task_line_the_hub_rows_show() -> None:
    """The card reads the snapshot's task_line, the headline the hub rows and the TUI show.

    The whole composed task, or its raw first line, showed a seed block's opener or a heading.
    """
    client = page.CLIENT_JS
    assert "add('task', s.task_line || '(none)')" in client
    assert "s.user_task || '').split(" not in client
    assert "add('task', s.user_task || '(none)')" not in client


def test_a_session_that_never_commits_shows_no_commit_card() -> None:
    """The Latest commit card is hidden for an ask or a plan, as the shells card is by count."""
    hidden = "cards.diff.parentElement.style.display = "
    assert hidden + "s.mode === 'ask' || s.mode === 'plan' ? 'none' : '';" in page.CLIENT_JS
