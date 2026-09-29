# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/btw` asks beside a run without interrupting it."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from agent6 import directive
from agent6.app import btw


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/btw what does h265 mean", "what does h265 mean"),
        ("  /btw  spaced  ", "spaced"),
        ("/btw", ""),  # nothing asked
        ("btw no slash", None),
        ("/btwx joined", None),
        ("steer text /btw not at the start", None),
    ],
)
def test_the_grammar_matches_only_a_leading_btw(text: str, expected: str | None) -> None:
    """The grammar matches only a leading /btw, never the English word mid-sentence."""
    assert directive.parse_btw(text) == expected


def _ask_dir(root: pathlib.Path, name: str, *, events: list[dict[str, object]]) -> pathlib.Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "ask"}), encoding="utf-8")
    (d / "logs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    # A started ask has a live worker.pid on disk; without one the status fold reads it as exited.
    if any(e.get("type") == "session.start" for e in events):
        (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    return d


def test_it_returns_as_soon_as_the_session_exists(tmp_path: pathlib.Path) -> None:
    """start_btw returns the moment the session is on disk, not when it has an answer."""
    asks = tmp_path / "sessions" / "asks"
    asks.mkdir(parents=True)
    launched: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def launch(cwd: pathlib.Path, argv: list[str], env: dict[str, str]) -> str:
        launched.append(argv)
        envs.append(env)
        _ask_dir(asks, "quiet-fox-AAAAAA", events=[{"type": "session.start"}])
        return ""

    session, err = btw.start_btw(
        "why h265",
        "parent-BBBBBB",
        cwd=tmp_path,
        launch=launch,
        list_asks=lambda: [d for d in asks.iterdir() if d.is_dir()],
    )
    assert err == ""
    assert session is not None and session.id == "quiet-fox-AAAAAA"
    # `--no-commands`: nobody can approve for a btw; `--` guards a question starting with a dash.
    assert launched == [["ask", "--no-commands", "--from", "parent-BBBBBB", "--", "why h265"]]


def test_a_bare_btw_asks_for_a_question_instead_of_opening_a_session(
    tmp_path: pathlib.Path,
) -> None:
    called: list[str] = []
    session, err = btw.start_btw(
        "",
        "parent-BBBBBB",
        cwd=tmp_path,
        launch=lambda *_a: called.append("x") or "",  # type: ignore[func-returns-value]
        list_asks=list,
    )
    assert session is None and "ask something" in err
    assert called == []


def test_a_launch_failure_is_reported_not_swallowed(tmp_path: pathlib.Path) -> None:
    def failing(cwd: pathlib.Path, argv: list[str], env: dict[str, str]) -> str:
        return "no host launcher"

    session, err = btw.start_btw("q", "p", cwd=tmp_path, launch=failing, list_asks=list)
    assert session is None and err == "no host launcher"


def test_the_answer_is_none_until_the_btw_finishes(tmp_path: pathlib.Path) -> None:
    d = _ask_dir(tmp_path, "quiet-fox-AAAAAA", events=[{"type": "session.start"}])
    assert btw.btw_answer(btw.BtwSession(id=d.name, dir=d, question="q")) is None


def test_the_answer_is_the_final_prose(tmp_path: pathlib.Path) -> None:
    """An ask ends by emitting its answer as prose, not via finish_session."""
    d = _ask_dir(
        tmp_path,
        "quiet-fox-AAAAAA",
        events=[
            {"type": "session.start"},
            {"type": "role.result", "text": "first thought"},
            {"type": "role.result", "text": "use ffmpeg -c:v libx265"},
            {"type": "session.end", "reason": "answered", "all_passed": True},
        ],
    )
    assert (
        btw.btw_answer(btw.BtwSession(id=d.name, dir=d, question="q")) == "use ffmpeg -c:v libx265"
    )


def test_a_btw_that_died_says_so_rather_than_rendering_blank(tmp_path: pathlib.Path) -> None:
    d = _ask_dir(
        tmp_path,
        "quiet-fox-AAAAAA",
        events=[
            {"type": "session.start"},
            {"type": "session.end", "reason": "crashed", "all_passed": False},
        ],
    )
    answer = btw.btw_answer(btw.BtwSession(id=d.name, dir=d, question="q"))
    assert answer is not None and "without an answer" in answer


def test_the_block_is_fenced_and_names_how_to_go_deeper() -> None:
    """The block is fenced and names how to go deeper.

    It prints into the run's view but is not part of it; a btw has no follow-up thread, so going
    deeper means resuming it as the ask it is.
    """
    block = btw.render_btw(
        btw.BtwSession(id="quiet-fox-AAAAAA", dir=pathlib.Path("/x"), question="why"), "because"
    )
    assert block.startswith("\n--- btw: why\n")
    assert "because" in block
    assert "agent6 resume quiet-fox-AAAAAA" in block


def test_a_btw_is_not_declared_dead_before_its_worker_starts(tmp_path: pathlib.Path) -> None:
    """A /btw is not declared dead before its worker starts.

    `start_btw` returns as soon as the session dir appears, a few ms before the child writes its
    worker pid; a live worker mid-preflight is the same not-yet window, and a dead one is a real
    ending ("died launching").
    """
    import os

    d = tmp_path / "sessions" / "asks" / "quiet-fox-AAAAAA"
    d.mkdir(parents=True)
    session = btw.BtwSession(id=d.name, dir=d, question="why h265")

    assert btw.btw_answer(session) is None, "a dir with no worker yet is not an ending"

    (d / "worker.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert btw.btw_answer(session) is None, "a live worker mid-preflight is not an ending"

    (d / "worker.pid").write_text("1\n", encoding="utf-8")  # foreign pid: the worker died
    answer = btw.btw_answer(session)
    assert answer is not None and "died launching" in answer


def test_a_btw_still_thinking_when_the_watcher_gives_up_is_said_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A /btw still thinking when the watcher gives up is said so.

    The give-up lands on the journal as its own block, naming how to read the answer later.
    """
    from agent6 import events as agent6_events
    from agent6.ui import btw as ui_btw

    d = _ask_dir(tmp_path, "quiet-fox-AAAAAA", events=[{"type": "session.start"}])
    monkeypatch.setattr(ui_btw, "_GIVE_UP_S", 0.05)
    monkeypatch.setattr(ui_btw, "_POLL_S", 0.01)
    logs = tmp_path / "run" / "logs.jsonl"
    logs.parent.mkdir()

    ui_btw._watch(btw.BtwSession(id=d.name, dir=d, question="q"), agent6_events.EventSink(logs))  # pyright: ignore[reportPrivateUsage]

    events = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    (answered,) = [e for e in events if e["type"] == "btw.answered"]
    assert "no answer after" in answered["block"]
    assert "agent6 sessions show quiet-fox-AAAAAA" in answered["block"]
