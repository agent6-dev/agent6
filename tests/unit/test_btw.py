# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/btw` asks beside a run without interrupting it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app.btw import BtwSession, btw_answer, render_btw, start_btw
from agent6.directive import parse_btw


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
    """A btw is a question asked beside the run, never steer text, so it must
    not be recognised mid-sentence where an operator meant the English word."""
    assert parse_btw(text) == expected


def _ask_dir(root: Path, name: str, *, events: list[dict[str, object]]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"version": 3, "mode": "ask"}), encoding="utf-8")
    (d / "logs.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return d


def test_it_returns_as_soon_as_the_session_exists(tmp_path: Path) -> None:
    """The run must not wait on it: start_btw returns the moment the session
    is on disk, not when it has an answer."""
    asks = tmp_path / "asks"
    asks.mkdir()
    launched: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def launch(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        launched.append(argv)
        envs.append(env)
        _ask_dir(asks, "quiet-fox-AAAAAA", events=[{"type": "run.start"}])
        return ""

    session, err = start_btw(
        "why h265",
        "parent-BBBBBB",
        cwd=tmp_path,
        launch=launch,
        list_asks=lambda: [d for d in asks.iterdir() if d.is_dir()],
    )
    assert err == ""
    assert session is not None and session.id == "quiet-fox-AAAAAA"
    # Seeded with the parent's context, and `--` so a question starting with a
    # dash cannot be read as a flag.
    # `--no-commands`: nobody can approve for a btw (no terminal of its own, the
    # parent mid-run), so the tools are withheld rather than offered-and-denied.
    # `--` guards a question starting with a dash.
    assert launched == [["ask", "--no-commands", "--from", "parent-BBBBBB", "--", "why h265"]]


def test_a_bare_btw_asks_for_a_question_instead_of_opening_a_session(tmp_path: Path) -> None:
    called: list[str] = []
    session, err = start_btw(
        "",
        "parent-BBBBBB",
        cwd=tmp_path,
        launch=lambda *_a: called.append("x") or "",  # type: ignore[func-returns-value]
        list_asks=list,
    )
    assert session is None and "ask something" in err
    assert called == []


def test_a_launch_failure_is_reported_not_swallowed(tmp_path: Path) -> None:
    def failing(cwd: Path, argv: list[str], env: dict[str, str]) -> str:
        return "no host launcher"

    session, err = start_btw("q", "p", cwd=tmp_path, launch=failing, list_asks=list)
    assert session is None and err == "no host launcher"


def test_the_answer_is_none_until_the_btw_finishes(tmp_path: Path) -> None:
    d = _ask_dir(tmp_path, "quiet-fox-AAAAAA", events=[{"type": "run.start"}])
    assert btw_answer(BtwSession(id=d.name, dir=d, question="q")) is None


def test_the_answer_is_the_final_prose(tmp_path: Path) -> None:
    """An ask ends by emitting its answer as prose, not via finish_run."""
    d = _ask_dir(
        tmp_path,
        "quiet-fox-AAAAAA",
        events=[
            {"type": "run.start"},
            {"type": "role.result", "text": "first thought"},
            {"type": "role.result", "text": "use ffmpeg -c:v libx265"},
            {"type": "run.end", "reason": "answered", "all_passed": True},
        ],
    )
    assert btw_answer(BtwSession(id=d.name, dir=d, question="q")) == "use ffmpeg -c:v libx265"


def test_a_btw_that_died_says_so_rather_than_rendering_blank(tmp_path: Path) -> None:
    d = _ask_dir(
        tmp_path,
        "quiet-fox-AAAAAA",
        events=[
            {"type": "run.start"},
            {"type": "run.end", "reason": "crashed", "all_passed": False},
        ],
    )
    answer = btw_answer(BtwSession(id=d.name, dir=d, question="q"))
    assert answer is not None and "without an answer" in answer


def test_the_block_is_fenced_and_names_how_to_go_deeper() -> None:
    """It prints INTO the run's view but is not part of it: an operator must
    never mistake it for the run's own output, and a btw has no follow-up
    thread -- going deeper means resuming it as the ask it is."""
    block = render_btw(BtwSession(id="quiet-fox-AAAAAA", dir=Path("/x"), question="why"), "because")
    assert block.startswith("\n--- btw: why\n")
    assert "because" in block
    assert "agent6 resume quiet-fox-AAAAAA" in block


def test_a_btw_is_not_declared_dead_before_its_worker_starts(tmp_path: Path) -> None:
    """`start_btw` returns as soon as the session DIR appears, which is a few
    ms before the child writes its worker pid. Reading that window as an ending
    made the watcher emit "(ended without an answer: created)" on its first
    poll and stop looking, while the btw ran on and answered."""
    from agent6.app.btw import BtwSession, btw_answer

    d = tmp_path / "asks" / "quiet-fox-AAAAAA"
    d.mkdir(parents=True)
    session = BtwSession(id=d.name, dir=d, question="why h265")

    assert btw_answer(session) is None, "a dir with no worker yet is not an ending"

    (d / "worker.pid").write_text("1\n", encoding="utf-8")
    assert btw_answer(session) is None
