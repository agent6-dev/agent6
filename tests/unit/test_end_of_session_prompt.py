# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A CLI run/plan session ends by asking for the next input (`/exit` finishes)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import _session_prompt as prompt_mod


def _seed_session(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, session_id: str = "test-run-AAAAAA"
) -> SessionLayout:
    """A real run dir under repo_root's state home, so resolution reaches the
    tty guard rather than short-circuiting on SessionIdError."""
    from agent6.ui.cli._common import _state_dir  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("AGENT6_STATE_HOME", str(repo_root / ".state"))
    layout = SessionLayout(state_dir=_state_dir(repo_root), session_id=session_id, subdir="runs")
    layout.session_dir.mkdir(parents=True, exist_ok=True)
    (layout.session_dir / "logs.jsonl").write_text(
        '{"type": "session.start", "ts": "2026-01-01T00:00:00Z"}\n', encoding="utf-8"
    )
    return layout


def _seen_resumes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_resume(_cfg: Path | None, session_id: str, **kw: object) -> int:
        calls.append((session_id, str(kw.get("steer", ""))))
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)
    return calls


def test_free_text_becomes_the_next_leg_then_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each answer is the next turn's operator instruction -- exactly what
    --steer carries -- so the session continues without retyping `resume`."""
    calls = _seen_resumes(monkeypatch)
    answers = iter(["now add the tests", "  ", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "now add the tests")]


def test_exit_leaves_the_session_resumable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/exit ends the prompting, never the session: nothing is sealed, so the
    printed line is the one that picks it back up."""
    _seen_resumes(monkeypatch)
    rc = prompt_mod.end_of_session_prompt(
        rc=3, session_id="runny-one-AAAAAA", ask=lambda _p: "/exit"
    )
    assert rc == 3
    assert "agent6 resume runny-one-AAAAAA" in capsys.readouterr().out


def test_eof_ends_like_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Walking away mid-prompt (Ctrl-D) is not an instruction."""
    calls = _seen_resumes(monkeypatch)

    def eof(_p: str) -> str:
        raise EOFError

    assert prompt_mod.end_of_session_prompt(rc=0, session_id="r-AAAAAA", ask=eof) == 0
    assert not calls


def test_a_failing_leg_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume that refuses (bad config, dirty tree) returns its own code
    rather than re-prompting over the failure."""

    def failing(_cfg: Path | None, _session_id: str, **_kw: object) -> int:
        return 2

    monkeypatch.setattr(prompt_mod, "_cmd_resume", failing)
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "keep going"

    assert prompt_mod.end_of_session_prompt(rc=0, session_id="r-AAAAAA", ask=ask) == 2
    assert len(asked) == 1


def test_no_terminal_ends_the_session_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A headless run (CI, a detached spawn) has nobody to type: it must end,
    not block on a prompt nothing will answer."""
    from agent6.ui import cli
    from agent6.ui.cli import _prompt_for_the_next_input  # pyright: ignore[reportPrivateUsage]

    # A real session dir so the ONLY short-circuit under test is the tty guard;
    # patch the bindings _prompt_for_the_next_input actually calls (imported
    # into `cli`, not the source module).
    layout = _seed_session(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "prompting_is_possible", lambda: False)
    called: list[str] = []

    def spy(**_kw: object) -> int:
        called.append("asked")
        return 0

    monkeypatch.setattr(cli, "end_of_session_prompt", spy)
    assert _prompt_for_the_next_input(None, 0, layout.session_id) == 0
    assert not called


def test_ask_sessions_do_not_prompt() -> None:
    """`agent6 ask` answers a question; a one-shot that becomes a conversation
    is a different feature: this is scoped to run and plan sessions."""
    import inspect

    from agent6.ui.cli import _dispatch_ask  # pyright: ignore[reportPrivateUsage]

    assert "_prompt_for_the_next_input" not in inspect.getsource(_dispatch_ask)


def test_a_backgrounded_run_is_not_stopped_by_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`agent6 run ... &` keeps a tty on stdin, so isatty() alone said "someone
    is there". Reading the terminal from a BACKGROUND process group raises
    SIGTTIN, which stops the job: the run suspended at the end instead of
    finishing, and needed `fg`. The same shape blocks forever wherever a tty is
    allocated with nobody at it (`docker run -t`, some CI runners).
    """
    monkeypatch.setattr(prompt_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(prompt_mod.sys.stdin, "fileno", lambda: 0)
    monkeypatch.setattr(prompt_mod.os, "getpgrp", lambda: 4242)

    def owner_is(pgrp: int) -> Callable[[int], int]:
        def tcgetpgrp(_fd: int) -> int:
            return pgrp

        return tcgetpgrp

    monkeypatch.setattr(prompt_mod.os, "tcgetpgrp", owner_is(1717))
    assert not prompt_mod.prompting_is_possible(), "prompted from a background process group"

    monkeypatch.setattr(prompt_mod.os, "tcgetpgrp", owner_is(4242))
    assert prompt_mod.prompting_is_possible(), "the foreground job must still prompt"


def test_a_refused_runs_discarded_id_ends_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal discards its husk, so the minted id matches nothing on disk;
    the follow-up prompt must end with the refusal's exit code, not crash on
    the resolver's SessionIdError."""
    from agent6.ui import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "prompting_is_possible", lambda: True)
    assert cli._prompt_for_the_next_input(None, 2, "gone-run-QQQQQQ") == 2  # pyright: ignore[reportPrivateUsage]
