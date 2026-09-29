# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A CLI run/plan session ends by asking for the next input (`/exit` finishes)."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.app._setup import BudgetOverrides, SandboxOverrides
from agent6.paths import state_dir
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import _session_prompt as prompt_mod


def _seed_session(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, session_id: str = "test-run-AAAAAA"
) -> SessionLayout:
    """A real run dir under repo_root's state home, so resolution reaches the tty guard."""
    monkeypatch.setenv("XDG_STATE_HOME", str(repo_root / ".state"))
    layout = SessionLayout(state_dir=state_dir(repo_root), session_id=session_id, subdir="runs")
    layout.session_dir.mkdir(parents=True, exist_ok=True)
    (layout.session_dir / "logs.jsonl").write_text(
        '{"type": "session.start", "ts": "2026-01-01T00:00:00Z"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )
    return layout


def _run_args(**overrides: object) -> argparse.Namespace:
    """`agent6 run` flags as argparse hands them over, defaults unless overridden."""
    fields: dict[str, object] = {
        "config": None,
        "max_usd": None,
        "max_tokens_fallback": None,
        "dangerously_disable_sandbox": False,
        "auto_approve": False,
        "no_commands": False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _seen_resumes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_resume(_cfg: Path | None, session_id: str, **kw: object) -> int:
        calls.append((session_id, str(kw.get("steer", ""))))
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)
    return calls


def test_follow_up_executions_run_under_the_invocations_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow-up at "next:" carries the run's overrides, such as `--max-usd`."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    seen: list[dict[str, object]] = []

    def fake_resume(_cfg: Path | None, session_id: str, **kw: object) -> int:
        seen.append(dict(kw))
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)
    answers = iter(["and a test", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    args = _run_args(max_usd=0.10, auto_approve=True)
    assert cli._prompt_for_the_next_input(args, 0, layout.session_id) == 0  # pyright: ignore[reportPrivateUsage]
    (execution,) = seen
    assert execution["steer"] == "and a test"
    assert execution["budget_overrides"] == BudgetOverrides.from_args(args)
    assert execution["sandbox_overrides"] == SandboxOverrides.from_args(args)


def test_free_text_becomes_the_next_execution_then_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each follow-up answer becomes the next turn's operator instruction, as `--steer` does."""
    calls = _seen_resumes(monkeypatch)
    answers = iter(["now add the tests", "  ", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "now add the tests")]


def test_a_malformed_directive_re_prompts_instead_of_spending_a_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare `/pin` at "next:" is refused by name and asked again, never run as an execution."""
    calls = _seen_resumes(monkeypatch)
    answers = iter(["/pin", "/pin keep the API stable", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "/pin keep the API stable")]
    assert "pin needs an instruction" in capsys.readouterr().err


def test_exit_leaves_the_session_resumable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """/exit ends the prompting, never the session; the printed line picks it back up."""
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


def test_a_failing_execution_stops_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resume that refuses (bad config, dirty tree) returns its own code, with no re-prompt."""

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
    """A headless run has nobody to type, so it ends instead of blocking on the prompt."""
    from agent6.ui.cli import _prompt_for_the_next_input  # pyright: ignore[reportPrivateUsage]

    # A real session dir, so the tty guard is the only short-circuit; patch what `cli` imports.
    layout = _seed_session(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: False)
    called: list[str] = []

    def spy(**_kw: object) -> int:
        called.append("asked")
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    assert _prompt_for_the_next_input(_run_args(), 0, layout.session_id) == 0
    assert not called


def test_ask_sessions_do_not_prompt() -> None:
    """`agent6 ask` stays a one-shot; the follow-up prompt is scoped to run and plan sessions."""
    import inspect

    from agent6.ui.cli import _dispatch_ask  # pyright: ignore[reportPrivateUsage]

    assert "_prompt_for_the_next_input" not in inspect.getsource(_dispatch_ask)


def test_a_backgrounded_run_is_not_stopped_by_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backgrounded run (`&`) ends without the prompt: a tty on stdin is not someone there.

    Reading the terminal from a background process group raises SIGTTIN and suspends the job; the
    same shape blocks forever wherever a tty is allocated with nobody at it.
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
    """A refusal discards its husk, and the follow-up prompt ends with the refusal's exit code."""
    from agent6.ui import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    assert cli._prompt_for_the_next_input(_run_args(), 2, "gone-run-QQQQQQ") == 2  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("mode", "asks"), [("run", True), ("plan", True), ("ask", False)])
def test_a_resumed_execution_ends_by_asking_like_a_fresh_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, asks: bool
) -> None:
    """A resumed run or plan asks "next:" the way a run does; a resumed ask stays a one-shot."""
    import json

    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="resumed-run-AAAAAA")
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": layout.session_id, "mode": mode}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)

    def _resumed(*_a: object, **_k: object) -> int:
        return 0

    monkeypatch.setattr("agent6.ui.cli.resume._cmd_resume", _resumed)
    asked: list[str] = []

    def spy(**kw: object) -> int:
        asked.append(str(kw["session_id"]))
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    args = _run_args(session_id="resumed-run", force=False, tui=False, preset="", steer="")
    assert cli._dispatch_resume(args) == 0  # pyright: ignore[reportPrivateUsage]
    assert asked == (["resumed-run-AAAAAA"] if asks else [])


@pytest.mark.parametrize("target", ["resumed-run", ""])
def test_resume_prompt_stays_on_the_session_selected_at_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A follow-up stays with the selected session when a concurrent session becomes newest."""
    from agent6.ui import cli

    selected = _seed_session(tmp_path, monkeypatch, session_id="resumed-run-AAAAAA")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)

    def fake_resume(*_args: object, **_kwargs: object) -> int:
        _seed_session(tmp_path, monkeypatch, session_id="resumed-run-BBBBBB")
        return 0

    monkeypatch.setattr("agent6.ui.cli.resume._cmd_resume", fake_resume)
    prompted: list[str] = []

    def spy(**kwargs: object) -> int:
        prompted.append(str(kwargs["session_id"]))
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    args = _run_args(session_id=target, force=False, tui=False, preset="", steer="")
    assert cli._dispatch_resume(args) == 0  # pyright: ignore[reportPrivateUsage]
    assert prompted == [selected.session_id]


def test_a_refused_execution_does_not_prompt_on_an_existing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused run whose explicit id points at an older session gets no follow-up prompt."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="existing-run-AAAAAA")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)

    def refused(*_args: object, **_kwargs: object) -> int:
        return 2

    monkeypatch.setattr("agent6.ui.cli.run._cmd_run", refused)
    monkeypatch.setattr("agent6.ui.cli.resume._cmd_resume", refused)

    def must_not_prompt(**_kwargs: object) -> int:
        pytest.fail("prompted after a refused execution")

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", must_not_prompt)
    common = {
        "session_id": layout.session_id,
        "tui": False,
        "config": None,
        "preset": "",
    }
    run_args = _run_args(
        **common,
        interactive=False,
        parallel="",
        standing="",
        seed_from="",
        task="new task",
        skill=[],
        pins=[],
        decompose=False,
    )
    plan_args = _run_args(**common, plan_command="run", task="new plan")
    resume_args = _run_args(**common, interactive=False, force=False, steer="")
    assert cli._dispatch_run(run_args) == 2  # pyright: ignore[reportPrivateUsage]
    assert cli._dispatch_plan(plan_args) == 2  # pyright: ignore[reportPrivateUsage]
    assert cli._dispatch_resume(resume_args) == 2  # pyright: ignore[reportPrivateUsage]


def test_a_execution_that_undoes_or_detaches_ends_the_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that parked never ran; the resume line it printed is the next step, not "next:"."""
    layout = _seed_session(tmp_path, monkeypatch, session_id="undone-run-AAAAAA")
    monkeypatch.chdir(tmp_path)
    asked: list[str] = []

    def fake_resume(_cfg: Path | None, _sid: str, **kw: object) -> int:
        # The execution forks back and ends the run as undone.
        (layout.session_dir / "logs.jsonl").write_text(
            '{"type": "session.start"}\n{"type": "session.end", "reason": "undone"}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(prompt_mod, "_cmd_resume", fake_resume)

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "/undo" if len(asked) == 1 else pytest.fail("asked again after the undo")

    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id=layout.session_id, session_dir=layout.session_dir, ask=ask
    )
    assert rc == 0 and len(asked) == 1


def test_a_detached_run_is_not_followed_by_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After `/detach` there is nothing to follow up on: the run continues in the background."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="detached-run-AAAAAA")
    (layout.session_dir / "logs.jsonl").write_text(
        '{"type": "session.start", "ts": "2026-01-01T00:00:00Z"}\n'
        '{"type": "loop.steer.detached"}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)

    def _must_not_prompt(**_kw: object) -> int:
        pytest.fail("prompted")

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", _must_not_prompt)
    args = _run_args()
    assert cli._prompt_for_the_next_input(args, 0, layout.session_id) == 0  # pyright: ignore[reportPrivateUsage]


def test_a_parked_start_is_not_followed_by_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that parked never ran; the resume line it printed is the next step."""
    import json

    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="parked-run-AAAAAA")
    (layout.session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": layout.session_id,
                "mode": "run",
                "user_task": "t",
                "parked_task": "t",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    asked: list[str] = []

    def spy(**_kw: object) -> int:
        asked.append("asked")
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    assert cli._prompt_for_the_next_input(_run_args(), 2, layout.session_id) == 2  # pyright: ignore[reportPrivateUsage]
    assert asked == []


def test_an_undone_run_is_not_followed_by_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/undo names the fork as the continuation; the undone run gets no "next:" prompt."""
    from agent6.ui import cli

    layout = _seed_session(tmp_path, monkeypatch, session_id="undone-run-AAAAAA")
    with (layout.session_dir / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "session.undone", "new_session_id": "fork-BBBBBB"}\n')
        fh.write('{"type": "session.end", "reason": "undone", "all_passed": false}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    asked: list[str] = []

    def spy(**_kw: object) -> int:
        asked.append("asked")
        return 0

    monkeypatch.setattr("agent6.ui.cli._session_prompt.end_of_session_prompt", spy)
    assert cli._prompt_for_the_next_input(_run_args(), 0, layout.session_id) == 0  # pyright: ignore[reportPrivateUsage]
    assert asked == []


def test_a_lone_slash_word_is_refused_not_sent_as_a_task(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lone slash word at "next:" re-prompts with a pointer instead of becoming a task.

    A live-run command points at steer, an unknown word (a typo, a REPL verb) at the prompt's own
    help. Multi-word slash input still rides as the execution's instruction, since directives like
    `/pin <text>` are the loop's to parse.
    """
    calls = _seen_resumes(monkeypatch)
    answers = iter(["/shells", "/cost", "now add the tests", "/exit"])
    rc = prompt_mod.end_of_session_prompt(
        rc=0, session_id="runny-one-AAAAAA", ask=lambda _p: next(answers)
    )
    assert rc == 0
    assert calls == [("runny-one-AAAAAA", "now add the tests")]
    err = capsys.readouterr().err
    assert "/shells is a composer command, not an instruction" in err
    assert "'/cost' is not sent as a task" in err


def test_i_with_tui_is_refused_before_a_execution_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`-i` with `--tui` is refused up front for `run` and `resume`: both want the terminal."""
    import agent6.ui.cli.resume as resume_mod
    import agent6.ui.cli.run as run_mod
    from agent6.ui import cli

    def _never(*_a: object, **_k: object) -> int:
        raise AssertionError("the execution must not start")

    monkeypatch.setattr(run_mod, "_cmd_run", _never)
    monkeypatch.setattr(resume_mod, "_cmd_resume", _never)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    run_args = _run_args(
        interactive=True,
        tui=True,
        parallel="",
        session_id="",
        standing="",
        seed_from="",
        task="do a thing",
        skill=[],
        pins=[],
        decompose=False,
    )
    assert cli._dispatch_run(run_args) == 2  # pyright: ignore[reportPrivateUsage]
    resume_args = _run_args(interactive=True, tui=True, session_id="runny-one-AAAAAA", steer="")
    assert cli._dispatch_resume(resume_args) == 2  # pyright: ignore[reportPrivateUsage]
    err = capsys.readouterr().err
    assert err.count("-i cannot combine with --tui") == 2


def _plan_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A `plan` whose execution is a fake writing a finished session; returns the prompts asked."""
    import agent6.ui.cli.run as run_mod

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    asked: list[str] = []

    def fake_cmd_run(*_a: object, **kw: object) -> int:
        layout = SessionLayout(
            state_dir=state_dir(tmp_path), session_id=str(kw["session_id"]), subdir="plans"
        )
        layout.session_dir.mkdir(parents=True, exist_ok=True)
        (layout.session_dir / "logs.jsonl").write_text(
            '{"type": "session.start", "ts": "2026-01-01T00:00:00Z"}\n'
            '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(run_mod, "_cmd_run", fake_cmd_run)
    monkeypatch.setattr("agent6.ui.cli._session_prompt.prompting_is_possible", lambda: True)
    monkeypatch.setattr("builtins.input", lambda p="": (asked.append(p), "/exit")[1])
    return asked


def test_plan_tui_does_not_hand_the_terminal_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`plan --tui` ends where the TUI ends, as `run --tui` does."""
    from agent6.ui.cli import main

    asked = _plan_harness(tmp_path, monkeypatch)
    assert main(["plan", "--tui", "do a thing"]) == 0
    assert asked == []
    asked = _plan_harness(tmp_path, monkeypatch)
    assert main(["plan", "do a thing"]) == 0
    assert asked == ["next (/exit to finish): "]
