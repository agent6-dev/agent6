# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The SessionFrontend an ACP client provides.

The rule every case here pins: a client that cannot be asked is never asked,
and the answer is the CAUTIOUS one. A session that cannot ask is a session that
does less, never one that does something unwatched.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from agent6 import events as agent6_events
from agent6.app import frontend
from agent6.tools import operator_prompts, schema
from agent6.ui.acp import frontend as acp_frontend


def _frontend(*, can_ask: bool = True, reply: str | None = "allow"):
    asked: list[tuple[str, tuple[str, ...], bool | None]] = []

    def _ask(
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        _call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        asked.append((prompt, options, standing))
        return reply

    front = acp_frontend.acp_frontend(
        ask=_ask,
        capabilities=frontend.FrontendCapabilities(can_ask=can_ask),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid, _flags: "",
    )
    return front, asked


def _prompts(
    front: frontend.SessionFrontend, session_dir: pathlib.Path, log: str = "logs.jsonl"
) -> operator_prompts.OperatorPrompts:
    """The gate over this front-end's approver and questioner, journaling into the session dir."""
    return operator_prompts.OperatorPrompts(
        approver=front.build_approver(session_dir),
        questioner=front.build_questioner(session_dir),
        journal=agent6_events.EventSink(session_dir / log).emit,
        session_dir=session_dir,
    )


def test_an_approval_becomes_a_request_to_the_editor(tmp_path: pathlib.Path) -> None:
    front, asked = _frontend()
    approve = _prompts(front, tmp_path).approve
    assert approve("Allow run_command: ls", scope="command") is True
    assert asked == [("Allow run_command: ls", ("allow", "deny"), True)]
    assert _journal(tmp_path / "logs.jsonl")[-1]["source"] == "acp"


def test_a_client_that_cannot_be_asked_gets_a_no(tmp_path: pathlib.Path) -> None:
    """Not a hang, and not an invented yes."""
    front, asked = _frontend(can_ask=False)
    approve = _prompts(front, tmp_path).approve
    assert approve("Allow run_command: rm -rf /") is False
    assert asked == [], "it must not even try"
    # The journal says nobody was asked, the CLI's word for a deny with no front-end to prompt.
    assert _journal(tmp_path / "logs.jsonl")[-1]["source"] == "headless"


def test_declining_is_a_no(tmp_path: pathlib.Path) -> None:
    front, _asked = _frontend(reply="deny")
    approve = _prompts(front, tmp_path).approve
    assert approve("Allow run_command: ls") is False


def test_a_question_carries_its_options_and_an_unanswered_one_is_empty(
    tmp_path: pathlib.Path,
) -> None:
    """The loop reads an empty answer as "the operator said nothing", which differs from a value."""
    front, asked = _frontend(reply="dark")
    ask_user = _prompts(front, tmp_path).ask
    assert ask_user(
        (schema.UserQuestion(question="Theme?", options=("dark", "light")),)
    ).answers == ("dark",)
    assert asked[0] == ("Theme?", ("dark", "light"), None)
    assert _journal(tmp_path / "logs.jsonl")[-1]["source"] == "acp"

    mute, _ = _frontend(can_ask=False)
    silent = _prompts(mute, tmp_path, "silent.jsonl").ask
    assert silent((schema.UserQuestion(question="Theme?"),)).answers == ("",)
    assert _journal(tmp_path / "silent.jsonl")[-1]["source"] == "headless"


def test_a_free_form_question_says_that_no_editor_could_answer_it(tmp_path: pathlib.Path) -> None:
    """ACP v1 has no free-form answer control, so a blank is not an answer.

    Silently returning an ordinary blank told the model an operator chose to say nothing.
    """
    front, asked = _frontend()
    answer = _prompts(front, tmp_path).ask((schema.UserQuestion(question="Which port?"),))
    assert asked == []
    assert answer.answers == ("",)
    assert operator_prompts.unanswered_note(answer)
    # `unseen` is per request: a blank beside an answered button would read as deliberate.
    mixed = (
        schema.UserQuestion(question="Proxy?", options=("yes", "no")),
        schema.UserQuestion(question="Port?"),
    )
    answer = _prompts(front, tmp_path).ask(mixed)
    assert asked == []
    assert answer.answers == ("", "")
    assert operator_prompts.unanswered_note(answer)


def test_the_unsandboxed_prompt_fires_only_when_it_is_true() -> None:
    """The lifecycle calls this on EVERY run; the "is this dangerous" test lives in the answer.

    Asking regardless told the editor a confined run was unsandboxed, a false statement about the
    run, on the one approval that must never become reflexive.
    """
    from agent6.config import Config

    front, asked = _frontend(reply="allow")
    assert front.confirm_unconfined_autorun("strict", Config()) is True
    assert asked == [], "a confined run is not dangerous and must not prompt"

    dangerous = Config.model_validate({"sandbox": {"isolation": "none", "run_commands": "yes"}})
    assert front.confirm_unconfined_autorun("none", dangerous) is True
    assert len(asked) == 1 and "UNSANDBOXED" in asked[0][0]


def test_an_unsandboxed_autorun_still_needs_a_human() -> None:
    from agent6.config import Config

    dangerous = Config.model_validate({"sandbox": {"isolation": "none", "run_commands": "yes"}})
    mute, _ = _frontend(can_ask=False)
    assert mute.confirm_unconfined_autorun("none", dangerous) is False
    denied, _ = _frontend(reply="deny")
    assert denied.confirm_unconfined_autorun("none", dangerous) is False


def test_an_approval_that_must_not_be_remembered_says_so(tmp_path: pathlib.Path) -> None:
    """A prompt with no scope is the fetch tool's off-list host, where a GET can carry data out.

    An editor that offers "always allow" needs something to key that decision on.
    """
    front, asked = _frontend(reply="allow once")
    approve = _prompts(front, tmp_path).approve
    assert approve("Allow fetch: evil.example /x") is True
    assert asked[-1][1:] == (("allow once", "deny"), False)

    approve("Allow run_command: ls", scope="command")
    assert asked[-1][1:] == (("allow", "deny"), True)


def test_the_editor_is_the_live_view_and_nothing_is_drawn() -> None:
    """The editor is the live view: deltas are emitted and no headless end block is printed.

    The lifecycle prints its headline and summary only when no live view rendered the run, and
    the reporter repeats every line to the editor, which already has the fold's done item.
    """
    front, _asked = _frontend()
    assert front.stream_modes(False) == (True, True)
    assert front.attach_console_view(None) is None  # pyright: ignore[reportArgumentType]
    assert front.should_spawn_tui(True, True, "run") is False


def test_the_steer_seam_is_inert() -> None:
    """ACP steers by prompting into a live session; a pause menu has no terminal to draw on."""
    front, _asked = _frontend()
    steer = front.make_steer_state(None, pathlib.Path("/x"), lambda: None)  # pyright: ignore[reportArgumentType]
    assert steer.requested() is False
    assert steer.prompt() is None
    assert steer.abort_pending() is False
    steer.clear()
    steer.restore()
    steer.reset_stage()


def test_parallel_lanes_are_not_spawned_into_a_single_pane() -> None:
    """`/parallel` fans out sibling runs.

    An ACP client renders ONE session, so lanes would run invisibly.
    """
    from agent6.config import Config

    front, _asked = _frontend()
    spawner = front.build_coordinator_spawner(
        Config(), pathlib.Path("/x"), pathlib.Path("/y"), "run", "r", None, False
    )
    assert spawner is None


def test_the_ask_repl_is_refused_rather_than_faked() -> None:
    """Two turn loops reading the same stdin is not something to paper over."""
    front, _asked = _frontend()
    with pytest.raises(RuntimeError, match="drives its own turns"):
        front.run_ask_repl(None, None, None, "q")  # pyright: ignore[reportArgumentType]


def test_steer_hooks_consume_a_seeded_resume_steer(tmp_path: pathlib.Path) -> None:
    """A later ACP prompt reaches the resumed run through the steer files.

    The frontend's hooks are the file bridge that reads them; inert hooks dropped the seeded
    instruction, so the resumed model re-finished the old task.
    """
    from agent6.sessions import ipc

    front, _ = _frontend()
    hooks = front.make_steer_state(
        None,  # pyright: ignore[reportArgumentType]
        tmp_path,
        lambda: None,  # pyright: ignore[reportArgumentType]
    )
    assert hooks.requested() is False
    ipc.request_steer(tmp_path)
    ipc.write_steer_answer(tmp_path, "append the line 'second turn'")
    assert hooks.requested() is True
    assert hooks.prompt() == "append the line 'second turn'"
    hooks.clear()
    assert hooks.requested() is False


def _journal(path: pathlib.Path) -> list[dict[str, object]]:
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_request_names_the_call_the_prompt_carries(tmp_path: pathlib.Path) -> None:
    """The editor is asked under the id the gate stamped on the prompt.

    With two calls in flight the gated one is not the newest, so an id re-derived from the
    front-end's own event stream names the wrong call.
    """
    calls: list[int | None] = []

    def _ask(
        prompt: str,
        options: tuple[str, ...],
        standing: bool | None,
        call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        calls.append(call_id)
        return options[0]

    front = acp_frontend.acp_frontend(
        ask=_ask,
        capabilities=frontend.FrontendCapabilities(can_ask=True),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid, _flags: "",
    )
    prompts = _prompts(front, tmp_path)
    events = agent6_events.EventSink(tmp_path / "logs.jsonl")
    events.emit("tool.call", name="run_command", args={"argv": ["ls"]}, call_id=1)
    events.emit("tool.call", name="read_file", args={"path": "x"}, call_id=2)
    assert prompts.approve("Allow run_command: ls", scope="command", call_id=1) is True
    assert prompts.ask(
        (schema.UserQuestion(question="Theme?", options=("dark", "light")),), call_id=1
    )
    prompts.approve("Run commands UNSANDBOXED on this host?")  # gates no call
    assert calls == [1, 1, None]


def test_a_multi_question_ask_shares_one_deadline(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One deadline covers a whole ask_user request, not one per question.

    Each question got the whole permission timeout, so a three-question request held a run
    three times the documented bound when the editor never answered.
    """
    from agent6.ui.acp import frontend as frontend_mod

    monkeypatch.setattr(frontend_mod, "PERMISSION_TIMEOUT_S", 0.0)
    held: list[bool] = []

    def _ask(
        _prompt: str,
        _options: tuple[str, ...],
        _standing: bool | None,
        _call_id: int | None,
        until: Callable[[], bool] | None = None,
    ) -> str | None:
        assert until is not None
        held.append(until())  # the request's deadline has passed: holds at once
        return None

    front = acp_frontend.acp_frontend(
        ask=_ask,
        capabilities=frontend.FrontendCapabilities(can_ask=True),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid, _flags: "",
    )
    answer = front.build_questioner(tmp_path)(
        operator_prompts.QuestionRequest(
            id="question-1",
            questions=(
                schema.UserQuestion(question="a?", options=("yes", "no")),
                schema.UserQuestion(question="b?", options=("yes", "no")),
            ),
            call_id=1,
        )
    )
    assert held == [True, True]
    assert (answer.answers, answer.source) == (("", ""), "acp")
