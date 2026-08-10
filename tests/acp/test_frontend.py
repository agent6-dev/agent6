# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The SessionFrontend an ACP client provides.

The rule every case here pins: a client that cannot be asked is never asked,
and the answer is the CAUTIOUS one. A session that cannot ask is a session that
does less, never one that does something unwatched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.app.frontend import FrontendCapabilities
from agent6.tools.schema import UserQuestion
from agent6.ui.acp.frontend import acp_frontend


def _frontend(*, can_ask: bool = True, reply: str | None = "allow"):
    asked: list[tuple[str, tuple[str, ...], bool | None]] = []

    def _ask(prompt: str, options: tuple[str, ...], standing: bool | None) -> str | None:
        asked.append((prompt, options, standing))
        return reply

    front = acp_frontend(
        ask=_ask,
        capabilities=FrontendCapabilities(can_ask=can_ask),
        agent6_exe=lambda: "agent6",
        spawn_detached_resume=lambda _cwd, _rid: "",
    )
    return front, asked


def test_an_approval_becomes_a_request_to_the_editor() -> None:
    front, asked = _frontend()
    approve = front.build_approver(Path("/x"), None)  # pyright: ignore[reportArgumentType]
    assert approve("Allow run_command: ls", scope="command") is True
    assert asked == [("Allow run_command: ls", ("allow", "deny"), True)]


def test_a_client_that_cannot_be_asked_gets_a_no() -> None:
    """Not a hang, and not an invented yes."""
    front, asked = _frontend(can_ask=False)
    approve = front.build_approver(Path("/x"), None)  # pyright: ignore[reportArgumentType]
    assert approve("Allow run_command: rm -rf /") is False
    assert asked == [], "it must not even try"


def test_declining_is_a_no() -> None:
    front, _asked = _frontend(reply="deny")
    approve = front.build_approver(Path("/x"), None)  # pyright: ignore[reportArgumentType]
    assert approve("Allow run_command: ls") is False


def test_a_question_carries_its_options_and_an_unanswered_one_is_empty() -> None:
    """The loop already reads an empty answer as "the operator said nothing",
    which is different from a value."""
    front, asked = _frontend(reply="dark")
    ask_user = front.build_questioner(Path("/x"), None)  # pyright: ignore[reportArgumentType]
    assert ask_user((UserQuestion(question="Theme?", options=("dark", "light")),)) == ("dark",)
    assert asked[0] == ("Theme?", ("dark", "light"), None)

    mute, _ = _frontend(can_ask=False)
    silent = mute.build_questioner(Path("/x"), None)  # pyright: ignore[reportArgumentType]
    assert silent((UserQuestion(question="Theme?"),)) == ("",)


def test_the_unsandboxed_prompt_fires_only_when_it_is_true() -> None:
    """The lifecycle calls this on EVERY run; the "is this dangerous" test
    lives in the answer. Asking regardless told the editor a confined run was
    unsandboxed -- a false statement about the run, on the one approval that
    must never become reflexive."""
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


def test_an_approval_that_must_not_be_remembered_says_so() -> None:
    """A prompt with no scope is the fetch tool's off-list host, where a GET
    can carry data out in its path. An editor that offers "always allow" needs
    something to key that decision on."""
    front, asked = _frontend(reply="allow once")
    approve = front.build_approver(Path("/x"), None)  # pyright: ignore[reportArgumentType]
    assert approve("Allow fetch: evil.example /x") is True
    assert asked[-1][1:] == (("allow once", "deny"), False)

    approve("Allow run_command: ls", scope="command")
    assert asked[-1][1:] == (("allow", "deny"), True)


def test_nothing_is_drawn_and_deltas_still_reach_the_journal() -> None:
    """An ACP client renders from session/update, so there is no console view
    -- but the deltas still have to be EMITTED for it to render."""
    front, _asked = _frontend()
    stream_text, console_stream = front.stream_modes(False)
    assert stream_text is True, "session/update has nothing to show otherwise"
    assert console_stream is False, "there is no console to echo to"
    assert front.should_spawn_tui(True, True, "run") is False


def test_the_steer_seam_is_inert() -> None:
    """ACP steers by prompting into a live session; a SIGINT pause menu has no
    terminal to draw on."""
    front, _asked = _frontend()
    steer = front.make_steer_state(None, Path("/x"), lambda: None)  # pyright: ignore[reportArgumentType]
    assert steer.requested() is False
    assert steer.prompt() is None
    assert steer.abort_pending() is False
    steer.clear()
    steer.restore()
    steer.reset_stage()


def test_parallel_lanes_are_not_spawned_into_a_single_pane() -> None:
    """`/parallel` fans out sibling runs. An ACP client renders ONE session, so
    lanes would run invisibly."""
    from agent6.config import Config

    front, _asked = _frontend()
    spawner = front.build_coordinator_spawner(
        Config(), Path("/x"), Path("/y"), "run", "r", None, False
    )
    assert spawner is None


def test_the_ask_repl_is_refused_rather_than_faked() -> None:
    """Two turn loops reading the same stdin is not something to paper over."""
    front, _asked = _frontend()
    with pytest.raises(RuntimeError, match="drives its own turns"):
        front.run_ask_repl(None, None, None, "q")  # pyright: ignore[reportArgumentType]
