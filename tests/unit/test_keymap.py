# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One table for the approval letters, read by every surface that offers them."""

from __future__ import annotations

from importlib import resources

from agent6.ui import keymap


def test_the_four_answers_and_their_order() -> None:
    assert [(e.key, e.answer) for e in keymap.APPROVAL_ANSWERS] == [
        ("y", "yes"),
        ("a", "session"),
        ("n", "no"),
        ("d", "session-deny"),
    ]
    assert [e.answer for e in keymap.APPROVAL_ANSWERS if e.standing] == ["session", "session-deny"]


def test_the_cli_prompt_line_is_unchanged() -> None:
    """The line an operator has learned: the plain answers, then the scoped pair explained."""
    assert keymap.approval_prompt_suffix(standing=True) == (
        "[y/N/a/d]  (a = allow all, d = deny all, this session): "
    )
    assert keymap.approval_prompt_suffix(standing=False) == "[y/N]: "


def test_a_letter_or_its_word_answers() -> None:
    for typed in ("y", "Y", " yes ", "YES"):
        assert keymap.answer_for(typed, standing=True) == "yes"
    for typed in ("a", "all", "always", "session"):
        assert keymap.answer_for(typed, standing=True) == "session"
    for typed in ("d", "deny", "never"):
        assert keymap.answer_for(typed, standing=True) == "session-deny"


def test_anything_else_denies() -> None:
    """`[y/N]` says so: the default answer is the safe one."""
    for typed in ("n", "", "   ", "maybe", "yolo"):
        assert keymap.answer_for(typed, standing=True) == "no"


def test_a_scoped_answer_needs_a_scoped_prompt() -> None:
    """A prompt nobody may answer for the session takes `a` as the letter it is, which denies."""
    assert keymap.answer_for("a", standing=False) == "no"
    assert keymap.answer_for("d", standing=False) == "no"
    assert keymap.answer_for("y", standing=False) == "yes"


def test_the_web_describes_the_shared_keys_correctly() -> None:
    """The web has no approval keys; its comment names the CLI's correctly."""
    js = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "the CLI's `x`" not in js
    deny_all = next(e for e in keymap.APPROVAL_ANSWERS if e.answer == "session-deny")
    assert f"`{deny_all.key}` at the CLI prompt" in js


def test_the_keymap_and_the_bridge_agree_on_the_four_values() -> None:
    """`record_answer` decides an answer's meaning without `ui.keymap`; the two are pinned here."""
    import pathlib
    from tempfile import mkdtemp

    from agent6.sessions import ipc

    values = [entry.answer for entry in keymap.APPROVAL_ANSWERS]
    assert values == ["yes", "session", "no", "session-deny"]

    # Which answers grant this call, and which persist for the scope.
    grants = {v for v in values if ipc.record_answer(pathlib.Path(mkdtemp()), v, scope=None)}
    assert grants == {"yes", "session"}
    assert {entry.answer for entry in keymap.APPROVAL_ANSWERS if entry.grants} == grants
    for entry in keymap.APPROVAL_ANSWERS:
        d = pathlib.Path(mkdtemp())
        ipc.record_answer(d, entry.answer, scope="command")
        persisted = ipc.session_allow_set(d, "command") or ipc.session_deny_set(d, "command")
        assert persisted == entry.standing, entry.answer


def test_an_unrecognised_answer_denies_and_persists_nothing() -> None:
    import pathlib
    from tempfile import mkdtemp

    from agent6.sessions import ipc

    d = pathlib.Path(mkdtemp())
    assert not ipc.record_answer(d, "truncated", scope="command")
    assert not ipc.session_allow_set(d, "command") and not ipc.session_deny_set(d, "command")


def test_completion_only_fires_on_a_line_that_is_one_word() -> None:
    """Tab completes a directive only where the whole line is one `/`-word, in every completer."""
    import inspect
    from importlib import resources

    from agent6.ui.cli import _menu_input  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.tui import composer

    assert [c for c, _ in composer.steer_suggestion_rows("/t", mode="steer")] == ["/task"]
    for line in ("fix it /t", "please /ta", "/task already typed "):
        assert composer.steer_suggestion_rows(line, mode="steer") == [], line

    # The pause menu rings the bell instead of opening its menu.
    guard = inspect.getsource(_menu_input._Reader.open_menu)
    assert '" " in self.line' in guard and 'startswith("/")' in guard

    # The web asks the same of the textarea's whole value.
    js = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    assert "v.startsWith('/') && !/\\s/.test(v)" in js


def test_one_key_never_means_two_things_on_one_screen() -> None:
    """No key, alias included, serves two actions."""
    for screen, table in keymap.SCREEN_KEYS.items():
        keys = [key for spec in table.values() for key in spec.split(",")]
        assert len(keys) == len(set(keys)), f"{screen} binds a key twice"


def test_refresh_is_r_wherever_a_screen_refreshes() -> None:
    """One letter, one meaning across screens."""
    for screen, table in keymap.SCREEN_KEYS.items():
        refreshers = [key for action, key in table.items() if action in ("refresh", "reload")]
        assert refreshers in ([], ["r"]), f"{screen} refreshes on {refreshers}, not r"


def test_the_approval_letters_are_free_on_every_view_of_a_session() -> None:
    """The four answer letters bind beside a run view's table; no table letter shadows an answer."""
    letters = {e.key for e in keymap.APPROVAL_ANSWERS}
    for screen in ("conversation", "dashboard", "machine watch"):
        taken = {key for spec in keymap.SCREEN_KEYS[screen].values() for key in spec.split(",")}
        assert not (taken & letters), f"{screen} takes an approval letter"
