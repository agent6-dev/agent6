from __future__ import annotations

import contextlib
import pathlib
from unittest import mock

import pytest

from agent6.ui.tui import conversation


@pytest.mark.parametrize(
    "pager, expected",
    [
        ("less -R", ["less", "-R"]),
        ("less -FRX", ["less", "-FRX"]),
        ("/usr/bin/less -F", ["/usr/bin/less", "-F", "-R"]),
        ("most -s", ["most", "-s"]),
        ('"/path with spaces/pager" -s', ["/path with spaces/pager", "-s"]),
        ("lesspipe", ["lesspipe"]),
        ("   ", ["less", "-R"]),
    ],
)
def test_pager_uses_shell_words_and_only_adds_missing_less_color_flag(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, pager: str, expected: list[str]
) -> None:
    """PAGER arguments stay separate, and less receives one raw-color option."""
    monkeypatch.setenv("PAGER", pager)
    screen = conversation.ConversationScreen(tmp_path / "logs.jsonl", title=lambda _ctx: "test")
    screen._content.append("transcript")  # pyright: ignore[reportPrivateUsage]
    app = mock.Mock()
    app.suspend.return_value = contextlib.nullcontext()

    def get_app(_self: conversation.ConversationScreen) -> mock.Mock:
        return app

    monkeypatch.setattr(conversation.ConversationScreen, "app", property(get_app))
    with mock.patch.object(conversation.subprocess, "run") as launch:
        screen.action_pager()
    launch.assert_called_once_with(expected, input="transcript", text=True, check=False)


def test_malformed_pager_is_reported_without_launching(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unbalanced PAGER quoting is a visible error rather than an exception from the action."""
    monkeypatch.setenv("PAGER", "less '")
    screen = conversation.ConversationScreen(tmp_path / "logs.jsonl", title=lambda _ctx: "test")
    with mock.patch.object(screen, "notify") as notice:
        screen.action_pager()
    assert "$PAGER" in notice.call_args.args[0]
    assert "No closing quotation" in notice.call_args.args[0]
