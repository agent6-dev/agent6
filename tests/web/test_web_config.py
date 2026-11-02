# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[web]` config section: secure by default (loopback), non-loopback opt-in."""

from __future__ import annotations

import pathlib

import pydantic
import pytest

from agent6.config import Config, WebConfig


def test_web_defaults_are_loopback() -> None:
    w = WebConfig()
    assert w.host == "127.0.0.1"
    assert w.port == 7658
    assert w.allow_non_loopback is False


def test_config_carries_web_section() -> None:
    assert Config().web.host == "127.0.0.1"


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "LOCALHOST"])
def test_loopback_hosts_need_no_optin(host: str) -> None:
    assert WebConfig(host=host).host == host


def test_non_loopback_rejected_without_optin() -> None:
    with pytest.raises(pydantic.ValidationError, match="allow_non_loopback"):
        WebConfig(host="0.0.0.0")


def test_non_loopback_allowed_with_optin() -> None:
    w = WebConfig(host="0.0.0.0", allow_non_loopback=True)
    assert w.host == "0.0.0.0"


def test_port_flag_is_held_to_the_same_bounds_as_the_config_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--port` goes through `[web].port`'s bounds.

    The flag skipped the schema the leaf is held to, so `--port 99999` reached
    `bind()` as an OverflowError crash report and `--port 0` bound an ephemeral
    port while printing the unreachable `:0` as the URL.
    """
    from agent6.ui.cli import web_cmds  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    for bad in (99999, 0, -1):
        assert (
            web_cmds._cmd_web(
                "", config_paths=(), host=None, port=bad, allow_non_loopback=False, secret=None
            )
            == 2
        )
        err = capsys.readouterr().err
        assert err.startswith("ERROR:") and "--port" in err


def test_non_loopback_flag_refusal_uses_the_shared_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli import web_cmds  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)

    assert (
        web_cmds._cmd_web(
            "",
            config_paths=(),
            host="0.0.0.0",
            port=None,
            allow_non_loopback=False,
            secret=None,
        )
        == 2
    )
    assert capsys.readouterr().err.startswith("REFUSING:")


@pytest.mark.parametrize("bucket", ["asks", "machines", "plans", "runs"])
def test_web_resolves_session_and_draft_prefixes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, bucket: str
) -> None:
    """The CLI passes the server a full id, so its exact-id deep link opens the requested page."""
    from unittest import mock

    from agent6 import paths
    from agent6.ui.cli import web_cmds

    monkeypatch.chdir(tmp_path)
    saved = paths.state_dir(tmp_path) / "sessions" / bucket / "quiet-run-AAAAAA"
    saved.mkdir(parents=True)
    (saved / "logs.jsonl").write_text('{"type": "session.start"}\n')
    with mock.patch.object(web_cmds, "run_web", return_value=0) as serve:
        assert (
            web_cmds._cmd_web(
                "quiet",
                config_paths=(),
                host=None,
                port=None,
                allow_non_loopback=False,
                secret=None,
            )
            == 0
        )
    assert serve.call_args.args[0] == saved.name


@pytest.mark.parametrize("target", ["unknown", "quiet"])
def test_web_refuses_unknown_and_ambiguous_targets_before_starting_the_server(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    """A bad target is not silently accepted as a hub link."""
    from unittest import mock

    from agent6 import paths
    from agent6.sessions import layout
    from agent6.ui.cli import web_cmds

    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
    for name in ["quiet-run-AAAAAA", "quiet-run-BBBBBB"]:
        saved = state / "sessions" / "runs" / name
        saved.mkdir(parents=True)
        (saved / "logs.jsonl").write_text('{"type": "session.start"}\n')
    (layout.machines_root(state) / "quiet").mkdir(parents=True)
    with mock.patch.object(web_cmds, "run_web", return_value=0) as serve:
        assert (
            web_cmds._cmd_web(
                target, config_paths=(), host=None, port=None, allow_non_loopback=False, secret=None
            )
            == 2
        )
    serve.assert_not_called()
    assert ("ambiguous" if target == "quiet" else "no session matches") in capsys.readouterr().err


@pytest.mark.parametrize("target", ["", "pipeline"])
def test_web_keeps_the_hub_and_exact_machine_targets(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from unittest import mock

    from agent6 import paths
    from agent6.sessions import layout
    from agent6.ui.cli import web_cmds

    monkeypatch.chdir(tmp_path)
    (layout.machines_root(paths.state_dir(tmp_path)) / "pipeline").mkdir(parents=True)
    with mock.patch.object(web_cmds, "run_web", return_value=0) as serve:
        assert (
            web_cmds._cmd_web(
                target, config_paths=(), host=None, port=None, allow_non_loopback=False, secret=None
            )
            == 0
        )
    assert serve.call_args.args[0] == target
