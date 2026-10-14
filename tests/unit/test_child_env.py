# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a process agent6 spawns outside the jail inherits.

One owner for the notify hooks and the MCP servers, so a provider key reaches neither.
"""

from __future__ import annotations

import pytest

from agent6 import child_env


def test_a_provider_key_never_reaches_a_spawned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider key never reaches a spawned child; an MCP server may log or forward its env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = child_env.curated_env()

    assert env["PATH"] == "/usr/bin"
    assert not [k for k in env if "API_KEY" in k or "TOKEN" in k]


def test_a_server_gets_exactly_the_variables_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server gets exactly the variables it names, and nobody writes a provider key down."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    env = child_env.curated_env(passthrough=("GITHUB_TOKEN",))

    assert env["GITHUB_TOKEN"] == "ghp-x"
    assert "ANTHROPIC_API_KEY" not in env


def test_a_name_that_is_not_set_is_simply_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty string would read as "configured but blank" to a server."""
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert "NOT_SET_ANYWHERE" not in child_env.curated_env(passthrough=("NOT_SET_ANYWHERE",))


def test_the_callers_facts_win_over_the_inherited_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_SESSION_ID", "stale")
    assert (
        child_env.curated_env(extra={"AGENT6_SESSION_ID": "fresh"})["AGENT6_SESSION_ID"] == "fresh"
    )


def test_an_unconfined_child_never_inherits_the_web_secret() -> None:
    """`AGENT6_WEB_SECRET` signs a browser in to `agent6 web`; a command the model runs gets none.

    Under `isolation = "none"` a command inherits the environment minus the provider keys;
    unless dropped with them, the secret would ride along.
    """
    from agent6 import child_env

    kept = child_env.without_provider_keys({"AGENT6_WEB_SECRET": "six-words-here", "PATH": "/bin"})

    assert kept == {"PATH": "/bin"}
