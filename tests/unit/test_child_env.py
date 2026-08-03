# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a process agent6 spawns outside the jail inherits.

The one owner for the notify hooks and the MCP servers, so their env-scope
claims cannot drift apart -- and so a provider key cannot reach either.
"""

from __future__ import annotations

import pytest

from agent6.child_env import curated_env


def test_a_provider_key_never_reaches_a_spawned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MCP server is third-party code that may log or forward its env, and
    it used to be handed the agent's FULL environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = curated_env()

    assert env["PATH"] == "/usr/bin"
    assert not [k for k in env if "API_KEY" in k or "TOKEN" in k]


def test_a_server_gets_exactly_the_variables_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming each one is the point: a provider key is never among them,
    because nobody would write it down."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    env = curated_env(passthrough=("GITHUB_TOKEN",))

    assert env["GITHUB_TOKEN"] == "ghp-x"
    assert "ANTHROPIC_API_KEY" not in env


def test_a_name_that_is_not_set_is_simply_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty string would read as "configured but blank" to a server."""
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert "NOT_SET_ANYWHERE" not in curated_env(passthrough=("NOT_SET_ANYWHERE",))


def test_the_callers_facts_win_over_the_inherited_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT6_SESSION_ID", "stale")
    assert curated_env(extra={"AGENT6_SESSION_ID": "fresh"})["AGENT6_SESSION_ID"] == "fresh"
