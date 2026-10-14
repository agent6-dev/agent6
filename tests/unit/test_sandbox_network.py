# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""sandbox.network: isolation compatibility, machine refusals, and the self-confined supervisor.

The supervisor subprocess runs a machine `agent` state self-confined.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest

from agent6 import git_ops, kinds
from agent6.app import confine, machine_agent, providers
from agent6.config import Config, ConfigError, validate_config
from agent6.harness import loop as harness_loop
from agent6.machine import AgentRequest
from agent6.tools import dispatch


def _cfg(network: str = "session") -> Config:
    return validate_config({"sandbox": {"network": network}})


# --- _is_loopback ----------------------------------------------------------


# --- check_network_support (isolation compatibility) ------------------------


@pytest.mark.parametrize("isolation", ["strict", "none"])
def test_check_network_support_allows_off_hardened(isolation: kinds.IsolationLevel) -> None:
    # `session` is only refused on hardened; strict supports it and none is unsandboxed
    # (warned elsewhere), so neither refuses here.
    assert confine.check_network_support(_cfg("session"), isolation) is None


def test_check_network_support_refuses_an_explicit_session_on_hardened() -> None:
    msg = confine.check_network_support(_cfg("session"), "hardened")
    assert msg is not None and "'session'" in msg
    assert confine.check_network_support(_cfg("auto"), "hardened") is None


def test_only_explicit_states_fails_naming_sandbox_network() -> None:
    """`only_explicit_states` is refused: a state sets its own network in its config table."""
    with pytest.raises(ConfigError, match=r"sandbox\.network"):
        _cfg("only_explicit_states")


# --- supervisor subprocess: machine_agent.run_one -------------------------


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    return tmp_path


def test_run_one_returns_finish_payload(
    iso: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.harness import _snapshot

    class _FakeWf:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _prompt: str) -> _snapshot.SessionResult:
            return _snapshot.SessionResult(
                reason="finish_session",
                completed=True,
                summary="done",
                iterations=1,
                tool_calls=0,
                finish_payload={"label": "ok"},
            )

    def _fake(*_a: object, **_k: object) -> object:
        return object()

    monkeypatch.setattr(harness_loop, "Harness", _FakeWf)
    monkeypatch.setattr(providers, "build_role_provider", _fake)
    monkeypatch.setattr(providers, "reviewer_seat_provider", _fake)
    monkeypatch.setattr(dispatch, "ToolDispatcher", _fake)

    req = machine_agent.MachineAgentRequest(
        cwd=iso,
        root=iso,
        overlay={},
        isolation="none",  # no real sandbox: landlock is a no-op
        transcript_dir=tmp_path / "t",
        request=AgentRequest(model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic"),
    )
    out = machine_agent.run_one(req)
    assert out.reason == "finish_session"
    assert out.payload == {"label": "ok"}


def _stub_loop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the agent loop in machine_agent; return a dict capturing dispatcher kwargs."""
    from agent6.harness import _snapshot

    class _FakeWf:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _prompt: str) -> _snapshot.SessionResult:
            return _snapshot.SessionResult(
                reason="finish_session",
                completed=True,
                summary="d",
                iterations=1,
                tool_calls=0,
                finish_payload={},
            )

    captured: dict[str, Any] = {}

    def _disp(**kw: object) -> object:
        captured.update(kw)
        return object()

    def _prov(*_a: object, **_k: object) -> object:
        return object()

    monkeypatch.setattr(harness_loop, "Harness", _FakeWf)
    monkeypatch.setattr(providers, "build_role_provider", _prov)
    monkeypatch.setattr(providers, "reviewer_seat_provider", _prov)
    monkeypatch.setattr(dispatch, "ToolDispatcher", _disp)
    return captured


def test_run_one_drops_out_of_cwd_protect_paths(
    iso: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_loop(monkeypatch)
    inside = iso / "m.asm.toml"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path.parent / "evil.asm.toml"
    outside.write_text("x", encoding="utf-8")
    req = machine_agent.MachineAgentRequest(
        cwd=iso,
        root=iso,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "t",
        protect_paths=(inside, outside),
        request=AgentRequest(model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic"),
    )
    machine_agent.run_one(req)
    # Only the in-cwd path survives the subprocess-boundary re-validation.
    assert captured["extra_protect_paths"] == (inside.resolve(),)


def test_run_one_exports_commit_identity(
    iso: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_loop(monkeypatch)
    for key in (
        "GIT_AUTHOR_NAME",
        "GIT_COMMITTER_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(key, raising=False)
    req = machine_agent.MachineAgentRequest(
        cwd=iso,
        root=iso,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "t",
        commit_identity=git_ops.CommitIdentity(name="Machine Bot", email="bot@example.com"),
        request=AgentRequest(
            model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic", mode="run"
        ),
    )
    out = machine_agent.run_one(req)
    assert out.reason == "finish_session"
    assert os.environ["GIT_AUTHOR_NAME"] == "Machine Bot"
    assert os.environ["GIT_COMMITTER_NAME"] == "Machine Bot"
    assert os.environ["GIT_AUTHOR_EMAIL"] == "bot@example.com"
    assert os.environ["GIT_COMMITTER_EMAIL"] == "bot@example.com"
