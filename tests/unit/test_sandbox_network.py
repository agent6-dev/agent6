# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent_network / tool_network: profile compatibility, machine refusals, and
the supervisor subprocess that runs a machine `agent` state self-confined."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.cli import machine_agent
from agent6.cli.egress import (
    _check_network_profile,  # pyright: ignore[reportPrivateUsage]
    _is_loopback,  # pyright: ignore[reportPrivateUsage]
    _maybe_start_egress,  # pyright: ignore[reportPrivateUsage]
)
from agent6.cli.machine_cmds import _machine_network_refusal  # pyright: ignore[reportPrivateUsage]
from agent6.config import Config, validate_config
from agent6.machine.model import ToolState
from agent6.types import SandboxProfile


def _cfg(agent_network: str = "providers", tool_network: str = "blocked") -> Config:
    return validate_config(
        {"sandbox": {"agent_network": agent_network, "tool_network": tool_network}}
    )


# --- _is_loopback ----------------------------------------------------------


def test_is_loopback() -> None:
    assert _is_loopback("127.0.0.1") and _is_loopback("localhost") and _is_loopback("::1")
    assert not _is_loopback("api.anthropic.com")


# --- _check_network_profile (profile compatibility) ------------------------


@pytest.mark.parametrize("profile", ["strict", "none"])
def test_check_network_profile_allows_off_hardened(profile: SandboxProfile) -> None:
    # local/carveouts only refused on hardened; strict supports them, none is
    # unsandboxed (warned elsewhere), so neither refuses here.
    assert _check_network_profile(_cfg("local", "blocked"), profile) is None
    assert _check_network_profile(_cfg("open", "carveouts"), profile) is None


def test_check_network_profile_refuses_local_on_hardened() -> None:
    assert "local" in (_check_network_profile(_cfg("local", "blocked"), "hardened") or "")


def test_check_network_profile_refuses_carveouts_on_hardened() -> None:
    assert "carveouts" in (_check_network_profile(_cfg("open", "carveouts"), "hardened") or "")


# --- _machine_network_refusal ----------------------------------------------

_TOOL = ToolState(kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"})
_NET_TOOL = ToolState(
    kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"}, allow_network=True
)


def test_refusal_networked_tool_under_blocked() -> None:
    msg = _machine_network_refusal(_cfg("providers", "blocked"), "strict", [_NET_TOOL], True)
    assert msg is not None and "allow_network" in msg


def test_refusal_providers_carveouts_strict_ok() -> None:
    # The headline combo: confined agent + audited networked tool, on strict.
    assert (
        _machine_network_refusal(_cfg("providers", "carveouts"), "strict", [_NET_TOOL], True)
        is None
    )


def test_refusal_blocked_tools_on_hardened() -> None:
    msg = _machine_network_refusal(_cfg("providers", "blocked"), "hardened", [_TOOL], False)
    assert msg is not None and "strict" in msg


def test_refusal_allowed_tools_on_hardened_ok() -> None:
    assert _machine_network_refusal(_cfg("open", "allowed"), "hardened", [_TOOL], False) is None


# --- _maybe_start_egress (local / open / non-strict short-circuits) --------


def test_egress_open_does_nothing() -> None:
    assert _maybe_start_egress(_cfg("open", "blocked"), "strict") == (None, None, None)


def test_egress_non_strict_defers_to_landlock() -> None:
    assert _maybe_start_egress(_cfg("providers", "blocked"), "hardened") == (None, None, None)


def test_egress_local_refuses_non_local_provider() -> None:
    cfg = validate_config(
        {
            "providers": {
                "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1"}
            },
            "sandbox": {"agent_network": "local"},
        }
    )
    broker, sock_dir, err = _maybe_start_egress(cfg, "strict")
    assert broker is None and sock_dir is None
    assert err is not None and "loopback" in err and "openrouter.ai" in err


# --- supervisor subprocess: machine_agent._run_one -------------------------


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    gdir = tmp_path / "g"
    gdir.mkdir()
    (gdir / "config.toml").write_text(
        '[providers.anthropic]\nkind = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    return tmp_path


def test_run_one_returns_finish_payload(
    iso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.workflows.loop import RunResult

    class _FakeWf:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _prompt: str) -> RunResult:
            return RunResult(
                reason="finish_run",
                completed=True,
                summary="done",
                iterations=1,
                tool_calls=0,
                finish_payload={"label": "ok"},
            )

    def _fake(*_a: object, **_k: object) -> object:
        return object()

    monkeypatch.setattr(machine_agent, "Workflow", _FakeWf)
    monkeypatch.setattr(machine_agent, "_build_role_provider", _fake)
    monkeypatch.setattr(machine_agent, "ToolDispatcher", _fake)

    req = {
        "cwd": str(iso),
        "root": str(iso),
        "overlay": {},
        "profile": "none",  # no real sandbox: egress/landlock are no-ops
        "transcript_dir": str(tmp_path / "t"),
        "request": {
            "model": "claude-x",
            "prompt": "go",
            "timeout_s": 5.0,
            "provider": "anthropic",
            "thinking": None,
            "temperature": None,
            "max_usd": None,
            "max_input_tokens": None,
            "max_output_tokens": None,
        },
    }
    out = machine_agent._run_one(req)  # pyright: ignore[reportPrivateUsage]
    assert out["reason"] == "finish_run"
    assert out["payload"] == {"label": "ok"}
