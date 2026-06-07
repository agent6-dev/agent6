# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent_network / tool_network: profile compatibility, machine refusals, and
the supervisor subprocess that runs a machine `agent` state self-confined."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _cfg(agent_network: str = "providers", tool_network: str = "block") -> Config:
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
    # local/only_explicit_states only refused on hardened; strict supports them,
    # none is unsandboxed (warned elsewhere), so neither refuses here.
    assert _check_network_profile(_cfg("local", "block"), profile) is None
    assert _check_network_profile(_cfg("open", "only_explicit_states"), profile) is None


def test_check_network_profile_refuses_local_on_hardened() -> None:
    assert "local" in (_check_network_profile(_cfg("local", "block"), "hardened") or "")


def test_check_network_profile_refuses_only_explicit_states_on_hardened() -> None:
    msg = _check_network_profile(_cfg("open", "only_explicit_states"), "hardened")
    assert msg is not None and "only_explicit_states" in msg


# --- _machine_network_refusal ----------------------------------------------

_TOOL = ToolState(kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"})
_NET_TOOL = ToolState(
    kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"}, allow_network="allow"
)
_BLOCK_TOOL = ToolState(
    kind="tool", command=("x",), timeout_secs=5, on={"ok": "s"}, allow_network="block"
)


def test_refusal_networked_tool_under_block() -> None:
    msg = _machine_network_refusal(_cfg("providers", "block"), "strict", [_NET_TOOL])
    assert msg is not None and "allow_network" in msg


def test_refusal_providers_explicit_states_strict_ok() -> None:
    # The headline combo: confined agent + audited networked tool, on strict.
    assert (
        _machine_network_refusal(_cfg("providers", "only_explicit_states"), "strict", [_NET_TOOL])
        is None
    )


def test_refusal_block_tools_on_hardened() -> None:
    msg = _machine_network_refusal(_cfg("providers", "block"), "hardened", [_TOOL])
    assert msg is not None and "strict" in msg


def test_refusal_explicit_block_state_on_hardened() -> None:
    # tool_network=allow runs auto/allow tools on hardened, but an explicit
    # allow_network="block" demand can't be honored there -> refuse.
    msg = _machine_network_refusal(_cfg("open", "allow"), "hardened", [_BLOCK_TOOL])
    assert msg is not None and "block" in msg


def test_refusal_allow_auto_tools_on_hardened_ok() -> None:
    assert _machine_network_refusal(_cfg("open", "allow"), "hardened", [_TOOL]) is None


# --- _maybe_start_egress (local / open / non-strict short-circuits) --------


def test_egress_open_does_nothing() -> None:
    assert _maybe_start_egress(_cfg("open", "block"), "strict") == (None, None, None)


def test_egress_non_strict_defers_to_landlock() -> None:
    assert _maybe_start_egress(_cfg("providers", "block"), "hardened") == (None, None, None)


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


def _stub_loop(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the agent loop in machine_agent; return a dict capturing dispatcher kwargs."""
    from agent6.workflows.loop import RunResult

    class _FakeWf:
        def __init__(self, **_kw: object) -> None:
            pass

        def run(self, _prompt: str) -> RunResult:
            return RunResult(
                reason="finish_run",
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

    monkeypatch.setattr(machine_agent, "Workflow", _FakeWf)
    monkeypatch.setattr(machine_agent, "_build_role_provider", _prov)
    monkeypatch.setattr(machine_agent, "ToolDispatcher", _disp)
    return captured


def test_run_one_drops_out_of_cwd_protect_paths(
    iso: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _stub_loop(monkeypatch)
    inside = iso / "m.asm.toml"
    inside.write_text("x", encoding="utf-8")
    outside = tmp_path.parent / "evil.asm.toml"
    outside.write_text("x", encoding="utf-8")
    req = {
        "cwd": str(iso),
        "root": str(iso),
        "overlay": {},
        "profile": "none",
        "transcript_dir": str(tmp_path / "t"),
        "protect_paths": [str(inside), str(outside)],
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
    machine_agent._run_one(req)  # pyright: ignore[reportPrivateUsage]
    # Only the in-cwd path survives the subprocess-boundary re-validation.
    assert captured["extra_protect_paths"] == (inside.resolve(),)


def test_egress_fails_closed_and_cleans_up_on_socket_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent6.cli.egress as eg

    sock = tmp_path / "egress-sock"

    def _fake_mkdtemp(**_k: object) -> str:
        sock.mkdir()
        return str(sock)

    def _boom(*_a: object, **_k: object) -> object:
        raise OSError("too many open files")

    monkeypatch.setattr(eg.tempfile, "mkdtemp", _fake_mkdtemp)
    monkeypatch.setattr(eg, "start_egress_broker", _boom)
    cfg = validate_config(
        {
            "providers": {"anthropic": {"kind": "anthropic"}},
            "sandbox": {"agent_network": "providers"},
        }
    )
    broker, sock_dir, err = _maybe_start_egress(cfg, "strict")
    assert broker is None and sock_dir is None
    assert err is not None and "too many open files" in err
    assert not sock.exists()  # the socket dir was cleaned up, not leaked
