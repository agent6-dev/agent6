# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 config get/set/unset/add/remove` + allow_urls egress wiring."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from agent6.cli.egress import (
    _allow_url_endpoints,  # pyright: ignore[reportPrivateUsage]
    _provider_endpoints,  # pyright: ignore[reportPrivateUsage]
)
from agent6.config import SandboxConfig, validate_config
from agent6.config_layer import resolved_state_dir
from agent6.sandbox.broker import Endpoint


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolated global config home + cwd inside a fresh repo."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(args: list[str]) -> int:
    from agent6.cli import main

    return main(args)


def _global_toml(tmp_path: Path) -> dict[str, object]:
    return tomllib.loads((tmp_path / "g" / "config.toml").read_text(encoding="utf-8"))


# --- set / get / unset (scalars) -------------------------------------------


def test_set_scalar_writes_global(iso: Path) -> None:
    assert _run(["config", "set", "sandbox.agent_network", "open"]) == 0
    assert _global_toml(iso)["sandbox"] == {"agent_network": "open"}  # type: ignore[comparison-overlap]


def test_set_bool_is_typed_not_string(iso: Path) -> None:
    assert _run(["config", "set", "sandbox.protect_git", "false"]) == 0
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["protect_git"] is False  # parsed as bool, not the string "false"


def test_set_rejects_invalid_enum_and_reverts(iso: Path) -> None:
    assert _run(["config", "set", "sandbox.agent_network", "providers"]) == 0
    assert _run(["config", "set", "sandbox.agent_network", "bogus"]) == 2
    # The bad write was reverted: the prior valid value survives.
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["agent_network"] == "providers"


def test_get_reports_value_and_source(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "sandbox.agent_network", "open"])
    capsys.readouterr()
    assert _run(["config", "get", "sandbox.agent_network"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.agent_network = open" in out
    assert "[global]" in out


def test_get_default_source_for_unset(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "get", "sandbox.protect_git"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.protect_git = true" in out
    assert "[default]" in out


def test_get_unknown_key_errors(iso: Path) -> None:
    assert _run(["config", "get", "sandbox.nope"]) == 2


def test_machine_get_on_malformed_toml_is_clean_error(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A malformed --machine file must produce a clean CONFIG ERROR (exit 2),
    # not an uncaught TOMLDecodeError traceback.
    bad = tmp_path / "broken.asm.toml"
    bad.write_text("this is = not valid [[[\n", encoding="utf-8")
    assert _run(["config", "get", "git.commit_strategy", "--machine", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "invalid TOML" in err


def test_unset_reverts_to_default(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "sandbox.protect_git", "false"])
    assert _run(["config", "unset", "sandbox.protect_git"]) == 0
    capsys.readouterr()
    _run(["config", "get", "sandbox.protect_git"])
    assert "[default]" in capsys.readouterr().out


def test_unset_missing_key_is_noop(iso: Path) -> None:
    _run(["config", "set", "sandbox.agent_network", "open"])  # create the file
    assert _run(["config", "unset", "sandbox.protect_git"]) == 0


def test_set_preserves_sibling_keys(iso: Path) -> None:
    _run(["config", "set", "sandbox.agent_network", "open"])
    _run(["config", "set", "sandbox.run_commands", "yes"])
    sandbox = _global_toml(iso)["sandbox"]
    assert sandbox == {"agent_network": "open", "run_commands": "yes"}  # type: ignore[comparison-overlap]


# --- repo target ------------------------------------------------------------


def test_set_repo_writes_repo_config(iso: Path) -> None:
    assert _run(["config", "set", "sandbox.agent_network", "open", "--repo"]) == 0
    repo_cfg = (resolved_state_dir(iso) / "config.toml").read_text(encoding="utf-8")
    assert "[sandbox]" in repo_cfg
    assert 'agent_network = "open"' in repo_cfg
    assert not (iso / "g" / "config.toml").is_file()


# --- add / remove (list field: allow_urls) ----------------------------------


def test_add_remove_allow_urls_roundtrip(iso: Path) -> None:
    assert _run(["config", "add", "sandbox.allow_urls", "a.com:8443"]) == 0
    assert _run(["config", "add", "sandbox.allow_urls", "https://b.com/v1"]) == 0
    assert _run(["config", "add", "sandbox.allow_urls", "a.com:8443"]) == 0  # dup: no-op
    cfg = validate_config(_global_toml(iso))
    assert cfg.sandbox.allow_urls == ("a.com:8443", "https://b.com/v1")

    assert _run(["config", "remove", "sandbox.allow_urls", "a.com:8443"]) == 0
    cfg = validate_config(_global_toml(iso))
    assert cfg.sandbox.allow_urls == ("https://b.com/v1",)


def test_add_invalid_allow_url_reverts(iso: Path) -> None:
    _run(["config", "add", "sandbox.allow_urls", "good.com"])
    assert _run(["config", "add", "sandbox.allow_urls", ""]) == 2  # empty rejected
    cfg = validate_config(_global_toml(iso))
    assert cfg.sandbox.allow_urls == ("good.com",)


def test_remove_absent_value_is_noop(iso: Path) -> None:
    _run(["config", "add", "sandbox.allow_urls", "a.com"])
    assert _run(["config", "remove", "sandbox.allow_urls", "z.com"]) == 0


# --- machine [config] overlay target ----------------------------------------


def _machine_file(tmp_path: Path) -> Path:
    p = tmp_path / "demo.asm.toml"
    p.write_text(
        '[machine]\nname = "demo"\nentry = "s"\n\n[states.s]\nkind = "terminal"\noutcome = "ok"\n',
        encoding="utf-8",
    )
    return p


def test_machine_overlay_set_and_get(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mf = _machine_file(iso)
    # A non-security knob is fine in a machine overlay (workflow tuning).
    assert _run(["config", "set", "workflow.critic", "on_verify_fail", "--machine", str(mf)]) == 0
    data = tomllib.loads(mf.read_text(encoding="utf-8"))
    assert data["config"] == {"workflow": {"critic": "on_verify_fail"}}  # type: ignore[comparison-overlap]
    # The original machine tables survive the edit.
    assert data["machine"]["name"] == "demo"  # type: ignore[index]
    capsys.readouterr()
    assert _run(["config", "get", "workflow.critic", "--machine", str(mf)]) == 0
    assert "[machine]" in capsys.readouterr().out


def test_machine_overlay_rejects_providers(iso: Path) -> None:
    mf = _machine_file(iso)
    assert _run(["config", "set", "providers.x.kind", "anthropic", "--machine", str(mf)]) == 2


def test_machine_overlay_rejects_sandbox(iso: Path) -> None:
    # Sandbox policy is an operator-only decision — a machine file (possibly
    # LLM-drafted/shared) must not weaken the jail via its [config] overlay.
    mf = _machine_file(iso)
    assert _run(["config", "set", "sandbox.agent_network", "open", "--machine", str(mf)]) == 2
    assert _run(["config", "add", "sandbox.allow_urls", "evil.com", "--machine", str(mf)]) == 2
    # ...but the same keys are settable in the global config.
    assert _run(["config", "set", "sandbox.agent_network", "open"]) == 0


def test_repo_and_machine_together_rejected(iso: Path) -> None:
    mf = _machine_file(iso)
    rc = _run(["config", "set", "sandbox.agent_network", "open", "--repo", "--machine", str(mf)])
    assert rc == 2


# --- egress endpoint wiring -------------------------------------------------


def test_allow_url_endpoints_parsed() -> None:
    cfg = validate_config(
        {"sandbox": {"allow_urls": ["a.com", "b.com:8443", "https://c.com/v1", "http://d:1234"]}}
    )
    eps = _allow_url_endpoints(cfg)
    assert eps == {
        Endpoint("a.com", 443),  # bare host -> https default
        Endpoint("b.com", 8443),
        Endpoint("c.com", 443),
        Endpoint("d", 1234),
    }


def test_allow_url_endpoints_empty_by_default() -> None:
    assert _allow_url_endpoints(validate_config({})) == set()


def test_effective_egress_unions_providers_and_allow_urls() -> None:
    cfg = validate_config(
        {
            "providers": {"anthropic": {"api_format": "anthropic"}},
            "sandbox": {"allow_urls": ["extra.com:9000"]},
        }
    )
    union = _provider_endpoints(cfg) | _allow_url_endpoints(cfg)
    assert Endpoint("extra.com", 9000) in union
    # The Anthropic provider endpoint is still present (union, not replace).
    assert any(ep.port == 443 for ep in union)


def test_sandboxconfig_allow_urls_default() -> None:
    assert SandboxConfig().allow_urls == ()
