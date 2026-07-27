# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 config get/set/unset/add/remove` + allow_urls egress wiring."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from agent6.app.egress import (
    _allow_url_endpoints,  # pyright: ignore[reportPrivateUsage]
    _provider_endpoints,  # pyright: ignore[reportPrivateUsage]
)
from agent6.config import SandboxConfig, validate_config
from agent6.config.layer import resolved_state_dir
from agent6.sandbox.broker import Endpoint


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolated global config home + cwd inside a fresh repo."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(args: list[str]) -> int:
    from agent6.ui.cli import main

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
    # A malformed --machine-file must produce a clean CONFIG ERROR (exit 2),
    # not an uncaught TOMLDecodeError traceback.
    bad = tmp_path / "broken.asm.toml"
    bad.write_text("this is = not valid [[[\n", encoding="utf-8")
    assert _run(["config", "get", "git.merge_strategy", "--machine-file", str(bad)]) == 2
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


def test_unset_last_leaf_drops_the_empty_table(iso: Path) -> None:
    # Unsetting a section's only key must not leave a dangling [sandbox]
    # header accreting in the file; a sibling key keeps the section.
    from agent6.paths import global_config_path

    _run(["config", "set", "git.auto_stash", "true"])
    _run(["config", "set", "sandbox.run_commands", "yes"])
    assert _run(["config", "unset", "sandbox.run_commands"]) == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[sandbox]" not in text
    assert "[git]" in text  # untouched sibling section survives
    _run(["config", "set", "git.auto_stash_pop", "true"])
    assert _run(["config", "unset", "git.auto_stash"]) == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[git]" in text  # still holds auto_stash_pop
    assert "auto_stash_pop" in text and "auto_stash =" not in text


def test_set_preserves_sibling_keys(iso: Path) -> None:
    _run(["config", "set", "sandbox.agent_network", "open"])
    _run(["config", "set", "sandbox.run_commands", "yes"])
    sandbox = _global_toml(iso)["sandbox"]
    assert sandbox == {"agent_network": "open", "run_commands": "yes"}  # type: ignore[comparison-overlap]


# --- top-level `profile` (the one section-less leaf) -------------------------


def test_set_top_level_profile_and_get(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "set", "profile", "ultra"]) == 0
    assert _global_toml(iso)["profile"] == "ultra"
    capsys.readouterr()
    assert _run(["config", "get", "profile"]) == 0
    out = capsys.readouterr().out
    assert "profile = ultra" in out
    assert "[global]" in out


def test_set_unknown_profile_name_reverts(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "set", "profile", "ultra"]) == 0
    assert _run(["config", "set", "profile", "porifle"]) == 2
    assert "unknown profile" in capsys.readouterr().err
    assert _global_toml(iso)["profile"] == "ultra"  # rolled back to the prior value


def test_unset_top_level_profile(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "profile", "quick"])
    assert _run(["config", "unset", "profile"]) == 0
    assert "profile" not in _global_toml(iso)
    capsys.readouterr()
    _run(["config", "get", "profile"])
    assert "[default]" in capsys.readouterr().out


def test_set_profile_heals_a_profile_table_typo(iso: Path) -> None:
    # A leftover `[profile]` TABLE (from `config set profile.<name>`) breaks the
    # config; the advertised fix `config set profile <name>` must heal it in one
    # step, not stack a bare key on top of the table (unparseable TOML, kept by
    # the lenient already-invalid path).
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text('[profile]\nporifle = "ultra"\n', encoding="utf-8")
    assert _run(["config", "set", "profile", "ultra"]) == 0
    assert _global_toml(iso) == {"profile": "ultra"}


def test_set_profile_table_typo_reports_profile_error(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `config set profile.porifle x` over a valid config must fail with the
    # profile-must-be-a-string message and roll back, even when the bare
    # `profile` key it collides with is already set.
    assert _run(["config", "set", "profile", "ultra"]) == 0
    assert _run(["config", "set", "profile.porifle", "x"]) == 2
    assert "must be a profile name string" in capsys.readouterr().err
    assert _global_toml(iso) == {"profile": "ultra"}


def test_set_profile_repo_targets_repo_config(iso: Path) -> None:
    assert _run(["config", "set", "profile", "quick", "--repo"]) == 0
    repo_cfg = (resolved_state_dir(iso) / "config.toml").read_text(encoding="utf-8")
    assert 'profile = "quick"' in repo_cfg
    assert not (iso / "g" / "config.toml").is_file()


def test_set_profile_machine_file_is_refused(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A machine [config] overlay must not smuggle a profile selection
    # (_forbid_layer_profile); the write is rolled back.
    mf = tmp_path / "m.asm.toml"
    mf.write_text("[config]\n", encoding="utf-8")
    assert _run(["config", "set", "profile", "ultra", "--machine-file", str(mf)]) == 2
    assert "profile" in capsys.readouterr().err
    assert "profile" not in mf.read_text(encoding="utf-8").replace("[config]", "")


# --- profiles listing ---------------------------------------------------------


def test_config_profiles_lists_builtins_and_user(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        'profile = "ultra"\n\n[profiles.myteam.review]\npanel_size = 2\n', encoding="utf-8"
    )
    assert _run(["config", "profiles"]) == 0
    out = capsys.readouterr().out
    for builtin in ("standard", "quick", "ultra", "paranoid"):
        assert builtin in out
    assert "selected" in out  # ultra marked as the selection, with its source
    assert "global" in out
    assert "review.panel_size = 3" in out  # ultra's contents are shown
    assert "myteam" in out  # user profile listed with its contents
    assert "review.panel_size = 2" in out


def test_config_profiles_none_selected(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "profiles"]) == 0
    out = capsys.readouterr().out
    assert "no profile selected" in out
    assert "standard" in out  # built-ins still listed


def test_config_profiles_user_shadow_replaces_builtin(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A user [profiles.ultra] REPLACES the built-in wholesale; the listing must
    # show the user's contents (not the dead built-in's) and say so.
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        "[profiles.ultra.review]\npanel_size = 9\n", encoding="utf-8"
    )
    assert _run(["config", "profiles"]) == 0
    out = capsys.readouterr().out
    assert "review.panel_size = 9" in out
    assert "review.panel_size = 3" not in out  # the built-in body is dead, not shown
    assert "replaces the built-in" in out


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
    # A non-security knob is fine in a machine overlay (review tuning).
    assert (
        _run(["config", "set", "review.trigger", "on_verify_fail", "--machine-file", str(mf)]) == 0
    )
    data = tomllib.loads(mf.read_text(encoding="utf-8"))
    assert data["config"] == {"review": {"trigger": "on_verify_fail"}}  # type: ignore[comparison-overlap]
    # The original machine tables survive the edit.
    assert data["machine"]["name"] == "demo"  # type: ignore[index]
    capsys.readouterr()
    assert _run(["config", "get", "review.trigger", "--machine-file", str(mf)]) == 0
    assert "[machine]" in capsys.readouterr().out


def test_machine_overlay_rejects_providers(iso: Path) -> None:
    mf = _machine_file(iso)
    assert _run(["config", "set", "providers.x.kind", "anthropic", "--machine-file", str(mf)]) == 2


def test_machine_overlay_rejects_sandbox(iso: Path) -> None:
    # Sandbox policy is an operator-only decision — a machine file (possibly
    # LLM-drafted/shared) must not weaken the jail via its [config] overlay.
    mf = _machine_file(iso)
    assert _run(["config", "set", "sandbox.agent_network", "open", "--machine-file", str(mf)]) == 2
    assert _run(["config", "add", "sandbox.allow_urls", "evil.com", "--machine-file", str(mf)]) == 2
    # ...but the same keys are settable in the global config.
    assert _run(["config", "set", "sandbox.agent_network", "open"]) == 0


def test_repo_and_machine_together_rejected(iso: Path) -> None:
    mf = _machine_file(iso)
    rc = _run(
        ["config", "set", "sandbox.agent_network", "open", "--repo", "--machine-file", str(mf)]
    )
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


def test_concurrent_list_adds_lose_no_element(iso: Path) -> None:
    """`config add` reads the current list well before it writes the extended
    one; two concurrent adds both read the same base and the later publish
    dropped the earlier element. The whole read-extend-revalidate cycle now
    holds the target's lock."""
    import threading

    n = 2
    per = 5
    barrier = threading.Barrier(n)

    def adder(i: int) -> None:
        barrier.wait()
        for j in range(per):
            assert _run(["config", "add", "sandbox.allow_urls", f"http://h{i}-{j}"]) == 0

    threads = [threading.Thread(target=adder, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    data = _global_toml(iso)
    sandbox = data.get("sandbox")
    assert isinstance(sandbox, dict)
    urls = set(sandbox.get("allow_urls", []))
    assert urls == {f"http://h{i}-{j}" for i in range(n) for j in range(per)}


def test_config_fill_serializes_against_a_concurrent_set(iso: Path) -> None:
    """`config fill` read the effective config, then published it with an
    unlocked, non-atomic write_text; a `config set` landing between the read
    and the write was overwritten by the stale snapshot (lost update). fill now
    holds the target's lock across load+publish, so a concurrent set blocks and
    lands after -- its value survives."""
    import threading
    import time
    from unittest import mock

    from agent6.ui.cli import config_cmds

    assert _run(["config", "set", "sandbox.memory_limit_mb", "512"]) == 0

    fill_holds_lock = threading.Event()
    release_fill = threading.Event()
    real_load = config_cmds.load_config_or_exit
    results: dict[str, object] = {}

    def gated_load(root: Path, cfg: Path | None) -> object:
        fill_holds_lock.set()  # reached inside `with locked_file(target)`
        release_fill.wait(timeout=5)
        return real_load(root, cfg)

    def run_fill() -> None:
        with mock.patch.object(config_cmds, "load_config_or_exit", gated_load):
            results["fill"] = _run(["config", "fill", "--force"])

    def run_set() -> None:
        fill_holds_lock.wait(timeout=5)
        results["set"] = _run(["config", "set", "sandbox.memory_limit_mb", "1234"])
        results["set_done"] = True

    tf = threading.Thread(target=run_fill, daemon=True)
    ts = threading.Thread(target=run_set, daemon=True)
    tf.start()
    ts.start()
    assert fill_holds_lock.wait(timeout=5)
    time.sleep(0.3)  # let the set reach (and block on) the target lock
    assert results.get("set_done") is None  # the set is queued behind fill
    release_fill.set()
    tf.join(timeout=10)
    ts.join(timeout=10)
    assert results["fill"] == 0 and results["set"] == 0
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["memory_limit_mb"] == 1234  # the set survived
