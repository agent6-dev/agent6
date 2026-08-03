# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 config get/set/unset/add/remove` + allow_urls egress wiring."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir


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


def test_set_bool_is_typed_not_string(iso: Path) -> None:
    assert _run(["config", "set", "sandbox.protect_git", "false"]) == 0
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["protect_git"] is False  # parsed as bool, not the string "false"


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
    _run(["config", "set", "git.run_repo_hooks", "true"])
    assert _run(["config", "unset", "git.auto_stash"]) == 0
    text = global_config_path().read_text(encoding="utf-8")
    assert "[git]" in text  # still holds run_repo_hooks
    assert "run_repo_hooks" in text and "auto_stash" not in text


# --- top-level `preset` (the one section-less leaf) -------------------------


def test_set_top_level_profile_and_get(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _global_toml(iso)["preset"] == "ultra"
    capsys.readouterr()
    assert _run(["config", "get", "preset"]) == 0
    out = capsys.readouterr().out
    assert "preset = ultra" in out
    assert "[global]" in out


def test_set_unknown_profile_name_reverts(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _run(["config", "set", "preset", "porifle"]) == 2
    assert "unknown preset" in capsys.readouterr().err
    assert _global_toml(iso)["preset"] == "ultra"  # rolled back to the prior value


def test_unset_top_level_profile(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "preset", "quick"])
    assert _run(["config", "unset", "preset"]) == 0
    assert "preset" not in _global_toml(iso)
    capsys.readouterr()
    _run(["config", "get", "preset"])
    assert "[default]" in capsys.readouterr().out


def test_set_profile_heals_a_profile_table_typo(iso: Path) -> None:
    # A leftover `[preset]` TABLE (from `config set preset.<name>`) breaks the
    # config; the advertised fix `config set preset <name>` must heal it in one
    # step, not stack a bare key on top of the table (unparseable TOML, kept by
    # the lenient already-invalid path).
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text('[preset]\nporifle = "ultra"\n', encoding="utf-8")
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _global_toml(iso) == {"preset": "ultra"}


def test_set_profile_table_typo_reports_profile_error(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `config set preset.porifle x` over a valid config must fail with the
    # preset-must-be-a-string message and roll back, even when the bare
    # `preset` key it collides with is already set.
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _run(["config", "set", "preset.porifle", "x"]) == 2
    assert "must be a preset name string" in capsys.readouterr().err
    assert _global_toml(iso) == {"preset": "ultra"}


def test_set_profile_repo_targets_repo_config(iso: Path) -> None:
    assert _run(["config", "set", "preset", "quick", "--repo"]) == 0
    repo_cfg = (resolved_state_dir(iso) / "config.toml").read_text(encoding="utf-8")
    assert 'preset = "quick"' in repo_cfg
    assert not (iso / "g" / "config.toml").is_file()


def test_set_profile_machine_file_is_refused(
    iso: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A machine [config] overlay must not smuggle a preset selection
    # (_forbid_layer_preset); the write is rolled back.
    mf = tmp_path / "m.asm.toml"
    mf.write_text("[config]\n", encoding="utf-8")
    assert _run(["config", "set", "preset", "ultra", "--machine-file", str(mf)]) == 2
    assert "preset" in capsys.readouterr().err
    assert "preset" not in mf.read_text(encoding="utf-8").replace("[config]", "")


# --- presets listing ---------------------------------------------------------


def test_config_profiles_lists_builtins_and_user(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        'preset = "ultra"\n\n[presets.myteam.review]\nconcurrency = 2\n', encoding="utf-8"
    )
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    for builtin in ("standard", "quick", "ultra", "paranoid"):
        assert builtin in out
    assert "selected" in out  # ultra marked as the selection, with its source
    assert "global" in out
    assert "review.concurrency = 3" in out  # ultra's contents are shown
    assert "myteam" in out  # user preset listed with its contents
    assert "review.concurrency = 2" in out


def test_config_profiles_none_selected(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    assert "no preset selected" in out
    assert "standard" in out  # built-ins still listed


def test_config_profiles_user_shadow_replaces_builtin(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A user [presets.ultra] REPLACES the built-in wholesale; the listing must
    # show the user's contents (not the dead built-in's) and say so.
    (iso / "g").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "config.toml").write_text(
        "[presets.ultra.review]\nconcurrency = 9\n", encoding="utf-8"
    )
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    assert "review.concurrency = 9" in out
    assert "review.concurrency = 3" not in out  # the built-in body is dead, not shown
    assert "replaces the built-in" in out


# --- repo target ------------------------------------------------------------


# --- add / remove (list field: allow_urls) ----------------------------------


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


# --- egress endpoint wiring -------------------------------------------------


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


def test_unset_refuses_a_leaf_inside_an_undeclared_table(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sandbox.protect_git = false` written as a dotted top-level key: unset
    said "nothing to unset" rc=0 while `config get` showed the leaf set -- the
    one write surface that lied about this shape (set refuses it, fix reports
    it stuck)."""
    (iso / "g").mkdir(parents=True, exist_ok=True)
    cfg = iso / "g" / "config.toml"
    cfg.write_text("sandbox.protect_git = false\n", encoding="utf-8")
    rc = _run(["config", "unset", "sandbox.protect_git"])
    assert rc == 2
    assert "cannot be unset on its own" in capsys.readouterr().err
    # The file is untouched: nothing was silently dropped or rewritten.
    assert cfg.read_text(encoding="utf-8") == "sandbox.protect_git = false\n"


def test_get_honours_the_global_config_flag(iso: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--config FILE` reaches `config get`, not just `config show`.

    `get` answered from the default/global stack while `show` reported the
    flag layer, so the two config readers disagreed about the same leaf."""
    explicit = iso / "x.toml"
    explicit.write_text("[review]\nperiod = 77\n", encoding="utf-8")
    assert _run(["--config", str(explicit), "config", "get", "review.period"]) == 0
    out = capsys.readouterr().out
    assert "review.period = 77" in out
    assert "[flag]" in out


def test_get_refuses_a_missing_global_config_file(iso: Path) -> None:
    """A `--config` file that does not exist is refused, as `config show`
    refuses it: answering from the defaults reports a value the named file
    never set."""
    assert _run(["--config", str(iso / "nope.toml"), "config", "get", "review.period"]) == 2


def test_get_refuses_a_machine_file_that_does_not_exist(iso: Path) -> None:
    """A missing overlay path read as an EMPTY overlay, so a typo'd
    --machine-file answered confidently from the stack below it at exit 0."""
    assert (
        _run(["config", "get", "--machine-file", str(iso / "nope.asm.toml"), "review.period"]) == 2
    )


def test_a_provider_leaf_error_names_every_valid_value(
    iso: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`api_format` is a discriminator with two legal values, and only the first
    member's complaint was reported -- telling someone configuring an
    OpenAI-compatible provider that 'anthropic' was the only option."""
    assert _run(["config", "set", "providers.p.api_format", "nonsense"]) == 2
    err = capsys.readouterr().err
    assert "anthropic" in err
    assert "openai" in err


def test_an_unreadable_config_refuses_rather_than_crashing(iso: Path) -> None:
    """A root-owned config after a sudo run is the operator's file, not a defect.

    The reader wrapped a TOML parse error but not an OSError, so a permission
    problem escaped as "unexpected PermissionError" with a saved traceback,
    "please report this", and exit 1."""
    gdir = iso / "g"
    gdir.mkdir(parents=True, exist_ok=True)
    cfg = gdir / "config.toml"
    cfg.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    cfg.chmod(0o000)
    try:
        assert _run(["config", "show"]) == 2
    finally:
        cfg.chmod(0o600)
