# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 config get/set/unset/add/remove` + allow_urls egress wiring."""

from __future__ import annotations

import pathlib
import tomllib

import pytest

from agent6 import paths
from agent6.config import io


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """Isolated global config home + cwd inside a fresh repo."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(args: list[str]) -> int:
    from agent6.ui.cli import main

    return main(args)


def _refuse(args: list[str]) -> int:
    """Run through the guarded entry point, where operator errors present as `ERROR:` and exit 2."""
    from agent6.ui.cli import cli_main

    return cli_main(args)


def _global_toml(tmp_path: pathlib.Path) -> dict[str, object]:
    return tomllib.loads((tmp_path / "g" / "agent6" / "config.toml").read_text(encoding="utf-8"))


# --- set / get / unset (scalars) -------------------------------------------


def test_set_bool_is_typed_not_string(iso: pathlib.Path) -> None:
    assert _run(["config", "set", "sandbox.protect_git", "false"]) == 0
    sandbox = _global_toml(iso)["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["protect_git"] is False  # parsed as bool, not the string "false"


def test_get_default_source_for_unset(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["config", "get", "sandbox.protect_git"]) == 0
    out = capsys.readouterr().out
    assert "sandbox.protect_git = true" in out
    assert "[default]" in out


def test_get_unknown_key_errors(iso: pathlib.Path) -> None:
    assert _run(["config", "get", "sandbox.nope"]) == 2


def test_machine_get_on_malformed_toml_is_clean_error(
    iso: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A malformed --machine-file is a clean ERROR at exit 2, not a TOMLDecodeError traceback.
    bad = tmp_path / "broken.asm.toml"
    bad.write_text("this is = not valid [[[\n", encoding="utf-8")
    assert _refuse(["config", "get", "git.merge_strategy", "--machine-file", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "invalid TOML" in err
    assert "report this" not in err


def test_unset_refuses_an_unknown_key(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown key is an error, not a successful no-op."""
    assert _run(["config", "unset", "nope.nope"]) == 2
    assert "unknown config key 'nope.nope'" in capsys.readouterr().err


def test_unset_reverts_to_default(iso: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "sandbox.protect_git", "false"])
    assert _run(["config", "unset", "sandbox.protect_git"]) == 0
    capsys.readouterr()
    _run(["config", "get", "sandbox.protect_git"])
    assert "[default]" in capsys.readouterr().out


def test_unset_last_leaf_drops_the_empty_table(iso: pathlib.Path) -> None:
    # Unsetting a section's only key leaves no dangling [sandbox] header; a sibling keeps it.

    _run(["config", "set", "git.dirty_tree", "stash"])
    _run(["config", "set", "sandbox.run_commands", "yes"])
    assert _run(["config", "unset", "sandbox.run_commands"]) == 0
    text = paths.global_config_path().read_text(encoding="utf-8")
    assert "[sandbox]" not in text
    assert "[git]" in text  # untouched sibling section survives
    _run(["config", "set", "git.run_repo_hooks", "true"])
    assert _run(["config", "unset", "git.dirty_tree"]) == 0
    text = paths.global_config_path().read_text(encoding="utf-8")
    assert "[git]" in text  # still holds run_repo_hooks
    assert "run_repo_hooks" in text and "dirty_tree" not in text


# --- top-level `preset` (the one section-less leaf) -------------------------


def test_set_top_level_profile_and_get(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _global_toml(iso)["preset"] == "ultra"
    capsys.readouterr()
    assert _run(["config", "get", "preset"]) == 0
    out = capsys.readouterr().out
    assert "preset = ultra" in out
    assert "[global]" in out


def test_set_unknown_profile_name_reverts(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _run(["config", "set", "preset", "porifle"]) == 2
    assert "unknown preset" in capsys.readouterr().err
    assert _global_toml(iso)["preset"] == "ultra"  # rolled back to the prior value


def test_unset_top_level_profile(iso: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(["config", "set", "preset", "quick"])
    assert _run(["config", "unset", "preset"]) == 0
    assert "preset" not in _global_toml(iso)
    capsys.readouterr()
    _run(["config", "get", "preset"])
    assert "[default]" in capsys.readouterr().out


def test_set_profile_heals_a_profile_table_typo(iso: pathlib.Path) -> None:
    # A leftover `[preset]` table breaks the config; `config set preset <name>` heals it.
    (iso / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "agent6" / "config.toml").write_text(
        '[preset]\nporifle = "ultra"\n', encoding="utf-8"
    )
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _global_toml(iso) == {"preset": "ultra"}


def test_set_profile_table_typo_reports_profile_error(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `config set preset.porifle x` fails with the preset-must-be-a-string message and rolls back.
    assert _run(["config", "set", "preset", "ultra"]) == 0
    assert _run(["config", "set", "preset.porifle", "x"]) == 2
    assert "must be a preset name string" in capsys.readouterr().err
    assert _global_toml(iso) == {"preset": "ultra"}


def test_set_profile_repo_targets_repo_config(iso: pathlib.Path) -> None:
    assert _run(["config", "set", "preset", "quick", "--repo"]) == 0
    repo_cfg = (paths.state_dir(iso) / "config.toml").read_text(encoding="utf-8")
    assert 'preset = "quick"' in repo_cfg
    assert not (iso / "g" / "agent6" / "config.toml").is_file()


def test_set_profile_machine_file_is_refused(
    iso: pathlib.Path, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A machine [config] overlay cannot smuggle a preset selection; the write is rolled back.
    mf = tmp_path / "m.asm.toml"
    mf.write_text("[config]\n", encoding="utf-8")
    assert _run(["config", "set", "preset", "ultra", "--machine-file", str(mf)]) == 2
    assert "preset" in capsys.readouterr().err
    assert "preset" not in mf.read_text(encoding="utf-8").replace("[config]", "")


# --- presets listing ---------------------------------------------------------


def test_config_profiles_lists_builtins_and_user(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (iso / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "agent6" / "config.toml").write_text(
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


def test_config_profiles_none_selected(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    assert "no preset selected" in out
    assert "standard" in out  # built-ins still listed


def test_config_profiles_user_shadow_replaces_builtin(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A user [presets.ultra] replaces the built-in wholesale; the listing shows the user's contents.
    (iso / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "agent6" / "config.toml").write_text(
        "[presets.ultra.review]\nconcurrency = 9\n", encoding="utf-8"
    )
    assert _run(["config", "presets"]) == 0
    out = capsys.readouterr().out
    assert "review.concurrency = 9" in out
    assert "review.concurrency = 3" not in out  # the built-in body is dead, not shown
    assert "replaces the built-in" in out


# --- repo target ------------------------------------------------------------


# --- add / remove (list fields) ---------------------------------------------


def test_repo_list_edits_keep_values_inherited_from_global(iso: pathlib.Path) -> None:
    assert _run(["config", "set", "sandbox.fetch_hosts", '["one.example", "two.example"]']) == 0

    assert _run(["config", "add", "--repo", "sandbox.fetch_hosts", "three.example"]) == 0
    repo = tomllib.loads((paths.state_dir(iso) / "config.toml").read_text(encoding="utf-8"))
    assert repo["sandbox"]["fetch_hosts"] == [  # type: ignore[index]
        "one.example",
        "two.example",
        "three.example",
    ]

    assert _run(["config", "remove", "--repo", "sandbox.fetch_hosts", "one.example"]) == 0
    repo = tomllib.loads((paths.state_dir(iso) / "config.toml").read_text(encoding="utf-8"))
    assert repo["sandbox"]["fetch_hosts"] == [  # type: ignore[index]
        "two.example",
        "three.example",
    ]


# --- machine [config] overlay target ----------------------------------------


def _machine_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "demo.asm.toml"
    p.write_text(
        '[machine]\nname = "demo"\nentry = "s"\n\n[states.s]\nkind = "terminal"\noutcome = "ok"\n',
        encoding="utf-8",
    )
    return p


def test_machine_overlay_set_and_get(iso: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_config_show_reads_through_a_machine_overlay(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config show` takes --machine-file like every other config verb."""
    mf = _machine_file(iso)
    assert (
        _run(["config", "set", "review.trigger", "on_verify_fail", "--machine-file", str(mf)]) == 0
    )
    capsys.readouterr()
    assert _run(["config", "show", "--machine-file", str(mf)]) == 0
    row = next(line for line in capsys.readouterr().out.splitlines() if "review.trigger" in line)
    assert "on_verify_fail" in row and "machine" in row, row
    assert _run(["config", "show", "review.trigger", "--machine-file", str(mf)]) == 0
    assert "on_verify_fail" in capsys.readouterr().out
    assert _refuse(["config", "show", "--machine-file", str(iso / "missing.asm.toml")]) == 2


def test_machine_overlay_rejects_providers(iso: pathlib.Path) -> None:
    mf = _machine_file(iso)
    assert _run(["config", "set", "providers.x.kind", "anthropic", "--machine-file", str(mf)]) == 2


@pytest.mark.parametrize(
    "command",
    [
        ["config", "show"],
        ["config", "get", "sandbox.network"],
        ["config", "fix"],
    ],
    ids=["show", "get", "fix"],
)
def test_machine_config_readers_refuse_a_hand_edited_protected_overlay(
    iso: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    command: list[str],
) -> None:
    machine = iso / "protected.asm.toml"
    machine.write_text('[config.sandbox]\nnetwork = "host"\n', encoding="utf-8")

    assert _refuse([*command, "--machine-file", str(machine)]) == 2
    err = capsys.readouterr().err
    assert "machine [config] overlays must not set sandbox.*" in err
    assert "operator-only" in err


# --- egress endpoint wiring -------------------------------------------------


def test_config_fill_has_no_repo_flag(iso: pathlib.Path) -> None:
    """`config fill` has no --repo flag.

    Filling the repo layer would shadow the global config permanently, defeating the layering
    `config show` explains; the global fill resolves defaults plus global only.
    """
    with pytest.raises(SystemExit) as exc:
        _run(["config", "fill", "--repo"])
    assert exc.value.code == 2


def test_config_fill_serializes_against_a_concurrent_set(iso: pathlib.Path) -> None:
    """`config fill` holds the target's lock across load and publish.

    A concurrent `config set` blocks and lands after, so its value survives.
    """
    import threading
    import time
    from unittest import mock

    from agent6.ui.cli import config_cmds

    assert _run(["config", "set", "sandbox.memory_limit_mb", "512"]) == 0

    fill_holds_lock = threading.Event()
    release_fill = threading.Event()
    real_load = config_cmds.load_global_only
    results: dict[str, object] = {}

    def gated_load() -> object:
        fill_holds_lock.set()  # reached inside `with locked_file(target)`
        release_fill.wait(timeout=5)
        return real_load()

    def run_fill() -> None:
        with mock.patch.object(config_cmds, "load_global_only", gated_load):
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
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unset refuses a leaf inside an undeclared table, as set refuses it and fix reports it.

    The shape: `sandbox.protect_git = false` written as a dotted top-level key.
    """
    (iso / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    cfg = iso / "g" / "agent6" / "config.toml"
    cfg.write_text("sandbox.protect_git = false\n", encoding="utf-8")
    rc = _refuse(["config", "unset", "sandbox.protect_git"])
    assert rc == 2
    assert "cannot be unset on its own" in capsys.readouterr().err
    # The file is untouched: nothing was silently dropped or rewritten.
    assert cfg.read_text(encoding="utf-8") == "sandbox.protect_git = false\n"


def test_add_rejects_a_value_masked_by_a_higher_layer(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config add` checks the whole new list as `config set` does.

    Under a masking repo overlay too.
    """
    assert _run(["config", "set", "--repo", "sandbox.fetch_hosts", '["ok.example"]']) == 0
    capsys.readouterr()
    rc = _refuse(["config", "add", "sandbox.fetch_hosts", "5"])
    assert rc == 2
    assert "fetch_hosts" in capsys.readouterr().err
    gcfg = iso / "g" / "agent6" / "config.toml"
    assert not gcfg.is_file() or "5" not in gcfg.read_text(encoding="utf-8")


def test_set_warns_when_another_layer_is_still_broken(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid write over a config broken in another layer lands.

    Exits 0 and warns with that error.
    """
    (iso / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "agent6" / "config.toml").write_text('[cli]\ninput = "x"\n', encoding="utf-8")
    rc = _run(["config", "set", "--repo", "sandbox.protect_git", "false"])
    assert rc == 0
    assert "a value this edit did not write" in capsys.readouterr().err


def test_get_honours_the_global_config_flag(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--config FILE` reaches `config get`, so `get` and `show` agree about the same leaf."""
    explicit = iso / "x.toml"
    explicit.write_text("[review]\nperiod = 77\n", encoding="utf-8")
    assert _run(["--config", str(explicit), "config", "get", "review.period"]) == 0
    out = capsys.readouterr().out
    assert "review.period = 77" in out
    assert "[flag]" in out


def test_get_refuses_a_missing_global_config_file(iso: pathlib.Path) -> None:
    """`get` refuses a missing `--config` file, as `show` does.

    Instead of answering from defaults.
    """
    assert _refuse(["--config", str(iso / "nope.toml"), "config", "get", "review.period"]) == 2


def test_get_refuses_a_machine_file_that_does_not_exist(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`get` refuses a --machine-file that does not exist instead of reading it as empty."""
    assert (
        _refuse(["config", "get", "--machine-file", str(iso / "nope.asm.toml"), "review.period"])
        == 2
    )
    assert "no such machine file" in capsys.readouterr().err


def test_fix_refuses_a_machine_file_that_does_not_exist(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = iso / "nope.asm.toml"

    assert _refuse(["config", "fix", "--machine-file", str(missing)]) == 2
    assert capsys.readouterr().err == f"ERROR: no such machine file: {missing}\n"


def test_set_refuses_a_machine_file_holding_a_protected_table(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`set` refuses a machine file holding a protected table, as get, show and fix do."""
    machine = iso / "protected.asm.toml"
    text = '[config.sandbox]\nnetwork = "host"\n'
    machine.write_text(text, encoding="utf-8")

    argv = ["config", "set", "harness.max_iterations", "3", "--machine-file", str(machine)]
    assert _refuse(argv) == 2
    assert "operator-only" in capsys.readouterr().err
    assert machine.read_text(encoding="utf-8") == text


def test_a_provider_leaf_error_names_every_valid_value(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A provider leaf error names every valid value of the `api_format` discriminator."""
    assert _run(["config", "set", "providers.p.api_format", "nonsense"]) == 2
    err = capsys.readouterr().err
    assert "anthropic" in err
    assert "openai" in err


def test_an_unreadable_config_refuses_rather_than_crashing(iso: pathlib.Path) -> None:
    """A root-owned config after a sudo run is refused as the operator's file.

    A crash report.
    """
    gdir = iso / "g" / "agent6"
    gdir.mkdir(parents=True, exist_ok=True)
    cfg = gdir / "config.toml"
    cfg.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    cfg.chmod(0o000)
    try:
        assert _refuse(["config", "show"]) == 2
    finally:
        cfg.chmod(0o600)


_CRASH_MARKERS = ("unexpected", "full traceback", "report this")


@pytest.mark.parametrize(
    "argv",
    [
        ["config", "set", "harness.max_iterations", "7"],
        ["config", "unset", "review.period"],
        ["config", "add", "sandbox.allow_urls", "https://example.com"],
        ["config", "remove", "sandbox.allow_urls", "https://example.com"],
    ],
    ids=["set", "unset", "add", "remove"],
)
def test_write_commands_refuse_an_unreadable_target(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    """Write commands refuse an unreadable target as the readers do."""
    gdir = iso / "g" / "agent6"
    gdir.mkdir(parents=True, exist_ok=True)
    cfg = gdir / "config.toml"
    cfg.write_text("[review]\nperiod = 7\n", encoding="utf-8")
    cfg.chmod(0o000)
    try:
        assert _refuse(argv) == 2
    finally:
        cfg.chmod(0o600)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert "config.toml" in err
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_a_write_command_bug_still_crash_reports(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unexpected exception inside `config set` keeps the crash report at exit 1."""

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(io, "upsert_toml_leaf", _boom)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    assert _refuse(["config", "set", "harness.max_iterations", "7"]) == 1
    err = capsys.readouterr().err
    assert "unexpected RuntimeError" in err
    tb_line = next(line for line in err.splitlines() if "full traceback:" in line)
    pathlib.Path(tb_line.split("full traceback:", 1)[1].strip()).unlink()


def test_set_of_an_unserializable_cli_value_refuses(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A value the writer cannot serialize (`2024-01-01`, a TOML date) is a refusal, not a crash."""
    assert _refuse(["config", "set", "harness.max_iterations", "2024-01-01"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert not any(marker in err for marker in _CRASH_MARKERS)


def test_commit_trailer_validates_placeholders_and_shape(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """[git.commit].trailer takes a trailer line with {model} as its one placeholder.

    An unknown placeholder or a shapeless string is refused at config set, not at commit time.
    """
    assert _run(["config", "set", "git.commit.trailer", "Assisted-by: agent6:{model}"]) == 0
    capsys.readouterr()
    assert _refuse(["config", "set", "git.commit.trailer", "Assisted-by: {agent}"]) == 2
    assert "agent" in capsys.readouterr().err
    assert _refuse(["config", "set", "git.commit.trailer", "By: {model} ({role})"]) == 2
    assert "role" in capsys.readouterr().err
    assert _refuse(["config", "set", "git.commit.trailer", "no trailer shape"]) == 2
    assert "Key: value" in capsys.readouterr().err


def test_checkpoint_style_refuses_combine(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The checkpoint table refuses `combine`, git's own squash message, while squash accepts it."""
    assert _run(["config", "set", "git.commit.squash.message", "combine"]) == 0
    assert _refuse(["config", "set", "git.commit.checkpoint.message", "combine"]) == 2


def test_coauthor_is_gone(iso: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Replaced by the trailer format string; no migration, pre-1.0."""
    assert _refuse(["config", "set", "git.commit.coauthor", "A <a@b>"]) == 2
    assert "trailer" in capsys.readouterr().err  # the did-you-mean points at it


def test_the_paired_context_thresholds_are_settable_together(iso: pathlib.Path) -> None:
    """The paired context thresholds move together in one validated inline-table upsert."""
    inline = "{ drop_at_chars = 200000, summarise_at_chars = 400000 }"
    assert _run(["config", "set", "context", inline]) == 0
    ctx = _global_toml(iso)["context"]
    assert isinstance(ctx, dict)
    assert ctx == {"drop_at_chars": 200000, "summarise_at_chars": 400000}


def test_setting_one_threshold_names_the_command_that_works(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Setting one threshold alone refuses and names the inline-table command that works."""
    assert _refuse(["config", "set", "context.drop_at_chars", "200000"]) == 2
    err = capsys.readouterr().err
    assert "config set context '{ drop_at_chars =" in err
    assert "summarise_at_chars =" in err


def _fill_and_read(iso: pathlib.Path) -> dict[str, object]:
    assert _run(["config", "fill", "--force"]) == 0
    return _global_toml(iso)


def test_fill_never_bakes_the_repo_layer_into_the_global_config(iso: pathlib.Path) -> None:
    """Fill writes the global file from defaults plus global, never this repo's overrides."""
    assert _run(["config", "set", "--repo", "sandbox.memory_limit_mb", "4321"]) == 0
    filled = _fill_and_read(iso)
    sandbox = filled["sandbox"]
    assert isinstance(sandbox, dict)
    assert sandbox["memory_limit_mb"] != 4321, "the repo layer leaked into the global config"


def test_fill_keeps_the_preset_selector_and_does_not_bake_its_effects(iso: pathlib.Path) -> None:
    """Fill keeps a selected preset selected and does not bake its effects."""
    assert _run(["config", "set", "preset", "quick"]) == 0
    filled = _fill_and_read(iso)
    assert filled.get("preset") == "quick", "the selector was dropped"
    review = filled["review"]
    assert isinstance(review, dict)
    # `quick` sets review.trigger = "off"; the filled value is the default and the preset applies.
    from agent6.config import Config

    assert review["trigger"] == Config().review.trigger


def test_fill_keeps_authored_preset_bodies(iso: pathlib.Path) -> None:
    (iso / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (iso / "g" / "agent6" / "config.toml").write_text(
        "[presets.myteam.review]\nconcurrency = 2\n", encoding="utf-8"
    )
    filled = _fill_and_read(iso)
    presets = filled["presets"]
    assert isinstance(presets, dict)
    assert "myteam" in presets


def test_config_path_lists_every_directory_agent6_writes_to(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config path` lists the config, repo and secrets files plus every directory agent6 writes to.

    Four XDG bases each holding a different thing is correct and unguessable.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(iso / "st"))
    monkeypatch.setenv("XDG_DATA_HOME", str(iso / "dt"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(iso / "ch"))
    assert _run(["config", "path"]) == 0
    out = capsys.readouterr().out
    for label in ("global config", "repo config", "secrets", "config dir", "cache"):
        assert f"{label}" in out
    assert str(iso / "st") in out  # state base
    assert str(paths.state_dir(iso)) in out  # this repo's own dir
    assert str(iso / "dt" / "agent6" / "skills") in out
    assert str(iso / "ch") in out


def test_top_level_help_names_the_directories(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 --help` ends with the four XDG dirs, resolved, and points at `config path`."""
    monkeypatch.setenv("XDG_STATE_HOME", str(iso / "st"))
    monkeypatch.setenv("XDG_DATA_HOME", str(iso / "dt"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(iso / "ch"))
    with pytest.raises(SystemExit) as exc:
        _run(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "directories" in out
    assert str(iso / "g") in out and str(iso / "st") in out
    assert str(iso / "dt") in out and str(iso / "ch") in out
    assert "agent6 config path" in out


def test_a_preset_leaf_with_a_valid_sibling_can_be_changed(iso: pathlib.Path) -> None:
    config = iso / "g" / "agent6" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "[presets.demo.context]\ndrop_at_chars = 100000\nsummarise_at_chars = 200000\n",
        encoding="utf-8",
    )

    assert _run(["config", "set", "presets.demo.context.drop_at_chars", "120000"]) == 0
    context = _global_toml(iso)["presets"]
    assert isinstance(context, dict)
    assert context["demo"]["context"] == {  # type: ignore[index]
        "drop_at_chars": 120000,
        "summarise_at_chars": 200000,
    }


def test_a_preset_leaf_is_validated_and_has_an_inverse(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A preset leaf is validated on set, seen by fix and removable by unset.

    An invalid `presets.demo.sandbox.network` otherwise surfaces only at `run --preset demo`.
    """
    assert _run(["config", "set", "presets.demo.sandbox.network", "banana"]) == 2
    assert "sandbox.network" in capsys.readouterr().err
    assert _run(["config", "set", "presets.demo.sandbox.nosuch", "1"]) == 2
    capsys.readouterr()
    assert _run(["config", "set", "presets.demo.sandbox.network", "host"]) == 0
    capsys.readouterr()
    assert _run(["config", "get", "presets.demo.sandbox.network"]) == 0
    assert "host" in capsys.readouterr().out
    assert _run(["config", "unset", "presets.demo.sandbox.network"]) == 0
    capsys.readouterr()
    assert _run(["config", "get", "presets.demo.sandbox.network"]) == 0
    assert "unset" in capsys.readouterr().out


def test_config_fix_reports_an_entry_it_cannot_auto_remove(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`config fix` says when an entry is not a leaf it can drop and exits non-zero."""
    from agent6.ui.cli import main

    gpath = paths.global_config_path()
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text('preset = "nope"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["config", "fix"]) == 2
    err = capsys.readouterr().err
    assert "not an auto-removable entry" in err and "nope" in err


def test_nonempty_table_leaf_stays_gettable_and_unsettable(
    iso: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dict-typed field is one config leaf even when its value is nonempty."""
    assert _run(["config", "set", "providers.demo.api_format", "openai"]) == 0
    assert (
        _run(
            [
                "config",
                "set",
                "providers.demo.extra_body",
                '{ route = { order = ["fast", "cheap"] } }',
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert _run(["config", "get", "providers.demo.extra_body"]) == 0
    assert "providers.demo.extra_body = {...}  [global]" in capsys.readouterr().out
    assert _run(["config", "show", "providers.demo.extra_body"]) == 0
    shown = capsys.readouterr().out
    assert "providers.demo.extra_body\n" in shown
    assert "providers.demo.extra_body.route" not in shown

    assert _run(["config", "unset", "providers.demo.extra_body"]) == 0
    providers = _global_toml(iso)["providers"]
    assert isinstance(providers, dict)
    demo = providers["demo"]
    assert isinstance(demo, dict)
    assert "extra_body" not in demo
