# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config presets: a named preset injected just above the config layer that selected it.

The preset overrides that config (a more specific config layer or a flag still wins); the most
specific preset source wins, and presets never stack.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.config import ConfigError, layer


def _write_repo_config(repo: pathlib.Path, toml: str) -> None:
    p = paths.repo_config_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(toml, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_preset_via_preset_field_expands_review_knobs(repo: pathlib.Path) -> None:
    _write_repo_config(repo, 'preset = "ultra"\n')
    cfg = layer.load_effective(repo).config
    assert cfg.review.trigger == "before_finish"
    assert cfg.review.seats == ("security", "correctness", "tests")
    assert cfg.review.decision == "veto"
    assert cfg.review.concurrency == 3  # seats run in parallel, not in series


def test_preset_via_flag_overrides_field(repo: pathlib.Path) -> None:
    _write_repo_config(repo, 'preset = "quick"\n')
    cfg = layer.load_effective(repo, preset="paranoid").config  # flag wins over the field
    assert len(cfg.review.seats) == 5
    assert cfg.review.tier == "explore"
    assert cfg.review.decision == "veto"
    assert cfg.review.concurrency == 5  # seats run in parallel, not in series


def test_repo_selected_preset_beats_same_layer_setting(repo: pathlib.Path) -> None:
    # The preset selected by the repo's top-level `preset` is injected ABOVE the
    # repo config, so it OVERRIDES a conflicting value set in the SAME repo config.
    _write_repo_config(repo, 'preset = "ultra"\n\n[review]\ndecision = "advisory"\n')
    cfg = layer.load_effective(repo).config
    assert cfg.review.decision == "veto"  # repo-selected preset wins
    assert cfg.review.seats == ("security", "correctness", "tests")  # rest of the preset applies


def test_custom_user_preset(repo: pathlib.Path) -> None:
    _write_repo_config(
        repo,
        'preset = "myteam"\n\n[presets.myteam.review]\n'
        'trigger = "before_finish"\nconcurrency = 2\n',
    )
    cfg = layer.load_effective(repo).config
    assert cfg.review.concurrency == 2 and cfg.review.trigger == "before_finish"


def test_only_a_flag_selected_preset_is_replayed_on_resume(repo: pathlib.Path) -> None:
    """Only a flag-selected preset is replayed on resume.

    A resumed or forked execution re-applies `--preset` but must not hand a config-selected name
    back as an override: `_select_preset` would call it a flag, which outranks every config layer,
    so a run whose repo config beat a global preset would come back from resume with the preset
    winning, gaining a blocking review veto the original never had. The stamp records which case it
    was.
    """
    from agent6.sessions import manifest

    assert manifest.HarnessStamp(preset="t", preset_from_flag=True).replay_preset == "t"
    assert manifest.HarnessStamp(preset="t").replay_preset == ""  # config-selected: re-resolves

    # Why it matters: the SAME files resolve differently when the name arrives
    # as a flag, which is exactly what the old replay did.
    _write_repo_config(repo, f'preset = "t"\n\n[review]\nconcurrency = 3\n\n{_PROFILE_T}')
    assert layer.load_effective(repo).config.review.concurrency == 5  # repo-selected preset wins
    _write_repo_config(repo, f"[review]\nconcurrency = 3\n\n{_PROFILE_T}")
    assert layer.load_effective(repo).config.review.concurrency == 3  # no selection: config wins
    assert layer.load_effective(repo, None, preset="t").config.review.concurrency == 5  # as a flag


def test_user_preset_named_standard_replaces_the_builtin(repo: pathlib.Path) -> None:
    """A user preset named `standard` replaces the built-in.

    A user table named after a built-in replaces it wholesale (docs/config.md, and resolve_preset's
    own "user presets win over built-ins" contract); short-circuiting the name to the empty built-in
    drops its overrides silently while `agent6 config presets` reports it selected and applied.
    """
    _write_repo_config(
        repo,
        'preset = "standard"\n\n[presets.standard]\nreview = { trigger = "before_finish",'
        " concurrency = 4 }\n",
    )
    cfg = layer.load_effective(repo).config
    assert cfg.review.trigger == "before_finish"
    assert cfg.review.concurrency == 4


def test_unknown_preset_errors(repo: pathlib.Path) -> None:
    _write_repo_config(repo, 'preset = "nope"\n')
    with pytest.raises(ConfigError, match="unknown preset"):
        layer.load_effective(repo)


def test_preset_table_instead_of_string_is_clear_error(repo: pathlib.Path) -> None:
    """A `[preset]` table instead of a string is a clear error.

    A mistyped `config set preset.porifle x` fails as "preset must be a string", never str()-coerced
    into `unknown preset "{'porifle': 'ultra'}"`.
    """
    _write_repo_config(repo, '[preset]\nporifle = "ultra"\n')
    with pytest.raises(ConfigError, match="must be a preset name string"):
        layer.load_effective(repo)


def test_no_preset_is_plain_defaults(repo: pathlib.Path) -> None:
    _write_repo_config(repo, "[review]\n")
    cfg = layer.load_effective(repo).config
    assert cfg.review.trigger == "off" and cfg.review.concurrency == 1


# ---------------------------------------------------------------------------
# Scope-nested precedence: a preset OVERRIDES config at its scope, but a
# more-specific config layer (or flag) overrides the preset; most-specific
# preset source wins, presets never stack.
#
# `review.concurrency` is the observable knob: default 1, the custom presets
# below set it to 5, and config layers set it to other distinct values.
# ---------------------------------------------------------------------------

# A custom preset [presets.t] that sets review.concurrency = 5 (distinct from
# both the default 1 and the config values used in each test).
_PROFILE_T = "[presets.t.review]\nconcurrency = 5\n"


@pytest.fixture
def global_config(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Point the global config at an isolated dir and return its path."""
    gdir = tmp_path / "global"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    return gdir / "agent6" / "config.toml"


def test_global_selected_preset_loses_to_repo_config(
    repo: pathlib.Path, global_config: pathlib.Path
) -> None:
    # Preset selected by GLOBAL top-level `preset` sits between global and repo
    # config, so a conflicting value in REPO config (more specific) wins.
    global_config.write_text(f'preset = "t"\n\n{_PROFILE_T}', encoding="utf-8")
    _write_repo_config(repo, "[review]\nconcurrency = 3\n")
    cfg = layer.load_effective(repo).config
    assert cfg.review.concurrency == 3  # repo config beats global-selected preset


def test_repo_selected_preset_beats_same_repo_config(repo: pathlib.Path) -> None:
    # Preset selected by REPO top-level `preset` sits ABOVE the repo config, so
    # a conflicting value in the SAME repo config loses to the preset.
    _write_repo_config(repo, f'preset = "t"\n\n[review]\nconcurrency = 3\n\n{_PROFILE_T}')
    cfg = layer.load_effective(repo).config
    assert cfg.review.concurrency == 5  # repo-selected preset wins


def test_flag_selected_preset_beats_config(repo: pathlib.Path) -> None:
    # --preset FLAG injects the preset above all config, so it beats a
    # conflicting value in config.
    _write_repo_config(repo, f"[review]\nconcurrency = 3\n\n{_PROFILE_T}")
    cfg = layer.load_effective(repo, preset="t").config
    assert cfg.review.concurrency == 5  # flag-selected preset wins


def test_flag_preset_loses_to_explicit_config_file(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    # --preset FLAG + an explicit --config FILE setting the same field: the
    # --config FILE sits ABOVE the flag-selected preset, so the file wins.
    _write_repo_config(repo, _PROFILE_T)  # custom preset defined in repo config
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[review]\nconcurrency = 7\n", encoding="utf-8")
    cfg = layer.load_effective(repo, explicit, preset="t").config
    assert cfg.review.concurrency == 7  # explicit --config FILE beats the preset


def test_no_stacking_only_most_specific_preset_applies(
    repo: pathlib.Path, global_config: pathlib.Path
) -> None:
    # Different presets at global (sets field X) and repo (sets field Y): only
    # the REPO preset applies; X falls back to its DEFAULT (no stacking).
    global_config.write_text(
        'preset = "g"\n\n[presets.g.review]\ntrigger = "before_finish"\n',
        encoding="utf-8",
    )
    _write_repo_config(
        repo,
        'preset = "r"\n\n[presets.r.review]\nconcurrency = 5\n',
    )
    cfg = layer.load_effective(repo).config
    assert cfg.review.concurrency == 5  # the repo preset applies
    assert cfg.review.trigger == "off"  # the global preset does NOT stack (default)


def test_no_preset_anywhere_is_plain_config(
    repo: pathlib.Path, global_config: pathlib.Path
) -> None:
    # Regression: with no preset selected anywhere, the result is identical to
    # plain config (the global/repo layers merge normally, preset is a no-op).
    global_config.write_text("[review]\nconcurrency = 4\n", encoding="utf-8")
    _write_repo_config(repo, '[review]\ntrigger = "before_finish"\n')
    cfg = layer.load_effective(repo).config
    assert cfg.review.concurrency == 4  # from global config
    assert cfg.review.trigger == "before_finish"  # from repo config
