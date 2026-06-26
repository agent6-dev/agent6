# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Config profiles: a named preset injected just above the config layer that
selected it, so the profile OVERRIDES that config (a more-specific config layer
or flag still wins); most-specific profile source wins, presets never stack."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import ConfigError
from agent6.config_layer import load_effective, repo_config_path_for


def _write_repo_config(repo: Path, toml: str) -> None:
    p = repo_config_path_for(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(toml, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_profile_via_profile_field_expands_review_knobs(repo: Path) -> None:
    _write_repo_config(repo, 'profile = "ultra"\n')
    cfg = load_effective(repo).config
    assert cfg.review.trigger == "before_finish"
    assert cfg.review.panel_size == 3
    assert cfg.review.decision == "veto"


def test_profile_via_flag_overrides_field(repo: Path) -> None:
    _write_repo_config(repo, 'profile = "quick"\n')
    cfg = load_effective(repo, profile="paranoid").config  # flag wins over the field
    assert cfg.review.panel_size == 5
    assert cfg.review.tier == "explore"
    assert cfg.review.decision == "veto"


def test_repo_selected_profile_beats_same_layer_setting(repo: Path) -> None:
    # The profile selected by the repo's top-level `profile` is injected ABOVE the
    # repo config, so it OVERRIDES a conflicting value set in the SAME repo config.
    _write_repo_config(repo, 'profile = "ultra"\n\n[review]\ndecision = "advisory"\n')
    cfg = load_effective(repo).config
    assert cfg.review.decision == "veto"  # repo-selected profile wins
    assert cfg.review.panel_size == 3  # the rest of the profile applies too


def test_custom_user_profile(repo: Path) -> None:
    _write_repo_config(
        repo,
        'profile = "myteam"\n\n[profiles.myteam.review]\n'
        'trigger = "before_finish"\npanel_size = 2\n',
    )
    cfg = load_effective(repo).config
    assert cfg.review.panel_size == 2 and cfg.review.trigger == "before_finish"


def test_unknown_profile_errors(repo: Path) -> None:
    _write_repo_config(repo, 'profile = "nope"\n')
    with pytest.raises(ConfigError, match="unknown profile"):
        load_effective(repo)


def test_no_profile_is_plain_defaults(repo: Path) -> None:
    _write_repo_config(repo, "[review]\n")
    cfg = load_effective(repo).config
    assert cfg.review.trigger == "off" and cfg.review.panel_size == 1


# ---------------------------------------------------------------------------
# Scope-nested precedence: a profile OVERRIDES config at its scope, but a
# more-specific config layer (or flag) overrides the profile; most-specific
# profile source wins, presets never stack.
#
# `review.panel_size` is the observable knob: default 1, the custom profiles
# below set it to 5, and config layers set it to other distinct values.
# ---------------------------------------------------------------------------

# A custom profile [profiles.t] that sets review.panel_size = 5 (distinct from
# both the default 1 and the config values used in each test).
_PROFILE_T = "[profiles.t.review]\npanel_size = 5\n"


@pytest.fixture
def global_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the global config at an isolated dir and return its path."""
    gdir = tmp_path / "global"
    gdir.mkdir()
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    return gdir / "config.toml"


def test_global_selected_profile_loses_to_repo_config(repo: Path, global_config: Path) -> None:
    # Profile selected by GLOBAL top-level `profile` sits between global and repo
    # config, so a conflicting value in REPO config (more specific) wins.
    global_config.write_text(f'profile = "t"\n\n{_PROFILE_T}', encoding="utf-8")
    _write_repo_config(repo, "[review]\npanel_size = 3\n")
    cfg = load_effective(repo).config
    assert cfg.review.panel_size == 3  # repo config beats global-selected profile


def test_repo_selected_profile_beats_same_repo_config(repo: Path) -> None:
    # Profile selected by REPO top-level `profile` sits ABOVE the repo config, so
    # a conflicting value in the SAME repo config loses to the profile.
    _write_repo_config(repo, f'profile = "t"\n\n[review]\npanel_size = 3\n\n{_PROFILE_T}')
    cfg = load_effective(repo).config
    assert cfg.review.panel_size == 5  # repo-selected profile wins


def test_flag_selected_profile_beats_config(repo: Path) -> None:
    # --profile FLAG injects the profile above all config, so it beats a
    # conflicting value in config.
    _write_repo_config(repo, f"[review]\npanel_size = 3\n\n{_PROFILE_T}")
    cfg = load_effective(repo, profile="t").config
    assert cfg.review.panel_size == 5  # flag-selected profile wins


def test_flag_profile_loses_to_explicit_config_file(repo: Path, tmp_path: Path) -> None:
    # --profile FLAG + an explicit --config FILE setting the same field: the
    # --config FILE sits ABOVE the flag-selected profile, so the file wins.
    _write_repo_config(repo, _PROFILE_T)  # custom profile defined in repo config
    explicit = tmp_path / "explicit.toml"
    explicit.write_text("[review]\npanel_size = 7\n", encoding="utf-8")
    cfg = load_effective(repo, explicit, profile="t").config
    assert cfg.review.panel_size == 7  # explicit --config FILE beats the profile


def test_no_stacking_only_most_specific_profile_applies(repo: Path, global_config: Path) -> None:
    # Different profiles at global (sets field X) and repo (sets field Y): only
    # the REPO profile applies; X falls back to its DEFAULT (no stacking).
    global_config.write_text(
        'profile = "g"\n\n[profiles.g.review]\ntrigger = "before_finish"\n',
        encoding="utf-8",
    )
    _write_repo_config(
        repo,
        'profile = "r"\n\n[profiles.r.review]\npanel_size = 5\n',
    )
    cfg = load_effective(repo).config
    assert cfg.review.panel_size == 5  # the repo profile applies
    assert cfg.review.trigger == "off"  # the global profile does NOT stack (default)


def test_no_profile_anywhere_is_plain_config(repo: Path, global_config: Path) -> None:
    # Regression: with no profile selected anywhere, the result is identical to
    # plain config (the global/repo layers merge normally, profile is a no-op).
    global_config.write_text("[review]\npanel_size = 4\n", encoding="utf-8")
    _write_repo_config(repo, '[review]\ntrigger = "before_finish"\n')
    cfg = load_effective(repo).config
    assert cfg.review.panel_size == 4  # from global config
    assert cfg.review.trigger == "before_finish"  # from repo config
