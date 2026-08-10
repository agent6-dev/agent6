# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 config show` renders TOML an operator can copy back out."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.config.layer import EffectiveConfig, load_effective
from agent6.viewmodel.config_view import render_key_detail, render_show


def test_a_top_level_scalar_is_not_dressed_as_a_table(tmp_path: Path) -> None:
    """`preset` is a bare top-level key, not a `[preset]` table.

    The renderer grouped every leaf by its first dotted segment, so a key with
    no dot became its own one-row "section" under a `[preset]` header. Copying
    that into a config file writes invalid TOML, and `config fill` -- the other
    half of the same feature -- already emits top-level scalars correctly.
    """
    out = render_show(load_effective(tmp_path, preset="quick"))

    assert "[preset]" not in out, "a scalar rendered as a TOML table header"
    assert "preset" in out, "the setting itself must still be shown"
    # It belongs above the tables, exactly where TOML requires it.
    assert out.index("preset") < out.index("["), "a top-level scalar must precede every section"


def test_config_presets_reads_the_explicit_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--config FILE` is a global flag every config subcommand honours -- except
    `presets`, which hardcoded None and silently listed only the built-ins.

    Silently: the file parsed, the preset was there, and the listing simply did
    not mention it.
    """
    from agent6.ui.cli import main

    cfg = tmp_path / "custom.toml"
    cfg.write_text('[presets.myfast.sandbox]\nrun_commands = "yes"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(cfg), "config", "presets"]) == 0
    assert "myfast" in capsys.readouterr().out, "presets ignored the explicit config file"


def test_a_filled_config_can_be_used_as_an_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fill` snapshots every effective value into one explicit file. That
    file has to be a config agent6 will actually load.

    It emitted the top-level `preset` selector, which the layer REFUSES from an
    explicit `--config` file -- so `agent6 config fill` produced a file that
    `agent6 --config <it>` rejected. `--parallel` was collateral: the
    orchestrator materializes each lane's config the same way, so every lane
    died before starting.

    A preset SELECTS other leaves; once they are materialized the selector is
    both redundant and, for a named preset, would apply twice.
    """
    from agent6.config.layer import materialize

    filled = tmp_path / "filled.toml"
    filled.write_text(materialize(load_effective(tmp_path).config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # The point: this must not raise.
    reloaded = load_effective(tmp_path, filled).config
    assert reloaded.agent6.config_version == 1


def test_config_fill_keeps_the_presets_the_file_defines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config fill` rewrites the operator's own config file. A `[presets.*]`
    table in it is meta-config -- stripped before validation, so absent from the
    `Config` the snapshot is rendered from -- and the rewrite dropped it.

    Silently, and with `--force` there is no earlier copy: `config presets`
    listed `myfast` before the fill and only the built-ins after. The leaves the
    preset selected survive (they are materialized), the definition did not, so
    `--preset myfast` stopped resolving at all.
    """
    from agent6.ui.cli import main

    cfg_home = tmp_path / "cfg"
    cfg_home.mkdir()
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(cfg_home))
    (cfg_home / "config.toml").write_text(
        'preset = "myfast"\n\n[presets.myfast.sandbox]\nrun_commands = "yes"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["config", "fill", "--force"]) == 0

    after = load_effective(tmp_path)
    assert after.config.sandbox.run_commands == "yes", "the preset stopped applying"
    text = (cfg_home / "config.toml").read_text(encoding="utf-8")
    assert "[presets.myfast" in text, f"config fill deleted the operator's preset:\n{text}"
    # The SELECTOR survives, and the preset's EFFECT is not baked: the filled
    # leaf is the default, with the preset still applying over it at runtime.
    # Baking it froze the old values while the selector -- what the operator
    # edits -- was dropped, so later preset edits did nothing.
    assert 'preset = "myfast"' in text
    assert 'run_commands = "ask"' in text, f"the preset's effect was baked in:\n{text}"


def test_descriptions_mode_prints_the_meaning_under_each_row() -> None:
    """`--descriptions` adds each leaf's meaning; the default stays values-only."""
    eff = EffectiveConfig(config=Config(), sources={}, layers=())
    assert "Cap on metered spend" not in render_show(eff)
    assert "Cap on metered spend" in render_show(eff, descriptions=True)


def test_key_detail_always_carries_the_meaning() -> None:
    """`config show <key>` is a deliberate ask about one key, so the meaning is
    part of the answer, no flag needed."""
    eff = EffectiveConfig(config=Config(), sources={}, layers=())
    detail = render_key_detail(eff, "budget.max_usd")
    assert detail is not None
    assert "meaning: Cap on metered spend" in detail
