# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An execution records its settings without keeping credentials or applying old values."""

from __future__ import annotations

import pathlib

from agent6 import paths
from agent6.app.session import setup
from agent6.config import layer
from agent6.sessions import event_log, journal, process
from agent6.viewmodel import execution_settings, receipt, telemetry
from agent6.viewmodel.session import log


def test_settings_record_the_rows_config_show_prints(tmp_path: pathlib.Path) -> None:
    """A flag, a preset and an override file keep distinct origins.

    The record masked every free-form string, so a commit name read `***` beside a
    `config show` that printed it.
    """
    cfg = paths.repo_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('[agent6]\npreset = "small"\n[presets.small.models.default]\neffort="high"\n')
    extra = tmp_path / "extra.toml"
    extra.write_text(
        '[providers.p]\nbackend="openai"\nbase_url="http://localhost:1"\n'
        'token_command=["echo", "private-token"]\n[providers.p.extra_body]\n'
        'field="private-body"\n[models.default]\nmodel="p/example"\n[budget]\nmax_usd=9\n'
        '[git.commit]\nname="smoke"\n'
    )
    eff = setup.load_session_config(
        tmp_path, (extra,), mode="run", budget_overrides=setup.BudgetOverrides(max_usd=3)
    )
    settings = execution_settings.observations(eff.config, eff)
    record = journal.ExecutionSettings(execution_id="one", settings=settings)
    assert "private-body" not in record.model_dump_json()
    rows = {r.key: r for r in record.settings}
    assert rows["providers.p.extra_body.field"].value == "***"
    assert rows["providers.p.token_command"].value == '["echo", "private-token"]'
    assert rows["git.commit.name"].value == "smoke"
    assert rows["budget.max_usd"].value == 3 and rows["budget.max_usd"].source == "flag"
    assert rows["budget.max_usd"].file is None
    assert rows["models.default.effort"].source == "preset small"
    assert rows["models.default.effort"].file == str(cfg)
    assert rows["models.default.model"].file == str(extra)
    assert "web.metrics_route" not in rows


def test_an_explicit_flag_is_the_source_even_when_the_file_value_matches(
    tmp_path: pathlib.Path,
) -> None:
    cfg = paths.repo_config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[providers.p]\nbackend="openai"\nbase_url="http://localhost:1"\n'
        '[models.default]\nmodel="p/m"\n[models.worker]\nmodel="p/m"\n'
        '[budget]\nmax_usd=3\n[sandbox]\nrun_commands="deny"\n'
    )
    eff = setup.load_session_config(
        tmp_path,
        (),
        mode="run",
        model="p/m",
        budget_overrides=setup.BudgetOverrides(max_usd=3),
        sandbox_overrides=setup.SandboxOverrides(no_commands=True),
    )
    rows = {r.key: r for r in execution_settings.observations(eff.config, eff)}
    for key in ("budget.max_usd", "models.worker.model", "sandbox.run_commands"):
        assert rows[key].source == "flag" and rows[key].file is None


def test_each_execution_keeps_its_own_settings_in_show_receipt_and_telemetry(
    tmp_path: pathlib.Path,
) -> None:
    """A second start keeps the earlier settings only as evidence."""
    config = paths.repo_config_path(tmp_path)
    config.parent.mkdir(parents=True, exist_ok=True)
    directory = tmp_path / "proof"
    directory.mkdir()
    events = event_log.EventSink(directory / "logs.jsonl")
    fold = telemetry.TelemetryFold("proof")
    for eid, cap in (("one", 1), ("two", 2)):
        config.write_text(f"[budget]\nmax_usd={cap}\n")
        eff = layer.load_effective(tmp_path)
        process.emit_session_start(
            events,
            directory,
            journal.SessionStart(execution_id=eid, user_task="work", mode="run"),
            settings=execution_settings.observations(eff.config, eff),
        )
    scan = log.scan_session_log(events.path)
    assert [r.execution_id for r in scan.settings] == ["one", "two"]
    assert [r.settings[0].value for r in scan.settings] == [1, 2]
    view = receipt.read_receipt(directory, tmp_path)
    assert view.settings == scan.settings
    assert "budget.max_usd" in receipt.render(view)
    import json

    observed = [fold.feed(json.loads(line)) for line in events.path.read_text().splitlines()]
    assert "budget.max_usd" in str(observed)
