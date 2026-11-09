from __future__ import annotations

import json
import pathlib

import pytest
from bench.coreagent import run

from agent6 import config
from agent6.sessions import event_log


@pytest.mark.parametrize("condition", sorted({*run.CONDITIONS, "verify_finish", "revise_auto"}))
def test_every_condition_loads_with_a_default_route(tmp_path: pathlib.Path, condition: str) -> None:
    """Benchmark configs use qualified routes and current condition values."""
    path = tmp_path / "config.toml"
    path.write_text(
        run._provider_block("chatgpt", "test-model", ["bash", "verify.sh"], "high")
        + run.CONDITIONS[condition].format(ROOT=run.ROOT),
        encoding="utf-8",
    )
    cfg = config.load_config(path)
    assert cfg.models.default is not None
    assert cfg.models.default.provider == "chatgpt"
    assert cfg.models.default.model == "test-model"
    assert cfg.models.default.effort == "high"
    assert cfg.models.worker is None
    assert cfg.models.helper is None
    assert cfg.harness.verify_command == ("bash", "verify.sh")
    assert cfg.harness.verify_when == ("finish" if condition == "verify_finish" else "never")
    assert cfg.prompt.revise_prompt == ("auto" if condition == "revise_auto" else "off")


@pytest.mark.parametrize("revised", [False, True])
def test_metrics_read_registered_revision_and_final_verification(
    tmp_path: pathlib.Path, revised: bool
) -> None:
    """A revision result and the final stop survive into the benchmark row."""
    log = tmp_path / "agent6" / "repo" / "sessions" / "runs" / "run" / "logs.jsonl"
    log.parent.mkdir(parents=True)
    sink = event_log.EventSink(log)
    sink.emit("loop.prompt_revision.call", mode="auto")
    if revised:
        sink.emit("loop.prompt_revision.result", raw_chars=10, revised_chars=20, questions=0)
    sink.emit(
        "session.stop", execution_id="e", reason="finish_session", turns=4, verification="passed"
    )
    metrics = run._extract_metrics(tmp_path)
    assert metrics["task_rewritten"] is revised
    assert metrics["verification"] == "passed"
    assert metrics["stop_reason"] == "finish_session"
    assert metrics["iterations"] == 4


def test_dry_run_builds_config_without_agent_or_grader(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry runs seed repos and configs without launching the installed agent."""
    monkeypatch.setattr(run, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(run, "AGENT6_BIN", "/no-agent-must-run")
    row = run.one_run(
        task="textkit",
        model="test-model",
        provider="chatgpt",
        condition="verify_finish",
        rep=0,
        budget=0.6,
        label="dry",
        effort="high",
        dry_run=True,
    )
    assert row["dry_run"] is True
    cfg = pathlib.Path(row["config"])
    assert config.load_config(cfg).harness.verify_when == "finish"
    assert (cfg.parent / ".git").is_dir()
    assert not (cfg.parent / "agent.log").exists()
    assert "score" not in row
    assert not pathlib.Path(row["state_home"]).is_relative_to(cfg.parent)
    assert json.loads(json.dumps(row))["condition"] == "verify_finish"
