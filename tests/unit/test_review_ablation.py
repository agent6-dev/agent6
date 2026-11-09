from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest
from bench.review import run_ab, stats

from agent6 import config


@pytest.mark.parametrize(
    "case",
    json.loads(
        (pathlib.Path(__file__).resolve().parents[2] / "bench/review/cases.json").read_text()
    )["cases"],
)
def test_review_fixtures_and_routes_are_runnable(
    tmp_path: pathlib.Path, case: dict[str, Any]
) -> None:
    """Each frozen case has a clean two-commit diff and executable supplied checks."""
    prepared = run_ab.prepare(case, tmp_path / case["id"], "chatgpt/test-model", "high", 0.0)
    cfg = config.load_config(prepared.config)
    assert cfg.models.default is not None
    assert cfg.models.default.model == "test-model"
    assert len(cfg.review.reviewers) == 3
    for seat in cfg.review.reviewers:
        route = seat.resolve(cfg.models.resolve("reviewer"))
        assert route is not None and route.effort == "high" and route.provider == "chatgpt"
    assert not prepared.state_home.is_relative_to(prepared.workdir)
    assert run_ab.git(prepared.workdir, "status", "--porcelain") == ""
    diff = run_ab.git(prepared.workdir, "diff", "HEAD~1..HEAD")
    assert case["change"].splitlines()[0] in diff
    assert "oracle" not in diff
    if case["id"] == "cross-file":
        assert 'parse("x")' in (prepared.workdir / "main.py").read_text()
        assert "main.py" not in run_ab.git(prepared.workdir, "diff", "--name-only", "HEAD~1..HEAD")
    subprocess.run([sys.executable, str(prepared.checks)], cwd=prepared.workdir, check=True)


def test_dry_run_never_launches_agent_or_driver(tmp_path: pathlib.Path) -> None:
    """The default dry run prepares all 36 cells without needing agent6 or credentials."""
    result = subprocess.run(
        [sys.executable, str(run_ab.ROOT / "run_ab.py"), "--dry-run", "--runs-root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "36 repos prepared; no model calls" in result.stdout
    assert len(list(tmp_path.glob("*/config.toml"))) == 36
    assert not list(tmp_path.rglob("logs.jsonl"))
    refused = subprocess.run(
        [sys.executable, str(run_ab.ROOT / "run_ab.py"), "--model", "p/m"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "--panel-driver" in refused.stderr


def test_metrics_do_not_call_every_grounded_block_true() -> None:
    """Grounding proves a citation, not the oracle match; true blocks need adjudication."""
    panel = {
        "seats": [
            {"verdict": "block", "risk": "", "risk_truncated": False},
            {"verdict": "pass", "risk": "x" * 1000, "risk_truncated": True},
            {"verdict": "abstain", "risk": "", "risk_truncated": False},
        ],
        "result": {"findings": [{"finding_id": "f", "finding": {"severity": "block"}}]},
    }
    row = run_ab.measures(panel, clean=False)
    assert row["grounded_blocks"] == 1
    assert row["grounded_true_blocks"] is None
    assert row["abstentions"] == 1
    assert row["missing_risk"] == 2
    assert row["risk_length_violations"] == 1
    assert run_ab.measures(panel, clean=True)["false_blocks"] == 1


@pytest.mark.parametrize("arm", ["A", "B"])
def test_panel_driver_records_real_grounding_and_receipt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    """The bench driver uses real panel grounding and reads its native three-seat receipt."""
    from unittest import mock

    from bench.review import panel_driver

    from agent6 import kinds
    from agent6.app import role_providers
    from agent6.providers import provider_types

    case = json.loads((run_ab.ROOT / "cases.json").read_text())["cases"][0]
    prepared = run_ab.prepare(case, tmp_path / "cell", "p/m", "high", 0.0)
    cfg = config.validate_config(
        {
            **config.load_config(prepared.config).model_dump(mode="json"),
            "providers": {"p": {"backend": "openai", "base_url": "http://127.0.0.1:1"}},
        }
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(prepared.state_home))
    monkeypatch.setenv("AGENT6_REVIEW_RISK", "off" if arm == "A" else "on")
    calls: list[str] = []

    def factory(*_args: Any, **kwargs: Any) -> Any:
        def call(**request: Any) -> provider_types.ProviderResponse:
            assert request["max_tokens"] == 1500
            assert request["temperature"] == 0.0
            calls.append(request["system"])
            kwargs["budget"].record(
                route=kinds.ModelRoute("p", "m"),
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                cost_usd=0.01,
            )
            reply: dict[str, Any] = dict(
                verdict="block",
                summary="traversal",
                findings=[
                    dict(
                        category="security",
                        severity="block",
                        title="Traversal",
                        detail="../ escapes files",
                        location=dict(path="case.py", side="head", start_line=2, end_line=2),
                    )
                ],
            )
            if arm == "B":
                reply["risk"] = "x" * 1001
            return provider_types.ProviderResponse(
                text=json.dumps(reply),
                tool_uses=(),
                stop_reason="end_turn",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                raw={},
            )

        return mock.Mock(call=mock.Mock(side_effect=call))

    monkeypatch.setattr(role_providers, "_provider_from_entry", factory)
    panel_driver.run(cfg, prepared.workdir, prepared.checks, case["supplied_checks"])
    row = run_ab.read_measurements(prepared.state_home, clean=False)
    assert len(calls) == 3
    assert all(('"risk":' in prompt) == (arm == "B") for prompt in calls)
    assert row["grounded_blocks"] == 1
    assert row["tokens_in"] == 30 and row["tokens_out"] == 15
    assert row["usd"] == pytest.approx(0.03)
    assert row["missing_risk"] == (3 if arm == "A" else 0)
    assert row["risk_length_violations"] == (3 if arm == "B" else 0)
    receipt = json.loads(pathlib.Path(row["receipt"]).read_text())
    assert receipt["verifications"][0]["outcome"] == "passed"
    assert not receipt["review"]["read_failure"]
    adjudicated = dict(row, run_id="cell", case="traversal")
    stats.adjudicate(
        [adjudicated],
        {"cell": {row["block_findings"][0]["finding_id"]: True}},
    )
    assert adjudicated["grounded_true_blocks"] == 1


def test_summary_pairs_by_case_and_repetition(capsys: pytest.CaptureFixture[str]) -> None:
    """Split and case summaries report matched B-A deltas and repetition ranges."""
    rows = [
        dict(
            label="b14",
            model="p/m",
            effort="high",
            temperature=0.0,
            split="evaluation",
            case="secret",
            rep=rep,
            arm=arm,
            grounded_true_blocks=value,
            elapsed_s=1.0,
        )
        for rep, arm, value in [(0, "A", 0), (0, "B", 2), (1, "A", 1), (1, "B", 2)]
    ]
    stats.summarize(rows)
    output = capsys.readouterr().out
    assert "evaluation: cases=1" in output
    assert "secret" in output and "reps=0..1" in output
    assert "B-A" in output and "1.500" in output
    with pytest.raises(ValueError, match="duplicate"):
        stats.summarize([*rows, rows[0]])
