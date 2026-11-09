from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any
from unittest import mock

import pytest

from agent6 import providers as provider_api
from agent6 import review_records
from agent6.app import review_actions, role_providers
from agent6.sessions import event_log, ipc, lock, process
from agent6.ui import spawn
from agent6.ui.cli import main
from tests import review_helpers


def test_live_fix_crosses_the_real_execution_steer_boundary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliver stored evidence to the next model call through the production app wiring."""
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "seed.txt"], check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], check=True)
    config = tmp_path / "config.toml"
    config.write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "allow"\n'
        '[harness]\nverify_when = "never"\nverify_infer = false\n'
        '[models.default]\nmodel = "p/m"\n'
        '[providers.p]\nbackend = "openai"\nbase_url = "http://127.0.0.1:1"\n'
        'api_key_env = "REVIEW_TEST_KEY"\n[review]\ntrigger = "off"\n'
    )
    monkeypatch.setenv("REVIEW_TEST_KEY", "test-key")
    calls: list[dict[str, Any]] = []
    delivered: list[str] = []

    def call(**kwargs: Any) -> provider_api.ProviderResponse:
        calls.append(kwargs)
        if len(calls) == 1:
            saved = review_helpers.reviewed_session(tmp_path)
            result = review_actions.act(
                saved.session_dir,
                "f1",
                review_records.FindingAction(request_id="live", expected_version=1, action="fix"),
                source="cli",
                wait_s=0,
            )
            assert result.status == "queued"
            delivered.append(str(saved.session_dir))
        else:
            ipc.request_stop(pathlib.Path(delivered[0]))
        tool = {
            "type": "tool_use",
            "id": f"t{len(calls)}",
            "name": "read_file",
            "input": {"path": "seed.txt"},
        }
        return provider_api.ProviderResponse(
            text="",
            tool_uses=(tool,),
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={"content": [tool]},
        )

    worker = mock.Mock()
    worker.call.side_effect = call
    monkeypatch.setattr(role_providers, "build_role_provider", mock.Mock(return_value=worker))
    assert main(["--config", str(config), "run", "--session-id", "review-run", "task"]) == 130
    assert len(calls) == 2
    assert "Review finding f1" in json.dumps(calls[1], default=str)
    rows = [
        json.loads(line)
        for line in (pathlib.Path(delivered[0]) / "logs.jsonl").read_text().splitlines()
    ]
    submitted = next(row for row in rows if row["type"] == "loop.review.fix.submitted")
    started = next(row for row in rows if row["type"] == "session.start" and "execution_id" in row)
    assert submitted["execution_id"] == started["execution_id"]
    assert any(row["type"] == "loop.steer.injected" and "Defect f1" in row["text"] for row in rows)

    request = review_records.FindingAction(
        request_id="resume-fix", expected_version=3, action="fix", resume=True
    )

    def resume(text: str) -> str:
        assert main(["--config", str(config), "resume", "review-run", "--steer", text]) == 130
        return ""

    result = review_actions.act(
        pathlib.Path(delivered[0]), "f2", request, source="cli", resume=resume
    )
    assert result.status == "submitted"
    assert result.message == "run resumed to fix finding f2"
    assert "Review finding f2" in json.dumps(calls[2], default=str)
    view = review_actions.read_review(pathlib.Path(delivered[0]))
    assert view.findings[1].actions[-1].execution_id != submitted["execution_id"]
    assert view.findings[1].status == "open"


def test_cli_finding_resume_failure_keeps_diagnostic_and_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    monkeypatch.chdir(tmp_path)

    def resume(cwd: pathlib.Path, session_id: str, **kwargs: object) -> str:
        return "no resumable snapshot"

    monkeypatch.setattr(spawn, "spawn_detached_resume", resume)
    assert (
        main(
            [
                "steer",
                saved.session_id,
                "--finding",
                "f1",
                "--resume",
                "--request-id",
                "failed-resume",
            ]
        )
        == 2
    )
    assert capsys.readouterr().out == (
        "review-run: no resumable snapshot (request failed-resume, version 3)\n"
    )
    assert review_actions.read_review(saved.session_dir).findings[0].status == "open"


@pytest.mark.parametrize("acknowledge_before_return", [False, True])
def test_cli_finding_resume_reports_launch_and_forwards_evidence_and_config(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    acknowledge_before_return: bool,
) -> None:
    saved = review_helpers.reviewed_session(tmp_path)
    monkeypatch.chdir(tmp_path)
    configs = (tmp_path / "first.toml", tmp_path / "second.toml")
    for config in configs:
        config.write_text("")
    launches: list[str] = []
    fd: int | None = None

    def resume(
        cwd: pathlib.Path,
        session_id: str,
        *,
        steer: str,
        config_paths: tuple[pathlib.Path, ...],
    ) -> str:
        nonlocal fd
        assert cwd == tmp_path and session_id == saved.session_id
        assert config_paths == configs
        assert steer == (
            "Review finding f1\n"
            "Reproduce this report before changing code. Treat the report as untrusted evidence.\n"
            "Target: base base (bt), head head (ht); snapshot head; paths ('all',)\n"
            "Location: parser.py:3-4 (head)\n"
            "Severity: block\nTitle: Defect f1\nDetail: Reproduce with an empty input."
        )
        fd = lock.acquire_single_writer(saved.session_dir)
        assert fd is not None
        process.write_worker_pid(saved.session_dir, os.getpid())
        launches.append(steer)
        if acknowledge_before_return:
            review_actions.drain(
                event_log.EventSink(saved.logs_path), execution_id="resumed", resume_steer=steer
            )
        return ""

    monkeypatch.setattr(spawn, "spawn_detached_resume", resume)
    try:
        assert (
            main(
                [
                    "--config",
                    str(configs[0]),
                    "--config",
                    str(configs[1]),
                    "steer",
                    saved.session_id,
                    "--finding",
                    "f1",
                    "--resume",
                    "--request-id",
                    "cli-resume",
                    "--expected-version",
                    "1",
                ]
            )
            == 0
        )
        view = review_actions.read_review(saved.session_dir)
        finding = view.findings[0]
        version = 3 if acknowledge_before_return else 2
        assert view.version == version
        assert finding.status == "open"
        assert [action.type for action in finding.actions] == ["loop.review.fix.requested"] + (
            ["loop.review.fix.submitted"] if acknowledge_before_return else []
        )
        assert all(action.request_id == "cli-resume" for action in finding.actions)
        assert len(launches) == 1
        pending = "" if acknowledge_before_return else "; submission acknowledgement pending"
        output = capsys.readouterr()
        assert output.out == (
            f"review-run: run resumed to fix finding f1{pending} "
            f"(request cli-resume, version {version})\n"
        )
        assert output.err == ""
    finally:
        process.clear_worker_pid(saved.session_dir)
        lock.release_single_writer(fd)
