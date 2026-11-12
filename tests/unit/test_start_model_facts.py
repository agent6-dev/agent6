from __future__ import annotations

import asyncio
import io
import json
import pathlib
from typing import Any

import httpx2
import pytest

from agent6 import paths
from agent6.app import stamps
from agent6.config import Config, layer
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import console_render, sessions_show
from agent6.ui.tui import app as tui_app
from agent6.ui.tui import dashboard_header
from agent6.viewmodel import policy, wire


@pytest.fixture
def recorded_run(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    config = paths.repo_config_path(repo)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.chatgpt]\nbackend = "chatgpt"\n'
        '[models.default]\nmodel = "chatgpt/gpt-6-astra"\n'
        '[models.reviewer]\neffort = "xhigh"\n',
        encoding="utf-8",
    )
    effective = layer.load_effective(repo)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="effort-run")
    layout.ensure()
    stamps.write_session_manifest(
        layout,
        session_id=layout.session_id,
        user_task="check effort",
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=effective.config,
        config_sources=effective.sources,
    )
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "check effort"}) + "\n",
        encoding="utf-8",
    )
    # A viewer must use the record even after the operator changes the config.
    config.write_text('[models.default]\neffort = "low"\n', encoding="utf-8")
    return layout.session_dir


def _assert_efforts(text: str) -> None:
    assert "worker chatgpt/gpt-6-astra effort provider default" in text
    assert "reviewer chatgpt/gpt-6-astra effort xhigh (repo)" in text
    assert "helper chatgpt/gpt-6-astra effort provider default" in text


def test_start_prints_unset_worker_and_explicit_reviewer_effort(recorded_run: pathlib.Path) -> None:
    """Sharing a route does not hide a role's different effort."""
    buf = io.StringIO()
    view = console_render.ConsoleView(
        buf, color=False, policy=lambda: policy.session_policy(recorded_run).line()
    )
    try:
        view.feed({"type": "session.start", "mode": "run", "user_task": "check effort"})
    finally:
        view.close()
    _assert_efforts(buf.getvalue())


def test_recorded_start_models_carry_effort(recorded_run: pathlib.Path) -> None:
    """The manifest distinguishes an omitted effort from an unrecorded one."""
    models = json.loads((recorded_run / "manifest.json").read_text(encoding="utf-8"))["models"]
    assert "effort" in models["driver"]
    assert models["driver"]["effort"] is None
    assert models["driver"]["effort_source"] == "provider default"
    assert models["reviewer"]["effort"] == "xhigh"
    assert models["reviewer"]["effort_source"] == "repo"
    assert models["helper"]["effort"] is None


def test_sessions_show_json_exposes_recorded_effort(
    recorded_run: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_show._cmd_status(recorded_run.name, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "models" in payload
    assert payload["models"]["driver"]["effort"] is None
    assert payload["models"]["driver"]["effort_source"] == "provider default"
    assert payload["models"]["reviewer"]["effort"] == "xhigh"


def test_sessions_show_prints_recorded_effort(
    recorded_run: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sessions_show._cmd_status(recorded_run.name) == 0
    _assert_efforts(capsys.readouterr().out)


def test_web_start_policy_reads_recorded_effort(recorded_run: pathlib.Path) -> None:
    line = wire.session_snapshot(recorded_run)["policy"]
    _assert_efforts(line)
    # The page keeps each setting together, so roles need separate wrap points.
    assert "worker chatgpt/gpt-6-astra effort provider default" in line.split(" · ")
    assert "reviewer chatgpt/gpt-6-astra effort xhigh (repo)" in line.split(" · ")


def test_dashboard_header_reads_recorded_effort(recorded_run: pathlib.Path) -> None:
    async def scenario() -> None:
        app = tui_app.Agent6TUI(recorded_run)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            header = app.screen.query_one(dashboard_header.RunHeader)
            _assert_efforts(str(header.content))

    asyncio.run(scenario())


def test_headless_launch_prints_effort_without_a_console_view(
    recorded_run: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-terminal launch prints the same recorded facts as a streaming console."""
    from agent6.models import cache
    from agent6.providers import transport
    from agent6.ui.cli import run

    config = paths.repo_config_path(pathlib.Path.cwd())
    config.write_text(
        '[sandbox]\nisolation = "none"\nrun_commands = "deny"\n'
        '[providers.local]\nbackend = "responses"\nauth_style = "none"\n'
        'base_url = "https://example.test/v1"\n'
        '[models.default]\nmodel = "local/gpt-6-astra"\n'
        '[models.reviewer]\neffort = "xhigh"\n',
        encoding="utf-8",
    )
    requests: list[dict[str, Any]] = []

    def respond(*_args: Any, **kwargs: Any) -> httpx2.Response:
        requests.append(json.loads(kwargs["content"]))
        return httpx2.Response(
            200,
            request=httpx2.Request("POST", "https://example.test/v1/responses"),
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Checked."}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )

    def list_models(*_args: object, **_kwargs: object) -> list[str]:
        return ["gpt-6-astra"]

    monkeypatch.setattr(cache, "list_models", list_models)
    monkeypatch.setattr(transport, "http_post", respond)
    assert run._cmd_run((), "Check effort", mode="ask") == 0
    assert requests and "reasoning" not in requests[-1]
    output = capsys.readouterr().err
    assert "worker local/gpt-6-astra effort provider default" in output
    assert "reviewer local/gpt-6-astra effort xhigh (repo)" in output
    assert "helper local/gpt-6-astra effort provider default" in output


@pytest.mark.parametrize("mode, role", [("run", "worker"), ("plan", "planner")])
def test_resume_records_inheritance_and_each_review_seat(
    recorded_run: pathlib.Path, mode: str, role: str
) -> None:
    """A seat with its own route does not inherit the reviewer's effort."""
    cfg = Config.model_validate(
        {
            "providers": {"chatgpt": {"backend": "chatgpt"}},
            "models": {
                "default": {"model": "chatgpt/gpt-6-astra", "effort": "max"},
                "reviewer": {"effort": "xhigh"},
            },
            "review": {
                "trigger": "before_finish",
                "reviewers": [
                    {"focus": "security"},
                    {"focus": "bugs", "effort": "off"},
                    {"focus": "bugs", "model": "chatgpt/other"},
                ],
            },
        }
    )
    # A plan resume keeps its original mode, as the lifecycle does.
    from agent6.sessions import manifest

    saved = manifest.read_manifest(recorded_run)
    stamps.write_manifest(recorded_run / "manifest.json", saved.model_copy(update={"mode": mode}))
    stamps.stamp_execution(
        recorded_run,
        cfg,
        mode,
        "full",
        config_sources={
            "models.default.effort": "global",
            "models.reviewer.effort": "repo",
            "review.reviewers": "repo",
        },
    )
    line = policy.session_policy(recorded_run).line()
    assert f"{role} chatgpt/gpt-6-astra effort max (models.default.effort, global)" in line
    assert "helper chatgpt/gpt-6-astra effort max (models.default.effort, global)" in line
    assert (
        "review seat 1 (security) chatgpt/gpt-6-astra effort xhigh (models.reviewer.effort, repo)"
        in line
    )
    assert "review seat 2 (bugs) chatgpt/gpt-6-astra effort off (repo)" in line
    # A pinned seat inherits the reviewer role's effort like an unpinned one.
    assert "review seat 3 (bugs) chatgpt/other effort xhigh (models.reviewer.effort, repo)" in line
    other_role = "worker" if role == "planner" else "planner"
    assert f"{other_role} chatgpt/" not in line


@pytest.mark.parametrize(
    "backend, model, expected",
    [
        ("openai", "deepseek/deepseek-r1", "low (backend default)"),
        ("openai", "plain-model", "provider default"),
        ("anthropic", "claude-sonnet-5", "off (backend default)"),
        ("responses", "gpt-6-astra", "provider default"),
        ("claude_code", "claude-sonnet-5", "provider default"),
    ],
)
def test_start_uses_the_provider_effort_rule(
    recorded_run: pathlib.Path, backend: str, model: str, expected: str
) -> None:
    cfg = Config.model_validate(
        {
            "providers": {
                "p": {
                    "backend": backend,
                    **({"base_url": "https://example.test/v1"} if backend == "responses" else {}),
                }
            },
            "models": {"default": {"model": f"p/{model}"}},
            "review": {"trigger": "off"},
        }
    )
    stamps.stamp_execution(recorded_run, cfg, "run", "full")
    line = policy.session_policy(recorded_run).line()
    assert f"worker p/{model} effort {expected}" in line
    assert "review seat" not in line


def test_old_model_facts_do_not_claim_provider_default(recorded_run: pathlib.Path) -> None:
    from agent6.sessions import manifest

    (recorded_run / "manifest.json").write_text(
        '{"mode": "run", "models": {"driver": {"provider": "p", "model": "m"}}}',
        encoding="utf-8",
    )
    assert policy.model_lines(manifest.read_manifest(recorded_run)) == (
        "worker p/m effort not recorded",
    )
