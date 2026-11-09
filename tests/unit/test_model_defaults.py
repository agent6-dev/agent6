from __future__ import annotations

import json
import pathlib
import subprocess
import tomllib
from typing import Any
from unittest import mock

import pytest

from agent6 import kinds, paths
from agent6.app import preflight, role_providers, session_runtime, session_setup
from agent6.config import layer, layer_data, model, validation
from agent6.harness import compaction_driver, compaction_elision, compaction_window
from agent6.machine import schema, semantics
from agent6.providers import provider_types
from agent6.sessions import event_log
from agent6.ui.cli import entry
from agent6.viewmodel import config_view
from tests.unit import test_machine_cli, test_machine_model, test_run_review


@pytest.mark.parametrize("role", ["worker", "planner", "reviewer", "helper"])
def test_roles_inherit_each_unset_leaf_only_from_default(role: kinds.RoleName) -> None:
    """A route override keeps default sampling settings, even beside a worker override."""
    cfg = model.Config.model_validate(
        {
            "models": {
                "default": {"model": "p/base", "effort": "high", "temperature": 0.7},
                "worker": {"model": "p/worker", "effort": "low"},
                role: {"model": "p/override"},
            }
        }
    )
    resolved = cfg.models.resolve(role)
    assert resolved is not None
    assert (resolved.model, resolved.effort, resolved.temperature) == ("override", "high", 0.7)
    copied = cfg.with_budget_overrides(max_usd=1)
    assert copied.models.resolve(role) == resolved


def test_unset_roles_ignore_worker_overrides() -> None:
    """A worker override supplies no field to another role."""
    cfg = model.Config.model_validate(
        {
            "models": {
                "default": {"model": "p/base", "effort": "high", "temperature": 0.7},
                "worker": {"model": "p/work", "effort": "off", "temperature": 0.0},
            }
        }
    )
    for role in ("helper", "planner", "reviewer"):
        assert cfg.models.resolve(role) == cfg.models.resolve("default")
    worker = cfg.models.resolve("worker")
    assert worker is not None and (worker.effort, worker.temperature) == ("off", 0.0)


def test_effort_only_override_keeps_default_route_through_copies() -> None:
    """A partial planner survives config copies without materializing inherited fields."""
    cfg = model.Config.model_validate(
        {"models": {"default": {"model": "p/base"}, "planner": {"effort": "max"}}}
    ).with_verify_command(("true",))
    route = cfg.models.resolve("planner")
    assert route is not None and (route.model, route.effort) == ("base", "max")
    assert cfg.models.source_role("planner") == "default"
    assert cfg.model_dump()["models"]["planner"] == {"effort": "max"}


@pytest.mark.parametrize("role", ["worker", "planner", "reviewer", "helper"])
def test_a_role_cannot_replace_the_required_default(role: kinds.RoleName) -> None:
    """A complete role override cannot make a config without a default runnable."""
    cfg = model.Config.model_validate(
        {"providers": {"p": {"backend": "openai"}}, "models": {"worker": {"model": "p/m"}}}
    )
    with pytest.raises(validation.ConfigError, match=r"models.default") as exc:
        cfg.require_runnable(role)
    assert "models.worker" in str(exc.value)
    assert "agent6 model default PROVIDER/MODEL" in str(exc.value)


@pytest.mark.parametrize("command", ["run", "ask", "plan", "machine", "create", "review"])
def test_entry_points_refuse_missing_default(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    """Every model entry point names the missing baseline before calling a provider."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    config = paths.global_config_path()
    config.parent.mkdir(parents=True)
    config.write_text('[providers.p]\nbackend = "openai"\n[models.worker]\nmodel = "p/m"\n')
    args = [command, "task"]
    if command == "machine":
        machine = tmp_path / "m.asm.toml"
        machine.write_text(test_machine_cli.AGENT_RUN_MACHINE)
        subprocess.run(["git", "add", str(machine)], check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "machine",
            ],
            check=True,
        )
        args = ["machine", "run", str(machine)]
    elif command == "create":
        args = ["machine", "create", "task"]
    elif command == "review":
        test_run_review._write_session(tmp_path)
        args = ["sessions", "review", "run-AAAA11"]
    with mock.patch.object(role_providers, "build_role_provider", side_effect=AssertionError):
        assert entry.cli_main(args) == 2
    error = capsys.readouterr().err
    assert "models.default" in error
    assert "agent6 model default PROVIDER/MODEL" in error


def test_config_show_attributes_inherited_leaves_to_the_default_layer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inherited leaves keep their actual origin rather than the override's layer."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text(
        '[models.default]\nmodel = "p/base"\neffort = "high"\ntemperature = 0.7\n'
    )
    repo_path = paths.repo_config_path(tmp_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    repo_path.write_text('[models.planner]\neffort = "max"\n[models.worker]\nmodel = "p/worker"\n')
    eff = layer.load_effective(tmp_path)
    shown = json.loads(config_view.render_show(eff, as_json=True))
    for role in ("worker", "planner", "reviewer", "helper"):
        assert shown[f"models.{role}.temperature"]["value"] == 0.7
        assert shown[f"models.{role}.temperature"]["source"] == "global"
    assert shown["models.planner.model"]["value"] == "p/base"
    assert shown["models.planner.model"]["source"] == "global"
    assert shown["models.planner.effort"]["source"] == "repo"
    assert shown["models.worker.model"]["source"] == "repo"


@pytest.mark.parametrize("target", [[], ["--repo"]])
def test_model_all_writes_default_and_four_overrides(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    target: list[str],
) -> None:
    """All writes five route tables at the selected layer."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert entry.main(["model", "all", "p/m", *target]) == 0
    path = paths.repo_config_path(tmp_path) if target else paths.global_config_path()
    assert tomllib.loads(path.read_text())["models"] == {
        role: {"model": "p/m"} for role in ("default", "worker", "planner", "reviewer", "helper")
    }


@pytest.mark.parametrize("key", ['model = "p/m"', 'effort = "high"', "temperature = 0.2"])
def test_machine_legacy_key_remedy_names_default(tmp_path: pathlib.Path, key: str) -> None:
    """Legacy state keys direct the operator to the state's default table."""
    path = tmp_path / "m.asm.toml"
    path.write_text(
        test_machine_model.VALID_MACHINE.replace(
            "[states.classify]\n", f"[states.classify]\n{key}\n"
        )
    )
    with pytest.raises(schema.MachineError, match=r"states.classify.config.models.default"):
        semantics.load_machine(path)


@pytest.mark.parametrize("role", kinds.MODEL_ROLES)
def test_machine_state_overlays_each_role(tmp_path: pathlib.Path, role: kinds.RoleName) -> None:
    """A state's partial role table inherits the operator's default route."""
    path = tmp_path / "m.asm.toml"
    path.write_text(
        test_machine_model.VALID_MACHINE
        + f'\n[states.classify.config.models.{role}]\neffort = "max"\n'
    )
    state = semantics.load_machine(path).states["classify"]
    assert isinstance(state, schema.AgentState)
    effective = layer_data.effective_from_layers(
        [
            layer_data.Layer("global", None, {"models": {"default": {"model": "p/base"}}}),
            layer_data.Layer("state", path, state.config),
        ],
        source="machine",
        presets=[],
    )
    resolved = effective.config.models.resolve(role)
    assert resolved is not None and (resolved.model, resolved.effort) == ("base", "max")
    assert effective.sources[f"models.{role}.effort"] == "state"


def test_helper_jobs_do_not_use_the_reviewer_route(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarising, gist distillation, revision and verify inference use the helper route."""
    cfg = model.Config.model_validate(
        {
            "providers": {"p": {"backend": "openai"}},
            "models": {
                "default": {"model": "p/base"},
                "helper": {"model": "p/help"},
                "reviewer": {"model": "p/review"},
            },
            "prompt": {"revise_prompt": "auto"},
            "review": {"trigger": "off"},
        }
    )
    sink = provider_types.TranscriptSink(tmp_path / "transcripts", objects_dir=tmp_path / "objects")
    events = event_log.EventSink(tmp_path / "events.jsonl")
    budget = session_setup.budget_tracker(cfg)
    built: list[tuple[str, str]] = []
    provider = mock.Mock()
    provider.call.return_value = provider_types.ProviderResponse(
        text="[]",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )

    def build(config: model.Config, role: kinds.RoleName, **kwargs: Any) -> Any:
        route = config.models.resolve(role)
        assert route is not None
        built.append((role, route.model))
        return provider

    monkeypatch.setattr(role_providers, "build_role_provider", build)
    session = session_runtime.build_session_providers(
        cfg, role="worker", events=events, transcript_sink=sink, stream_text=False
    )
    assert session.summariser_provider is not None
    compactor = compaction_driver.Compactor(
        settings=compaction_window.CompactionSettings(summariser=session.summariser_provider),
        provider=session.provider,
        curator=None,
        mode="run",
        dag_available=False,
        decisions=lambda: "",
        compact_requested=lambda: None,
        compact_clear=lambda: None,
        log=lambda _: None,
        emit=lambda *_a, **_kw: None,
        emit_graph_snapshot=lambda: None,
    )
    compactor.distill_gists((compaction_elision.GistRequest(path="a.py", content="print(1)"),))
    assert provider.call.call_count == 1
    role_providers.build_prompt_reviser_provider(
        cfg, transcript_sink=sink, budget=budget, events=events
    )
    (tmp_path / "AGENTS.md").write_text("Tests use the project's custom runner.\n")
    preflight.infer_verify_if_unset(
        cfg, tmp_path, mode="run", events=events, transcript_sink=sink, budget=budget
    )
    assert built == [("worker", "base"), ("helper", "help"), ("helper", "help"), ("helper", "help")]
