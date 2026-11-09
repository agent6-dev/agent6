from __future__ import annotations

import pathlib
import tomllib

import pytest

from agent6 import paths
from agent6.config import layer, layer_data, model, validation, write
from agent6.machine import schema, semantics
from agent6.ui.cli import entry
from agent6.ui.web import model as web_model
from tests.unit import test_machine_model


@pytest.mark.parametrize("role", ["worker", "planner", "reviewer"])
def test_old_role_provider_names_replacement(tmp_path: pathlib.Path, role: str) -> None:
    """Each role refuses the split route with its qualified replacement."""
    path = tmp_path / "config.toml"
    path.write_text(f'[models.{role}]\nprovider = "chatgpt"\nmodel = "gpt-6-astra"\n')
    with pytest.raises(validation.ConfigError) as exc:
        model.load_config(path)
    assert (
        f'[models.{role}] provider and model are one setting now: model = "chatgpt/gpt-6-astra"'
        in str(exc.value)
    )


def test_old_reviewer_provider_names_replacement(tmp_path: pathlib.Path) -> None:
    """A roster entry refuses the split route rather than accepting an alias."""
    path = tmp_path / "config.toml"
    path.write_text('[[review.reviewers]]\nfocus = "tests"\nprovider = "p"\nmodel = "vendor/id"\n')
    with pytest.raises(validation.ConfigError) as exc:
        model.load_config(path)
    assert "review.reviewers" in str(exc.value)
    assert 'model = "p/vendor/id"' in str(exc.value)


def test_scalar_route_round_trip_and_reviewer_inheritance() -> None:
    """Serialization preserves one route leaf and the slash-containing model id."""
    cfg = model.Config.model_validate(
        {
            "models": {"worker": {"model": "p/vendor/id", "effort": "high"}},
            "review": {"reviewers": [{"focus": "tests", "model": "p/other"}, {"focus": "api"}]},
        }
    )
    worker = cfg.models.resolve("worker")
    assert worker is not None and (worker.provider, worker.model) == ("p", "vendor/id")
    pinned, inherited = (seat.resolve(worker) for seat in cfg.review.reviewers)
    assert pinned is not None and (pinned.provider, pinned.model) == ("p", "other")
    assert inherited == worker
    dumped = cfg.model_dump()
    assert dumped["models"]["worker"] == {
        "model": "p/vendor/id",
        "effort": "high",
    }
    assert model.Config.model_validate(dumped) == cfg
    assert "models.worker.provider" not in layer_data.config_leaves(cfg)
    assert layer_data.config_leaves(cfg)["models.worker.model"] == "p/vendor/id"


def test_scalar_route_set_show_payload_and_unset(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI writes, provenance and the web payload share a single qualified leaf."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "global"))
    assert entry.main(["config", "set", "models.worker.model", "p/vendor/id"]) == 0
    assert tomllib.loads(paths.global_config_path().read_text())["models"]["worker"] == {
        "model": "p/vendor/id"
    }
    assert entry.main(["config", "show", "models.worker"]) == 0
    shown = capsys.readouterr().out
    assert "p/vendor/id" in shown
    payload = web_model.config_payload(tmp_path)
    assert payload["models.worker.model"]["value"] == "p/vendor/id"
    assert payload["models.worker.model"]["source"] == "global"
    assert "models.worker.provider" not in payload
    assert write.unset_config_value(tmp_path, "models.worker.model").error is None
    assert layer.load_effective(tmp_path).config.models.worker is None


def test_a_masked_model_write_still_requires_a_qualified_route(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo override cannot hide a malformed model written to global config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "global"))
    assert write.set_config_value(tmp_path, "models.worker.model", "p/original") is None
    assert (
        write.set_config_value(tmp_path, "models.worker.model", "p/override", to_repo=True) is None
    )
    before = paths.global_config_path().read_text()
    error = write.set_config_value(tmp_path, "models.worker.model", "unqualified")
    assert error is not None and "provider/model" in error
    assert paths.global_config_path().read_text() == before


def test_membership_is_checked_after_layers_merge(tmp_path: pathlib.Path) -> None:
    """A route can name a provider supplied by a later layer."""
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text('[providers.a]\nbackend = "openai"\n[models.worker]\nmodel = "b/id"\n')
    second.write_text('[providers.b]\nbackend = "openai"\n')
    cfg = layer.load_effective(tmp_path, (first, second)).config
    assert cfg.models.worker is not None and cfg.models.worker.provider == "b"


def test_pinned_reviewer_membership_is_checked_after_merge(tmp_path: pathlib.Path) -> None:
    """A pinned reviewer must name an effective provider, not one from an isolated layer."""
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text(
        '[providers.a]\nbackend = "openai"\n'
        '[[review.reviewers]]\nfocus = "tests"\nmodel = "b/vendor/id"\n'
    )
    second.write_text('[providers.b]\nbackend = "openai"\n')
    with pytest.raises(validation.ConfigError, match=r"review.reviewers.*providers.b"):
        layer.load_effective(tmp_path, (first,))
    seat = layer.load_effective(tmp_path, (first, second)).config.review.reviewers[0]
    route = seat.resolve(None)
    assert route is not None and (route.provider, route.model) == ("b", "vendor/id")


def test_machine_state_scalar_route_and_old_key_refusal(tmp_path: pathlib.Path) -> None:
    """A pinned state uses the same qualified setting and migration refusal."""
    path = tmp_path / "m.asm.toml"
    prefix = test_machine_model.VALID_MACHINE + "\n[states.classify.config.models.worker]\n"
    path.write_text(prefix + 'model = "p/vendor/id"\n')
    state = semantics.load_machine(path).states["classify"]
    assert isinstance(state, schema.AgentState)
    assert state.config["models"]["worker"] == {"model": "p/vendor/id"}
    path.write_text(prefix + 'provider = "p"\nmodel = "vendor/id"\n')
    with pytest.raises(schema.MachineError) as exc:
        semantics.load_machine(path)
    assert "states.classify.config.models.worker" in str(exc.value)
    assert 'model = "p/vendor/id"' in str(exc.value)
