"""Lane snapshots preserve resolved sampling without inheriting child model overrides."""

from __future__ import annotations

import pathlib
import tomllib

import pytest

from agent6 import kinds, lane_records, paths
from agent6.app import lane_launch, stamps, worktrees
from agent6.app.session import setup as session_setup
from agent6.config import Config, ConfigError, RoleModel
from tests import lane_env


@pytest.fixture
def lane(tmp_path: pathlib.Path) -> lane_records.LaneSpec:
    return lane_records.LaneSpec(
        lane=1,
        session_id="fan-l1",
        worktree=tmp_path / "lane-1",
        settings=kinds.ModelSettings(kinds.ModelRoute("p", "lane")),
    )


def resolved_config() -> Config:
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {
                "default": {"model": "p/base", "effort": "high", "temperature": 0.7},
                "planner": {"model": "p/plan", "effort": "off", "temperature": 0.0},
                "reviewer": {"effort": "low", "temperature": 0.2},
            },
        }
    )
    # A resolved in-memory selection, not a new TOML spelling for None.
    return cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"worker": RoleModel(effort=None, temperature=None)}
            )
        }
    )


def test_four_explicit_routes_without_default_are_not_runnable(
    lane: lane_records.LaneSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All executable roles configured still require the public default-route baseline."""
    cfg = Config.model_validate(
        {
            "providers": {"p": {"backend": "openai", "base_url": "http://localhost:1/v1"}},
            "models": {
                role: {"model": f"p/{role}"} for role in kinds.MODEL_ROLES if role != "default"
            },
        }
    )
    written = worktrees.write_lane_config(
        cfg, lane, explicit_leaves=frozenset(), into=lane.worktree.parent / "launch"
    )
    assert "default" not in tomllib.loads(written.path.read_text())["models"]
    for name in (lane_launch.ENV, "AGENT6_SUBRUN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError, match=r"No models\.default\.model configured"):
        session_setup.load_session_config(lane.worktree, (written.path,), mode="run")
    lane_env.mark_lane(monkeypatch, written.path.parent, config=written.path)
    with pytest.raises(ConfigError, match="lane snapshot"):
        session_setup.load_session_config(lane.worktree, (written.path,), mode="run")


def test_snapshot_writes_resolved_roles(lane: lane_records.LaneSpec) -> None:
    path = worktrees.write_lane_config(
        resolved_config(), lane, explicit_leaves=frozenset(), into=lane.worktree.parent / "launch"
    ).path
    models = tomllib.loads(path.read_text())["models"]
    assert models == {
        "default": {"model": "p/base"},
        "worker": {"model": "p/lane"},
        "planner": {"model": "p/plan", "effort": "off", "temperature": 0.0},
        "reviewer": {"model": "p/base", "effort": "low", "temperature": 0.2},
        "helper": {"model": "p/base", "effort": "high", "temperature": 0.7},
    }


def test_snapshot_reloads_resolved_sampling(
    lane: lane_records.LaneSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = resolved_config()
    path = worktrees.write_lane_config(
        cfg, lane, explicit_leaves=frozenset(), into=lane.worktree.parent / "launch"
    ).path
    lane_env.mark_lane(monkeypatch, path.parent, config=path)
    effective = session_setup.load_session_config(lane.worktree, (path,), mode="run")
    worker = effective.config.models.resolve("worker")
    assert worker is not None
    assert worker.effort is None
    assert worker.temperature == 0.0  # TOML cannot preserve the selected None.
    assert worker.model == "lane"
    for role in ("planner", "reviewer", "helper"):
        assert effective.config.models.resolve(role) == cfg.models.resolve(role)


@pytest.mark.parametrize("internal", [True, False])
def test_only_internal_snapshot_discards_global_model_contributions(
    lane: lane_records.LaneSpec, monkeypatch: pytest.MonkeyPatch, internal: bool
) -> None:
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_text(
        '[models.default]\nmodel = "p/global"\neffort = "high"\ntemperature = 0.7\n'
        '[models.worker]\neffort = "max"\n'
        '[sandbox]\nisolation = "partial"\nprotect_git = "off"\n'
        "[budget]\nmax_usd = 19.0\n"
    )
    # Already-resolved snapshot shape isolates the loader loss from the writer loss.
    path = lane.worktree.parent / "lane-1-config.toml"
    path.write_text(
        '[providers.p]\nbackend = "openai"\nbase_url = "http://localhost:1/v1"\n'
        + "".join(f'[models.{r}]\nmodel = "p/base"\n' for r in kinds.MODEL_ROLES)
        + '[sandbox]\nprotect_git = "on"\n[budget]\nmax_usd = 3.5\n'
    )
    monkeypatch.setenv("AGENT6_SUBRUN", "1")
    if internal:
        lane_env.mark_lane(monkeypatch, path.parent, config=path)
    effective = session_setup.load_session_config(lane.worktree, (path,), mode="run")
    worker = effective.config.models.resolve("worker")
    assert worker is not None
    assert worker.effort == (None if internal else "max")
    assert worker.temperature == (0.0 if internal else 0.7)
    assert effective.sources["models.worker.effort"] == ("default" if internal else "global")
    assert effective.config.sandbox.isolation == "partial"
    assert effective.sources["sandbox.isolation"] == "global"
    assert effective.config.sandbox.protect_git == "on"
    assert effective.sources["sandbox.protect_git"] == f"flag ({path})"
    assert "sandbox.home" not in effective.explicit_leaves
    assert effective.config.budget.max_usd == 3.5
    assert effective.sources["budget.max_usd"] == f"flag ({path})"


@pytest.mark.parametrize("temperature", [None, 0.0, 0.6])
def test_snapshot_notes_only_unrepresentable_temperatures(
    lane: lane_records.LaneSpec, temperature: float | None
) -> None:
    cfg = resolved_config()
    cfg = cfg.model_copy(
        update={
            "models": cfg.models.model_copy(update={"worker": RoleModel(temperature=temperature)})
        }
    )
    # A route selector keeps the worker's sampling: the selected temperature rides along.
    routed = lane_records.LaneSpec(
        lane=1, session_id="fan-l1", worktree=lane.worktree, settings=cfg.lane_settings("p/lane")
    )
    written = worktrees.write_lane_config(
        cfg, routed, explicit_leaves=frozenset(), into=routed.worktree.parent / "launch"
    )
    assert written.notes == (
        ("models.worker.temperature: selected None; TOML key omitted; reload uses 0.0.",)
        if temperature is None
        else ()
    )


def test_snapshot_discards_preset_models_but_keeps_other_preset_values(
    lane: lane_records.LaneSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_text(
        '[agent6]\npreset = "custom"\n[presets.custom.models.worker]\neffort = "max"\n'
        '[presets.custom.sandbox]\nprotect_git = "off"\n'
    )
    written = worktrees.write_lane_config(
        resolved_config(), lane, explicit_leaves=frozenset(), into=lane.worktree.parent / "launch"
    )
    lane_env.mark_lane(monkeypatch, written.path.parent, config=written.path)
    effective = session_setup.load_session_config(lane.worktree, (written.path,), mode="run")
    worker = effective.config.models.resolve("worker")
    assert worker is not None and worker.effort is None
    assert effective.sources["models.worker.effort"] == "default"
    assert effective.config.sandbox.protect_git == "off"
    assert effective.sources["sandbox.protect_git"] == "preset"
    assert "custom" in effective.presets


@pytest.mark.parametrize("raw", ["{}", "[1]", "[null]", "not json", '"note"'])
def test_lineage_refuses_malformed_config_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, raw: str
) -> None:
    """A launch file whose notes are not a list of strings refuses the lineage read."""
    path = tmp_path / "lane-1.json"
    lane_launch.write_private(
        path,
        '{"worktree": "/x/lane-1", "session_id": "fan-l1", "task": "t",'
        ' "config": "/x/lane-1-config.toml", "parent": "co", "group": "fan", "lane": 1,'
        f' "notes": {raw}}}',
    )
    monkeypatch.setenv(lane_launch.ENV, str(path))
    with pytest.raises(ConfigError, match="lane launch"):
        stamps.parallel_lineage()


@pytest.mark.parametrize(
    ("broken", "refusal"),
    [
        ("path", "lane snapshot"),
        ("envelope", "lane launch"),
        ("lineage", "lane launch"),
        ("models", "lane snapshot"),
        ("default_sampling", "lane snapshot"),
    ],
)
def test_marked_snapshot_refuses_mismatch_or_malformed_models(
    lane: lane_records.LaneSpec, monkeypatch: pytest.MonkeyPatch, broken: str, refusal: str
) -> None:
    path = worktrees.write_lane_config(
        resolved_config(), lane, explicit_leaves=frozenset(), into=lane.worktree.parent / "launch"
    ).path
    envelope = lane_env.mark_lane(monkeypatch, path.parent, config=path)
    if broken == "path":
        lane_env.mark_lane(monkeypatch, path.parent, config=path.with_suffix(".other"))
    elif broken == "envelope":
        monkeypatch.setenv(lane_launch.ENV, str(path.parent / "missing.json"))
    elif broken == "lineage":
        envelope.write_text(envelope.read_text().replace('"lane": 1', '"lane": 0'))
    elif broken == "models":
        path.write_text(
            '[providers.p]\nbackend = "openai"\nbase_url = "http://localhost:1/v1"\n'
            '[models.default]\nmodel = "p/base"\n'
        )
    else:
        path.write_text(
            '[providers.p]\nbackend = "openai"\nbase_url = "http://localhost:1/v1"\n'
            '[models.default]\nmodel = "p/base"\neffort = "max"\n'
            + "".join(
                f'[models.{r}]\nmodel = "p/base"\n' for r in kinds.MODEL_ROLES if r != "default"
            )
        )
    with pytest.raises(ConfigError, match=refusal):
        session_setup.load_session_config(lane.worktree, (path,), mode="run")
