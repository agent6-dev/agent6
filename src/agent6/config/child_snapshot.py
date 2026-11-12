"""Serialize resolved child models and load them without lower-layer model inheritance."""

from __future__ import annotations

import dataclasses
import pathlib

from agent6 import kinds
from agent6.config import harness_config, layer, layer_data, layer_presets, validation
from agent6.config import model as config_model


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """The child's TOML and the sampling values it cannot represent."""

    text: str
    notes: tuple[str, ...]


def build(cfg: config_model.Config, *, explicit_leaves: frozenset[str]) -> Snapshot:
    """Materialize resolved roles, keeping only the route in `models.default`.

    Args:
        cfg: The child config, including its worker route override.
        explicit_leaves: The parent's explicit settings, including resets to defaults.

    Returns:
        The selective config serialization and immutable loss notes.
    """
    roles: dict[str, harness_config.RoleModel] = {}
    notes: list[str] = []
    for role in kinds.MODEL_ROLES:
        resolved = cfg.models.resolve(role)
        if resolved is None:
            continue
        if role == "default":
            roles[role] = harness_config.RoleModel(provider=resolved.provider, model=resolved.model)
            continue
        roles[role] = resolved.model_copy(
            update={"effort": resolved.effort, "temperature": resolved.temperature}
        )
        if resolved.temperature is None:
            notes.append(
                f"models.{role}.temperature: selected None; TOML key omitted; reload uses 0.0."
            )
    snapshot = cfg.model_copy(update={"models": cfg.models.model_copy(update=roles)})
    return Snapshot(layer.materialize(snapshot, explicit_leaves=explicit_leaves), tuple(notes))


def marked_path(config_paths: tuple[pathlib.Path, ...], snapshot: pathlib.Path) -> pathlib.Path:
    """Require the child's sole `--config` to be the snapshot its envelope names.

    Args:
        config_paths: The child command's config paths.
        snapshot: The snapshot the child's envelope names.

    Returns:
        The snapshot path as the command gave it.

    Raises:
        ConfigError: The command's config paths are not that one file.
    """
    if len(config_paths) != 1 or config_paths[0].resolve() != snapshot.resolve():
        raise validation.ConfigError(
            "internal child snapshot requires the child's sole --config to be the snapshot its"
            " envelope names"
        )
    return config_paths[0]


def load(
    repo_root: pathlib.Path, path: pathlib.Path, *, preset: str = ""
) -> layer_data.EffectiveConfig:
    """Replace models with the snapshot after preset selection; merge everything else normally.

    Args:
        repo_root: The child's workspace.
        path: The explicitly identified snapshot file.
        preset: The invocation's preset flag.

    Returns:
        The validated config and provenance, with discarded model sources removed.

    Raises:
        ConfigError: A layer or the snapshot is malformed.
    """
    layers = layer_data.discover_layers(repo_root, (path,))
    models = layers[-1].data.get("models")
    if (
        not isinstance(models, dict)
        or set(models) != set(kinds.MODEL_ROLES)
        or any(not isinstance(entry, dict) or not entry.get("model") for entry in models.values())
        or set(models["default"]) != {"model"}
    ):
        raise validation.ConfigError(
            "internal child snapshot must contain every resolved model role and only the"
            " route in models.default"
        )
    presets = layer_presets.preset_names(layers)
    layers = layer_presets.apply_preset(layers, preset)
    layers = [
        source
        if source.name == "flag" and source.path == path
        else dataclasses.replace(
            source, data={k: v for k, v in source.data.items() if k != "models"}
        )
        for source in layers
    ]
    return layer_data.effective_from_layers(
        layers, source="(internal child snapshot)", presets=presets
    )
