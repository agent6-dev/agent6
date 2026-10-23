# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve config presets and list their names, bodies and origins."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Iterable
from typing import Any

from agent6.config import _layer_data
from agent6.config import model as config_model

# The generated config table prints each note.
BUILTIN_PRESET_NOTES: dict[str, str] = {
    "no_review": "turns off the review panel",
    "review": "three review seats that can block the finish",
    "deep_review": "five repo-scope review seats that can block the finish",
}


BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "no_review": {
        "review": {"trigger": "off"},
    },
    "review": {
        "review": {
            "trigger": "before_finish",
            # The seats share one model, so a model quorum above one is unreachable.
            "block_on": "any",
            "reviewers": [{"focus": "security"}, {"focus": "correctness"}, {"focus": "tests"}],
            "concurrency": 3,
        },
    },
    "deep_review": {
        "review": {
            "trigger": "before_finish",
            "block_on": "any",
            "scope": "repo",
            "reviewers": [
                {"focus": "security"},
                {"focus": "correctness"},
                {"focus": "tests"},
                {"focus": "edge-cases"},
                {"focus": "over-engineering"},
            ],
            "concurrency": 5,
        },
    },
}


def resolve_preset(name: str, user_presets: dict[str, Any]) -> dict[str, Any]:
    """Return the overrides a preset applies.

    A user preset replaces a built-in of the same name wholesale.

    Args:
        name: The preset name; "none" selects nothing.
        user_presets: The `[presets.*]` tables, merged.

    Returns:
        The nested config dict; {} for "none".

    Raises:
        ConfigError: The name is unknown, or the user table is not a table.
    """
    config_model.preset_name(name)
    if name == "none":
        return {}
    if name in user_presets:
        prof = user_presets[name]
        if not isinstance(prof, dict):
            raise config_model.ConfigError(
                f"[presets.{name}] must be a table, got {type(prof).__name__}"
            )
        return prof
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    known = ", ".join(sorted({"none", *BUILTIN_PRESETS, *user_presets}))
    raise config_model.ConfigError(f"unknown preset {name!r}. Known presets: {known}.")


def preset_names(layers: Iterable[_layer_data.Layer]) -> list[str]:
    """Return the preset names a chooser offers.

    Args:
        layers: The layers whose `[presets.*]` tables add to the built-ins.

    Returns:
        The names, sorted and de-duplicated.
    """
    names: set[str] = {"none", *BUILTIN_PRESETS}
    for layer in layers:
        prof = layer.data.get("presets")
        if isinstance(prof, dict):
            names.update(prof.keys())
    return sorted(names)


def available_preset_names(
    repo_root: pathlib.Path, explicit_paths: tuple[pathlib.Path, ...] = ()
) -> list[str]:
    """Return the preset names over the repo's discovered layers.

    A config-read failure degrades to the built-ins alone, so a chooser never blocks on a
    bad config.

    Args:
        repo_root: The repo.
        explicit_paths: The `--config FILE` layers in command-line order.

    Returns:
        The names, sorted.
    """
    layers: list[_layer_data.Layer] = []
    with contextlib.suppress(Exception):
        layers = _layer_data.discover_layers(repo_root, explicit_paths)
    return preset_names(layers)


@dataclasses.dataclass(frozen=True, slots=True)
class PresetInfo:
    """One preset as `config presets` shows it.

    Attributes:
        name: The preset name.
        overrides: The nested config dict it applies; {} applies no overrides.
        origin: `built-in`, `global`, `repo` or `global+repo`.
        replaces_builtin: A user preset with a built-in's name replaces it whole.
    """

    name: str
    overrides: dict[str, Any]
    origin: str
    replaces_builtin: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PresetCatalog:
    """Everything `config presets` lists.

    Attributes:
        presets: The built-ins in definition order, then the user's, sorted.
        selected: The selected preset's name; "none" selects no preset.
        source: `repo`, `global` or `default`.
    """

    presets: tuple[PresetInfo, ...]
    selected: str
    source: str


def preset_catalog(
    repo_root: pathlib.Path, explicit_paths: tuple[pathlib.Path, ...] = ()
) -> PresetCatalog:
    """Build the `config presets` listing.

    A user table replaces a same-named built-in, so only the effective body is reported.
    An audit surface, so unlike `available_preset_names` a broken config raises.

    Args:
        repo_root: The repo.
        explicit_paths: The `--config FILE` layers in command-line order.

    Returns:
        The catalog.

    Raises:
        ConfigError: A layer is unreadable, a preset name is unknown or reserved, or a body
            is not a table.
    """
    layers = _layer_data.discover_layers(repo_root, explicit_paths)
    _, user = strip_presets(layers)
    origins: dict[str, str] = {}
    for layer in layers:
        prof = layer.data.get("presets")
        if not isinstance(prof, dict):
            continue
        for name, body in prof.items():
            if not isinstance(body, dict):
                raise config_model.ConfigError(
                    f"[presets.{name}] must be a table, got {type(body).__name__}"
                )
            origins[name] = f"{origins[name]}+{layer.name}" if name in origins else layer.name
    selected, source = select_preset(layers, "")
    resolve_preset(selected, user)
    builtins = tuple(
        PresetInfo(name, user[name], origins[name], replaces_builtin=True)
        if name in user
        else PresetInfo(name, body, "built-in", replaces_builtin=False)
        for name, body in BUILTIN_PRESETS.items()
    )
    customs = tuple(
        PresetInfo(name, user[name], origins[name], replaces_builtin=False)
        for name in sorted(user)
        if name not in BUILTIN_PRESETS
    )
    return PresetCatalog((*builtins, *customs), selected, source)


def own_preset(layer: _layer_data.Layer) -> str:
    """Return a layer's own raw top-level `preset`, or "".

    Args:
        layer: The layer.

    Returns:
        The preset name the layer itself sets; "" when it sets none.

    Raises:
        ConfigError: The value is empty or not a string.
    """
    raw = layer.data.get("preset")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        shape = "a [preset] table" if isinstance(raw, dict) else f"a {type(raw).__name__}"
        raise config_model.ConfigError(
            f"top-level `preset` in the {layer.name} config must be a preset name"
            f' string (e.g. preset = "review"), got {shape};'
            f" set it with `agent6 config set preset <name>`."
        )
    return config_model.preset_name(raw)


def select_preset(cleaned: list[_layer_data.Layer], preset_override: str) -> tuple[str, str]:
    """Pick the selected preset, most specific source first, never stacking.

    Args:
        cleaned: The layers with their `[presets.*]` tables stripped.
        preset_override: The `--preset` flag, or "".

    Returns:
        `(name, source)`: the flag, else the repo layer's own `preset`, else the global
        layer's, else `("none", "default")`.

    Raises:
        ConfigError: A layer's `preset` is not a string.
    """
    if preset_override:
        return preset_override, "flag"
    by_name = {layer.name: layer for layer in cleaned}
    for source in ("repo", "global"):
        layer = by_name.get(source)
        if layer is not None and (name := own_preset(layer)):
            return name, source
    return "none", "default"


def insert_preset(
    cleaned: list[_layer_data.Layer], preset: _layer_data.Layer, source: str
) -> list[_layer_data.Layer]:
    """Splice the preset layer in at the position its source dictates.

    Args:
        cleaned: The layers with their `[presets.*]` tables stripped.
        preset: The preset layer.
        source: `global` or `repo` put it right after that layer; `flag` puts it just below
            a `--config FILE` or machine overlay, else last.

    Returns:
        The layers with the preset inserted.
    """
    out: list[_layer_data.Layer] = []
    inserted = False
    for layer in cleaned:
        if source == "flag" and not inserted and layer.name in ("flag", "machine", "state"):
            out.append(preset)
            inserted = True
        out.append(layer)
        if not inserted and layer.name == source:  # source in {"global", "repo"}
            out.append(preset)
            inserted = True
    if not inserted:
        out.append(preset)
    return out


def strip_presets(
    layers: list[_layer_data.Layer],
) -> tuple[list[_layer_data.Layer], dict[str, Any]]:
    """Strip the `[presets.*]` tables out of the layers and merge them.

    Presets are meta-config the Config schema forbids, so every path that validates a layer
    strips them first.

    Args:
        layers: The layers.

    Returns:
        The stripped layers, and the user presets merged low to high.

    Raises:
        ConfigError: A layer defines the reserved `[presets.none]` table.
    """
    cleaned: list[_layer_data.Layer] = []
    user_presets: dict[str, Any] = {}
    for layer in layers:
        data = dict(layer.data)
        prof = data.pop("presets", None)
        if isinstance(prof, dict):
            if "none" in prof:
                raise config_model.ConfigError(
                    '[presets.none] is refused: "none" is reserved for selecting no preset.'
                )
            user_presets = _layer_data.deep_merge(user_presets, prof)
        cleaned.append(_layer_data.Layer(layer.name, layer.path, data))
    return cleaned, user_presets


def apply_preset(layers: list[_layer_data.Layer], preset_override: str) -> list[_layer_data.Layer]:
    """Strip the `[presets]` tables and splice the selected preset in above its selector.

    The precedence, lowest first: default, global, a global-selected preset, repo, a
    repo-selected preset, a `--preset` preset, `--config FILE`, the machine overlay.

    Args:
        layers: The layers.
        preset_override: The `--preset` flag, or "".

    Returns:
        The layers ready to merge.

    Raises:
        ConfigError: The selected preset is unknown, or a preset value is malformed.
    """
    cleaned, user_presets = strip_presets(layers)
    name, source = select_preset(cleaned, preset_override)
    overrides = resolve_preset(name, user_presets)
    if source == "flag":
        cleaned.append(_layer_data.Layer("flag", None, {"preset": name}))
    if not overrides:
        return cleaned
    return insert_preset(cleaned, _layer_data.Layer("preset", None, overrides), source)
