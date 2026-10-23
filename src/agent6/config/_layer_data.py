# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Discover config layers, merge their values and attribute each leaf."""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from collections.abc import Callable
from typing import Any, Literal

import pydantic

from agent6 import paths
from agent6.config import _base, io
from agent6.config import model as config_model

LayerName = Literal["default", "preset", "global", "repo", "flag", "machine", "state"]


@dataclasses.dataclass(frozen=True, slots=True)
class Layer:
    """One config source.

    Attributes:
        name: The layer name.
        path: The file it was read from; None for a preset or an in-memory overlay.
        data: The raw parsed table.
    """

    name: LayerName
    path: pathlib.Path | None
    data: dict[str, Any]

    @property
    def source(self) -> str:
        """Identify the layer, including the path for command-line files."""
        return f"flag ({self.path})" if self.name == "flag" and self.path else self.name

    @property
    def described(self) -> str:
        """Name the layer as a sentence does: `the repo config`, `--config FILE`, `the preset`."""
        if self.name == "flag":
            return f"--config {self.path}"
        if self.name in ("global", "repo"):
            return f"the {self.name} config"
        return f"the {self.name}" if self.name == "preset" else f"the {self.name} overlay"


@dataclasses.dataclass(frozen=True, slots=True)
class EffectiveConfig:
    """One load: the validated config with the provenance of every leaf.

    Attributes:
        config: The validated config.
        sources: The layer name that set each dotted leaf.
        layers: The layers that contributed, in precedence order.
        presets: The preset names this load knew, built-ins and `[presets.*]` tables, sorted.
    """

    config: config_model.Config
    sources: dict[str, str]
    layers: tuple[Layer, ...]
    presets: tuple[str, ...] = ()

    @property
    def explicit_leaves(self) -> frozenset[str]:
        """The leaves a config layer set: a demand to honor or refuse, not a default to degrade."""
        return frozenset(leaf for leaf, layer in self.sources.items() if layer != "default")


def read_toml(path: pathlib.Path) -> dict[str, Any]:
    """Parse a config file.

    Args:
        path: The file.

    Returns:
        The parsed table.

    Raises:
        ConfigError: The file is not valid TOML or cannot be read (a root-owned file after a
            sudo run, a directory at the path): the operator's file, so a refusal and never a
            crash report.
    """
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise _base.ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise _base.ConfigError(f"Config file cannot be read ({path}): {exc}") from exc


def forbid_layer_preset(layer_name: str, data: dict[str, Any]) -> None:
    """Refuse a top-level `preset` key in a layer that cannot select one.

    Only the global and repo configs and `--preset` select one; merged in from elsewhere the
    key would show as effective while never applying, and honoring it would let a machine
    overlay pick a preset that loosens the sandbox.

    Args:
        layer_name: The layer, for the message.
        data: The layer's raw table.

    Raises:
        ConfigError: The table holds a top-level `preset`.
    """
    if "preset" in data:
        raise _base.ConfigError(
            f"top-level `preset` selects a config preset only from the global/repo"
            f" config or the --preset flag, not the {layer_name} config; use"
            f" --preset <name> or set it in your repo/global config."
        )


def discover_layers(
    repo_root: pathlib.Path, explicit_paths: tuple[pathlib.Path, ...]
) -> list[Layer]:
    """Read the config files that exist, in precedence order.

    Args:
        repo_root: The repo whose state-dir config is the `repo` layer.
        explicit_paths: The `--config FILE` layers in command-line order.

    Returns:
        The layers, lowest precedence first.

    Raises:
        ConfigError: A file is unreadable or invalid, a file in `explicit_paths` is
            missing, or one carries a top-level `preset`.
    """
    layers: list[Layer] = []
    gpath = paths.global_config_path()
    if gpath.is_file():
        layers.append(Layer("global", gpath, read_toml(gpath)))
    rpath = paths.repo_config_path(repo_root)
    if rpath.is_file():
        layers.append(Layer("repo", rpath, read_toml(rpath)))
    for explicit_path in explicit_paths:
        if not explicit_path.is_file():
            raise _base.ConfigError(f"--config file not found: {explicit_path}")
        data = read_toml(explicit_path)
        forbid_layer_preset("--config", data)
        layers.append(Layer("flag", explicit_path, data))
    return layers


def format_changed(val: object, existing: object) -> bool:
    """Return whether an override replaces the existing dict whole instead of merging.

    The one rule the merge and the provenance walk share: a provider entry whose `api_format`
    changes between layers replaces, or the lower layer's format-specific keys would survive
    as an `extra_forbidden` error under the new format.

    Args:
        val: The overriding value.
        existing: The value beneath it.

    Returns:
        True when both are dicts whose `api_format` differs.
    """
    return (
        isinstance(val, dict)
        and isinstance(existing, dict)
        and "api_format" in val
        and "api_format" in existing
        and val.get("api_format") != existing.get("api_format")
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge an override into a base, recursing into dicts unless `format_changed`.

    Args:
        base: The lower-precedence table.
        override: The higher-precedence table.

    Returns:
        A new merged table; neither input is modified.
    """
    out = dict(base)
    for key, val in override.items():
        existing = out.get(key)
        if isinstance(val, dict) and isinstance(existing, dict):
            out[key] = val if format_changed(val, existing) else deep_merge(existing, val)
        else:
            out[key] = val
    return out


def merge_layers(layers: list[Layer]) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge the layers and stamp per-leaf provenance in the same walk.

    One walk, so the two never diverge: on a wholesale replace the stale sub-provenance dies
    with the subtree before the winner's leaves are stamped.

    Args:
        layers: The layers, lowest precedence first.

    Returns:
        The merged table, and the layer name that set each dotted leaf.
    """
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}

    def walk(
        base: dict[str, Any], override: dict[str, Any], layer_name: str, prefix: tuple[str, ...]
    ) -> dict[str, Any]:
        out = dict(base)
        for key, val in override.items():
            path = io.render_key((*prefix, key))
            existing = out.get(key)
            if (
                isinstance(val, dict)
                and isinstance(existing, dict)
                and not format_changed(val, existing)
            ):
                out[key] = walk(existing, val, layer_name, (*prefix, key))
            else:
                for stale in [k for k in sources if k == path or k.startswith(f"{path}.")]:
                    del sources[stale]
                out[key] = val
                if isinstance(val, dict) and val:
                    for leaf in flatten_leaves(val, prefix=(*prefix, key)):
                        sources[leaf] = layer_name
                else:
                    sources[path] = layer_name
        return out

    for layer in layers:
        merged = walk(merged, layer.data, layer.source, ())
    return merged, sources


def flatten_leaves(data: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf paths.

    A list, an array of tables included, is one leaf.

    Args:
        data: The nested table.
        prefix: The literal components above every path.

    Returns:
        The leaves by dotted path.
    """
    out: dict[str, Any] = {}
    for key, val in data.items():
        path = io.render_key((*prefix, key))
        if isinstance(val, dict) and val:
            out.update(flatten_leaves(val, prefix=(*prefix, key)))
        else:
            out[path] = val
    return out


def config_leaves(config: config_model.Config) -> dict[str, Any]:
    """Flatten a validated config to dotted leaves.

    A name-keyed section of models (`providers`) is traversed.
    Other mappings yield one leaf per entry, keeping each entry's value whole.
    An empty mapping stays one leaf.

    Args:
        config: The validated config.

    Returns:
        The leaves by dotted path.
    """
    out: dict[str, Any] = {}

    def walk(model: pydantic.BaseModel, prefix: tuple[str, ...]) -> None:
        for name in type(model).model_fields:
            value = getattr(model, name)
            parts = (*prefix, name)
            path = io.render_key(parts)
            if isinstance(value, pydantic.BaseModel):
                walk(value, parts)
            elif isinstance(value, dict) and value:
                for key, entry in value.items():
                    if isinstance(entry, pydantic.BaseModel):
                        walk(entry, (*parts, key))
                    else:
                        out[io.render_key((*parts, key))] = entry
            elif isinstance(value, tuple):
                out[path] = tuple(
                    entry.model_dump(exclude_none=True)
                    if isinstance(entry, pydantic.BaseModel)
                    else entry
                    for entry in value
                )
            else:
                out[path] = value

    walk(config, ())
    return out


def leaf_fix_hint(
    layers: list[Layer], source_of_leaf: dict[str, str]
) -> Callable[[str, str], str | None]:
    """Build the locator `validate_config` appends to each error line.

    Args:
        layers: The layers.
        source_of_leaf: The layer name that set each dotted leaf.

    Returns:
        A function from (dotted leaf, pydantic error type) to a line naming the layer that
        set it, with the file and the fix command when the layer has one (an unknown key
        points at `agent6 config fix`); None for a built-in default.
    """
    by_name = {layer.source: layer for layer in layers}

    def locate(leaf: str, error_type: str) -> str | None:
        layer = by_name.get(source_of_leaf.get(leaf, ""))
        if layer is None:
            return None
        if layer.path is None:
            # A preset's override or a file-less machine overlay: no single file to point at.
            return f"    set by the {layer.name} layer"
        if layer.name == "flag":
            fix = f"edit {layer.path}"
        elif error_type == "extra_forbidden":
            fix = "agent6 config fix"
        elif layer.name == "repo":
            fix = f"agent6 config set --repo {leaf} <value>"
        else:
            fix = f"agent6 config set {leaf} <value>"
        return f"    set in the {layer.name} config: {layer.path}\n    fix: {fix}"

    return locate


def effective_from_layers(
    layers: list[Layer], *, source: str, presets: list[str]
) -> EffectiveConfig:
    """Merge the layers, validate, and attribute every effective leaf to its layer.

    Args:
        layers: The layers, lowest precedence first.
        source: The name validation errors report.
        presets: The preset names this load knew.

    Returns:
        The effective config.

    Raises:
        ConfigError: The merged config fails validation.
    """
    merged, source_of_leaf = merge_layers(layers)
    config = config_model.validate_config(
        merged, source=source, locate=leaf_fix_hint(layers, source_of_leaf)
    )
    effective_leaves = config_leaves(config)
    layer_order = [layer.source for layer in layers]

    def source_for(leaf: str) -> str:
        contributors = {
            source
            for path, source in source_of_leaf.items()
            if path == leaf or path.startswith(f"{leaf}.")
        }
        return next((name for name in reversed(layer_order) if name in contributors), "default")

    sources = {leaf: source_for(leaf) for leaf in effective_leaves}
    return EffectiveConfig(
        config=config, sources=sources, layers=tuple(layers), presets=tuple(presets)
    )
