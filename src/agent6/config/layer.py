# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Layered config resolution and per-leaf provenance.

The layers, lowest precedence first: `default` (the model's defaults), `global`
(`$XDG_CONFIG_HOME/agent6/config.toml`), `repo` (`<state-base>/<repo-id>/config.toml`),
`flag` (`--config FILE`) and `machine` (a machine's per-state overlay). The raw tables are
deep-merged in that order and validated once; every leaf remembers the layer that set it.
A selected preset is spliced in just above the layer that selected it, per `_apply_preset`.
"""

from __future__ import annotations

import contextlib
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from agent6.config.io import (
    format_toml_value,
    read_toml_file,
    read_toml_leaf,
    toml_key,
)
from agent6.config.model import (
    Config,
    ConfigError,
    validate_config,
)
from agent6.paths import (
    global_config_path,
    repo_config_path,
)

LayerName = Literal["default", "preset", "global", "repo", "flag", "machine"]

# The `config show` and `config fill` order, from the model so a new section is never omitted.
SECTION_ORDER = tuple(Config.model_fields)


@dataclass(frozen=True, slots=True)
class Layer:
    """One config source.

    Attributes:
        name: The layer name.
        path: The file it was read from; None for a preset or an in-memory overlay.
        data: The raw parsed table.
    """

    name: LayerName
    path: Path | None
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    """One load: the validated config with the provenance of every leaf.

    Attributes:
        config: The validated config.
        sources: The layer name that set each dotted leaf.
        layers: The layers that contributed, in precedence order.
        presets: The preset names this load knew, built-ins and `[presets.*]` tables, sorted.
    """

    config: Config
    sources: dict[str, str]
    layers: tuple[Layer, ...]
    presets: tuple[str, ...] = ()

    @property
    def explicit_leaves(self) -> frozenset[str]:
        """The leaves a config layer set: a demand to honor or refuse, not a default to degrade."""
        return frozenset(leaf for leaf, layer in self.sources.items() if layer != "default")


def _read_toml(path: Path) -> dict[str, Any]:
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
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file cannot be read ({path}): {exc}") from exc


def _forbid_layer_preset(layer_name: str, data: dict[str, Any]) -> None:
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
        raise ConfigError(
            f"top-level `preset` selects a config preset only from the global/repo"
            f" config or the --preset flag, not the {layer_name} config; use"
            f" --preset <name> or set it in your repo/global config."
        )


def discover_layers(repo_root: Path, explicit_path: Path | None) -> list[Layer]:
    """Read the config files that exist, in precedence order.

    Args:
        repo_root: The repo whose state-dir config is the `repo` layer.
        explicit_path: The `--config FILE` layer, or None.

    Returns:
        The layers, lowest precedence first.

    Raises:
        ConfigError: A file is unreadable or invalid, `explicit_path` is missing, or it
            carries a top-level `preset`.
    """
    layers: list[Layer] = []
    gpath = global_config_path()
    if gpath.is_file():
        layers.append(Layer("global", gpath, _read_toml(gpath)))
    rpath = repo_config_path(repo_root)
    if rpath.is_file():
        layers.append(Layer("repo", rpath, _read_toml(rpath)))
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigError(f"--config file not found: {explicit_path}")
        data = _read_toml(explicit_path)
        _forbid_layer_preset("--config", data)
        layers.append(Layer("flag", explicit_path, data))
    return layers


# The docs table and the `--preset` help print each note.
BUILTIN_PRESET_NOTES: dict[str, str] = {
    "standard": "the plain defaults, no review panel",
    "quick": "no review panel: fast and cheap",
    "ultra": "a three-seat review panel that vetoes the finish it does not pass",
    "paranoid": "five explore-tier review seats vetoing the finish: maximum scrutiny",
}
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {},
    "quick": {
        "review": {"trigger": "off"},
    },
    "ultra": {
        "review": {
            "trigger": "before_finish",
            # veto, not quorum: the seats share one model, and the gate counts blocks per model.
            "decision": "veto",
            "seats": ["security", "correctness", "tests"],
            "concurrency": 3,
        },
    },
    "paranoid": {
        "review": {
            "trigger": "before_finish",
            "decision": "veto",
            "tier": "explore",
            "seats": [
                "security",
                "correctness",
                "tests",
                "edge-cases",
                "over-engineering",
            ],
            "concurrency": 5,
        },
    },
}


def resolve_preset(name: str, user_presets: dict[str, Any]) -> dict[str, Any]:
    """Return the overrides a preset applies.

    A user preset wins over a built-in of the same name, `standard` included: it is an empty
    built-in like any other, so a user table of that name replaces it as the docs promise.

    Args:
        name: The preset name; "" selects nothing.
        user_presets: The `[presets.*]` tables, merged.

    Returns:
        The nested config dict; {} for "".

    Raises:
        ConfigError: The name is unknown, or the user table is not a table.
    """
    if not name:
        return {}
    if name in user_presets:
        prof = user_presets[name]
        if not isinstance(prof, dict):
            raise ConfigError(f"[presets.{name}] must be a table, got {type(prof).__name__}")
        return prof
    if name in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name]
    known = ", ".join(sorted({*BUILTIN_PRESETS, *user_presets}))
    raise ConfigError(f"unknown preset {name!r}. Known presets: {known}.")


def preset_names(layers: Iterable[Layer]) -> list[str]:
    """Return the preset names a chooser offers.

    Args:
        layers: The layers whose `[presets.*]` tables add to the built-ins.

    Returns:
        The names, sorted and de-duplicated.
    """
    names: set[str] = set(BUILTIN_PRESETS)
    for layer in layers:
        prof = layer.data.get("presets")
        if isinstance(prof, dict):
            names.update(prof.keys())
    return sorted(names)


def available_preset_names(repo_root: Path, explicit_path: Path | None = None) -> list[str]:
    """Return the preset names over the repo's discovered layers.

    A config-read failure degrades to the built-ins alone, so a chooser never blocks on a
    bad config.

    Args:
        repo_root: The repo.
        explicit_path: The `--config FILE` layer, or None.

    Returns:
        The names, sorted.
    """
    layers: list[Layer] = []
    with contextlib.suppress(Exception):
        layers = discover_layers(repo_root, explicit_path)
    return preset_names(layers)


@dataclass(frozen=True, slots=True)
class PresetInfo:
    """One preset as `config presets` shows it.

    Attributes:
        name: The preset name.
        overrides: The nested config dict it applies; {} for the plain defaults.
        origin: `built-in`, `global`, `repo` or `global+repo`.
        replaces_builtin: A user preset with a built-in's name replaces it whole.
    """

    name: str
    overrides: dict[str, Any]
    origin: str
    replaces_builtin: bool


@dataclass(frozen=True, slots=True)
class PresetCatalog:
    """Everything `config presets` lists.

    Attributes:
        presets: The built-ins in definition order, then the user's, sorted.
        selected: The selected preset's name; "" when none is selected anywhere.
        source: `repo`, `global` or `none`.
    """

    presets: tuple[PresetInfo, ...]
    selected: str
    source: str


def preset_catalog(repo_root: Path, explicit_path: Path | None = None) -> PresetCatalog:
    """Build the `config presets` listing.

    A user table replaces a same-named built-in, so only the effective body is reported.
    An audit surface, so unlike `available_preset_names` a broken config raises.

    Args:
        repo_root: The repo.
        explicit_path: The `--config FILE` layer, or None.

    Returns:
        The catalog.

    Raises:
        ConfigError: A layer is unreadable, or a `[presets.<name>]` is not a table.
    """
    layers = discover_layers(repo_root, explicit_path)
    user: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}
    for layer in layers:
        prof = layer.data.get("presets")
        if not isinstance(prof, dict):
            continue
        for name, body in prof.items():
            if not isinstance(body, dict):
                raise ConfigError(f"[presets.{name}] must be a table, got {type(body).__name__}")
            user[name] = _deep_merge(user.get(name, {}), body)
            origins[name] = f"{origins[name]}+{layer.name}" if name in origins else layer.name
    selected, source = _select_preset(layers, "")
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


def _format_changed(val: object, existing: object) -> bool:
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge an override into a base, recursing into dicts unless `_format_changed`.

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
            out[key] = val if _format_changed(val, existing) else _deep_merge(existing, val)
        else:
            out[key] = val
    return out


def _merge_layers(layers: list[Layer]) -> tuple[dict[str, Any], dict[str, str]]:
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
        base: dict[str, Any], override: dict[str, Any], layer_name: str, prefix: str
    ) -> dict[str, Any]:
        out = dict(base)
        for key, val in override.items():
            path = f"{prefix}{key}"
            existing = out.get(key)
            if (
                isinstance(val, dict)
                and isinstance(existing, dict)
                and not _format_changed(val, existing)
            ):
                out[key] = walk(existing, val, layer_name, f"{path}.")
            else:
                for stale in [k for k in sources if k == path or k.startswith(f"{path}.")]:
                    del sources[stale]
                out[key] = val
                if isinstance(val, dict) and val:
                    for leaf in flatten_leaves(val, prefix=f"{path}."):
                        sources[leaf] = layer_name
                else:
                    sources[path] = layer_name
        return out

    for layer in layers:
        merged = walk(merged, layer.data, layer.name, "")
    return merged, sources


def flatten_leaves(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf paths.

    A list, an array of tables included, is one leaf.

    Args:
        data: The nested table.
        prefix: The dotted prefix of every path.

    Returns:
        The leaves by dotted path.
    """
    out: dict[str, Any] = {}
    for key, val in data.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            out.update(flatten_leaves(val, prefix=f"{path}."))
        else:
            out[path] = val
    return out


def config_leaves(config: Config) -> dict[str, Any]:
    """Flatten a validated config to dotted leaves.

    A name-keyed section of models (`providers`) is traversed; a dict-typed value (a
    provider's `extra_body`) is one leaf, whatever it holds.

    Args:
        config: The validated config.

    Returns:
        The leaves by dotted path.
    """
    out: dict[str, Any] = {}

    def walk(model: BaseModel, prefix: str) -> None:
        for name in type(model).model_fields:
            value = getattr(model, name)
            path = f"{prefix}{name}"
            if isinstance(value, BaseModel):
                walk(value, f"{path}.")
            elif (
                isinstance(value, dict)
                and value
                and all(isinstance(entry, BaseModel) for entry in value.values())
            ):
                for key, entry in value.items():
                    walk(entry, f"{path}.{key}.")
            else:
                out[path] = value

    walk(config, "")
    return out


def _leaf_fix_hint(
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
    by_name = {layer.name: layer for layer in layers}

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


def _effective_from_layers(
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
    merged, source_of_leaf = _merge_layers(layers)
    config = validate_config(merged, source=source, locate=_leaf_fix_hint(layers, source_of_leaf))
    effective_leaves = config_leaves(config)
    layer_order = [layer.name for layer in layers]

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


def _own_preset(layer: Layer) -> str:
    """Return a layer's own raw top-level `preset`, or "".

    Args:
        layer: The layer.

    Returns:
        The preset name the layer itself sets; "" when it sets none.

    Raises:
        ConfigError: The value is not a string (a `[preset]` table from a typo'd `config set`).
    """
    raw = layer.data.get("preset")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        shape = "a [preset] table" if isinstance(raw, dict) else f"a {type(raw).__name__}"
        raise ConfigError(
            f"top-level `preset` in the {layer.name} config must be a preset name"
            f' string (e.g. preset = "ultra"), got {shape};'
            f" set it with `agent6 config set preset <name>`."
        )
    return raw


def _select_preset(cleaned: list[Layer], preset_override: str) -> tuple[str, str]:
    """Pick the selected preset, most specific source first, never stacking.

    Args:
        cleaned: The layers with their `[presets.*]` tables stripped.
        preset_override: The `--preset` flag, or "".

    Returns:
        `(name, source)`: the flag, else the repo layer's own `preset`, else the global
        layer's, else `("", "none")`.

    Raises:
        ConfigError: A layer's `preset` is not a string.
    """
    if preset_override:
        return preset_override, "flag"
    by_name = {layer.name: layer for layer in cleaned}
    for source in ("repo", "global"):
        layer = by_name.get(source)
        if layer is not None and (name := _own_preset(layer)):
            return name, source
    return "", "none"


def _insert_preset(cleaned: list[Layer], preset: Layer, source: str) -> list[Layer]:
    """Splice the preset layer in at the position its source dictates.

    Args:
        cleaned: The layers with their `[presets.*]` tables stripped.
        preset: The preset layer.
        source: `global` or `repo` put it right after that layer; `flag` puts it just below
            a `--config FILE` or machine overlay, else last.

    Returns:
        The layers with the preset inserted.
    """
    out: list[Layer] = []
    inserted = False
    for layer in cleaned:
        if source == "flag" and not inserted and layer.name in ("flag", "machine"):
            out.append(preset)
            inserted = True
        out.append(layer)
        if not inserted and layer.name == source:  # source in {"global", "repo"}
            out.append(preset)
            inserted = True
    if not inserted:
        out.append(preset)
    return out


def _strip_presets(layers: list[Layer]) -> tuple[list[Layer], dict[str, Any]]:
    """Strip the `[presets.*]` tables out of the layers and merge them.

    Presets are meta-config the Config schema forbids, so every path that validates a layer
    strips them first.

    Args:
        layers: The layers.

    Returns:
        The stripped layers, and the user presets merged low to high.
    """
    cleaned: list[Layer] = []
    user_presets: dict[str, Any] = {}
    for layer in layers:
        data = dict(layer.data)
        prof = data.pop("presets", None)
        if isinstance(prof, dict):
            user_presets = _deep_merge(user_presets, prof)
        cleaned.append(Layer(layer.name, layer.path, data))
    return cleaned, user_presets


def _apply_preset(layers: list[Layer], preset_override: str) -> list[Layer]:
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
    cleaned, user_presets = _strip_presets(layers)
    name, source = _select_preset(cleaned, preset_override)
    overrides = resolve_preset(name, user_presets)
    if not overrides:
        return cleaned
    return _insert_preset(cleaned, Layer("preset", None, overrides), source)


def load_effective(
    repo_root: Path, explicit_path: Path | None = None, *, preset: str = ""
) -> EffectiveConfig:
    """Load the effective config of a repo.

    Args:
        repo_root: The repo.
        explicit_path: The `--config FILE` layer, or None.
        preset: The `--preset` flag, or "".

    Returns:
        The effective config.

    Raises:
        ConfigError: A layer is unreadable or the merged config is invalid.
    """
    layers = discover_layers(repo_root, explicit_path)
    presets = preset_names(layers)
    layers = _apply_preset(layers, preset)
    return _effective_from_layers(layers, source="(merged config layers)", presets=presets)


def load_global_only() -> EffectiveConfig:
    """Load the defaults plus the global config: what `agent6 config fill` materializes.

    A fill writes the global file, so the repo layer would follow the operator to every
    other repo, and a preset's effects would freeze as explicit values while its selector
    kept applying.

    Returns:
        The effective config with no repo layer and no preset applied.

    Raises:
        ConfigError: The global config is unreadable or invalid.
    """
    gpath = global_config_path()
    layers = [Layer("global", gpath, _read_toml(gpath))] if gpath.is_file() else []
    cleaned, _ = _strip_presets(layers)
    return _effective_from_layers(cleaned, source="(global config)", presets=preset_names(layers))


def load_effective_with_overlay(
    repo_root: Path, overlay: dict[str, Any], *, explicit_path: Path | None = None
) -> EffectiveConfig:
    """Load the effective config with a machine's `[config]` table as the highest layer.

    Args:
        repo_root: The repo.
        overlay: The machine's `[config]` table; its leaves are attributed to `machine`.
        explicit_path: The `--config FILE` layer, which sits under the overlay.

    Returns:
        The effective config.

    Raises:
        ConfigError: A layer is unreadable, the overlay selects a preset, or the merged
            config is invalid.
    """
    layers = discover_layers(repo_root, explicit_path)
    if overlay:
        _forbid_layer_preset("machine overlay", overlay)
        layers = [*layers, Layer("machine", None, overlay)]
    presets = preset_names(layers)
    layers = _apply_preset(layers, "")
    return _effective_from_layers(
        layers, source="(merged config layers + machine overlay)", presets=presets
    )


@dataclass(frozen=True, slots=True)
class InvalidEntry:
    """One invalid config leaf that `config fix` can drop, and where it lives.

    Attributes:
        leaf: The dotted config leaf, or a table name.
        value: The offending value, read back from the file.
        layer: `global`, `repo` or `machine`.
        path: The file to edit.
        file_key: The dotted key within that file: the leaf, or `config.<leaf>` in a machine
            overlay.
        is_table: The whole `[leaf]` table is dropped, not one leaf.
        error_type: The pydantic error type, which tells an unknown table from a partial one.
    """

    leaf: str
    value: Any
    layer: LayerName
    path: Path
    file_key: str
    is_table: bool = False
    error_type: str = ""


@dataclass(frozen=True, slots=True)
class ConfigDiagnosis:
    """The `config fix` diagnosis of the on-disk config.

    Attributes:
        removable: The invalid leaves that map to a file and can be dropped.
        blocked: A message for what fix cannot drop, or None; both empty means valid.
    """

    removable: tuple[InvalidEntry, ...]
    blocked: str | None


def _fix_scope_layers(repo_root: Path, machine: Path | None) -> list[Layer]:
    """Return the layers `config fix` repairs, with presets applied as in a real load.

    Args:
        repo_root: The repo.
        machine: A machine file whose `[config]` overlay tops the layers, or None.

    Returns:
        The layers ready to merge.

    Raises:
        ConfigError: A layer is unreadable, or the overlay selects a preset.
    """
    layers = discover_layers(repo_root, None)
    if machine is not None:
        overlay = read_toml_file(machine).get("config", {})
        if isinstance(overlay, dict) and overlay:
            _forbid_layer_preset("machine overlay", overlay)
            layers = [*layers, Layer("machine", machine, overlay)]
    return _apply_preset(layers, "")


def _merge_with_origin(layers: list[Layer]) -> tuple[dict[str, Any], dict[str, Layer]]:
    """Deep-merge the layers and map each dotted leaf to the layer that set it.

    Args:
        layers: The layers, lowest precedence first.

    Returns:
        The merged table, and the layer of each leaf.
    """
    merged, sources = _merge_layers(layers)
    by_name = {layer.name: layer for layer in layers}
    return merged, {leaf: by_name[name] for leaf, name in sources.items() if name in by_name}


def _removable_for(loc: str, origin: dict[str, Layer]) -> tuple[str, Layer, bool] | None:
    """Return what to drop for a validation error at a location.

    Three shapes: the location is a file leaf; it is under a file leaf (the longest present
    prefix); or it is an ancestor table of file leaves, an unknown whole table reported at
    the table (`[cli]` at `cli` while the file holds `cli.input`), dropped whole.

    Args:
        loc: The error's dotted location.
        origin: The layer of each dotted leaf.

    Returns:
        `(file_key, layer, is_table)`, or None when no config file is at fault.
    """
    parts = loc.split(".") if loc else []
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in origin:
            return cand, origin[cand], False
    prefix = f"{loc}." if loc else ""
    child = next((k for k in origin if prefix and k.startswith(prefix)), None)
    if child is not None:
        return loc, origin[child], True  # an extra whole table -> drop the table
    return None


def _diagnose_errors(
    exc: ValidationError, origin: dict[str, Layer], *, only_layer: str | None
) -> ConfigDiagnosis:
    """Turn validation errors into droppable entries and a note for the rest.

    Args:
        exc: The validation error.
        origin: The layer of each dotted leaf.
        only_layer: When set, an error in any other layer is reported, not dropped.

    Returns:
        The diagnosis; an error from a default or a preset is blocked.
    """
    removable: list[InvalidEntry] = []
    blocked: list[str] = []
    seen: set[str] = set()
    for issue in exc.errors():
        loc = ".".join(str(part) for part in issue["loc"])
        note = f"  - {loc or '<root>'}: {issue['msg']}"
        match = _removable_for(loc, origin)
        if match is None:
            blocked.append(note)
            continue
        key, layer, is_table = match
        if key in seen:
            continue
        seen.add(key)
        if layer.path is None or (only_layer is not None and layer.name != only_layer):
            blocked.append(note)
            continue
        file_key = f"config.{key}" if layer.name == "machine" else key
        value = read_toml_leaf(read_toml_file(layer.path), file_key)
        removable.append(
            InvalidEntry(
                leaf=key,
                value=value,
                layer=layer.name,
                path=layer.path,
                file_key=file_key,
                is_table=is_table,
                error_type=issue["type"],
            )
        )
    return ConfigDiagnosis(tuple(removable), "\n".join(blocked) if blocked else None)


def _diagnose_layers(layers: list[Layer], *, only_layer: str | None) -> ConfigDiagnosis:
    """Validate a layer prefix and locate its invalid entries.

    Args:
        layers: The layers, lowest precedence first.
        only_layer: When set, an error in any other layer is reported, not dropped.

    Returns:
        The diagnosis; a `ConfigError` from validation is blocked with its message.
    """
    merged, origin = _merge_with_origin(layers)
    try:
        Config.model_validate(merged)
    except ConfigError as exc:
        return ConfigDiagnosis((), str(exc))
    except ValidationError as exc:
        return _diagnose_errors(exc, origin, only_layer=only_layer)
    return ConfigDiagnosis((), None)


def _diagnose_presets(repo_root: Path) -> ConfigDiagnosis:
    """Diagnose every user preset, selected or not.

    Each layer prefix is validated so a repo override cannot hide a stale value in the
    global preset. A partial table may be completed by the next layer or the selecting
    config, so only an unknown table counts among the whole-table errors.

    Args:
        repo_root: The repo.

    Returns:
        The first preset's diagnosis that holds a droppable entry, else a clean one.

    Raises:
        ConfigError: A layer is unreadable.
    """
    by_name: dict[str, list[Layer]] = {}
    for layer in discover_layers(repo_root, None):
        presets = layer.data.get("presets")
        if not isinstance(presets, dict):
            continue
        for name, body in presets.items():
            if not isinstance(body, dict):
                if layer.path is None:  # discover_layers always attaches file paths
                    continue
                return ConfigDiagnosis(
                    (
                        InvalidEntry(
                            leaf=f"presets.{name}",
                            value=body,
                            layer=layer.name,
                            path=layer.path,
                            file_key=f"presets.{name}",
                            error_type="dict_type",
                        ),
                    ),
                    None,
                )
            by_name.setdefault(name, []).append(Layer(layer.name, layer.path, body))
    for name, layers in by_name.items():
        for end in range(1, len(layers) + 1):
            diagnosis = _diagnose_layers(layers[:end], only_layer=None)
            leaves = [
                entry
                for entry in diagnosis.removable
                if not entry.is_table or entry.error_type == "extra_forbidden"
            ]
            if not leaves:
                continue
            return ConfigDiagnosis(
                tuple(
                    InvalidEntry(
                        leaf=f"presets.{name}.{entry.leaf}",
                        value=read_toml_leaf(
                            read_toml_file(entry.path), f"presets.{name}.{entry.file_key}"
                        ),
                        layer=entry.layer,
                        path=entry.path,
                        file_key=f"presets.{name}.{entry.file_key}",
                        is_table=entry.is_table,
                        error_type=entry.error_type,
                    )
                    for entry in leaves
                ),
                diagnosis.blocked,
            )
    return ConfigDiagnosis((), None)


def find_invalid_entries(repo_root: Path, *, machine: Path | None = None) -> ConfigDiagnosis:
    """Diagnose the on-disk config for `agent6 config fix`.

    Each file-layer prefix is checked before the selected preset is inserted: a preset may
    complete a partial table, but must not hide a stale scalar in the file that selected it.

    Args:
        repo_root: The repo.
        machine: When set, only its `[config]` overlay entries are droppable; a global or
            repo problem the merge surfaces is reported, not touched.

    Returns:
        The diagnosis; an unknown preset name, unreadable TOML, or a value only a default or
        preset carries is blocked with its message.
    """
    only = "machine" if machine is not None else None
    try:
        if machine is None and (preset_diagnosis := _diagnose_presets(repo_root)).removable:
            return preset_diagnosis
        if machine is None:
            cleaned, _ = _strip_presets(discover_layers(repo_root, None))
            for end in range(1, len(cleaned) + 1):
                diagnosis = _diagnose_layers(cleaned[:end], only_layer=None)
                leaves = tuple(entry for entry in diagnosis.removable if not entry.is_table)
                if leaves:
                    return ConfigDiagnosis(leaves, None)
        layers = _fix_scope_layers(repo_root, machine)
    except ConfigError as exc:
        return ConfigDiagnosis((), str(exc))
    return _diagnose_layers(layers, only_layer=only)


def leaf_keys(eff: EffectiveConfig) -> list[str]:
    """Return every dotted leaf path of the effective config, sorted.

    Args:
        eff: The effective config.

    Returns:
        The paths.
    """
    return sorted(config_leaves(eff.config))


def effective_leaf(eff: EffectiveConfig, dotted_key: str) -> tuple[Any, str] | None:
    """Return a leaf's value and the layer that set it, as `config show` reports them.

    Args:
        eff: The effective config.
        dotted_key: The leaf, or `presets.<name>.<leaf>` for a preset's own value.

    Returns:
        `(value, source)`, the source `default` when no layer set it and `unset` for a preset
        leaf no layer authored; None when the key is not a leaf.
    """
    leaves = config_leaves(eff.config)
    parts = dotted_key.split(".")
    if parts[0] == "presets" and len(parts) > 2:
        # A preset leaf may name a role or provider absent from the effective config.
        name, leaf = parts[1], ".".join(parts[2:])
        for layer in reversed(eff.layers):
            table = _file_presets(layer.path).get(name)
            if isinstance(table, dict) and (value := read_toml_leaf(table, leaf)) is not None:
                return value, f"preset {name} ({layer.name})"
        if leaf in leaves:
            return None, "unset"
        return None
    if dotted_key not in leaves:
        return None
    return leaves[dotted_key], eff.sources.get(dotted_key, "default")


def _emit_table(path: str, data: dict[str, Any], lines: list[str]) -> None:
    """Emit one TOML table, recursing into its subtables and arrays of tables.

    A None value is skipped: an unset optional field materializes as absent.

    Args:
        path: The table's dotted name.
        data: The table.
        lines: The output, appended to.
    """
    scalars = {
        k: v
        for k, v in data.items()
        if v is not None and not isinstance(v, dict) and not _is_table_array(v)
    }
    subtables = {k: v for k, v in data.items() if isinstance(v, dict) and v}
    arraytables = {k: v for k, v in data.items() if _is_table_array(v)}
    # A pure parent table ([providers], [models]) stays implicit: no empty header.
    is_leaf = not subtables and not arraytables
    if scalars or is_leaf:
        lines.append(f"[{path}]")
        for key, value in scalars.items():
            lines.append(f"{toml_key(key)} = {format_toml_value(value)}")
        lines.append("")
    for key, sub in subtables.items():
        _emit_table(f"{path}.{toml_key(key)}" if path else toml_key(key), sub, lines)
    for key, arr in arraytables.items():
        for item in arr:
            lines.append(f"[[{path}.{toml_key(key)}]]" if path else f"[[{toml_key(key)}]]")
            for k2, v2 in item.items():
                if v2 is not None:
                    lines.append(f"{toml_key(k2)} = {format_toml_value(v2)}")
            lines.append("")


def _is_table_array(value: Any) -> bool:
    """Return whether the value is a non-empty list of dicts.

    Args:
        value: Any config value.

    Returns:
        True for an array of tables.
    """
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, dict) for v in value)
    )


def materialize(
    config: Config,
    *,
    keep_presets_from: Path | None = None,
    keep_preset_selector: bool = False,
) -> str:
    """Render the resolved config as a complete TOML document, every value explicit.

    Args:
        config: The validated config.
        keep_presets_from: A file whose own `[presets.*]` tables the document carries; no
            `Config` holds them, and a fill rewriting the operator's file would otherwise
            delete definitions it cannot see.
        keep_preset_selector: Keep the top-level `preset`. A `--config FILE` layer refuses
            one (a lane's snapshot drops it); the global config `config fill` writes keeps
            it, or the fill would deselect the preset and freeze the current values.

    Returns:
        The TOML text.
    """
    data = config.model_dump(mode="python")
    data = data if keep_preset_selector else {k: v for k, v in data.items() if k != "preset"}
    lines: list[str] = [
        "# agent6 effective config, materialized by `agent6 config fill`.",
        "# Every value below is explicit; edit freely.",
        "",
    ]
    ordered = [s for s in SECTION_ORDER if s in data]
    ordered += [s for s in data if s not in SECTION_ORDER]
    # A top-level scalar (`preset`) must precede every `[section]` in TOML.
    for section in ordered:
        value = data[section]
        if value is not None and not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f"{section} = {format_toml_value(value)}")
    if lines[-1] != "":
        lines.append("")
    for section in ordered:
        value = data[section]
        if isinstance(value, dict):
            if not value:
                continue
            _emit_table(section, value, lines)
        elif _is_table_array(value):
            for item in value:
                lines.append(f"[[{section}]]")
                for k2, v2 in item.items():
                    if v2 is not None:
                        lines.append(f"{toml_key(k2)} = {format_toml_value(v2)}")
                lines.append("")
    if kept := _file_presets(keep_presets_from):
        _emit_table("presets", kept, lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def _file_presets(path: Path | None) -> dict[str, Any]:
    """Return the `[presets.*]` tables a file defines itself.

    Args:
        path: The file, or None.

    Returns:
        The tables by name; {} when the file is absent or defines none.
    """
    if path is None or not path.is_file():
        return {}
    presets = _read_toml(path).get("presets")
    return presets if isinstance(presets, dict) else {}
