# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Layered config resolution and per-leaf provenance.

The layers, lowest precedence first: `default` (the model's defaults), `global`
(`$XDG_CONFIG_HOME/agent6/config.toml`), `repo` (`<state-base>/<repo-id>/config.toml`),
`flag` (`--config FILE`), `machine` (a machine's `[config]` overlay) and `state` (one
state's `config` table). The raw tables are
deep-merged in that order and validated once; every leaf remembers the layer that set it.
A selected preset is spliced in just above the layer that selected it, per `_presets.apply_preset`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pydantic

from agent6 import paths
from agent6.config import _base, _layer_data, _presets, io
from agent6.config import model as config_model

# The `config show` and `config fill` order, from the model so a new section is never omitted.
SECTION_ORDER = tuple(config_model.Config.model_fields)


def load_effective(
    repo_root: pathlib.Path, explicit_paths: tuple[pathlib.Path, ...] = (), *, preset: str = ""
) -> _layer_data.EffectiveConfig:
    """Load the effective config of a repo.

    Args:
        repo_root: The repo.
        explicit_paths: The `--config FILE` layers in command-line order.
        preset: The `--preset` flag, or "".

    Returns:
        The effective config.

    Raises:
        ConfigError: A layer is unreadable or the merged config is invalid.
    """
    layers = _layer_data.discover_layers(repo_root, explicit_paths)
    presets = _presets.preset_names(layers)
    layers = _presets.apply_preset(layers, preset)
    return _layer_data.effective_from_layers(
        layers, source="(merged config layers)", presets=presets
    )


def shadowing_layer(
    eff: _layer_data.EffectiveConfig, leaf: str, target: _layer_data.LayerName
) -> _layer_data.Layer | None:
    """Return the layer above the target that sets a leaf, or None.

    A write to the target changes what a run reads only when no layer above it sets the leaf.
    A target file that does not exist yet sits where it would load: a global one under every
    layer, a repo one above the global layer and the preset that layer selected.

    Args:
        eff: The loaded config.
        leaf: The dotted leaf.
        target: The layer a write lands in, `global` or `repo`.

    Returns:
        The layer whose value a run reads over the target's; None when the leaf is a table or
        the target's own value, a lower layer's or the default is in force.
    """
    found = effective_leaf(eff, leaf)
    source = found[1] if found is not None else None
    if source is None:
        return None
    names = [lyr.name for lyr in eff.layers]
    if target in names:
        above = eff.layers[names.index(target) + 1 :]
    elif target == "global":
        above = eff.layers
    else:
        above = eff.layers[len([n for n in names[:2] if n in ("global", "preset")]) :]
    return next((lyr for lyr in above if lyr.source == source), None)


def load_global_only() -> _layer_data.EffectiveConfig:
    """Load the defaults plus the global config: what `agent6 config fill` materializes.

    A fill writes the global file, so the repo layer would follow the operator to every
    other repo, and a preset's effects would freeze as explicit values while its selector
    kept applying.

    Returns:
        The effective config with no repo layer and no preset applied.

    Raises:
        ConfigError: The global config is unreadable or invalid.
    """
    gpath = paths.global_config_path()
    layers = (
        [_layer_data.Layer("global", gpath, _layer_data.read_toml(gpath))]
        if gpath.is_file()
        else []
    )
    cleaned, _ = _presets.strip_presets(layers)
    return _layer_data.effective_from_layers(
        cleaned, source="(global config)", presets=_presets.preset_names(layers)
    )


def load_effective_with_overlay(
    repo_root: pathlib.Path,
    overlay: dict[str, Any],
    *,
    explicit_paths: tuple[pathlib.Path, ...] = (),
    state_overlay: dict[str, Any] | None = None,
) -> _layer_data.EffectiveConfig:
    """Load the effective config with a machine's overlays as the highest layers.

    Args:
        repo_root: The repo.
        overlay: The machine's `[config]` table; its leaves are attributed to `machine`.
        explicit_paths: The `--config FILE` layers, which sit under the overlays.
        state_overlay: One state's `config` table, above the machine's; its leaves are
            attributed to `state`.

    Returns:
        The effective config.

    Raises:
        ConfigError: A layer is unreadable, an overlay selects a preset, or the merged
            config is invalid.
    """
    layers = _layer_data.discover_layers(repo_root, explicit_paths)
    if overlay:
        _layer_data.forbid_layer_preset("machine overlay", overlay)
        layers = [*layers, _layer_data.Layer("machine", None, overlay)]
    if state_overlay:
        _layer_data.forbid_layer_preset("state overlay", state_overlay)
        layers = [*layers, _layer_data.Layer("state", None, state_overlay)]
    presets = _presets.preset_names(layers)
    layers = _presets.apply_preset(layers, "")
    return _layer_data.effective_from_layers(
        layers, source="(merged config layers + machine overlay)", presets=presets
    )


@dataclasses.dataclass(frozen=True, slots=True)
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
    layer: _layer_data.LayerName
    path: pathlib.Path
    file_key: str
    is_table: bool = False
    error_type: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigDiagnosis:
    """The `config fix` diagnosis of the on-disk config.

    Attributes:
        removable: The invalid leaves that map to a file and can be dropped.
        blocked: A message for what fix cannot drop, or None; both empty means valid.
    """

    removable: tuple[InvalidEntry, ...]
    blocked: str | None


def _fix_scope_layers(
    repo_root: pathlib.Path, machine: pathlib.Path | None
) -> list[_layer_data.Layer]:
    """Return the layers `config fix` repairs, with presets applied as in a real load.

    Args:
        repo_root: The repo.
        machine: A machine file whose `[config]` overlay tops the layers, or None.

    Returns:
        The layers ready to merge.

    Raises:
        ConfigError: A layer is unreadable, or the overlay selects a preset.
    """
    layers = _layer_data.discover_layers(repo_root, ())
    if machine is not None:
        overlay = io.read_toml_file(machine).get("config", {})
        if isinstance(overlay, dict) and overlay:
            _layer_data.forbid_layer_preset("machine overlay", overlay)
            layers = [*layers, _layer_data.Layer("machine", machine, overlay)]
    return _presets.apply_preset(layers, "")


def _merge_with_origin(
    layers: list[_layer_data.Layer],
) -> tuple[dict[str, Any], dict[str, _layer_data.Layer]]:
    """Deep-merge the layers and map each dotted leaf to the layer that set it.

    Args:
        layers: The layers, lowest precedence first.

    Returns:
        The merged table, and the layer of each leaf.
    """
    merged, sources = _layer_data.merge_layers(layers)
    by_name = {layer.source: layer for layer in layers}
    return merged, {leaf: by_name[name] for leaf, name in sources.items() if name in by_name}


def _removable_for(
    loc: str, origin: dict[str, _layer_data.Layer]
) -> tuple[str, _layer_data.Layer, bool] | None:
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
    parts = io.split_key(loc) if loc else []
    for i in range(len(parts), 0, -1):
        cand = io.render_key(parts[:i])
        if cand in origin:
            return cand, origin[cand], False
    prefix = f"{loc}." if loc else ""
    child = next((k for k in origin if prefix and k.startswith(prefix)), None)
    if child is not None:
        return loc, origin[child], True  # an extra whole table -> drop the table
    return None


def _diagnose_errors(
    exc: pydantic.ValidationError, origin: dict[str, _layer_data.Layer], *, only_layer: str | None
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
        loc = io.render_key(str(part) for part in issue["loc"])
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
        file_key = io.render_key(("config", *io.split_key(key))) if layer.name == "machine" else key
        value = io.read_toml_leaf(io.read_toml_file(layer.path), file_key)
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


def _diagnose_layers(layers: list[_layer_data.Layer], *, only_layer: str | None) -> ConfigDiagnosis:
    """Validate a layer prefix and locate its invalid entries.

    Args:
        layers: The layers, lowest precedence first.
        only_layer: When set, an error in any other layer is reported, not dropped.

    Returns:
        The diagnosis; a `ConfigError` from validation is blocked with its message.
    """
    merged, origin = _merge_with_origin(layers)
    try:
        config_model.Config.model_validate(merged)
    except _base.ConfigError as exc:
        return ConfigDiagnosis((), str(exc))
    except pydantic.ValidationError as exc:
        return _diagnose_errors(exc, origin, only_layer=only_layer)
    return ConfigDiagnosis((), None)


def _diagnose_presets(repo_root: pathlib.Path) -> ConfigDiagnosis:
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
    by_name: dict[str, list[_layer_data.Layer]] = {}
    for layer in _layer_data.discover_layers(repo_root, ()):
        presets = layer.data.get("presets")
        if not isinstance(presets, dict):
            continue
        for name, body in presets.items():
            if not isinstance(body, dict):
                if layer.path is None:  # _layer_data.discover_layers always attaches file paths
                    continue
                return ConfigDiagnosis(
                    (
                        InvalidEntry(
                            leaf=io.render_key(("presets", name)),
                            value=body,
                            layer=layer.name,
                            path=layer.path,
                            file_key=io.render_key(("presets", name)),
                            error_type="dict_type",
                        ),
                    ),
                    None,
                )
            by_name.setdefault(name, []).append(_layer_data.Layer(layer.name, layer.path, body))
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
                        leaf=io.render_key(("presets", name, *io.split_key(entry.leaf))),
                        value=io.read_toml_leaf(
                            io.read_toml_file(entry.path),
                            io.render_key(("presets", name, *io.split_key(entry.file_key))),
                        ),
                        layer=entry.layer,
                        path=entry.path,
                        file_key=io.render_key(("presets", name, *io.split_key(entry.file_key))),
                        is_table=entry.is_table,
                        error_type=entry.error_type,
                    )
                    for entry in leaves
                ),
                diagnosis.blocked,
            )
    return ConfigDiagnosis((), None)


def find_invalid_entries(
    repo_root: pathlib.Path, *, machine: pathlib.Path | None = None
) -> ConfigDiagnosis:
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
            cleaned, _ = _presets.strip_presets(_layer_data.discover_layers(repo_root, ()))
            for end in range(1, len(cleaned) + 1):
                diagnosis = _diagnose_layers(cleaned[:end], only_layer=None)
                leaves = tuple(entry for entry in diagnosis.removable if not entry.is_table)
                if leaves:
                    return ConfigDiagnosis(leaves, None)
        layers = _fix_scope_layers(repo_root, machine)
    except _base.ConfigError as exc:
        return ConfigDiagnosis((), str(exc))
    return _diagnose_layers(layers, only_layer=only)


def leaf_keys(eff: _layer_data.EffectiveConfig) -> list[str]:
    """Return every dotted leaf path of the effective config, sorted.

    Args:
        eff: The effective config.

    Returns:
        The paths.
    """
    return sorted(_layer_data.config_leaves(eff.config))


def effective_leaf(eff: _layer_data.EffectiveConfig, dotted_key: str) -> tuple[Any, str] | None:
    """Return a leaf's value and the layer that set it, as `config show` reports them.

    Args:
        eff: The effective config.
        dotted_key: The leaf, or `presets.<name>.<leaf>` for a preset's own value.

    Returns:
        `(value, source)`, the source `default` when no layer set it and `unset` for a preset
        leaf no layer authored; None when the key is not a leaf.
    """
    leaves = _layer_data.config_leaves(eff.config)
    parts = io.split_key(dotted_key)
    dotted_key = io.render_key(parts)
    if parts[0] == "presets" and len(parts) > 2:
        # A preset leaf may name a role or provider absent from the effective config.
        name, leaf = parts[1], io.render_key(parts[2:])
        for layer in reversed(eff.layers):
            table = _file_presets(layer.path).get(name)
            if isinstance(table, dict) and (value := io.read_toml_leaf(table, leaf)) is not None:
                return value, f"preset {name} ({layer.name})"
        if leaf in leaves:
            return None, "unset"
        return None
    if dotted_key in leaves:
        return leaves[dotted_key], eff.sources.get(dotted_key, "default")
    nested = _layer_data.flatten_leaves(eff.config.model_dump(mode="python"))
    if dotted_key not in nested:
        return None
    _, sources = _layer_data.merge_layers(list(eff.layers))
    return nested[dotted_key], sources.get(dotted_key, "default")


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
            lines.append(f"{io.toml_key(key)} = {io.format_toml_value(value)}")
        lines.append("")
    parts = io.split_key(path) if path else []
    for key, sub in subtables.items():
        _emit_table(io.render_key((*parts, key)), sub, lines)
    for key, arr in arraytables.items():
        for item in arr:
            lines.append(f"[[{io.render_key((*parts, key))}]]")
            for k2, v2 in item.items():
                if v2 is not None:
                    lines.append(f"{io.toml_key(k2)} = {io.format_toml_value(v2)}")
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
    config: config_model.Config,
    *,
    explicit_leaves: frozenset[str] | None = None,
    keep_presets_from: pathlib.Path | None = None,
    keep_preset_selector: bool = False,
) -> str:
    """Render the resolved config as TOML, optionally omitting unset defaults.

    Args:
        config: The validated config.
        explicit_leaves: Write these dotted leaves and values differing from `Config()`.
            None writes every value, as `config fill` does.
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
    if explicit_leaves is not None:
        defaults = _layer_data.config_leaves(config_model.Config())
        selected = {
            leaf
            for leaf, value in _layer_data.config_leaves(config).items()
            if leaf in explicit_leaves or leaf not in defaults or value != defaults[leaf]
        }

        def select_leaves(table: dict[str, Any], prefix: str = "") -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in table.items():
                path = f"{prefix}{key}"
                if path in selected:
                    out[key] = value
                elif isinstance(value, dict) and (sub := select_leaves(value, f"{path}.")):
                    out[key] = sub
            return out

        data = select_leaves(data)
    data = data if keep_preset_selector else {k: v for k, v in data.items() if k != "preset"}
    lines: list[str] = [
        "# agent6 effective config."
        if explicit_leaves is not None
        else "# agent6 effective config, materialized by `agent6 config fill`.",
        "# Every value below is explicit; edit freely.",
        "",
    ]
    ordered = [s for s in SECTION_ORDER if s in data]
    ordered += [s for s in data if s not in SECTION_ORDER]
    # A top-level scalar (`preset`) must precede every `[section]` in TOML.
    for section in ordered:
        value = data[section]
        if value is not None and not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f"{section} = {io.format_toml_value(value)}")
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
                        lines.append(f"{io.toml_key(k2)} = {io.format_toml_value(v2)}")
                lines.append("")
    if kept := _file_presets(keep_presets_from):
        _emit_table("presets", kept, lines)
    return "\n".join(lines).rstrip("\n") + "\n"


def _file_presets(path: pathlib.Path | None) -> dict[str, Any]:
    """Return the `[presets.*]` tables a file defines itself.

    Args:
        path: The file, or None.

    Returns:
        The tables by name; {} when the file is absent or defines none.
    """
    if path is None or not path.is_file():
        return {}
    presets = _layer_data.read_toml(path).get("presets")
    return presets if isinstance(presets, dict) else {}
