# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Layered config resolution + auditing for agent6.

Config is assembled from up to four layers, lowest precedence first:

1. ``default`` — the secure defaults baked into the pydantic model,
2. ``global``  — ``$XDG_CONFIG_HOME/agent6/config.toml`` (user-wide),
3. ``repo``    — ``./.agent6/config.toml`` (this repository),
4. ``flag``    — an explicit ``--config FILE`` (power users / CI).

Raw TOML dicts are deep-merged in that order and validated **once**, so a
repo can override a single field without restating the rest. Every leaf
remembers which layer last set it, which powers ``agent6 config show`` —
the audit surface that makes the effective config and its provenance
obvious at a glance.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent6.config import Config, ConfigError, validate_config
from agent6.paths import global_config_path, repo_config_path

LayerName = Literal["default", "global", "repo", "flag"]

# Display order for `config show` / `config fill` (model definition order).
_SECTION_ORDER = (
    "agent6",
    "providers",
    "models",
    "sandbox",
    "git",
    "workflow",
    "budget",
    "notify",
    "mcp",
)


@dataclass(frozen=True, slots=True)
class Layer:
    name: LayerName
    path: Path | None
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    config: Config
    sources: dict[str, str]  # dotted leaf path -> layer name
    layers: tuple[Layer, ...]  # the layers that actually contributed (existing files)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML ({path}): {exc}") from exc


def discover_layers(repo_root: Path, explicit_path: Path | None) -> list[Layer]:
    """The config layers that exist, in precedence order (low -> high)."""
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
        layers.append(Layer("flag", explicit_path, _read_toml(explicit_path)))
    return layers


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf paths.

    Lists (incl. arrays of tables) are treated as leaves — their provenance
    is the whole array, not individual elements.
    """
    out: dict[str, Any] = {}
    for key, val in data.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            out.update(_flatten(val, prefix=f"{path}."))
        else:
            out[path] = val
    return out


def load_effective(repo_root: Path, explicit_path: Path | None = None) -> EffectiveConfig:
    """Merge + validate all layers and record per-leaf provenance."""
    layers = discover_layers(repo_root, explicit_path)
    merged: dict[str, Any] = {}
    source_of_leaf: dict[str, str] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer.data)
        for leaf in _flatten(layer.data):
            source_of_leaf[leaf] = layer.name
    config = validate_config(merged, source="(merged config layers)")
    # Source map over the *effective* config: every leaf the model
    # produced, attributed to the layer that set it or "default".
    effective_leaves = _flatten(config.model_dump(mode="python"))
    sources: dict[str, str] = {}
    for leaf in effective_leaves:
        sources[leaf] = source_of_leaf.get(leaf, "default")
    return EffectiveConfig(config=config, sources=sources, layers=tuple(layers))


# ---------------------------------------------------------------------------
# Rendering: `config show`
# ---------------------------------------------------------------------------


def _fmt_value(val: Any) -> str:
    if val is None:
        return "(unset)"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        if not val:
            return "[]"
        return "[" + ", ".join(_fmt_value(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{...}" if val else "{}"
    return str(val)


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "\u2026"


def render_show(eff: EffectiveConfig, *, as_json: bool = False) -> str:
    """Render the effective config + provenance.

    Plain mode is a section-grouped, fixed-width 3-column table
    (key / value / source) with a leading ``*`` on rows that override the
    default — no box drawing, so it never wraps badly. ``*`` rows are the
    ones to eyeball. JSON mode emits ``{leaf: {value, source}}``.
    """
    leaves = _flatten(eff.config.model_dump(mode="python"))
    if as_json:
        payload = {
            leaf: {"value": leaves[leaf], "source": eff.sources.get(leaf, "default")}
            for leaf in leaves
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str)

    # Group leaves by top-level section, preserving a friendly order.
    by_section: dict[str, list[str]] = {}
    for leaf in leaves:
        section = leaf.split(".", 1)[0]
        by_section.setdefault(section, []).append(leaf)
    ordered_sections = [s for s in _SECTION_ORDER if s in by_section]
    ordered_sections += [s for s in by_section if s not in _SECTION_ORDER]

    # Column widths (capped to avoid wide-terminal sprawl / narrow wrap).
    key_w = min(max((len(leaf) for leaf in leaves), default=10) + 1, 40)
    val_w = 40
    lines: list[str] = []
    sources_seen: set[str] = set()
    for section in ordered_sections:
        section_leaves = by_section[section]
        if not section_leaves:
            continue
        lines.append(f"[{section}]")
        for leaf in section_leaves:
            value = _fmt_value(leaves[leaf])
            source = eff.sources.get(leaf, "default")
            sources_seen.add(source)
            mark = " " if source == "default" else "*"
            short_key = _truncate(leaf, key_w)
            short_val = _truncate(value, val_w)
            lines.append(f"{mark} {short_key:<{key_w}} {short_val:<{val_w}} {source}")
        lines.append("")
    legend_layers = ", ".join(
        f"{lyr.name}={lyr.path}" for lyr in eff.layers if lyr.path is not None
    )
    lines.append("source: default | " + (legend_layers or "(no config files; all defaults)"))
    lines.append("* = overrides the built-in default")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Materialize: `config fill`
# ---------------------------------------------------------------------------


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(v) for v in value) + "]"
    return str(value)


def _emit_table(path: str, data: dict[str, Any], lines: list[str]) -> None:
    """Emit one TOML table (and recurse into sub-tables / arrays of tables).

    ``None`` values are skipped — an unset optional field materializes as
    absent, i.e. "use the default".
    """
    scalars = {
        k: v
        for k, v in data.items()
        if v is not None and not isinstance(v, dict) and not _is_table_array(v)
    }
    subtables = {k: v for k, v in data.items() if isinstance(v, dict) and v}
    arraytables = {k: v for k, v in data.items() if _is_table_array(v)}
    # Emit a header for this path only when it carries scalar keys, or when
    # it is a genuine leaf table (no children at all). A pure parent table
    # like [providers] / [models] is left implicit so we don't print an
    # empty header above its [providers.<name>] children.
    is_leaf = not subtables and not arraytables
    if scalars or is_leaf:
        lines.append(f"[{path}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    for key, sub in subtables.items():
        _emit_table(f"{path}.{key}" if path else key, sub, lines)
    for key, arr in arraytables.items():
        for item in arr:
            lines.append(f"[[{path}.{key}]]" if path else f"[[{key}]]")
            for k2, v2 in item.items():
                if v2 is not None and not isinstance(v2, dict):
                    lines.append(f"{k2} = {_toml_scalar(v2)}")
            lines.append("")


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, dict) for v in value)
    )


def materialize(config: Config) -> str:
    """Render the fully-resolved config as a complete TOML document.

    Used by ``agent6 config fill`` to snapshot every effective value into
    one explicit file (handy before tightening defaults or for an audit).
    """
    data = config.model_dump(mode="python")
    lines: list[str] = [
        "# agent6 effective config, materialized by `agent6 config fill`.",
        "# Every value below is explicit; edit freely.",
        "",
    ]
    ordered = [s for s in _SECTION_ORDER if s in data]
    ordered += [s for s in data if s not in _SECTION_ORDER]
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
                    if v2 is not None and not isinstance(v2, dict):
                        lines.append(f"{k2} = {_toml_scalar(v2)}")
                lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
