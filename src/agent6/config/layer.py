# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Layered config resolution + auditing for agent6.

Config is assembled from up to four layers, lowest precedence first:

1. ``default``, the secure defaults baked into the pydantic model,
2. ``global`` , ``$XDG_CONFIG_HOME/agent6/config.toml`` (user-wide),
3. ``repo``   , the per-repo config under the state dir (out of the workspace,
   ``<state-base>/<repo-id>/config.toml``; see ``agent6.paths.state_dir``),
4. ``flag``   , an explicit ``--config FILE`` (power users / CI).

Raw TOML dicts are deep-merged in that order and validated **once**, so a
repo can override a single field without restating the rest. Every leaf
remembers which layer last set it, which powers ``agent6 config show``,
the audit surface that makes the effective config and its provenance
obvious at a glance.

A selected ``profile`` preset is injected just ABOVE the config layer that
SELECTED it (``--profile`` flag / repo / global top-level ``profile``), so the
profile OVERRIDES that config while a more-specific config layer (or an explicit
``--config FILE`` / machine overlay) still overrides the profile. Only the
most-specific source's profile is injected -- global and repo presets never
stack. See :func:`_apply_profile`.
"""

from __future__ import annotations

import contextlib
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from agent6.config.io import (
    parse_cli_value,
    remove_toml_leaf,
    upsert_toml_leaf,
    upsert_toml_table,
)
from agent6.config.model import (
    BUILTIN_PROFILES,
    AnthropicProviderEntry,
    Config,
    ConfigError,
    Deployment,
    OpenAIProviderEntry,
    resolve_profile,
    validate_config,
)
from agent6.paths import (
    chown_to_real_user,
    global_config_path,
    repo_config_path,
    state_dir,
)

LayerName = Literal["default", "profile", "global", "repo", "flag", "machine"]

# Display order for `config show` / `config fill`, derived FROM the Config model's
# field declaration order so a new section can never be silently omitted. Scalar
# top-level fields (e.g. `profile`) carry no `[section]` table and are rendered
# inline by their parent, so the section ordering only needs the table names; we
# keep every field name here and the lookups below tolerate non-section entries.
SECTION_ORDER = tuple(Config.model_fields)


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


def _global_state_dir() -> str | None:
    """Read ``[agent6].state_dir`` (the state BASE) from the GLOBAL config only.

    Resolved *before* the layered merge because it locates the directory the
    per-repo config lives in. Honored only from the global config;
    ``_forbid_repo_state_dir`` rejects it in any other layer.
    """
    gpath = global_config_path()
    if not gpath.is_file():
        return None
    data = _read_toml(gpath)
    section = data.get("agent6")
    if isinstance(section, dict):
        sd = section.get("state_dir")
        if isinstance(sd, str):
            # Raw pre-model read locating the per-repo config dir, so it runs
            # BEFORE the Config model (which also validates state_dir). Apply
            # the same absolute-path check here; fail loudly, don't drift.
            if not Path(sd).expanduser().is_absolute():
                raise ConfigError(
                    f"[agent6].state_dir in {gpath} must be an absolute path, got {sd!r}"
                )
            return sd
    return None


def _forbid_repo_state_dir(layer_name: str, data: dict[str, Any]) -> None:
    """Refuse ``state_dir`` in a repo/flag/overlay layer (global-only setting)."""
    section = data.get("agent6")
    if isinstance(section, dict) and "state_dir" in section:
        raise ConfigError(
            f"[agent6].state_dir may only be set in the global config"
            f" ({global_config_path()}), not in the {layer_name} config — it"
            " locates the directory the per-repo config itself lives in."
        )


def resolved_state_dir(repo_root: Path) -> Path:
    """The per-repo state dir for *repo_root*, honoring the global base override."""
    return state_dir(repo_root, _global_state_dir())


def repo_config_path_for(repo_root: Path) -> Path:
    """The per-repo config path for *repo_root* (out of the workspace)."""
    return repo_config_path(repo_root, _global_state_dir())


def discover_layers(repo_root: Path, explicit_path: Path | None) -> list[Layer]:
    """The config layers that exist, in precedence order (low -> high).

    The repo config lives out of the workspace under the state dir, whose base
    comes from the global config's ``[agent6].state_dir`` (or the XDG default).
    """
    layers: list[Layer] = []
    gpath = global_config_path()
    if gpath.is_file():
        layers.append(Layer("global", gpath, _read_toml(gpath)))
    base = _global_state_dir()
    rpath = repo_config_path(repo_root, base)
    if rpath.is_file():
        data = _read_toml(rpath)
        _forbid_repo_state_dir("repo", data)
        layers.append(Layer("repo", rpath, data))
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigError(f"--config file not found: {explicit_path}")
        data = _read_toml(explicit_path)
        _forbid_repo_state_dir("--config", data)
        layers.append(Layer("flag", explicit_path, data))
    return layers


def available_profile_names(repo_root: Path, explicit_path: Path | None = None) -> list[str]:
    """Profile names a chooser can offer: the built-ins plus the user's custom
    ``[profiles.<name>]`` tables (read from the config layers, the same source
    ``--profile`` resolves against), sorted + de-duplicated. A config-read failure
    degrades to the built-ins alone, so a caller (e.g. the TUI's new-work chooser)
    never blocks on a bad config."""
    names: set[str] = set(BUILTIN_PROFILES)
    with contextlib.suppress(Exception):
        for layer in discover_layers(repo_root, explicit_path):
            prof = layer.data.get("profiles")
            if isinstance(prof, dict):
                names.update(prof.keys())
    return sorted(names)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        existing = out.get(key)
        if isinstance(val, dict) and isinstance(existing, dict):
            # A discriminated dict (e.g. a [providers.<name>] entry) whose
            # `api_format` changes between layers must REPLACE, not deep-merge:
            # the lower layer's format-specific keys (an anthropic prompt_caching,
            # say) are invalid under the new format and would otherwise survive
            # the merge and surface as a confusing extra_forbidden error.
            if (
                "api_format" in val
                and "api_format" in existing
                and val.get("api_format") != existing.get("api_format")
            ):
                out[key] = val
            else:
                out[key] = _deep_merge(existing, val)
        else:
            out[key] = val
    return out


def flatten_leaves(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted leaf paths.

    Lists (incl. arrays of tables) are treated as leaves, their provenance
    is the whole array, not individual elements.
    """
    out: dict[str, Any] = {}
    for key, val in data.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            out.update(flatten_leaves(val, prefix=f"{path}."))
        else:
            out[path] = val
    return out


def _leaf_fix_hint(
    layers: list[Layer], source_of_leaf: dict[str, str]
) -> Callable[[str], str | None]:
    """Locator for validate_config: a dotted leaf -> "set in <layer> <path>; fix:
    <command>", or None when the value came from a built-in default (nothing to
    fix). Lets a stale value name the exact file and the command to correct it."""
    by_name = {layer.name: layer for layer in layers}

    def locate(leaf: str) -> str | None:
        layer = by_name.get(source_of_leaf.get(leaf, ""))
        if layer is None or layer.path is None:
            return None
        if layer.name == "repo":
            fix = f"agent6 config set --repo {leaf} <value>"
        elif layer.name == "flag":
            fix = f"edit {layer.path}"
        else:
            fix = f"agent6 config set {leaf} <value>"
        return f"    set in the {layer.name} config: {layer.path}\n    fix: {fix}"

    return locate


def _effective_from_layers(layers: list[Layer], *, source: str) -> EffectiveConfig:
    """Merge *layers* low->high, validate, and build the per-leaf source map."""
    merged: dict[str, Any] = {}
    source_of_leaf: dict[str, str] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer.data)
        for leaf in flatten_leaves(layer.data):
            source_of_leaf[leaf] = layer.name
    config = validate_config(merged, source=source, locate=_leaf_fix_hint(layers, source_of_leaf))
    # Source map over the *effective* config: every leaf the model
    # produced, attributed to the layer that set it or "default".
    effective_leaves = flatten_leaves(config.model_dump(mode="python"))
    sources = {leaf: source_of_leaf.get(leaf, "default") for leaf in effective_leaves}
    return EffectiveConfig(config=config, sources=sources, layers=tuple(layers))


def _own_profile(layer: Layer) -> str:
    """A layer's OWN raw top-level ``profile`` (not the merged value), or ""."""
    return str(layer.data.get("profile", "") or "")


def _select_profile(cleaned: list[Layer], profile_override: str) -> tuple[str, str]:
    """Pick the (profile name, source) most-specific first from each layer's OWN
    raw top-level ``profile`` (never stacking global+repo): the ``--profile``
    flag, else the ``repo`` layer's field, else the ``global`` layer's field,
    else ("", "none")."""
    if profile_override:
        return profile_override, "flag"
    by_name = {layer.name: layer for layer in cleaned}
    for source in ("repo", "global"):
        layer = by_name.get(source)
        if layer is not None and (name := _own_profile(layer)):
            return name, source
    return "", "none"


def _insert_profile(cleaned: list[Layer], preset: Layer, source: str) -> list[Layer]:
    """Splice *preset* into *cleaned* at the position for its *source*.

    ``global``/``repo`` -> right AFTER that config layer (so the profile
    overrides it but the more-specific config layer / flag still wins). ``flag``
    (``--profile``) -> just BELOW an explicit ``--config FILE`` / machine overlay
    if present (those still win), else appended last (overrides all config).
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


def _apply_profile(layers: list[Layer], profile_override: str) -> list[Layer]:
    """Strip ``[profiles]`` tables out of the user layers (they are meta-config,
    not part of the validated Config) and inject the selected profile preset
    just ABOVE the config layer that SELECTED it, so the profile OVERRIDES that
    config while a more-specific config layer (or an explicit ``--config FILE`` /
    machine overlay) still overrides the profile. Only the most-specific source's
    profile is injected -- global and repo presets never stack.

    Source is chosen by :func:`_select_profile` (``--profile`` flag > repo's own
    top-level ``profile`` > global's own), and the preset is spliced in by
    :func:`_insert_profile`. Resulting precedence (low->high): default <
    global-config < [profile if global-selected] < repo-config <
    [profile if repo-selected] < [profile if --flag] < flag(``--config FILE``) <
    machine-overlay.
    """
    cleaned: list[Layer] = []
    user_profiles: dict[str, Any] = {}
    for layer in layers:
        data = dict(layer.data)
        prof = data.pop("profiles", None)
        if isinstance(prof, dict):
            user_profiles = _deep_merge(user_profiles, prof)
        cleaned.append(Layer(layer.name, layer.path, data))

    name, source = _select_profile(cleaned, profile_override)
    overrides = resolve_profile(name, user_profiles)
    if not overrides:
        return cleaned
    return _insert_profile(cleaned, Layer("profile", None, overrides), source)


def load_effective(
    repo_root: Path, explicit_path: Path | None = None, *, profile: str = ""
) -> EffectiveConfig:
    """Merge + validate all layers and record per-leaf provenance. A named
    ``profile`` (CLI flag or top-level ``profile``) is injected just above the
    config layer that selected it, so it OVERRIDES that config (a more-specific
    config layer / flag still wins); ``[profiles.<name>]`` tables in config
    define custom ones. See :func:`_apply_profile`."""
    layers = discover_layers(repo_root, explicit_path)
    layers = _apply_profile(layers, profile)
    return _effective_from_layers(layers, source="(merged config layers)")


def load_effective_with_overlay(repo_root: Path, overlay: dict[str, Any]) -> EffectiveConfig:
    """Like :func:`load_effective` but with *overlay* as the highest layer.

    Used by `agent6 machine run` to apply a machine file's ``[config]``
    table on top of the repo/global/default layers. The overlay is merged
    and validated exactly like a config file; its leaves are labelled
    ``machine`` in the provenance map (``config show`` style).
    """
    layers = discover_layers(repo_root, None)
    if overlay:
        _forbid_repo_state_dir("machine overlay", overlay)
        layers = [*layers, Layer("machine", None, overlay)]
    # Apply the selected profile (and strip [profiles] tables) just like
    # load_effective, so a user's [profiles.<name>] + [workflow].profile work
    # under `machine run` / `config --machine` instead of failing validation.
    layers = _apply_profile(layers, "")
    return _effective_from_layers(layers, source="(merged config layers + machine overlay)")


def leaf_keys(eff: EffectiveConfig) -> list[str]:
    """Every dotted leaf path in the effective config, sorted (for completion)."""
    return sorted(flatten_leaves(eff.config.model_dump(mode="python")))


def effective_leaf(eff: EffectiveConfig, dotted_key: str) -> tuple[Any, str] | None:
    """The ``(value, source-layer)`` for *dotted_key*, or None if it is not a leaf.

    Mirrors `config show`: the value comes from the merged+validated config and
    the source is the layer that set it (``default`` when no layer did).
    """
    leaves = flatten_leaves(eff.config.model_dump(mode="python"))
    if dotted_key not in leaves:
        return None
    return leaves[dotted_key], eff.sources.get(dotted_key, "default")


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

    ``None`` values are skipped, an unset optional field materializes as
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


def materialize(config: Config, *, for_repo: bool = False) -> str:
    """Render the fully-resolved config as a complete TOML document.

    Used by ``agent6 config fill`` to snapshot every effective value into
    one explicit file (handy before tightening defaults or for an audit).
    When ``for_repo`` is set, the global-only ``[agent6].state_dir``
    is dropped (it is invalid in a per-repo config).
    """
    data = config.model_dump(mode="python")
    if for_repo and isinstance(data.get("agent6"), dict):
        data["agent6"].pop("state_dir", None)
    lines: list[str] = [
        "# agent6 effective config, materialized by `agent6 config fill`.",
        "# Every value below is explicit; edit freely.",
        "",
    ]
    ordered = [s for s in SECTION_ORDER if s in data]
    ordered += [s for s in data if s not in SECTION_ORDER]
    # Top-level scalar fields (e.g. `profile`) carry no table header and must
    # precede every `[section]` in TOML, so emit them first as bare key=value.
    for section in ordered:
        value = data[section]
        if value is not None and not isinstance(value, dict) and not _is_table_array(value):
            lines.append(f"{section} = {_toml_scalar(value)}")
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
                    if v2 is not None and not isinstance(v2, dict):
                        lines.append(f"{k2} = {_toml_scalar(v2)}")
                lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Shared edit path: the `config` CLI and the TUI/web config editors write
# through these, so a value set from any UI is parsed, persisted (comment-
# preserving), re-validated, and rolled back on failure identically.
# ---------------------------------------------------------------------------


def _write_target(repo_root: Path, *, to_repo: bool) -> Path:
    return repo_config_path_for(repo_root) if to_repo else global_config_path()


def _revalidate(repo_root: Path, target: Path, prior: str | None) -> str | None:
    """Re-load the merged config after an edit; restore *prior* (or delete a
    freshly-created file) and return the error string if the edit is invalid."""
    try:
        load_effective(repo_root, None)
    except ConfigError as exc:
        if prior is None:
            target.unlink(missing_ok=True)
        else:
            target.write_text(prior, encoding="utf-8")
        return str(exc)
    return None


def set_config_value(
    repo_root: Path, dotted_key: str, raw_value: str, *, to_repo: bool = False
) -> str | None:
    """Set one leaf in the global (or, with *to_repo*, the repo) config.

    *raw_value* is interpreted exactly as ``config set`` interprets a CLI value
    (``true``/numbers/arrays parse; a bare word stays a string). Returns an
    error string when the edit produced an invalid config (the file is rolled
    back and left as it was), else None.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    try:
        upsert_toml_leaf(target, dotted_key, parse_cli_value(raw_value))
    except ValueError as exc:
        return str(exc)
    err = _revalidate(repo_root, target, prior)
    if err is None:
        chown_to_real_user(target.parent)
        chown_to_real_user(target)
    return err


def set_config_table(
    repo_root: Path,
    table: str,
    fields: dict[str, str | bool | None],
    *,
    to_repo: bool = False,
) -> str | None:
    """Insert/replace a whole ``[table]`` block in one shot (e.g. a new
    ``[providers.<name>]`` entry from the TUI's add-provider form). Revalidates
    the merged config and rolls the file back on failure. Returns an error string
    on invalid config, else None. ``None`` field values are omitted."""
    target = _write_target(repo_root, to_repo=to_repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = target.read_text(encoding="utf-8") if target.is_file() else None
    try:
        upsert_toml_table(target, table, fields)
    except ValueError as exc:
        return str(exc)
    err = _revalidate(repo_root, target, prior)
    if err is None:
        chown_to_real_user(target.parent)
        chown_to_real_user(target)
    return err


def provider_choices() -> dict[str, list[str]]:
    """Fixed-choice fields for the add-provider form, read from the schema so
    they never drift: the api_format discriminator (per provider subclass) and
    the deployment profiles."""
    formats: list[str] = []
    for model in (AnthropicProviderEntry, OpenAIProviderEntry):
        formats.extend(get_args(model.model_fields["api_format"].annotation))
    return {"api_format": formats, "deployment": list(get_args(Deployment))}


# Known provider presets, keyed by the conventional provider NAME used as the
# [providers.<name>] table key. Maps a name to its api_format and, for
# OpenAI-compatible hosts, the default base_url. Both `agent6 connect` and the
# TUI add-provider form consult this so well-known names (openrouter, ollama)
# land on the right host instead of the bare (api_format, deployment) fallback
# in `config._default_base_url` -- which only knows api.openai.com for the
# `openai` format and would otherwise point an "openrouter" provider at OpenAI.
# Advanced deployments (vertex/azure/token_command) are hand-edited per CONFIG.md.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "anthropic": {"api_format": "anthropic"},
    "openai": {"api_format": "openai", "base_url": "https://api.openai.com/v1"},
    "openrouter": {"api_format": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "ollama": {"api_format": "openai", "base_url": "http://localhost:11434/v1"},
}


def unset_config_value(repo_root: Path, dotted_key: str, *, to_repo: bool = False) -> str | None:
    """Remove one leaf so it reverts to the next layer / built-in default.

    Re-validates and rolls back on failure. Returns None on success, including
    the no-op case where the key was not set in the target file.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return None
    prior = target.read_text(encoding="utf-8")
    try:
        removed = remove_toml_leaf(target, dotted_key)
    except ValueError as exc:
        return str(exc)
    if not removed:
        return None
    err = _revalidate(repo_root, target, prior)
    if err is None:
        chown_to_real_user(target)
    return err
