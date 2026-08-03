# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Safe mutation of config files: validate a written value standalone, upsert or
remove it through the comment-preserving TOML surgery in ``io``, then revalidate
the merged config and roll back if this edit broke it. The ``set_config_*`` API
the ``config`` CLI and the TUI/web/init/connect editors all write through, so a
value set from any surface is validated and rolled back identically."""

from __future__ import annotations

import contextlib
import difflib
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from agent6.config.io import (
    parse_cli_value,
    read_toml_file,
    remove_toml_leaf,
    upsert_toml_leaf,
    upsert_toml_table,
)
from agent6.config.layer import (
    flatten_leaves,
    leaf_keys,
    load_effective,
    repo_config_path_for,
)
from agent6.config.model import (
    AnthropicProviderEntry,
    Config,
    ConfigError,
    Deployment,
    OpenAIProviderEntry,
)
from agent6.paths import chown_to_real_user, global_config_path, mkdir_for_real_user
from agent6.portable import atomic_write, locked_file


def _write_target(repo_root: Path, *, to_repo: bool) -> Path:
    return repo_config_path_for(repo_root) if to_repo else global_config_path()


def _prepare_write_target(repo_root: Path, *, to_repo: bool) -> Path:
    """The config file to write, its directory created and handed back to the
    real operator. Under ``sudo`` the dir is created as root; the handover is at
    creation, so a failed or killed write never strands a root-owned dir a later
    non-root write cannot create its atomic-write temp file in."""
    target = _write_target(repo_root, to_repo=to_repo)
    mkdir_for_real_user(target.parent)
    return target


@contextlib.contextmanager
def _writing_config(target: Path) -> Generator[bool]:
    """Hold the config write lock, handing *target* back to the real operator on
    every exit path. Yields whether the lock is actually held (see
    :func:`agent6.portable.locked_file`), for :func:`_revalidate`'s rollback.
    Under ``sudo`` every publish -- including the rollback's ``atomic_write`` onto
    a new inode -- creates the file as root, so the handover is unconditional."""
    with locked_file(target) as held:
        try:
            yield held
        finally:
            chown_to_real_user(target)


def target_unparseable(target: Path) -> bool:
    """Whether *target* itself is no longer valid TOML (a missing file is fine)."""
    try:
        read_toml_file(target)
    except ConfigError:
        return True
    return False


def config_is_valid(repo_root: Path) -> bool:
    """Whether the merged config currently loads. Measured BEFORE a write so
    :func:`_revalidate` can tell "this edit broke it" from "it was already
    broken elsewhere"."""
    try:
        load_effective(repo_root, None)
    except ConfigError:
        return False
    return True


# Appended to a validation error when the config lock FAILED OPEN (a stale
# root-owned .lock): a snapshot restore without the lock could erase a
# concurrent writer's update, so the write is kept and the operator undoes it.
_KEPT_NO_LOCK = (
    "(kept as written: the config lock could not be taken, so an automatic"
    " rollback might erase a concurrent edit; undo by hand or run `agent6 config fix`)"
)

PROVIDER_MEMBERS = (AnthropicProviderEntry, OpenAIProviderEntry)


def provider_field_error(key: str, leaf: str, value: object) -> str | None:
    """Validate a ``providers.<name>.<leaf>`` write against the union members
    directly, since a minimal standalone dict lacks the entry's discriminator.
    A leaf on no member is an unknown key (the members' own field pool is the
    did-you-mean universe); a value every owning member rejects is invalid.
    None when some member accepts it, so a partial entry stays writable."""
    fields = sorted({f for m in PROVIDER_MEMBERS for f in m.model_fields})
    if leaf not in fields:
        close = difflib.get_close_matches(leaf, fields, n=2)
        hint = f". Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        return f"unknown provider key {key!r}{hint} (see `agent6 config show`)"
    errors: list[str] = []
    for member in PROVIDER_MEMBERS:
        if leaf not in member.model_fields:
            continue
        fmt = get_args(member.model_fields["api_format"].annotation)[0]
        try:
            # Validate the leaf against the whole member so its @field_validators
            # run; seed the api_format discriminator the member requires. Only a
            # rejection AT this leaf counts -- a complaint about another unset
            # field means the leaf itself is acceptable to this member.
            member.model_validate({"api_format": fmt, leaf: value})
            return None
        except ValidationError as exc:
            leaf_errs = [e["msg"] for e in exc.errors() if e["loc"] == (leaf,)]
            if not leaf_errs:
                return None
            errors.append(leaf_errs[0])
    return f"{key}: {errors[0]}" if errors else None


def unknown_key_error(key: str) -> str:
    """A human message for a key the schema forbids, with a did-you-mean.

    The pool is usually the SCHEMA defaults: this runs after the unknown key
    was already written, so the merged config no longer loads and the live
    branch (which would add real provider tables) only survives when a higher
    layer masks the write."""
    try:
        pool = leaf_keys(load_effective(Path.cwd(), None))
    except ConfigError:
        pool = sorted(flatten_leaves(Config().model_dump(mode="python")))
    close = difflib.get_close_matches(key, pool, n=2)
    hint = f". Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
    return f"unknown config key {key!r}{hint} (see `agent6 config show`)"


def written_value_error(key: str, value: object) -> str | None:
    """Validate the just-written ``key = value`` against the Config model on its
    own (a minimal dict, defaults for the rest), independent of the layer merge.
    A write of an invalid value into a layer that a HIGHER layer masks (e.g. a
    global set the repo overlay shadows) would otherwise validate the merged
    config -- where the value is hidden -- and land the bad value in the file,
    only to explode later where the mask is absent. THE one owner every writer
    uses (`config set` and the engine-level set_config_* the TUI, web, init and
    connect drive), so all of them validate the written value identically.
    Rejects only when the error sits exactly at *key*, or at a parent of it for
    the schema's extra_forbidden (an unknown key or section), so a partial
    dynamic entry (a provider filled in over several sets) is not falsely
    reverted."""
    if key == "presets" or key.startswith("presets."):
        # [presets.*] is meta-config the loader strips BEFORE validation
        # (_apply_preset), so the Config schema forbids it by design; the
        # standalone check would falsely reject every legitimate preset write.
        # The merged re-validation still catches a preset body that breaks.
        return None
    parts = key.split(".")
    if parts[0] == "providers" and len(parts) == 3:
        return provider_field_error(key, parts[2], value)
    nested: dict[str, object] = {}
    cur = nested
    for part in parts[:-1]:
        child: dict[str, object] = {}
        cur[part] = child
        cur = child
    cur[parts[-1]] = value
    try:
        Config.model_validate(nested)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            if err["type"] == "extra_forbidden" and (loc == key or key.startswith(loc + ".")):
                # An unknown top-level section errors at the SECTION (a parent
                # loc), not the leaf; both deserve the same friendly message,
                # not pydantic-speak or the merged-layer dump.
                return unknown_key_error(key)
            if loc == key:
                msg = err["msg"]
                if err["type"] == "bool_parsing":
                    msg = f"expected true or false, got {value!r}"
                return f"{key}: {msg}"
    except ConfigError as exc:
        return str(exc)
    return None


def _revalidate(
    repo_root: Path,
    target: Path,
    prior: str | None,
    *,
    was_valid: bool,
    held: bool = True,
    written: Sequence[tuple[str, object]] = (),
) -> str | None:
    """Re-load the merged config after an edit; restore *prior* (or delete a
    freshly-created file) and return the error string if THIS edit broke it. The
    caller holds ``locked_file`` across the whole write+revalidate+rollback
    cycle, so the atomic rollback cannot restore a snapshot over a concurrent
    writer's update. When the lock FAILED OPEN (*held* False: a stale root-owned
    ``.lock``) the restore could do exactly that, so the edit is kept and the
    error surfaced -- the keep-and-warn contract an already-invalid config gets.

    A config that was ALREADY invalid keeps the edit: rolling back on any error
    would let a stale value in an unedited layer refuse every write. The
    pre-existing error still surfaces on the next run, and `agent6 config fix`
    removes it.

    *written* is the ``(key, value)`` pairs this edit wrote, each validated
    STANDALONE via :func:`written_value_error` so a value a higher layer masks in
    the merge is caught here, not left to explode once the mask is gone."""

    def _rollback(err: str) -> str:
        # A snapshot restore without the lock (held False) could erase a
        # concurrent writer's update, so keep the write and say so instead.
        if not held:
            return f"{err}\n{_KEPT_NO_LOCK}"
        if prior is None:
            target.unlink(missing_ok=True)
        else:
            atomic_write(target, prior)
        return err

    for wkey, wvalue in written:
        value_err = written_value_error(wkey, wvalue)
        if value_err is not None:
            return _rollback(value_err)
    try:
        load_effective(repo_root, None)
    except ConfigError as exc:
        # A target that no longer PARSES is always this write's doing, never a
        # stale value in another layer: keeping it leaves a config no command
        # can read.
        if not was_valid and not target_unparseable(target):
            return None  # broken before this edit; not ours to refuse
        return _rollback(str(exc))
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
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with _writing_config(target) as held:
        prior = target.read_text(encoding="utf-8") if target.is_file() else None
        was_valid = config_is_valid(repo_root)
        parsed = parse_cli_value(raw_value)
        try:
            upsert_toml_leaf(target, dotted_key, parsed)
        except ValueError as exc:
            return str(exc)
        return _revalidate(
            repo_root, target, prior, was_valid=was_valid, held=held, written=[(dotted_key, parsed)]
        )


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
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with _writing_config(target) as held:
        prior = target.read_text(encoding="utf-8") if target.is_file() else None
        was_valid = config_is_valid(repo_root)
        try:
            upsert_toml_table(target, table, fields)
        except ValueError as exc:
            return str(exc)
        return _revalidate(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            # PER-LEAF: written_value_error reports an error only at loc == key,
            # so a whole-table (key, dict) would hide every leaf-level error;
            # per-leaf also routes providers.<name>.<leaf> through
            # provider_field_error.
            written=[(f"{table}.{k}", v) for k, v in fields.items() if v is not None],
        )


def provider_choices() -> dict[str, list[str]]:
    """Fixed-choice fields for the add-provider form, read from the schema so
    they never drift: the api_format discriminator (per provider subclass) and
    the deployment presets."""
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


def set_config_leaves(
    repo_root: Path,
    table: str,
    fields: dict[str, str | bool | None],
    *,
    to_repo: bool = False,
) -> str | None:
    """Upsert individual ``[table]`` leaves, preserving sibling keys and comments
    verbatim -- the UPDATE counterpart to :func:`set_config_table`'s whole-block
    replace. One revalidate+rollback wraps all the leaf writes, so a bad merged
    config restores the prior file whole. ``None`` field values are omitted."""
    target = _prepare_write_target(repo_root, to_repo=to_repo)
    with _writing_config(target) as held:
        prior = target.read_text(encoding="utf-8") if target.is_file() else None
        was_valid = config_is_valid(repo_root)
        try:
            for key, val in fields.items():
                if val is not None:
                    upsert_toml_leaf(target, f"{table}.{key}", val)
        except ValueError as exc:
            # A leaf whose ancestor is a header-less table: the surgery refuses
            # (its sibling writers catch this too). Restore the prior file so a
            # partial multi-leaf write doesn't land, and return the message --
            # the caller (connect / TUI / init) prints it, never a traceback.
            if prior is not None:
                atomic_write(target, prior)
            return str(exc)
        return _revalidate(
            repo_root,
            target,
            prior,
            was_valid=was_valid,
            held=held,
            written=[(f"{table}.{k}", v) for k, v in fields.items() if v is not None],
        )


def unset_config_value(repo_root: Path, dotted_key: str, *, to_repo: bool = False) -> str | None:
    """Remove one leaf so it reverts to the next layer / built-in default.

    Re-validates and rolls back on failure. Returns None on success, including
    the no-op case where the key was not set in the target file.
    """
    target = _write_target(repo_root, to_repo=to_repo)
    if not target.is_file():
        return None
    with _writing_config(target) as held:
        prior = target.read_text(encoding="utf-8")
        was_valid = config_is_valid(repo_root)
        try:
            removed = remove_toml_leaf(target, dotted_key)
        except ValueError as exc:
            return str(exc)
        if not removed:
            return None
        return _revalidate(repo_root, target, prior, was_valid=was_valid, held=held)
