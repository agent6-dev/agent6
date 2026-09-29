# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Content handlers: agent6_docs, read_file, list_dir, apply_edit and apply_patch.

All run in-process, never through the jail, so the write handlers carry their own
protected-path guard (`refuse_protected_writes`): `.git` under `protect_git`, an in-repo
virtualenv or installed-package tree, and the operator's `extra_protect_paths`.
"""

from __future__ import annotations

import pathlib
from typing import Any

from agent6.config import Config
from agent6.tools import (
    _agent6_docs,
    _edit_diag,
    _path_safety,
    errors,
    patch_apply,
    results,
    schema,
)
from agent6.tools import index as tools_index

# What read_file pulls into memory; a bigger file returns its capped prefix with truncated=True.
MAX_READ_CHARS = 5_000_000


def agent6_docs(raw: dict[str, Any]) -> results.ToolResult:
    """Return the index of agent6's bundled docs, or one doc capped at 60k characters.

    Args:
        raw: The tool call's arguments.

    Returns:
        The index when no name is given, else the doc's content.

    Raises:
        ToolError: The name is not a bundled doc.
    """
    args = schema.Agent6DocsInput.model_validate(raw)
    available = _agent6_docs.list_agent6_docs()
    if not args.name:
        return results.DocsIndexResult(available=tuple(available))
    content = _agent6_docs.read_agent6_doc(args.name)
    if content is None:
        raise errors.ToolError(
            f"unknown agent6 doc {args.name!r}; available: {', '.join(available) or '(none)'}"
        )
    cap = 60_000
    return results.DocsContentResult(
        name=args.name,
        content=content[:cap],
        size=len(content),
        truncated=len(content) > cap,
    )


def read_file(ws: _path_safety.Workspace, raw: dict[str, Any]) -> results.ReadFileResult:
    """Read a text file, whole or one page of lines, capped at `MAX_READ_CHARS`.

    Args:
        ws: The workspace the path resolves in.
        raw: The tool call's arguments.

    Returns:
        The content or the requested slice, with the line counts of the capped prefix.

    Raises:
        ToolError: The path is not a file, not UTF-8, or binary.
    """
    args = schema.ReadFileInput.model_validate(raw)
    sp = ws.resolve_read(args.path)
    if not sp.abs_path.is_file():
        raise errors.ToolError(f"Not a file: {args.path}")
    try:
        # One char past the cap detects the overflow; pagination works on the capped prefix.
        full = _path_safety.read_contained(sp, limit_chars=MAX_READ_CHARS + 1)
    except UnicodeDecodeError as exc:
        raise errors.ToolError(f"File is not UTF-8 text: {args.path}") from exc
    read_truncated = len(full) > MAX_READ_CHARS
    if read_truncated:
        full = full[:MAX_READ_CHARS]
    # Some binary payloads decode as UTF-8; a NUL byte is what binary means in practice.
    if "\x00" in full:
        raise errors.ToolError(f"File is binary (contains NUL bytes): {args.path}")
    # One split owns every line count, so a full read and a later page of the file agree.
    lines = full.splitlines(keepends=True)
    if args.start_line == 1 and args.limit is None:
        return results.ReadFileResult(
            content=full,
            size=len(full.encode("utf-8")),
            lines_total=len(lines),
            truncated=read_truncated,
        )
    first = args.start_line - 1  # 1-based on the wire, 0-based slice
    end = len(lines) if args.limit is None else min(len(lines), first + args.limit)
    sliced = lines[first:end]
    slice_text = "".join(sliced)
    return results.ReadFileResult(
        content=slice_text,
        size=len(slice_text.encode("utf-8")),
        lines_total=len(lines),
        start_line=args.start_line,
        lines_returned=len(sliced),
        truncated=read_truncated,
    )


def list_dir(ws: _path_safety.Workspace, raw: dict[str, Any]) -> results.ListDirResult:
    """List a directory's entries, sorted, capped at `LIST_DIR_CAP`.

    Args:
        ws: The workspace the path resolves in.
        raw: The tool call's arguments.

    Returns:
        The visible names (directories with a trailing slash), how many entries are hidden,
        and whether the list was cut.

    Raises:
        ToolError: The path is not a directory.
    """
    args = schema.ListDirInput.model_validate(raw)
    sp = ws.resolve_read(args.path)
    if not sp.abs_path.is_dir():
        raise errors.ToolError(f"Not a directory: {args.path}")
    listing = sorted(_path_safety.list_contained(sp), key=lambda e: e.name)
    # A hidden entry is dropped from the names but counted, so the listing stays true.
    visible = [e for e in listing if not ws.is_denied(sp.abs_path / e.name)]
    return results.ListDirResult(
        entries=tuple(e.name + "/" if e.is_dir else e.name for e in visible[: schema.LIST_DIR_CAP]),
        hidden=len(listing) - len(visible),
        truncated=len(visible) > schema.LIST_DIR_CAP,
    )


def _under_project_dir(path: pathlib.Path, dir_name: str) -> bool:
    """Return whether the workspace-relative path is the top-level directory or lies under it."""
    return _path_safety.path_within(path, pathlib.Path(dir_name))


def _refuse_protected_write(
    candidate: str, dir_name: str, *, why: str, resolved: _path_safety.SafePath | None = None
) -> None:
    """Refuse an in-process edit into the workspace's own top-level directory.

    The edit tools write outside the jail, so without this an LLM could rewrite `.git/hooks/*`
    or `.git/config` and get code executed on the next `git` invocation; the strict jail's
    read-only bind of `.git` never covers an in-process write. A nested `.git` (a vendored
    repo, a submodule gitlink) is content. Reads stay allowed. Both the raw candidate and the
    post-symlink relative path are checked, so `./decoy -> .git` cannot launder a write.

    Args:
        candidate: The path the model gave.
        dir_name: The top-level directory refused.
        why: The reason named in the refusal.
        resolved: The contained path, when the caller has resolved it.

    Raises:
        ToolError: The candidate or its resolution lies under the directory.
    """
    if _under_project_dir(pathlib.Path(candidate), dir_name):
        raise errors.ToolError(f"Refusing to write under {dir_name}/ ({why}): {candidate!r}")
    if resolved is not None and _under_project_dir(resolved.rel_path, dir_name):
        raise errors.ToolError(
            f"Refusing to write under {dir_name}/ ({why}) via symlink: {candidate!r} "
            f"resolves to {resolved.rel_path!s}"
        )


def _refuse_env_write(candidate: str, resolved: _path_safety.SafePath) -> None:
    """Refuse an in-process edit into an in-repo virtualenv or installed-package tree.

    These are the operator's environment, not source: a run rewriting an editable-install
    `.pth` to make a verify pass corrupts the venv, and the damage never shows in the diff
    because venvs are gitignored. A directory holding `pyvenv.cfg` is a virtualenv root; a
    `site-packages` ancestor is an installed tree. Reads stay allowed. The check walks the
    post-symlink path so a decoy symlink cannot launder the write.

    Args:
        candidate: The path the model gave.
        resolved: The contained path.

    Raises:
        ToolError: The path lies inside a virtualenv or a `site-packages` tree.
    """
    ancestors = [resolved.abs_path, *resolved.abs_path.parents]
    for anc in ancestors:
        if _path_safety.fold_name(anc.name) == "site-packages":
            raise errors.ToolError(
                f"Refusing to write into an installed-package tree (site-packages): "
                f"{candidate!r}. Installed packages are environment, not source; "
                f"editing them corrupts the operator's virtualenv."
            )
    # Ancestors only: the target itself is the file being written.
    for anc in resolved.abs_path.parents:
        try:
            if (anc / "pyvenv.cfg").is_file():
                raise errors.ToolError(
                    f"Refusing to write inside a virtualenv ({anc.name}/): {candidate!r}. "
                    f"A venv is environment, not source; editing it corrupts the "
                    f"operator's setup and never shows in the run's diff."
                )
        except OSError:
            continue


def refuse_protected_writes(
    path: str,
    config: Config,
    extra_protect_paths: tuple[pathlib.Path, ...],
    resolved: _path_safety.SafePath | None = None,
) -> None:
    """Refuse an in-process edit into a protected location, at both isolation levels.

    `.git` under `protect_git`, a virtualenv or installed-package tree, and the extra protect
    paths (a machine bundle's `.asm.toml` and `scripts/`), which the jail marks read-only for
    `run_command` but an in-process edit would otherwise rewrite.

    Args:
        path: The path the model gave.
        config: The run's config.
        extra_protect_paths: The operator's and the machine's protected paths.
        resolved: The contained path, when the caller has resolved it.

    Raises:
        ToolError: The path lies in a protected location.
    """
    if config.sandbox.protect_git:
        _refuse_protected_write(path, ".git", why="git history/metadata", resolved=resolved)
    if resolved is not None:
        _refuse_env_write(path, resolved)
    if resolved is not None and extra_protect_paths:
        target = resolved.abs_path
        for prot in extra_protect_paths:
            if _path_safety.path_within(target, prot):
                raise errors.ToolError(
                    f"Refusing to write to a protected path (machine bundle): {path!r} "
                    f"resolves under {prot}"
                )


def _existing_text(sp: _path_safety.SafePath, rel_path: str) -> str | None:
    """Return the file's current text, or None when it does not exist yet.

    Raises:
        ToolError: The path exists but is not a file (an OSError would leak the host path into
            the transcript), or the file is larger than the edit tools take.
    """
    if not sp.abs_path.exists():
        return None
    if not sp.abs_path.is_file():
        raise errors.ToolError(f"Not a file: {rel_path}")
    # Refused rather than truncated: a partial read must never become a whole-file write.
    text = _path_safety.read_contained(sp, limit_chars=MAX_READ_CHARS + 1)
    if len(text) > MAX_READ_CHARS:
        raise errors.ToolError(
            f"{rel_path} is larger than the edit tools take ({MAX_READ_CHARS:,} chars);"
            " change it with run_command"
        )
    return text


def _preview(
    sp: _path_safety.SafePath,
    path: str,
    existing: str | None,
    new_content: str,
    *,
    applied: list[str] | None = None,
    deleting: bool = False,
    healed: tuple[str, ...] = (),
) -> results.PreviewResult:
    """Return the dry-run result, its byte counts measured as the write would land on disk."""
    like = None if existing is None else _path_safety.read_bytes_contained(sp)
    return _edit_diag.preview_result(
        path,
        existing,
        new_content,
        bytes_before=0 if like is None else len(like),
        bytes_after=len(_path_safety.disk_bytes(new_content, like=like)),
        applied=applied,
        deleting=deleting,
        healed=healed,
    )


def apply_edit(
    ws: _path_safety.Workspace,
    config: Config,
    extra_protect_paths: tuple[pathlib.Path, ...],
    index: tools_index.SymbolIndex | None,
    raw: dict[str, Any],
) -> results.ToolResult:
    """Apply string replacements or a whole-file write to one file.

    Args:
        ws: The workspace the path resolves in.
        config: The run's config.
        extra_protect_paths: The operator's and the machine's protected paths.
        index: The symbol index to notify of the change, when one exists.
        raw: The tool call's arguments.

    Returns:
        The edits applied, or the preview when `preview` is set.

    Raises:
        ToolError: The path is protected, a create targets an existing file, a replace targets
            a missing one, an `old_string` is absent or not unique, or nothing would be written.
    """
    args = schema.ApplyEditInput.model_validate(raw)
    refuse_protected_writes(args.path, config, extra_protect_paths)
    sp = ws.resolve_write(args.path)
    refuse_protected_writes(args.path, config, extra_protect_paths, sp)
    applied: list[str] = []
    existing = _existing_text(sp, args.path)
    new_content = existing
    for i, edit in enumerate(args.edits):
        if edit.kind in schema.WHOLE_FILE_KINDS:
            if edit.kind == "create" and existing is not None:
                raise errors.ToolError(
                    f"create requested but file already exists: {args.path}"
                    ' (kind="overwrite" replaces it whole)'
                )
            new_content = edit.new_string
            applied.append(edit.kind)
        else:
            if new_content is None:
                raise errors.ToolError(f"replace requested but file does not exist: {args.path}")
            first = new_content.find(edit.old_string)
            if first == -1:
                # A uniform indent shift is healed; anything else gets the closest on-disk text.
                fuzzy = _edit_diag.indent_tolerant_replacement(
                    new_content, edit.old_string, edit.new_string
                )
                if fuzzy is not None:
                    new_content = fuzzy
                    applied.append("replace~indent")
                    continue
                raise errors.ToolError(
                    _edit_diag.edit_mismatch_error(args.path, i, new_content, edit.old_string)
                )
            second = new_content.find(edit.old_string, first + 1)
            if second != -1:
                detail = (
                    "overlapping matches"
                    if second < first + len(edit.old_string)
                    else f"{new_content.count(edit.old_string)} matches"
                )
                raise errors.ToolError(
                    f"old_string is not unique in {args.path} "
                    f"(edit #{i}, {detail}); add more surrounding "
                    f"context to make it unique"
                )
            new_content = new_content.replace(edit.old_string, edit.new_string, 1)
            applied.append("replace")
    if new_content is None:
        raise errors.ToolError("No content to write")
    if args.preview:
        return _preview(sp, args.path, existing, new_content, applied=applied)
    _path_safety.write_contained(sp, new_content)
    if index is not None:
        index.mark_changed(sp.abs_path)
    return results.EditResult(
        applied=tuple(applied), path=str(sp.rel_path), created=existing is None
    )


def _first_repeated(paths: list[pathlib.Path]) -> pathlib.Path | None:
    """Return the first path two sections of one patch both target, or None."""
    seen: set[pathlib.Path] = set()
    for path in paths:
        if path in seen:
            return path
        seen.add(path)
    return None


def _stage_patch_section(
    ws: _path_safety.Workspace,
    config: Config,
    extra_protect_paths: tuple[pathlib.Path, ...],
    *,
    path_arg: str,
    section: str,
) -> tuple[_path_safety.SafePath, str, str | None, str | None, tuple[str, ...]]:
    """Resolve, security-check and apply one single-file patch section in memory.

    Args:
        ws: The workspace the path resolves in.
        config: The run's config.
        extra_protect_paths: The operator's and the machine's protected paths.
        path_arg: The explicit `path` argument, or "" to take the patch header's.
        section: The section's text.

    Returns:
        The contained path, the target as named, the existing text (None for a new file), the
        new text (None when the section deletes the file), and the healed hunks.

    Raises:
        ToolError: The header names no path, the target is protected, an explicit `path`
            disagrees with the header, the patch does not apply, or a delete targets no file.
    """
    # The target is resolved and protected-path-checked either way, so deriving it never widens.
    try:
        derived_path = patch_apply.patch_target_path(section)
    except patch_apply.PatchError as exc:
        raise errors.ToolError(f"apply_patch failed for {path_arg or '<unknown>'}: {exc}") from exc
    target = path_arg or derived_path
    # Security checks on the write location come before the header-agreement check.
    refuse_protected_writes(target, config, extra_protect_paths)
    sp = ws.resolve_write(target)
    refuse_protected_writes(target, config, extra_protect_paths, sp)
    if path_arg and path_arg != derived_path:
        raise errors.ToolError(
            f"apply_patch: `path` argument {path_arg!r} disagrees with the patch "
            f"header path {derived_path!r}; emit them consistently or omit `path`"
        )
    existing = _existing_text(sp, target)
    try:
        applier = (
            patch_apply.apply_v4a_text
            if patch_apply.is_v4a_patch(section)
            else patch_apply.apply_patch_text
        )
        _, new_content, healed = applier(section, existing)
    except patch_apply.PatchError as exc:
        raise errors.ToolError(f"apply_patch failed for {target}: {exc}") from exc
    if new_content is None and existing is None:
        raise errors.ToolError(f"cannot delete {target}: not a file")
    return sp, target, existing, new_content, healed


def apply_patch(
    ws: _path_safety.Workspace,
    config: Config,
    extra_protect_paths: tuple[pathlib.Path, ...],
    index: tools_index.SymbolIndex | None,
    raw: dict[str, Any],
) -> results.ToolResult:
    """Apply a unified or V4A patch over one or more files, all or nothing.

    Nothing is written until every section applied cleanly in memory.

    Args:
        ws: The workspace the paths resolve in.
        config: The run's config.
        extra_protect_paths: The operator's and the machine's protected paths.
        index: The symbol index to notify of the changes, when one exists.
        raw: The tool call's arguments.

    Returns:
        The bytes written per file and the files deleted, or the preview when `preview` is set.

    Raises:
        ToolError: `path` is given for a multi-file patch, a section fails to stage, one file
            appears in several sections, or a write fails (naming what already changed).
    """
    args = schema.ApplyPatchInput.model_validate(raw)
    sections = patch_apply.split_patch_files(args.patch)
    if len(sections) > 1 and args.path:
        raise errors.ToolError(
            f"apply_patch: `path` argument {args.path!r} is ambiguous for a "
            f"{len(sections)}-file patch; omit `path` (each file names itself)"
        )
    staged: list[tuple[_path_safety.SafePath, str, str | None]] = []
    seen_paths: list[pathlib.Path] = []
    previews: list[results.PreviewResult] = []
    healed_all: list[str] = []
    for section in sections:
        sp, target, existing, new_content, healed = _stage_patch_section(
            ws, config, extra_protect_paths, path_arg=args.path, section=section
        )
        healed_all.extend(healed)
        seen_paths.append(sp.abs_path)
        if args.preview:
            # A deletion previews as the full-removal diff (bytes_after 0).
            previews.append(
                _preview(
                    sp,
                    target,
                    existing,
                    new_content or "",
                    deleting=new_content is None,
                    healed=healed,
                )
            )
            continue
        staged.append((sp, target, new_content))
    if (dupe := _first_repeated(seen_paths)) is not None:
        # Each section stages from disk, so over one file the last write would win silently.
        count = sum(1 for path in seen_paths if path == dupe)
        raise errors.ToolError(
            f"apply_patch: {dupe} appears in {count} sections of one patch;"
            " a section reads the file as it is on disk, so only the last would"
            " land. Send one section per file, with every hunk for it inside."
        )
    if args.preview:
        if len(previews) == 1:
            return previews[0]
        return results.PreviewResult(
            path=previews[0].path,
            diff="".join(pv.diff for pv in previews),
            hunks=sum(pv.hunks for pv in previews),
            bytes_before=sum(pv.bytes_before for pv in previews),
            bytes_after=sum(pv.bytes_after for pv in previews),
            truncated=any(pv.truncated for pv in previews),
            files=tuple(pv.path for pv in previews),
            healed=tuple(healed_all),
        )
    # The writes are not all-or-nothing, so a failure part way names what already changed.
    landed: list[str] = []
    written: dict[str, int] = {}
    for sp, _target, new_content in staged:
        try:
            if new_content is None:
                _path_safety.unlink_contained(sp)
                if index is not None:
                    index.mark_deleted(sp.abs_path)
            else:
                written[str(sp.rel_path)] = _path_safety.write_contained(sp, new_content)
                if index is not None:
                    index.mark_changed(sp.abs_path)
        except (OSError, errors.ToolError) as exc:
            detail = (exc.strerror or str(exc)) if isinstance(exc, OSError) else str(exc)
            changed = f"; already changed: {', '.join(landed)}" if landed else ""
            raise errors.ToolError(f"apply_patch: {sp.rel_path}: {detail}{changed}") from exc
        landed.append(str(sp.rel_path))
    rows = tuple(written.items())
    deleted = tuple(str(sp.rel_path) for sp, _t, new in staged if new is None)
    return results.PatchResult(
        path=str(staged[0][0].rel_path),
        bytes_written=sum(b for _p, b in rows),
        files=rows if len(staged) > 1 else (),
        deleted=deleted,
        healed=tuple(healed_all),
    )
