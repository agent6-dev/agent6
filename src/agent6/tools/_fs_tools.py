# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Content access & write handlers: agent6_docs, read_file, list_dir, grep,
apply_edit, apply_patch.

All of these run in-process (never through ``agent6.sandbox.jail``), so the
write handlers (apply_edit/apply_patch) carry their own protected-path guard:
``.git`` (when ``protect_git``), an in-repo virtualenv / installed-package
tree, and any operator/machine ``extra_protect_paths`` -- none of which the
jail's mount-based protections cover for an in-process write. See
``refuse_protected_writes``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from agent6.config import Config
from agent6.tools._agent6_docs import list_agent6_docs, read_agent6_doc
from agent6.tools._edit_diag import (
    edit_mismatch_error,
    indent_tolerant_replacement,
    preview_result,
)
from agent6.tools._grep_safety import reject_pathological_regex
from agent6.tools._path_safety import SafePath, resolve_in_root
from agent6.tools.errors import ToolError
from agent6.tools.index import SymbolIndex
from agent6.tools.patch_apply import (
    PatchError,
    apply_patch_text,
    apply_v4a_text,
    is_v4a_patch,
    patch_target_path,
)
from agent6.tools.results import (
    DocsContentResult,
    DocsIndexResult,
    EditResult,
    GrepResult,
    ListDirResult,
    PatchResult,
    ReadFileResult,
    ToolResult,
)
from agent6.tools.schema import (
    Agent6DocsInput,
    ApplyEditInput,
    ApplyPatchInput,
    GrepInput,
    ListDirInput,
    ReadFileInput,
)

# Wall-clock budget for a single grep over the tree; the regex-shape screening
# (tools/_grep_safety.py) closes the catastrophic-backtracking cases, and this
# bounds the rest.
MAX_GREP_WALL_S = 10.0


def agent6_docs(raw: dict[str, Any]) -> ToolResult:
    args = Agent6DocsInput.model_validate(raw)
    available = list_agent6_docs()
    if not args.name:
        return DocsIndexResult(available=tuple(available))
    content = read_agent6_doc(args.name)
    if content is None:
        raise ToolError(
            f"unknown agent6 doc {args.name!r}; available: {', '.join(available) or '(none)'}"
        )
    cap = 60_000
    return DocsContentResult(
        name=args.name,
        content=content[:cap],
        truncated=len(content) > cap,
    )


def read_file(root: Path, raw: dict[str, Any]) -> ReadFileResult:
    args = ReadFileInput.model_validate(raw)
    sp = resolve_in_root(root, args.path)
    if not sp.abs_path.is_file():
        raise ToolError(f"Not a file: {args.path}")
    try:
        full = sp.abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"File is not UTF-8 text: {args.path}") from exc
    if args.offset == 0 and args.limit is None:
        return ReadFileResult(content=full, size=len(full), lines_total=full.count("\n") + 1)
    lines = full.splitlines(keepends=True)
    end = len(lines) if args.limit is None else min(len(lines), args.offset + args.limit)
    slice_text = "".join(lines[args.offset : end])
    return ReadFileResult(
        content=slice_text,
        size=len(slice_text),
        lines_total=len(lines),
        offset=args.offset,
        lines_returned=end - args.offset,
    )


def list_dir(root: Path, raw: dict[str, Any]) -> ListDirResult:
    args = ListDirInput.model_validate(raw)
    sp = resolve_in_root(root, args.path)
    if not sp.abs_path.is_dir():
        raise ToolError(f"Not a directory: {args.path}")
    entries: list[str] = []
    for entry in sorted(sp.abs_path.iterdir()):
        suffix = "/" if entry.is_dir() else ""
        entries.append(entry.name + suffix)
    return ListDirResult(entries=tuple(entries))


def grep(root: Path, raw: dict[str, Any]) -> GrepResult:
    args = GrepInput.model_validate(raw)
    sp = resolve_in_root(root, args.path)
    reject_pathological_regex(args.pattern)
    try:
        pat = re.compile(args.pattern, re.IGNORECASE if args.case_insensitive else 0)
    except re.error as exc:
        raise ToolError(f"Invalid regex: {exc}") from exc
    deadline = time.monotonic() + MAX_GREP_WALL_S
    hits: list[dict[str, Any]] = []
    targets: list[Path]
    # Skip hidden files/dirs (.git, ...) only when they are BELOW
    # the requested path, so an explicit `grep <pat> .github/` (or a hidden
    # file named directly) is still searched. `skip_base` is the requested
    # directory; for an explicitly-named file there is nothing to skip.
    skip_base: Path | None
    if sp.abs_path.is_file():
        targets = [sp.abs_path]
        skip_base = None
    else:
        targets = [p for p in sp.abs_path.rglob("*") if p.is_file()]
        skip_base = sp.abs_path
    root_resolved = root.resolve()
    for path in targets:
        if skip_base is not None and any(
            part.startswith(".") for part in path.relative_to(skip_base).parts
        ):
            continue
        # Contain each target like read_file contains its leaf: rglob yields
        # in-repo symlinks whose destination can be anywhere on the host, and
        # this read runs in-process (outside the jail). Resolve and require
        # the real file to still be under root; skip escapees.
        try:
            path.resolve().relative_to(root_resolved)
        except (OSError, ValueError):
            continue
        if time.monotonic() > deadline:
            # Bound total wall-clock: a pathological pattern/large tree can't
            # hang the run. Report partial hits rather than stalling.
            return GrepResult(hits=tuple(hits), truncated=True, timeout=True)
        try:
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(),
                start=1,
            ):
                # Re-check the wall-clock inside the line loop too: the
                # between-files check alone can't bound a large single file.
                # (It still can't interrupt one in-progress C-level match —
                # the static screen above is the defence for that.)
                if time.monotonic() > deadline:
                    return GrepResult(hits=tuple(hits), truncated=True, timeout=True)
                if pat.search(line):
                    hits.append(
                        {
                            "path": str(path.relative_to(root)),
                            "line": lineno,
                            "text": line[:500],
                        }
                    )
                    if len(hits) >= 500:
                        return GrepResult(hits=tuple(hits), truncated=True)
        except OSError:
            continue
    return GrepResult(hits=tuple(hits), truncated=False)


def _refuse_protected_write(
    candidate: str, dir_name: str, *, why: str, resolved: SafePath | None = None
) -> None:
    """Refuse an in-process ``apply_edit`` / ``apply_patch`` into a protected
    top-level directory.

    ``.git`` (when ``protect_git``): the edit tools write **in-process, outside
    the jail**, so without this an LLM could create or rewrite ``.git/hooks/*``
    or ``.git/config`` (e.g. ``core.fsmonitor``) and get code executed outside
    the sandbox on the next ``git`` invocation, or corrupt git history --
    defeating ``protect_git`` entirely (the strict jail's RO bind of ``.git``
    never covers these in-process writes). Reads stay allowed. (Run state lives
    out of the workspace, so it is unreachable by edits and needs no guard.)

    Checks both the raw candidate string AND the post-symlink-resolution relative
    path, so a symlink ``./decoy -> .git`` can't launder a write past the prefix
    check.
    """
    parts = Path(candidate).parts
    if parts and parts[0] == dir_name:
        raise ToolError(f"Refusing to write under {dir_name}/ ({why}): {candidate!r}")
    if resolved is not None:
        rel_parts = resolved.rel_path.parts
        if rel_parts and rel_parts[0] == dir_name:
            raise ToolError(
                f"Refusing to write under {dir_name}/ ({why}) via symlink: {candidate!r} "
                f"resolves to {resolved.rel_path!s}"
            )


def _refuse_env_write(candidate: str, resolved: SafePath) -> None:
    """Refuse an in-process edit into an in-repo virtualenv or installed-package
    tree. These are the operator's ENVIRONMENT, not source: a run editing them
    (e.g. rewriting an editable-install ``.pth`` to make an in-jail verify pass)
    silently corrupts the operator's venv, and since venvs are gitignored the
    damage never shows in ``runs diff`` / merge. Observed live: a run rewrote
    ``.venv/.../_editable_impl_*.pth`` from the host path to the jail's
    ``/workspace`` and broke ``import`` on the host afterward.

    A directory holding ``pyvenv.cfg`` is a virtualenv root (the canonical
    marker, name-agnostic: ``.venv`` / ``venv`` / ``env``); a ``site-packages``
    ancestor is an installed tree. Reads stay allowed; only writes are refused.
    The check walks the post-symlink-resolution path so a decoy symlink can't
    launder the write."""
    ancestors = [resolved.abs_path, *resolved.abs_path.parents]
    for anc in ancestors:
        if anc.name == "site-packages":
            raise ToolError(
                f"Refusing to write into an installed-package tree (site-packages): "
                f"{candidate!r}. Installed packages are environment, not source; "
                f"editing them corrupts the operator's virtualenv."
            )
    # A venv root is an ancestor DIRECTORY containing pyvenv.cfg. Check ancestors
    # of the target (not the target itself, which is the file being written).
    for anc in resolved.abs_path.parents:
        try:
            if (anc / "pyvenv.cfg").is_file():
                raise ToolError(
                    f"Refusing to write inside a virtualenv ({anc.name}/): {candidate!r}. "
                    f"A venv is environment, not source; editing it corrupts the "
                    f"operator's setup and never shows in the run's diff."
                )
        except OSError:
            continue


def refuse_protected_writes(
    path: str,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    resolved: SafePath | None = None,
) -> None:
    """Refuse an in-process edit into a protected location (it bypasses the
    jail entirely). ``.git`` under ``protect_git``, a virtualenv / installed
    package tree (see ``_refuse_env_write``), plus any operator/machine
    protect paths (a machine bundle's ``.asm.toml`` + ``scripts/``), which the
    jail marks read-only for ``run_command`` but the in-process edit tools
    would otherwise let a ``mode="run"`` state rewrite -- persisting a payload
    for the next run. Applies on both sandbox profiles."""
    if config.sandbox.protect_git:
        _refuse_protected_write(path, ".git", why="git history/metadata", resolved=resolved)
    if resolved is not None:
        _refuse_env_write(path, resolved)
    if resolved is not None and extra_protect_paths:
        target = resolved.abs_path
        for prot in extra_protect_paths:
            if target == prot or prot in target.parents:
                raise ToolError(
                    f"Refusing to write to a protected path (machine bundle): {path!r} "
                    f"resolves under {prot}"
                )


def apply_edit(
    root: Path,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    index: SymbolIndex | None,
    raw: dict[str, Any],
) -> ToolResult:
    args = ApplyEditInput.model_validate(raw)
    refuse_protected_writes(args.path, config, extra_protect_paths)
    sp = resolve_in_root(root, args.path)
    refuse_protected_writes(args.path, config, extra_protect_paths, sp)
    # Write-outside-cwd is enforced by resolve_in_root already (root == cwd).
    applied: list[str] = []
    existing = sp.abs_path.read_text(encoding="utf-8") if sp.abs_path.exists() else None
    new_content = existing
    for i, edit in enumerate(args.edits):
        if edit.kind == "create":
            if existing is not None and i == 0:
                raise ToolError(f"create requested but file already exists: {args.path}")
            new_content = edit.new_string
            applied.append("create")
        else:
            if new_content is None:
                raise ToolError(f"replace requested but file does not exist: {args.path}")
            occurrences = new_content.count(edit.old_string)
            if occurrences == 0:
                # Weak models most often miss by indentation depth alone
                # (right lines, wrong leading whitespace). If old_string
                # matches EXACTLY ONE region up to a uniform indent shift,
                # apply it (the shift is verified to reproduce that region
                # byte-for-byte first, so a wrong region can't be edited) --
                # saving a full round-trip. Otherwise hand the model the exact
                # closest on-disk text so it retries directly; re-reading the
                # whole file is the dominant small-model time sink on a botched
                # edit.
                fuzzy = indent_tolerant_replacement(new_content, edit.old_string, edit.new_string)
                if fuzzy is not None:
                    new_content = fuzzy
                    applied.append("replace~indent")
                    continue
                raise ToolError(edit_mismatch_error(args.path, i, new_content, edit.old_string))
            if occurrences > 1:
                raise ToolError(
                    f"old_string is not unique in {args.path} "
                    f"(edit #{i}, {occurrences} matches); add more surrounding "
                    f"context to make it unique"
                )
            new_content = new_content.replace(edit.old_string, edit.new_string, 1)
            applied.append("replace")
    if new_content is None:
        raise ToolError("No content to write")
    if args.preview:
        return preview_result(args.path, existing, new_content, applied=applied)
    sp.abs_path.parent.mkdir(parents=True, exist_ok=True)
    sp.abs_path.write_text(new_content, encoding="utf-8")
    if index is not None:
        index.mark_changed(sp.abs_path)
    return EditResult(applied=tuple(applied), path=str(sp.rel_path))


def apply_patch(
    root: Path,
    config: Config,
    extra_protect_paths: tuple[Path, ...],
    index: SymbolIndex | None,
    raw: dict[str, Any],
) -> ToolResult:
    args = ApplyPatchInput.model_validate(raw)
    v4a = is_v4a_patch(args.patch)
    # The write location: the explicit `path` arg if given, else derived from
    # the patch headers (V4A always embeds it; GPT-family models omit `path`).
    # Either way it is resolved + protected-path-checked below, so deriving it
    # from the patch never widens where a write can land.
    try:
        derived_path = patch_target_path(args.patch)
    except PatchError as exc:
        raise ToolError(f"apply_patch failed for {args.path or '<unknown>'}: {exc}") from exc
    target = args.path or derived_path
    # Security checks on the write location come first (absolute path, repo
    # escape, protected dirs), before the lower-priority model-confusion
    # check that an explicit `path` matches the patch header.
    refuse_protected_writes(target, config, extra_protect_paths)
    sp = resolve_in_root(root, target)
    refuse_protected_writes(target, config, extra_protect_paths, sp)
    if args.path and args.path != derived_path:
        raise ToolError(
            f"apply_patch: `path` argument {args.path!r} disagrees with the patch "
            f"header path {derived_path!r}; emit them consistently or omit `path`"
        )
    existing = sp.abs_path.read_text(encoding="utf-8") if sp.abs_path.exists() else None
    try:
        if v4a:
            _, new_content = apply_v4a_text(args.patch, existing)
        else:
            _, new_content = apply_patch_text(args.patch, existing)
    except PatchError as exc:
        raise ToolError(f"apply_patch failed for {target}: {exc}") from exc
    if args.preview:
        return preview_result(target, existing, new_content)
    sp.abs_path.parent.mkdir(parents=True, exist_ok=True)
    sp.abs_path.write_text(new_content, encoding="utf-8")
    if index is not None:
        index.mark_changed(sp.abs_path)
    return PatchResult(path=str(sp.rel_path), bytes_written=len(new_content))
