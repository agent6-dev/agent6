"""Ground review citations, classify their lines and read rule blobs against immutable trees."""

from __future__ import annotations

import dataclasses
import itertools
import os
import pathlib
import re
import shutil
import tempfile
from typing import Literal

from agent6 import errors, review_records
from agent6.tools import _path_safety
from agent6.vcs import (
    _git_command,
    _review_submodules,
    git_files,
    git_history,
    git_policy,
    git_read,
)

# Diff grounding: the (path, line) citations this diff supports.
# Both the old-side and new-side line numbers, so a deletion grounds against the pre-image path.
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GIT_HDR_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def _unquote_git_path(p: str) -> str:
    """Return the path with git's C-string quoting stripped, so it matches a citation."""
    p = p.strip()
    if len(p) >= 2 and p.startswith('"') and p.endswith('"'):
        p = p[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return p


def _hdr_path(raw: str) -> str:
    """Return the path of a `--- ` or `+++ ` header line, or "" for /dev/null."""
    target = _unquote_git_path(raw[4:].split("\t", 1)[0])
    return "" if target == "/dev/null" else re.sub(r"^[ab]/", "", target)


@dataclasses.dataclass(frozen=True, slots=True)
class Hunk:
    """One hunk as it addresses one path.

    A side with no lines spans the line the change sits at, so a citation of it grounds.

    Attributes:
        old: The pre-image lines it replaced, an inclusive span; None when that side is another
            file's (a rename's other name) or absent (a created file).
        new: The post-image lines it produced, likewise; None for a rename's other name or a
            deleted file.
    """

    old: tuple[int, int] | None
    new: tuple[int, int] | None


def _span(start: str, count: str | None) -> tuple[int, int]:
    """Return the inclusive span of a hunk side from its header start and count."""
    s, n = int(start), int(count) if count is not None else 1
    return (s, s + max(n, 1) - 1)


Hunks = dict[tuple[str, Literal["base", "head"]], list[Hunk]]


def diff_hunks(diff: str) -> Hunks:
    """Map each touched path and side to its hunks, for grounding and dedup.

    An in-place hunk carries both sides under its path. A rename's, a created file's or a
    deleted file's hunk carries each side under that side's path, so deleted code grounds at
    its old line number and neither name grounds the other's. A `+++ ` line counts as a header
    only after a `--- ` line. A file touched without hunks (binary, a pure rename, a mode
    change) is recorded with none, so a path-only citation of it grounds and a line citation
    does not. A type change's deletion and addition each contribute their present side.

    Args:
        diff: The unified diff.

    Returns:
        The hunks by repository path and side.
    """
    hunks: Hunks = {}
    newpath = oldpath = ""
    lines = diff.splitlines()
    for i, raw in enumerate(lines):
        if g := _GIT_HDR_RE.match(raw):
            if oldpath:
                hunks.setdefault((oldpath, "base"), [])
            if newpath:
                hunks.setdefault((newpath, "head"), [])
            oldpath = _unquote_git_path(g.group(1))
            newpath = _unquote_git_path(g.group(2))
            continue
        if raw.startswith("new file mode "):
            oldpath = ""
        elif raw.startswith("deleted file mode "):
            newpath = ""
        # A "--- " header is always followed by "+++ "; the lookahead rejects a deleted "-- " line.
        if raw.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            oldpath, newpath = _hdr_path(raw), _hdr_path(lines[i + 1])
            continue
        m = _HUNK_RE.match(raw)
        if m:
            old, new = _span(m.group(1), m.group(2)), _span(m.group(3), m.group(4))
            paths: dict[Literal["base", "head"], str] = {"base": oldpath, "head": newpath}
            for side, path in paths.items():
                if path:
                    hunks.setdefault((path, side), []).append(
                        Hunk(
                            old=old if path == oldpath else None,
                            new=new if path == newpath else None,
                        )
                    )
    if oldpath:
        hunks.setdefault((oldpath, "base"), [])
    if newpath:
        hunks.setdefault((newpath, "head"), [])
    return hunks


# Evidence reads are bounded; larger blobs and patches retain an explicit unknown reason.
_EVIDENCE_MAX_BYTES = 4 * 1024 * 1024


class LineOriginError(Exception):
    """The available diff or blobs cannot prove a line's origin."""


def _blob_lines(root: pathlib.Path, tree: str, name: str) -> list[bytes]:
    if not name or not git_read.tree_entry(root, tree, name):
        return []
    blob = git_files.blob_at(root, tree, name, max_bytes=_EVIDENCE_MAX_BYTES)
    if blob is None:
        raise LineOriginError("The cited file is not a regular file on both present sides.")
    if blob.data is None:
        raise LineOriginError("The file exceeds the 4 MiB line-origin evidence limit.")
    if git_history.looks_binary(blob.data):
        raise LineOriginError("Binary files have no line-origin evidence.")
    parts = blob.data.split(b"\n")
    return [line + b"\n" for line in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _unchanged_spans(patch: str, old_size: int, new_size: int) -> list[tuple[int, int, int]]:
    """Return unchanged (base start, head start, count) spans using Git's line accounting.

    Args:
        patch: The single-file diff between the base and head blobs.
        old_size: The base blob's line count.
        new_size: The head blob's line count.

    Returns:
        Base start, head start and line count for each unchanged span.

    Raises:
        LineOriginError: The patch is incomplete or does not describe one exact mapping.
    """
    if sum(line.startswith("diff --git ") for line in patch.split("\n")) > 1:
        raise LineOriginError("The path has more than one diff section.")
    spans: list[tuple[int, int, int]] = []
    old = new = 1
    lines = patch.split("\n")
    for index, raw in enumerate(lines):
        match = _HUNK_RE.match(raw)
        if match is None:
            continue
        old_start, old_end = _span(match[1], match[2])
        new_start, new_end = _span(match[3], match[4])
        # An empty side starts after the header's anchor line.
        old_start += int(match[2] == "0")
        new_start += int(match[4] == "0")
        gap = old_start - old
        if gap < 0 or new_start - new != gap:
            raise LineOriginError("The diff has an ambiguous line mapping.")
        spans.append((old, new, gap))
        old, new = old_start, new_start
        for body_index in range(index + 1, len(lines)):
            if old > old_end and new > new_end:
                break
            line = lines[body_index]
            if line.startswith(" "):
                spans.append((old, new, 1))
                old += 1
                new += 1
            elif line.startswith("-"):
                old += 1
            elif line.startswith("+"):
                new += 1
            elif not line.startswith("\\ No newline"):
                raise LineOriginError("The diff has incomplete line evidence.")
        if old != old_end + 1 or new != new_end + 1:
            raise LineOriginError("The diff line counts do not match its hunks.")
    tail = old_size - old + 1
    if tail < 0 or new_size - new + 1 != tail:
        raise LineOriginError("The diff line counts do not match its blobs.")
    spans.append((old, new, tail))
    return spans


def _mapped_range(
    spans: list[tuple[int, int, int]], start: int, end: int, side: Literal["base", "head"]
) -> tuple[int, int] | None:
    """Return the opposite-side range, or None when any cited line changed.

    Args:
        spans: Unchanged spans as (base start, head start, count).
        start: The first cited line, one-based.
        end: The last cited line, inclusive.
        side: The cited side.

    Returns:
        The inclusive opposite-side range, or None if any cited line changed.

    Raises:
        LineOriginError: The unchanged range maps to disjoint lines.
    """
    pieces: list[tuple[int, int]] = []
    for old, new, count in spans:
        source, dest = (old, new) if side == "base" else (new, old)
        lo, hi = max(start, source), min(end, source + count - 1)
        if lo <= hi:
            pieces.append((dest + lo - source, hi - lo + 1))
    if sum(count for _, count in pieces) != end - start + 1:
        return None
    if any(a + count != b for (a, count), (b, _) in itertools.pairwise(pieces)):
        raise LineOriginError("The cited range maps to disjoint lines at the base or head.")
    return pieces[0][0], pieces[-1][0] + pieces[-1][1] - 1


def _classify(
    root: pathlib.Path,
    target: review_records.ReviewTarget,
    location: review_records.ReviewLocation,
    hide: tuple[str, ...],
) -> review_records.LineOrigin:
    root = (git_read.toplevel(root) or root).resolve()
    workspace = _path_safety.host_workspace(root, hide)
    lexical_hidden = tuple(pathlib.Path(os.path.abspath(p)) for p in hide)  # noqa: PTH100
    workspace = dataclasses.replace(workspace, denied=(*workspace.denied, *lexical_hidden))
    # Rename detection and line counts can read every changed blob, not just the citation.
    names = _git_command.nul_fields(
        root,
        "diff-tree",
        "-r",
        "-z",
        "--no-renames",
        "--name-only",
        target.base_tree,
        target.head_tree,
    )
    if any(workspace.is_denied(root / name) for name in (location.path, *filter(None, names))):
        raise LineOriginError("sandbox.hide_paths prevents complete line-origin evidence.")
    start, end = location.start_line, location.end_line
    if start is None or end is None:
        raise LineOriginError("A file-only citation has no line-origin evidence.")
    changed = next(
        (
            file
            for file in git_history.changed_files(root, target.base_tree, target.head_tree)
            if location.path
            == (file.old_path or file.path if location.side == "base" else file.path)
        ),
        None,
    )
    if changed is None:
        raise LineOriginError("The cited path is outside the reviewed diff.")
    old_path, new_path = changed.old_path or changed.path, changed.path
    before = _blob_lines(root, target.base_tree, old_path)
    after = _blob_lines(root, target.head_tree, new_path)
    cited = before if location.side == "base" else after
    if end > len(cited):
        raise LineOriginError("The cited range is outside the recorded blob.")
    patch, truncated = git_history.file_patch(
        root,
        target.base_tree,
        target.head_tree,
        new_path,
        changed.old_path,
        max_bytes=_EVIDENCE_MAX_BYTES,
    )
    if truncated:
        raise LineOriginError("The patch exceeds the 4 MiB line-origin evidence limit.")
    spans = _unchanged_spans(patch, len(before), len(after))
    mapped = _mapped_range(spans, start, end, location.side)
    if mapped is None:
        return review_records.LineOrigin(
            kind="changed", reason="The cited range includes changed lines."
        )
    base_range, head_range = (
        ((start, end), mapped) if location.side == "base" else (mapped, (start, end))
    )
    base_start, base_end = base_range
    head_start, head_end = head_range
    if before[base_start - 1 : base_end] != after[head_start - 1 : head_end]:
        raise LineOriginError("The mapped line bytes differ between the base and head blobs.")
    if not target.base_commit or git_read.tree_sha(root, target.base_commit) != target.base_tree:
        raise LineOriginError("The reviewed base has no matching commit history.")
    if _git_command.run(root, "rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        raise LineOriginError("Shallow history cannot provide complete base attribution.")
    blamed = git_files.blame_lines(
        root,
        target.base_commit,
        old_path,
        b"",
        start_line=base_start,
        lines=base_end - base_start + 1,
        hide=hide,
    )
    if [row.line for row in blamed] != list(range(base_start, base_end + 1)) or not all(
        row.sha for row in blamed
    ):
        raise LineOriginError("Blame did not attribute every cited base line.")
    return review_records.LineOrigin(
        kind="pre_existing",
        reason="The cited lines exist at the base. This does not date the defect.",
        base_location=review_records.ReviewLocation(
            path=old_path, side="base", start_line=base_start, end_line=base_end
        ),
        commits=tuple(sorted({row.sha for row in blamed})),
    )


def classify_line_origin(
    root: pathlib.Path,
    target: review_records.ReviewTarget,
    location: review_records.ReviewLocation,
    *,
    hide: tuple[str, ...],
) -> review_records.LineOrigin:
    """Classify a cited range against immutable blobs and bounded base blame.

    Args:
        root: A directory inside the repository.
        target: The immutable comparison being reviewed.
        location: The cited path, side and line range.
        hide: `[sandbox].hide_paths`, refused for line-origin evidence.

    Returns:
        Evidence with a base range and commits, or a changed or unknown reason.
    """
    try:
        return _classify(root, target, location, hide)
    except errors.GitError as exc:
        return review_records.LineOrigin(reason=exc.safe_message)
    except (LineOriginError, OSError) as exc:
        return review_records.LineOrigin(reason=str(exc))


def rule_blob(
    root: pathlib.Path, tree: str, name: str, *, max_bytes: int
) -> tuple[review_records.RuleSource, str] | None:
    """Read an optional regular rule blob, refusing links, non-text and oversized content.

    Args:
        root: The repository root.
        tree: The immutable reviewed base tree.
        name: The repository-relative rule path.
        max_bytes: The remaining byte allowance for the combined rule set.

    Returns:
        The source and text, or None when absent.

    Raises:
        GitError: A present path cannot supply a whole regular rule file.
        UnicodeDecodeError: The blob is not UTF-8 text.
    """
    parts = pathlib.PurePosixPath(name).parts
    for i in range(1, len(parts)):
        parent = "/".join(parts[:i])
        entry = git_read.tree_entry(root, tree, parent)
        if not entry:
            return None
        if not entry.startswith("040000 tree "):
            raise errors.GitError(f"{parent} is not a directory in the base tree")
    entry = git_read.tree_entry(root, tree, name)
    if not entry:
        return None
    if not entry.startswith(("100644 blob ", "100755 blob ")):
        raise errors.GitError("not a regular file in the base tree (symlinks are refused)")
    blob = git_files.blob_at(root, tree, name, max_bytes=max_bytes)
    if blob is None:
        raise errors.GitError("rule blob is unavailable")
    if blob.data is None:
        raise errors.GitError("combined rules exceed 64 KiB; reduce the base-tree rule files")
    text = blob.data.decode("utf-8")
    if "\x00" in text:
        raise errors.GitError("rule blob contains NUL bytes")
    return review_records.RuleSource(name, entry.split()[2], len(blob.data)), text


@dataclasses.dataclass(frozen=True, slots=True)
class Comparison:
    """Keep the diff with the immutable trees it compares."""

    target: review_records.ReviewTarget
    diff: str
    hidden_paths: tuple[str, ...] = ()
    dirty_submodules: tuple[str, ...] = ()


def _comparison_commits(root: pathlib.Path, base: str, head: str) -> tuple[str | None, str | None]:
    """Resolve review endpoints, using the merge base unless the range has two dots.

    Args:
        root: The review's working directory.
        base: A base revision or range, or "" for the working tree against HEAD.
        head: The head revision when base is a single revision.

    Returns:
        The base and head commits, with no head commit for a worktree review.

    Raises:
        GitError: A revision or merge base cannot be resolved.
    """
    if not base:
        return git_history.resolve_commit(root, "HEAD") or None, None
    if "..." in base:
        left, right = base.split("...", 1)
        merge_base = True
    elif ".." in base:
        left, right = base.split("..", 1)
        merge_base = False
    else:
        left, right, merge_base = base, head, True
    base_commit = git_history.resolve_commit(root, left or "HEAD")
    head_commit = git_history.resolve_commit(root, right or "HEAD")
    if not base_commit or not head_commit:
        raise errors.GitError("review base or head is not a commit")
    if merge_base:
        base_commit = _git_command.run(root, "merge-base", base_commit, head_commit).stdout.strip()
    return base_commit, head_commit


def comparison(
    root: pathlib.Path,
    *,
    base: str,
    head: str,
    paths: tuple[str, ...],
    hide: tuple[str, ...] = (),
) -> Comparison:
    """Resolve a review range once and diff its immutable trees without changing the index.

    Args:
        root: The review's working directory.
        base: A base revision or range, or "" for the working tree against HEAD.
        head: The head revision when base is a single revision.
        paths: The working-directory-relative pathspecs limiting the review.
        hide: `[sandbox].hide_paths`, excluded before snapshotting or diffing.

    Returns:
        The comparison and its immutable target.

    Raises:
        GitError: A revision or its tree cannot be resolved.
    """
    base_commit, head_commit = _comparison_commits(root, base, head)
    repo = (git_read.toplevel(root) or root).resolve()
    base_tree = git_read.tree_sha(repo, base_commit) if base_commit else git_read.EMPTY_TREE
    head_tree = git_read.tree_sha(repo, head_commit) if head_commit else base_tree
    workspace = _path_safety.host_workspace(repo, hide)
    lexical_hidden = tuple(pathlib.Path(os.path.abspath(p)) for p in hide)  # noqa: PTH100
    workspace = dataclasses.replace(workspace, denied=(*workspace.denied, *lexical_hidden))
    specs = paths
    # Exclusion-only selections span the repository in both listings and diffs.
    if all(re.match(r"^:(?:[!^]|/[!^]|\((?:[^,)]+,)*exclude(?:,|\)))", p) for p in paths):
        specs = (":/", *paths)
    exclusions: list[str] = []
    dirty_submodules: tuple[str, ...] = ()
    for denied in sorted(set(workspace.denied)):
        if not _path_safety.path_within(denied, repo) or denied == repo:
            continue
        name = pathlib.PurePosixPath(*denied.parts[len(repo.parts) :]).as_posix()
        exclusions.append(f":(top,exclude,icase,literal){name}")
    with tempfile.TemporaryDirectory(prefix="agent6-review-") as directory:
        tmp = pathlib.Path(directory)
        env = {"GIT_INDEX_FILE": str(tmp / "index")}
        if head_commit:
            _git_command.run(repo, "read-tree", head_tree, env_extra=env)
        else:
            index = pathlib.Path(
                _git_command.run(
                    repo, "rev-parse", "--path-format=absolute", "--git-path", "index"
                ).stdout.rstrip("\n")
            )
            if index.is_file():
                # The index mtime tells Git which cached entries need a racy-entry recheck.
                shutil.copy2(index, tmp / "index")
        listing = (
            "ls-files",
            "-z",
            "--full-name",
            "--cached",
            *([] if head_commit else ["--others", "--exclude-standard"]),
        )
        coverage_listing = (*listing, *([f"--with-tree={base_tree}"] if base_commit else []))
        names = sorted(
            set(_git_command.nul_fields(root, *coverage_listing, "--", *specs, env_extra=env))
            - {""}
        )
        hidden = tuple(name for name in names if workspace.is_denied(repo / name))
        selected = tuple(sorted(set(names).difference(hidden)))
        if hidden and selected:
            filtered = set(
                _git_command.nul_fields(
                    root, *coverage_listing, "--", *specs, *exclusions, env_extra=env
                )
            ) - {""}
            if filtered != set(selected):
                raise errors.GitError("Git pathspecs cannot enforce the review's hidden paths")
        if not head_commit and selected:
            dirty_submodules = _review_submodules.dirty_paths(repo, selected, workspace)
            changed = set(
                _git_command.nul_fields(
                    root,
                    "ls-files",
                    "-z",
                    "--full-name",
                    "--modified",
                    "--others",
                    "--exclude-standard",
                    "--",
                    *specs,
                    *exclusions,
                    env_extra=env,
                )
            ) - {""}
            _git_command.run(
                repo,
                "update-index",
                "--add",
                "--remove",
                "--replace",
                "-z",
                "--stdin",
                stdin_text="".join(f"{name}\0" for name in sorted(changed)),
                env_extra=env,
            )
            unmerged = {
                entry.partition("\t")[2]
                for entry in _git_command.nul_fields(repo, "ls-files", "-u", "-z", env_extra=env)
                if entry
            }
            if unmerged:
                entries = (
                    _git_command.nul_fields(repo, "ls-tree", "-r", "-z", base_tree)
                    if base_commit
                    else []
                )
                restored = "".join(
                    f"0 {'0' * len(base_tree)}\t{name}\0" for name in sorted(unmerged)
                ) + "".join(
                    f"{entry}\0" for entry in entries if entry.partition("\t")[2] in unmerged
                )
                _git_command.run(
                    repo, "update-index", "-z", "--index-info", stdin_text=restored, env_extra=env
                )
            head_tree = _git_command.run(repo, "write-tree", env_extra=env).stdout.strip()
    target = review_records.ReviewTarget(
        base_commit,
        base_tree,
        head_commit,
        head_tree,
        head_commit or "",
        selected if paths or hidden else (),
        "revisions" if base else "worktree",
    )
    diff = (
        _git_command.run(
            root,
            "diff",
            *git_policy.DIFF_SAFETY_FLAGS,
            "--no-relative",
            base_tree,
            head_tree,
            "--",
            *specs,
            *exclusions,
        ).stdout
        if selected
        else ""
    )
    return Comparison(target, diff, hidden, dirty_submodules)
