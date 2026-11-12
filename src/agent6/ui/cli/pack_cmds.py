"""Carry this home's state with the workspace."""

from __future__ import annotations

import pathlib

from agent6 import errors, pack, paths
from agent6.vcs import git_ops, git_read


def cmd_pack(*, repo_config: bool = False) -> int:
    """Write a workspace pack, kept out of git by the repository's own exclude file.

    A project's `.gitignore` is shared with people who do not use agent6, so it is never
    written: `.agent6/` goes into `.git/info/exclude` unless git already ignores it.

    Args:
        repo_config: Fold global settings into repo config and omit global config and UI files.

    Returns:
        Zero after writing the pack.

    Raises:
        PackError: The directory is not a git checkout; `.agent6` is tracked, un-ignored by a
            rule, or holds another origin's pack or an unreadable manifest; a run, exec or
            machine of the checkout is live; a path is a link or special file; or a file cannot
            be read or written.
        GitError: Git cannot inspect the repository.
    """
    root = paths.project_root(pathlib.Path.cwd(), allow_pack=True)
    if not (root / ".git").exists():
        raise errors.PackError(f"{root} is not a git checkout; a pack carries a checkout's state")
    if not git_read.pack_is_ignored(root):
        try:
            excluded = git_ops.exclude_locally(root, ".agent6/")
        except (OSError, UnicodeDecodeError) as exc:
            raise errors.PackError(f"{root / '.git/info/exclude'}: {exc}") from exc
        if excluded is not None:
            paths.chown_to_real_user(excluded)
            print(f"Excluded .agent6/ in {excluded}; .gitignore is untouched.")
    if refusal := git_read.pack_exclusion_error(root):
        raise errors.PackError(refusal)
    manifest = pack.pack(
        root,
        repo_config=repo_config,
        on_fold=lambda key: print(f"Moved to repo config: {key}"),
    )
    print(f"Packed {len(manifest.files)} files into {root / '.agent6'}; home copies kept.")
    return 0


def cmd_unpack() -> int:
    """Import a workspace pack into this home and remove it on success.

    Returns:
        Zero after importing the pack.
    """
    root = paths.project_root(pathlib.Path.cwd(), allow_pack=True)
    result = pack.unpack(root)
    print(f"Unpacked: {len(result.moved)} moved, {len(result.kept)} kept.")
    if not any(
        paths.global_config_dir() / name in result.moved for name in ("config.toml", "ui.toml")
    ):
        print("No global config.toml or ui.toml written.")
    for label, files in (("moved", result.moved), ("kept", result.kept)):
        for file in files:
            print(f"  {label}: {file}")
    print(f"State re-keyed: {result.origin.workspace} -> {root}")
    return 0
