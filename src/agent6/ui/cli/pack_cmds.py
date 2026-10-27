"""Carry this home's state with the workspace."""

from __future__ import annotations

import pathlib

from agent6 import errors, pack, paths
from agent6.ui.cli import init


def cmd_pack() -> int:
    """Write a workspace pack and gitignore it.

    Returns:
        Zero after writing the pack.

    Raises:
        PackError: The pack or its gitignore entry cannot be written.
    """
    root = paths.project_root(pathlib.Path.cwd())
    try:
        ignore = root / ".gitignore"
        text = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        if ".agent6/" not in {line.strip() for line in text.splitlines()}:
            print(init.append_gitignore(root, [".agent6/"]))
            paths.chown_to_real_user(ignore)
    except (OSError, UnicodeDecodeError) as exc:
        raise errors.PackError(f"{root / '.gitignore'}: {exc}") from exc
    manifest = pack.pack(root)
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
    for label, files in (("moved", result.moved), ("kept", result.kept)):
        for file in files:
            print(f"  {label}: {file}")
    print(f"State re-keyed: {result.origin.workspace} -> {root}")
    return 0
