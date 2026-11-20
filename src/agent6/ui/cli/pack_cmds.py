"""Carry this home's state with the workspace."""

from __future__ import annotations

import pathlib

from agent6 import errors, pack, paths, portable
from agent6.ui import keymap
from agent6.ui.cli import common, steer_tty
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


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


def _shown(text: object) -> str:
    """Return a pack's own text for one terminal line: quoted whole unless it prints as it is."""
    return str(text) if str(text).isprintable() else repr(str(text))


def _preview_lines(root: pathlib.Path, preview: pack.Preview) -> list[str]:
    """Return the listing `unpack` prints before it asks.

    Args:
        root: The project root.
        preview: What the unpack would do.

    Returns:
        Every file to install with its destination and, under a config file, every setting
        it sets; then the files this home already holds.
    """
    origin = preview.origin
    moves = [file for file in preview.files if file.outcome == "move"]
    keeps = [file for file in preview.files if file.outcome == "keep"]
    lines = [
        f"Pack in {root / '.agent6'}. It says it is from {origin.host!r}:{origin.home!r},"
        f" packed at {preview.packed_at}.",
        f"It would install {common.plural(len(moves), 'file')} into this home:",
    ]
    marked = 0
    for file in moves:
        lines.append(f"  {_shown(file.name)} -> {_shown(file.home)}")
        for setting in preview.settings.get(file.name, ()):
            marked += setting.operator_only
            mark = "  [operator-only]" if setting.operator_only else ""
            value = setting.value or "(a secret, not shown)"
            lines.append(f"      {_shown(setting.key)} = {value}{mark}")
    if keeps:
        lines.append(f"It would keep {common.plural(len(keeps), 'file')} this home already has:")
        lines.extend(f"  {_shown(file.name)} = {_shown(file.home)}" for file in keeps)
    if marked:
        lines.append(
            f"{common.plural(marked, 'setting')} marked [operator-only]: a key under `sandbox`,"
            " or one no machine file may set, such as a host command, a provider or a"
            " credential."
        )
    return lines


def cmd_unpack(*, yes: bool = False) -> int:
    """List what a workspace pack would install into this home, ask, then import it.

    Args:
        yes: Install without asking.

    Returns:
        Zero after importing the pack or after a declined question; 2 when there is no
        terminal to ask on and `yes` is unset.
    """
    root = paths.project_root(pathlib.Path.cwd(), allow_pack=True)
    preview = pack.preview(root)
    for line in _preview_lines(root, preview):
        print(line)
    if not yes and any(file.outcome == "move" for file in preview.files):
        if not portable.has_controlling_tty():
            common.refuse(
                "unpack asks before it installs, and there is no terminal to ask on. Pass"
                " --yes to install what is listed above."
            )
            return 2
        typed = steer_tty.tty_prompt(
            f"Install them? {keymap.approval_prompt_suffix(standing=False)}",
            fall_back_to_stdin=False,
        )
        if keymap.answer_for(typed or "", standing=False) != "yes":
            print("Nothing was installed; the pack stays.")
            return 0
    result = pack.unpack(root, shown=preview)
    print(f"Unpacked: {len(result.moved)} moved, {len(result.kept)} kept.")
    if not any(
        paths.global_config_dir() / name in result.moved for name in ("config.toml", "ui.toml")
    ):
        print("No global config.toml or ui.toml written.")
    print(f"State re-keyed: {result.origin.workspace} -> {root}")
    return 0
