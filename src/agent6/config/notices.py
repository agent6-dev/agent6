"""Config editors report the layers masking their writes."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

from agent6.config import layer, model


def write_message(
    repo_root: pathlib.Path,
    keys: Iterable[str],
    message: str,
    *,
    to_repo: bool = False,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> str:
    """Append the layers masking a successful config edit to its confirmation.

    Args:
        repo_root: The repo whose config the editor shows.
        keys: The leaves written or removed, or a table removed whole.
        message: The editor's confirmation.
        to_repo: The edit landed in the repo config instead of the global config.
        config_paths: The editor's `--config` files in command-line order.

    Returns:
        The confirmation with CLI-worded shadow notes, unchanged when the config cannot load.
    """
    try:
        eff = layer.load_effective(repo_root, config_paths)
    except model.ConfigError:
        return message
    leaves = sorted(
        {leaf for key in keys for leaf in eff.sources if leaf == key or leaf.startswith(key + ".")}
    )
    for leaf in leaves:
        above = layer.shadowing_layer(eff, leaf, "repo" if to_repo else "global")
        if above is not None:
            tail = (
                " while that flag is used."
                if above.name == "flag"
                else "; a run here reads its value."
            )
            message += f"\nnote: {above.described} overrides {leaf}{tail}"
    return message
