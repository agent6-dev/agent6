"""Validate workspace pack manifests without importing filesystem path resolution."""

from __future__ import annotations

import os
import pathlib
import socket
from collections.abc import Callable, Iterator
from typing import Annotated, Literal

import pydantic

from agent6 import errors

Hash = Annotated[str, pydantic.Field(pattern=r"^[0-9a-f]{64}$")]
CONFIG_FILES = ("config.toml", "secrets.toml", "ui.toml")
# A pack is `pack.json` and these directories inside `.agent6/`; every other name there is not its.
AREAS = ("config", "repo", "skills")
# The state a pack never carries, so no manifest may name it: this home's locks, its record
# of the last unpack, and the sessions' own worktrees.
REPO_LEFT_OUT = ("locks", "pack-base.json", "worktrees")


class Origin(pydantic.BaseModel):
    """Identify the exporting home and workspace."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    host: str = pydantic.Field(min_length=1)
    home: str
    repo_id: str = pydantic.Field(min_length=1)
    workspace: str

    @pydantic.field_validator("home", "workspace")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        """Require an absolute path.

        Args:
            value: The value from the manifest.

        Returns:
            The validated absolute path.

        Raises:
            ValueError: The path is relative.
        """
        if not pathlib.Path(value).is_absolute():
            raise ValueError("must be an absolute path")
        return value


class PackedFile(pydantic.BaseModel):
    """Record a file's exported hash, timestamp and common ancestor."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    sha256: Hash
    mtime: pydantic.AwareDatetime
    base: Hash | None


def _pack_name(value: str) -> str:
    """Validate a pack-relative file name.

    Args:
        value: The value from the manifest.

    Returns:
        The validated name.

    Raises:
        ValueError: The name escapes the transfer layout.
    """
    parts = value.split("/")
    valid = len(parts) >= 2 and all(p not in ("", ".", "..") for p in parts)
    valid = valid and "\\" not in value and "\0" not in value
    valid = valid and (
        (parts[0] == "config" and len(parts) == 2 and parts[1] in CONFIG_FILES)
        or (parts[0] == "skills")
        or (parts[0] == "repo" and parts[1] not in REPO_LEFT_OUT)
    )
    if not valid:
        raise ValueError(
            "must name config/{config,ui,secrets}.toml, skills/* or repo/*"
            f" (not {', '.join(REPO_LEFT_OUT)})"
        )
    return value


PackName = Annotated[str, pydantic.AfterValidator(_pack_name)]


class Manifest(pydantic.BaseModel):
    """Validate the portable state manifest at its trust boundary."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    packed_at: pydantic.AwareDatetime
    origin: Origin
    files: dict[PackName, PackedFile]


def check_path(path: pathlib.Path, *, regular: bool = False) -> None:
    """Refuse links in transfer paths and non-regular file inputs.

    Args:
        path: The transfer file or directory.
        regular: Require an existing regular file.

    Raises:
        PackError: The path is unsafe or unreadable.
    """
    try:
        for part in (path, *path.parents):
            if part.is_symlink():
                raise errors.PackError(f"{part}: symlink paths cannot be packed or unpacked")
        if regular and not path.is_file():
            raise errors.PackError(f"{path}: expected a regular file")
    except OSError as exc:
        raise errors.PackError(f"{path}: {exc}") from exc


def read(root: pathlib.Path) -> Manifest:
    """Read a pack manifest, naming any invalid field.

    Args:
        root: The resolved project root.

    Returns:
        The validated manifest.

    Raises:
        PackError: The manifest is missing, unreadable or invalid.
    """
    file = root / ".agent6" / "pack.json"
    if not os.path.lexists(file):
        raise errors.PackError(f"{file.parent}: no pack here; agent6 pack writes one")
    check_path(file, regular=True)
    try:
        return Manifest.model_validate_json(file.read_bytes())
    except (OSError, pydantic.ValidationError) as exc:
        raise errors.PackError(f"{file}: {exc}") from exc


def is_own(origin: Origin, home: pathlib.Path) -> bool:
    """Return whether the origin names this host and effective home."""
    return origin.host == socket.gethostname() and origin.home == str(home)


def check_origin(root: pathlib.Path) -> None:
    """Refuse any pack before a command reads this home's state.

    The origin and time are the manifest's own words, which a model can write: the refusal
    quotes them as what the pack says.

    Args:
        root: The resolved project root.

    Raises:
        PackError: A pack is present or its manifest is invalid.
    """
    if not (root / ".agent6/pack.json").exists():
        return
    manifest = read(root)
    origin = manifest.origin
    time = manifest.model_dump(mode="json")["packed_at"]
    raise errors.PackError(
        f"{root / '.agent6'}: a pack is present, and no command starts beside one. The pack"
        f" says it is from {origin.host!r}:{origin.home!r}, packed at {time}. Run `agent6"
        " unpack` in that workspace, which lists what it would install and asks first; or"
        " delete a pack you did not make from that directory: pack.json, config/, repo/ and"
        " skills/."
    )


def pack_dirs(root: pathlib.Path, *, onerror: Callable[[OSError], None]) -> Iterator[pathlib.Path]:
    """Walk a project tree for the directories that hold an `.agent6`, the root's first.

    The walk follows no link and enters neither `.git` nor `.agent6`.

    Args:
        root: The project root.
        onerror: Receives each directory the walk cannot list.

    Yields:
        Each directory with an `.agent6` directory in it.
    """
    for directory, children, _files in os.walk(root, followlinks=False, onerror=onerror):
        if ".agent6" in children:
            yield pathlib.Path(directory)
        children[:] = sorted(name for name in children if name not in (".git", ".agent6"))
