"""Validate workspace pack manifests without importing filesystem path resolution."""

from __future__ import annotations

import pathlib
import socket
from typing import Annotated, Literal

import pydantic

from agent6 import errors

Hash = Annotated[str, pydantic.Field(pattern=r"^[0-9a-f]{64}$")]
CONFIG_FILES = ("config.toml", "secrets.toml", "ui.toml")


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
        or (parts[0] == "repo" and parts[1] != "locks")
    )
    if not valid:
        raise ValueError(
            "must name config/{config,ui,secrets}.toml, skills/* or repo/* (not locks)"
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
        f"{root / '.agent6'}: pack from {origin.host}:{origin.home}, packed at {time}. "
        "Run agent6 unpack in that workspace or delete its .agent6/ directory."
    )
