# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[container]` table: which container runs the commands, and how it starts and stops."""

from __future__ import annotations

import pathlib
from typing import Annotated

import pydantic

from agent6.config import validation


def _container_path(value: str) -> str:
    """Refuse a workspace path that is not an absolute, normal container path.

    Args:
        value: The configured path, empty for unset.

    Returns:
        The path unchanged.

    Raises:
        ValueError: The path is relative, holds `..` or a control character.
    """
    if not value:
        return value
    path = pathlib.PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or any(ord(c) < 32 for c in value):
        raise ValueError("container.workspace must be an absolute container path")
    return value


ContainerPath = Annotated[str, pydantic.AfterValidator(_container_path)]


class ContainerConfig(pydantic.BaseModel):
    """The `[container]` table; `sandbox.isolation = "container"` selects it."""

    model_config = validation.MODEL_CONFIG

    exec: validation.Argv = pydantic.Field(
        default=(),
        description=(
            "argv prefix the launcher invocation is appended to, no shell; empty: find this "
            "workspace's running Docker or Podman dev container by its labels; set: run inside "
            'a container of your choice, such as `["docker", "exec", "-i", "--user", "vscode", '
            '"agent6-dev"]`, `["sudo", "-n", "docker", "exec", "-i", "agent6-dev"]`, '
            '`["podman", "exec", "-i", "NAME"]` or `["devcontainer", "exec", '
            '"--workspace-folder", "DIR"]`; never rewritten, never given a tty'
        ),
    )
    runtime: validation.Argv = pydantic.Field(
        default=(),
        description=(
            "argv prefix for the runtime's CLI, used by discovery and by the exec it derives; "
            "empty: the `docker`, then `podman`, found on PATH; a host whose docker needs sudo "
            'sets `["sudo", "-n", "docker"]`'
        ),
    )
    workspace: ContainerPath = pydantic.Field(
        default="",
        description=(
            "workspace path inside the container; empty: the discovered bind mount's path, or "
            "the host path under `container.exec`; set: that path, which creates no mount"
        ),
    )
    start: validation.Argv = pydantic.Field(
        default=(),
        description=(
            "operator command that starts the container, as argv without a shell; empty: "
            "agent6 starts nothing and refuses when no dev container runs; run once when no "
            "container runs, before discovery runs again; separate from "
            "`workspace.setup_command`, which runs inside the prepared workspace"
        ),
    )
    stop: validation.Argv = pydantic.Field(
        default=(),
        description=(
            "operator command that stops a container the run started, as argv without a shell; "
            "runs after the run and every run attached to the container settled their jobs and "
            "records; empty: the container is left running; attaching alone grants no stop"
        ),
    )
