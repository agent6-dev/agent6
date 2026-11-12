# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The facts a fan-out hands a lane, in one file named by `AGENT6_LANE`.

The lane reads it at its config load and its manifest write, so the grouping survives a
coordinator death; a run outside a lane has no envelope.
"""

from __future__ import annotations

import os
import pathlib

import pydantic

from agent6.config import ConfigError
from agent6.sessions import manifest

ENV = "AGENT6_LANE"


class LaneEnvelope(pydantic.BaseModel, frozen=True, extra="forbid"):
    """What a lane knows about itself that no config layer holds.

    **Support: internal.** Written by the coordinator for one spawn, read by that lane.

    Attributes:
        config: The lane's config snapshot; the lane's sole `--config` must name it.
        notes: The snapshot's serialization loss notes, recorded in the lane's manifest.
        coordinator: The session every listing nests the lane under.
        group: The fan-out the lane belongs to.
        lane: The lane's number, from 1.
        gate_source: Where the verify gate in the snapshot came from; "" for none.
        operator_task: The task's words before composition; None when they are the task.
    """

    config: pathlib.Path
    notes: tuple[str, ...] = ()
    coordinator: str = pydantic.Field(min_length=1)
    group: str = pydantic.Field(min_length=1)
    lane: int = pydantic.Field(ge=1)
    gate_source: manifest.GateSource = ""
    operator_task: str | None = None


def write(path: pathlib.Path, envelope: LaneEnvelope) -> None:
    """Write the envelope the lane's `AGENT6_LANE` names.

    Args:
        path: The file.
        envelope: The lane's facts.
    """
    path.write_text(envelope.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read() -> LaneEnvelope | None:
    """Return the envelope `AGENT6_LANE` names, or None outside a lane.

    Returns:
        The lane's facts, or None when the variable is unset.

    Raises:
        ConfigError: The variable names a missing or malformed file.
    """
    raw = os.environ.get(ENV)
    if not raw:
        return None
    try:
        return LaneEnvelope.model_validate_json(pathlib.Path(raw).read_text(encoding="utf-8"))
    except (OSError, pydantic.ValidationError) as exc:
        raise ConfigError(f"invalid lane envelope {raw}: {exc}") from exc
