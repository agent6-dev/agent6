# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Mark the test process as a lane: write an envelope and point `AGENT6_LANE` at it."""

from __future__ import annotations

import pathlib

import pytest

from agent6.app import lane_envelope
from agent6.sessions import manifest


def mark_lane(
    monkeypatch: pytest.MonkeyPatch,
    where: pathlib.Path,
    *,
    config: pathlib.Path | None = None,
    coordinator: str = "co",
    group: str = "fan",
    lane: int = 1,
    gate_source: manifest.GateSource = "",
    notes: tuple[str, ...] = (),
    operator_task: str | None = None,
) -> pathlib.Path:
    """Write a lane envelope under `where` and set `AGENT6_LANE` and `AGENT6_SUBRUN`.

    Args:
        monkeypatch: The test's monkeypatch.
        where: A directory for the envelope file.
        config: The snapshot the envelope names; a placeholder path when the test loads none.
        coordinator: The coordinator session id.
        group: The fan-out id.
        lane: The lane number.
        gate_source: The gate's origin.
        notes: The snapshot's loss notes.
        operator_task: The operator's task before composition.

    Returns:
        The envelope's path.
    """
    where.mkdir(parents=True, exist_ok=True)
    path = where / f"lane-{lane}.json"
    lane_envelope.write(
        path,
        lane_envelope.LaneEnvelope(
            config=config if config is not None else where / f"lane-{lane}-config.toml",
            notes=notes,
            coordinator=coordinator,
            group=group,
            lane=lane,
            gate_source=gate_source,
            operator_task=operator_task,
        ),
    )
    monkeypatch.setenv(lane_envelope.ENV, str(path))
    monkeypatch.setenv("AGENT6_SUBRUN", "1")
    return path
