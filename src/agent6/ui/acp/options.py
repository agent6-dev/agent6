# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build an ACP session's config options: a model picker and an effort picker.

An ACP session only opens runs, so it offers no mode.
"""

from __future__ import annotations

import pathlib
import typing
from typing import Any

from agent6 import kinds
from agent6.config import harness_config
from agent6.models import choices
from agent6.viewmodel import config_view

MODEL = "model"
EFFORT = "effort"
# The effort value that takes the config's own.
INHERIT = "inherit"
# The level each other effort value names: the levels a role's `effort` allows.
EFFORT_LEVELS: dict[str, kinds.EffortLevel] = {
    level: level for level in typing.get_args(kinds.EffortLevel)
}


def config_options(
    cwd: pathlib.Path,
    config_paths: tuple[pathlib.Path, ...],
    session_dir: pathlib.Path | None,
    *,
    mode: str,
    model: str,
    effort: kinds.EffortLevel | None,
) -> list[dict[str, Any]]:
    """Return a session's complete `configOptions` list: the model, then the effort.

    Args:
        cwd: The session's directory, whose layered config applies.
        config_paths: The `--config FILE` layers in command-line order.
        session_dir: The session's recorded run; None before its first prompt.
        mode: The session mode, whose role the options set.
        model: The route picked for the session, or "".
        effort: The effort picked for the session; None takes the config's.

    Returns:
        Two flat selects. The model's current value is the route the next prompt runs: the
        picked one, else the run's recorded one, else the config's. No option when none of
        the three names a route.
    """
    route = model or choices.session_route(cwd, config_paths, mode, session_dir)
    if not route:
        return []
    schema = config_view.field_schema(harness_config.RoleModel, ["effort"])
    meanings: dict[str, str] = schema.choice_meanings if schema is not None else {}
    return [
        {
            "configId": MODEL,
            "name": "Model",
            "description": "The worker's provider/model route, from the next prompt.",
            "category": "model",
            "type": "select",
            "currentValue": route,
            "options": [
                {"value": choice, "name": choice}
                for choice in sorted({*choices.available_routes(cwd, config_paths), route})
            ],
        },
        {
            "configId": EFFORT,
            "name": "Effort",
            "description": "The worker's reasoning effort, from the next prompt.",
            "category": "thought_level",
            "type": "select",
            "currentValue": effort or INHERIT,
            "options": [
                {"value": INHERIT, "name": INHERIT, "description": "Take the config's effort"},
                *(
                    {"value": level, "name": level, "description": meanings.get(level, "")}
                    for level in EFFORT_LEVELS
                ),
            ],
        },
    ]
