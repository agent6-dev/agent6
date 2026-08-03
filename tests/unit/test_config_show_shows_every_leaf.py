# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every config leaf is auditable through `agent6 config show`.

A field the renderer does not walk into is a setting an operator cannot see the
value or the origin of -- and for a security-sensitive one, cannot audit at all.
The invariant was stated and unchecked, so a new nested field could be added
without anyone noticing it never appeared.
"""

from __future__ import annotations

from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from agent6.config import Config
from agent6.config.layer import EffectiveConfig
from agent6.viewmodel.config_view import render_show

# Sections keyed by an operator-chosen NAME. A leaf under them only exists once
# an entry does, so the walk needs one to have something to look for.
_POPULATED: dict[str, Any] = {
    "providers": {"acme": {"api_format": "openai", "base_url": "https://example.invalid/v1"}},
    "mcp": {
        "enabled": True,
        "servers": {
            "notes": {
                "command": ["true"],
                "sandbox": {"read_paths": ["/usr"], "network": "none"},
            }
        },
    },
}


def _leaf_paths(model: BaseModel, prefix: str = "") -> set[str]:
    """Every dotted leaf path of a populated model instance."""
    leaves: set[str] = set()
    for name in type(model).model_fields:
        value = getattr(model, name)
        path = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            leaves |= _leaf_paths(value, f"{path}.")
        elif isinstance(value, dict) and value:
            for key, entry in value.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(entry, BaseModel):
                    leaves |= _leaf_paths(entry, f"{path}.{key}.")
                else:
                    leaves.add(path)
        else:
            leaves.add(path)
    return leaves


def _optional_model(annotation: Any) -> bool:
    return (
        any(isinstance(arg, type) and issubclass(arg, BaseModel) for arg in get_args(annotation))
        and get_origin(annotation) is not None
    )


def test_every_leaf_of_a_populated_config_is_rendered() -> None:
    config = Config.model_validate(_POPULATED)
    shown = render_show(EffectiveConfig(config=config, sources={}, layers=()))

    missing = sorted(path for path in _leaf_paths(config) if path not in shown)
    assert not missing, f"`config show` does not render these leaves: {missing}"


def test_the_new_mcp_network_knob_is_among_them() -> None:
    """The field this test was written for: a nested leaf under a name-keyed
    section, two levels down, which is where a renderer stops walking."""
    config = Config.model_validate(_POPULATED)
    assert "mcp.servers.notes.sandbox.network" in _leaf_paths(config)


@pytest.mark.parametrize(
    ("path", "safe"),
    [
        ("sandbox.run_commands", "ask"),
        ("sandbox.tool_network", "auto"),
        ("sandbox.isolation", "auto"),
        ("sandbox.protect_git", True),
        ("mcp.enabled", False),
    ],
)
def test_security_sensitive_defaults_are_the_safe_value(path: str, safe: object) -> None:
    """Secure by default: these are the ones a review would check first, so a
    change to any of them has to change this list too."""
    node: Any = Config()
    for part in path.split("."):
        node = getattr(node, part)
    assert node == safe
