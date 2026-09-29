# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every config leaf is auditable through `agent6 config show`.

A field the renderer does not walk into cannot be seen or audited.
"""

from __future__ import annotations

import json
from typing import Any

import pydantic
import pytest

from agent6.config import Config, layer
from agent6.viewmodel import config_view

# Sections keyed by an operator-chosen name; a leaf under them exists only once an entry does.
_POPULATED: dict[str, Any] = {
    "providers": {"acme": {"api_format": "openai", "base_url": "https://example.invalid/v1"}},
    "mcp": {
        "enabled": True,
        "servers": {
            "notes": {
                "command": ["true"],
                "sandbox": {"read_paths": ["/usr"], "network": "session"},
            }
        },
    },
}


def _leaf_paths(model: pydantic.BaseModel, prefix: str = "") -> set[str]:
    """Every dotted leaf path of a populated model instance."""
    leaves: set[str] = set()
    for name in type(model).model_fields:
        value = getattr(model, name)
        path = f"{prefix}{name}"
        if isinstance(value, pydantic.BaseModel):
            leaves |= _leaf_paths(value, f"{path}.")
        elif isinstance(value, dict) and value:
            for key, entry in value.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(entry, pydantic.BaseModel):
                    leaves |= _leaf_paths(entry, f"{path}.{key}.")
                else:
                    leaves.add(path)
        else:
            leaves.add(path)
    return leaves


def _rendered_keys(config: Config) -> set[str]:
    """Return the leaf paths the view-model walks to, keyed exactly.

    A substring scan would let a dropped `sandbox.network` row hide behind
    `mcp.servers.notes.sandbox.network`.
    """
    view = config_view.build_config_view(
        layer.EffectiveConfig(config=config, sources={}, layers=())
    )
    return {s.key for s in view.settings}


def test_every_leaf_of_a_populated_config_is_rendered() -> None:
    config = Config.model_validate(_POPULATED)
    missing = sorted(_leaf_paths(config) - _rendered_keys(config))
    assert not missing, f"`config show` does not render these leaves: {missing}"


def test_the_new_mcp_network_knob_is_among_them() -> None:
    """The renderer reaches a nested leaf two levels under a name-keyed section."""
    config = Config.model_validate(_POPULATED)
    assert "mcp.servers.notes.sandbox.network" in _rendered_keys(config)


@pytest.mark.parametrize(
    ("path", "safe"),
    [
        ("sandbox.run_commands", "ask"),
        ("sandbox.network", "auto"),
        ("sandbox.isolation", "auto"),
        ("sandbox.protect_git", True),
        ("mcp.enabled", False),
    ],
)
def test_security_sensitive_defaults_are_the_safe_value(path: str, safe: object) -> None:
    """Security-sensitive defaults are the safe value; a change to any of them changes this list."""
    node: Any = Config()
    for part in path.split("."):
        node = getattr(node, part)
    assert node == safe


def test_every_rendered_leaf_carries_its_meaning() -> None:
    """The JSON view describes every leaf it renders, unset section holders included."""
    for config in (Config(), Config.model_validate(_POPULATED)):
        eff = layer.EffectiveConfig(config=config, sources={}, layers=())
        view = json.loads(config_view.render_show(eff, as_json=True))
        undescribed = sorted(k for k, leaf in view.items() if not leaf["description"])
        assert not undescribed, f"leaves with no description: {undescribed}"
