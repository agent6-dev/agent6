# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every config leaf is auditable through `agent6 config show`.

A field the renderer does not walk into is a setting an operator cannot see the
value or the origin of -- and for a security-sensitive one, cannot audit at all.
The invariant was stated and unchecked, so a new nested field could be added
without anyone noticing it never appeared.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from agent6.config import Config
from agent6.config.layer import EffectiveConfig
from agent6.viewmodel.config_view import build_config_view, render_show

# Sections keyed by an operator-chosen NAME. A leaf under them only exists once
# an entry does, so the walk needs one to have something to look for.
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


def _rendered_keys(config: Config) -> set[str]:
    """The leaf paths the view-model actually walks to, keyed exactly -- not a
    substring scan, under which a dropped `sandbox.network` row hides behind
    `mcp.servers.notes.sandbox.network`."""
    view = build_config_view(EffectiveConfig(config=config, sources={}, layers=()))
    return {s.key for s in view.settings}


def test_every_leaf_of_a_populated_config_is_rendered() -> None:
    config = Config.model_validate(_POPULATED)
    missing = sorted(_leaf_paths(config) - _rendered_keys(config))
    assert not missing, f"`config show` does not render these leaves: {missing}"


def test_the_new_mcp_network_knob_is_among_them() -> None:
    """A nested leaf under a name-keyed section, two levels down, where a
    renderer might stop walking: the RENDERER reaches it, not just the schema."""
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
    """Secure by default: these are the ones a review would check first, so a
    change to any of them has to change this list too."""
    node: Any = Config()
    for part in path.split("."):
        node = getattr(node, part)
    assert node == safe


def test_every_rendered_leaf_carries_its_meaning() -> None:
    """The JSON view (so the web editor's hover text and `--descriptions`)
    describes every leaf it renders, unset section holders included:
    `models.worker` shown as (unset) is exactly where "what is this?" is asked."""
    for config in (Config(), Config.model_validate(_POPULATED)):
        eff = EffectiveConfig(config=config, sources={}, layers=())
        view = json.loads(render_show(eff, as_json=True))
        undescribed = sorted(k for k, leaf in view.items() if not leaf["description"])
        assert not undescribed, f"leaves with no description: {undescribed}"
