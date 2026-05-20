# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared helpers for sub-agents."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agent6.providers import Provider, ProviderError, ProviderResponse


class SubAgentError(Exception):
    """A sub-agent failed to produce a validated structured output."""


T = TypeVar("T", bound=BaseModel)


def call_for_model[T: BaseModel](
    provider: Provider,
    *,
    system: str,
    user: str,
    output_model: type[T],
    max_tokens: int = 4096,
) -> T:
    """Call the model asking for JSON matching *output_model*. Validate strictly."""
    schema_hint = json.dumps(output_model.model_json_schema(), indent=2)
    user_with_schema = (
        f"{user}\n\n"
        "Respond with ONLY a JSON object that conforms exactly to this schema. "
        "Do not include code fences or commentary.\n\n"
        f"Schema:\n{schema_hint}"
    )
    try:
        resp: ProviderResponse = provider.call(
            system=system,
            messages=[{"role": "user", "content": user_with_schema}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        raise SubAgentError(f"provider call failed: {exc}") from exc
    text = _strip_fences(resp.text.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubAgentError(
            f"sub-agent did not return valid JSON: {exc}; head={text[:300]!r}"
        ) from exc
    try:
        return output_model.model_validate(parsed)
    except ValidationError as exc:
        raise SubAgentError(f"sub-agent JSON failed schema validation: {exc}") from exc


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text
