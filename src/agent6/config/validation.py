# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared config validation policy and operator-facing errors."""

from __future__ import annotations

import json
from typing import Annotated, Any

import pydantic
import pydantic.fields

from agent6 import errors, kinds


class ConfigError(errors.OperatorError):
    """The config file is missing, malformed or invalid: the operator's file, so a refusal."""


def model_route(spec: str) -> kinds.ModelRoute:
    """Parse a persisted, fully qualified route at its first slash.

    Args:
        spec: The provider/model setting.

    Returns:
        The provider and verbatim model id.

    Raises:
        ValueError: The provider or model id is missing.
    """
    provider, slash, model = spec.partition("/")
    if not slash or not provider.strip() or not model.strip():
        raise ValueError('model must be fully qualified: model = "provider/model"')
    return kinds.ModelRoute(provider, model)


def refuse_split_model(data: Any, where: str) -> None:
    """Refuse a settings table holding the old provider leaf.

    Args:
        data: The raw settings table.
        where: The table's path.

    Raises:
        ValueError: The table holds provider, with or without model.
    """
    if isinstance(data, dict) and "provider" in data:
        route = f"{data['provider']}/{data.get('model', '<model>')}"
        raise ValueError(
            f"[{where}] provider and model are one setting now: model = {json.dumps(route)}"
        )


# strict: TOML delivers native types, so a typo must not coerce ("5" is not an int).
# allow_inf_nan=False: an inf timeout raises OverflowError; an inf or nan budget cap never binds.
MODEL_CONFIG = pydantic.ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

# The list-to-tuple conversion is the only one strict mode keeps; items stay uncoerced.
StrTuple = Annotated[tuple[str, ...], pydantic.Field(strict=False)]


def _argv_elements(v: tuple[str, ...]) -> tuple[str, ...]:
    """Refuse an empty argv element.

    Args:
        v: The argv.

    Returns:
        The argv unchanged.

    Raises:
        ValueError: An element is empty or whitespace.
    """
    if any(not arg.strip() for arg in v):
        raise ValueError("argv elements must be non-empty strings")
    return v


# Command argv fields: an empty element is a typo; an empty tuple means "unset".
Argv = Annotated[StrTuple, pydantic.AfterValidator(_argv_elements)]


def is_argv_field(info: pydantic.fields.FieldInfo) -> bool:
    """Return whether a field is an `Argv`, by the validator the alias carries.

    Args:
        info: The field's schema.

    Returns:
        True for an `Argv` field.
    """
    return any(
        isinstance(meta, pydantic.AfterValidator) and meta.func is _argv_elements
        for meta in info.metadata
    )
