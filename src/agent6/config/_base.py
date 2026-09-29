# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one pydantic ConfigDict every config model shares."""

from __future__ import annotations

from typing import Annotated

import pydantic

# strict: TOML delivers native types, so a typo must not coerce ("5" is not an int).
# allow_inf_nan=False: an infinite timeout or budget raises a raw OverflowError downstream.
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
