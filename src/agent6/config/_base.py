# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one pydantic ConfigDict every config model shares."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field

# strict: a config typo must not coerce ("true" is not a bool, "5" is not an
# int, a bool is not a number); TOML already delivers native types, so lax
# coercion only ever laundered mistakes. allow_inf_nan=False: an infinite
# timeout or budget is never a real setting, and inf raised raw OverflowError
# downstream. Deliberate conversions stay as explicit mode="before"
# validators on their own fields.
MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

# TOML arrays arrive as Python lists; converting to the frozen tuple is the
# ONE container conversion strict mode keeps. Items still validate without
# scalar coercion (an int in a string array stays refused).
StrTuple = Annotated[tuple[str, ...], Field(strict=False)]


def _argv_elements(v: tuple[str, ...]) -> tuple[str, ...]:
    if any(not arg.strip() for arg in v):
        raise ValueError("argv elements must be non-empty strings")
    return v


# Command argv fields (`verify_command`, `metric.command`, notify hooks, MCP
# `command`). An empty ELEMENT is always a typo; an empty TUPLE stays valid
# where the field means "unset".
Argv = Annotated[StrTuple, AfterValidator(_argv_elements)]
