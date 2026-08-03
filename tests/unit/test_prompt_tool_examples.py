# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every call-shaped tool example in the model-facing prompt text names real
schema fields: a prompt teaching `tool(field, ...)` with a renamed or deleted
field walks the model into a validation failure on its first try."""

from __future__ import annotations

import re

import agent6.prompts.loop as loop_prompts
from agent6.tools.schema import ALL_TOOLS, ASK_EXTRA_TOOLS, LOOP_EXTRA_TOOLS, PLAN_EXTRA_TOOLS

_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(\s*([a-z_][^()]*)\)")
_ARG = re.compile(r"\s*([a-z_][a-z0-9_]*)")


def test_prompt_tool_call_examples_use_schema_field_names() -> None:
    fields = {
        cls.TOOL_NAME: set(cls.model_fields)
        for cls in ALL_TOOLS + LOOP_EXTRA_TOOLS + PLAN_EXTRA_TOOLS + ASK_EXTRA_TOOLS
    }
    bad: list[str] = []
    checked = 0
    for attr in dir(loop_prompts):
        value = getattr(loop_prompts, attr)
        if not isinstance(value, str):
            continue
        for call in _CALL.finditer(value):
            tool, arglist = call.group(1), call.group(2)
            if tool not in fields:
                continue
            checked += 1
            for part in arglist.split(","):
                arg = _ARG.match(part)
                if arg is not None and arg.group(1) not in fields[tool]:
                    bad.append(f"{attr}: {tool}({arglist}) -- {arg.group(1)!r} is not a field")
    assert checked, "no call-shaped tool examples found; the guard is dead"
    assert not bad, "prompt examples name unknown tool fields:\n" + "\n".join(bad)
