# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tool input schemas — pydantic models converted to JSON Schema for Anthropic."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    TOOL_NAME: ClassVar[str] = ""
    TOOL_DESCRIPTION: ClassVar[str] = ""


class ReadFileInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "read_file"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Read a text file from the repository. Path is relative to the repo root."
    )

    path: str = Field(min_length=1)


class ListDirInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "list_dir"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "List entries in a directory (non-recursive). Path is relative to the repo root."
    )

    path: str = Field(default=".")


class GrepInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "grep"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Search for a regex pattern in files under a path. Returns matching lines with paths."
    )

    pattern: str = Field(min_length=1)
    path: str = Field(default=".")
    case_insensitive: bool = False


class ApplyEditInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "apply_edit"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Apply one or more old_string/new_string edits to a file. Old string must be unique."
        " Use kind='create' (with empty old_string) to create a new file."
    )

    path: str = Field(min_length=1)
    edits: tuple[EditPair, ...] = Field(min_length=1)


class EditPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(pattern="^(replace|create)$")
    old_string: str = ""
    new_string: str = ""


class RunVerifyInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_verify_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run the user-declared verify command in the sandbox. No arguments."
    )


class RunCommandInput(_ToolInput):
    TOOL_NAME: ClassVar[str] = "run_command"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Run a command in the sandbox. argv must be an array of strings (no shell)."
        " Requires `run_commands` capability != 'no' in config; if 'ask', the user is prompted."
    )

    argv: tuple[str, ...] = Field(min_length=1)


ApplyEditInput.model_rebuild()

ALL_TOOLS: tuple[type[_ToolInput], ...] = (
    ReadFileInput,
    ListDirInput,
    GrepInput,
    ApplyEditInput,
    RunVerifyInput,
    RunCommandInput,
)


def schemas_as_provider_tools() -> list[dict[str, Any]]:
    """Emit Anthropic-API-shape tool descriptors. (kept dict-typed to avoid circular import)"""
    out: list[dict[str, Any]] = []
    for cls in ALL_TOOLS:
        schema = cls.model_json_schema()
        # Anthropic wants the schema directly, not wrapped, with "type" present.
        schema.setdefault("type", "object")
        out.append(
            {
                "name": cls.TOOL_NAME,
                "description": cls.TOOL_DESCRIPTION,
                "input_schema": schema,
            }
        )
    return out
