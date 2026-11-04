"""Prepare transcript conversation lines for history search without provider prompts."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class SearchText:
    """Search input, its source file per line, and files whose evidence could not be read."""

    text: str
    sources: tuple[pathlib.Path, ...]
    errors: tuple[str, ...]


def search_text(targets: list[pathlib.Path]) -> SearchText:
    """Read conversation content from every target's provider transcripts.

    System and developer messages carry the prompt and skill index, not conversation.

    Args:
        targets: The session directories.

    Returns:
        JSON lines for rg, their source paths, and read errors.
    """
    lines: list[str] = []
    sources: list[pathlib.Path] = []
    errors: list[str] = []
    for target in targets:
        for path in sorted((target / "transcripts").glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                content: dict[str, Any] = (
                    {"outcome": record["outcome"]} if "outcome" in record else {}
                )
                for side in ("request", "response"):
                    body = record[side]["body"]
                    if isinstance(body, str):
                        try:
                            body = json.loads(body)
                        except json.JSONDecodeError:
                            content[side] = body
                            continue
                    content[side] = {
                        key: value
                        for key, value in body.items()
                        if key not in ("instructions", "system", "tools", "tool_choice")
                    }
                    for key in ("messages", "input"):
                        if isinstance(messages := content[side].get(key), list):
                            content[side][key] = [
                                message
                                for message in messages
                                if not isinstance(message, dict)
                                or message.get("role") not in ("system", "developer")
                            ]
                kept = json.dumps(content, indent=2, ensure_ascii=False).splitlines()
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RecursionError,
            ) as exc:
                errors.append(f"{path}: {exc}")
                continue
            lines.extend(kept)
            sources.extend([path] * len(kept))
    return SearchText("\n".join(lines), tuple(sources), tuple(errors))
