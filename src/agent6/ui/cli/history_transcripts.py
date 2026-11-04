"""Prepare transcript conversation lines for history search without provider prompts."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Iterator
from typing import Any

from agent6.sessions import objects


@dataclasses.dataclass(frozen=True, slots=True)
class SearchText:
    """One transcript's search input or its read error."""

    path: pathlib.Path
    text: str = ""
    error: str = ""


def search_text(targets: list[pathlib.Path]) -> Iterator[SearchText]:
    """Read conversation content from every target's provider transcripts.

    System and developer messages carry the prompt and skill index, not conversation.

    Args:
        targets: The session directories.

    Yields:
        One file's JSON lines for rg or its read error.
    """
    for target in targets:
        for path in sorted((target / "transcripts").glob("*.json")):
            try:
                record = objects.materialize(
                    json.loads(path.read_text(encoding="utf-8")),
                    target / "objects",
                    record_path=path,
                )
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
                text = json.dumps(content, indent=2, ensure_ascii=False) + "\n"
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RecursionError,
            ) as exc:
                yield SearchText(path, error=f"{path}: {exc}")
                continue
            yield SearchText(path, text=text)
