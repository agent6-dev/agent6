"""Read content and token usage from an Anthropic Messages reply."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent6.providers import provider_types


def non_negative_integer(value: Any, field_name: str) -> int:
    """Return a response value as a non-negative integer.

    Args:
        value: The decoded value; a bool or a fractional float is refused.
        field_name: The field's name, for the error.

    Returns:
        The integer.

    Raises:
        ProviderError: The value is not one.
    """  # noqa: DOC501  # the TypeError is raised and caught in the same try
    try:
        if isinstance(value, bool):
            raise TypeError
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise provider_types.ProviderError(
            f"Anthropic response {field_name} was not a non-negative integer"
        ) from exc
    if count < 0 or (isinstance(value, float) and not value.is_integer()):
        raise provider_types.ProviderError(
            f"Anthropic response {field_name} was not a non-negative integer"
        )
    return count


def usage_count(usage: Mapping[str, Any], key: str) -> int:
    """Return one usage count; 0 when absent."""
    value = usage.get(key)
    if value is None:
        return 0
    return non_negative_integer(value, f"usage.{key}")


def usage_mapping(value: Any) -> Mapping[str, Any]:
    """Return a `usage` object, empty when absent.

    Args:
        value: The `usage` field's value.

    Returns:
        The object; empty for None.

    Raises:
        ProviderError: The value is present but not an object.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise provider_types.ProviderError("Anthropic response usage was not an object")
    return value


def response_string(value: Any, field_name: str, *, empty: bool = True) -> str:
    """Return a response field as a string.

    Args:
        value: The field's value.
        field_name: The field's name, for the error.
        empty: Whether a blank string is accepted.

    Returns:
        The value, unchanged.

    Raises:
        ProviderError: The value is not a string, or is blank when one is required.
    """
    if not isinstance(value, str) or (not empty and not value.strip()):
        qualifier = "a nonempty string" if not empty else "a string"
        raise provider_types.ProviderError(f"Anthropic response {field_name} was not {qualifier}")
    return value


def billed_usage(value: Any) -> Mapping[str, Any]:
    """Read the billed input and output counts.

    Args:
        value: The reply's usage object.

    Returns:
        The compaction iterations' counts or the ordinary message usage.
    """
    usage = usage_mapping(value)
    iterations = usage.get("iterations") or []
    parts = [
        part for part in iterations if isinstance(part, dict) and part.get("type") == "compaction"
    ]
    if not parts:
        return usage
    return {
        key: sum(usage_count(part, key) for part in parts)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
        if any(key in part for part in parts)
    }


def require_metered_usage(usage: object, *, source: str) -> None:
    """Refuse a budgeted call whose usage cannot be metered.

    A gateway with usage tracking off returns all-zero counts, so presence is not
    enough; the input side must be positive with the cache counters summed in,
    since a fully cached turn reports `input_tokens: 0`.

    Args:
        usage: The response's `usage` value.
        source: The name that leads the error.

    Raises:
        ProviderError: No positive input usage (retryable: a usage-less reply is a
            gateway integrity failure, not a request defect).
    """
    if isinstance(usage, Mapping):
        usage = billed_usage(usage)
        total_input = (
            usage_count(usage, "input_tokens")
            + usage_count(usage, "cache_read_input_tokens")
            + usage_count(usage, "cache_creation_input_tokens")
        )
        if total_input > 0:
            return
    raise provider_types.ProviderError(
        f"{source} reported no usage input tokens (usage.input_tokens missing or 0); "
        "budgeted runs require provider usage accounting"
    )


def parse_response(data: dict[str, Any]) -> provider_types.ProviderResponse:
    """Parse one Messages body.

    Args:
        data: The response body.

    Returns:
        The response in agent6's canonical shape.

    Raises:
        ProviderError: The body is malformed (retryable).
    """
    content = data.get("content") or []
    if not isinstance(content, list):
        raise provider_types.ProviderError(
            f"Anthropic response `content` was {type(content).__name__}, not a"
            " list (malformed 2xx from upstream gateway)"
        )
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    tool_use_ids: set[str] = set()
    summary_block: dict[str, Any] | None = None
    for block in content:
        if not isinstance(block, dict):
            raise provider_types.ProviderError(
                "Anthropic response content block was not an object"
                " (malformed 2xx from upstream gateway)"
            )
        block_type = response_string(block.get("type"), "content block type", empty=False)
        if block_type == "text":
            text_parts.append(response_string(block.get("text", ""), "content text"))
        elif block_type == "tool_use":
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                raise provider_types.ProviderError(
                    "Anthropic response content tool_use.input was not an object"
                )
            tool_use_id = response_string(block.get("id"), "content tool_use.id", empty=False)
            if tool_use_id in tool_use_ids:
                raise provider_types.ProviderError(
                    f"Anthropic response had duplicate content tool_use.id {tool_use_id!r}"
                )
            tool_use_ids.add(tool_use_id)
            tool_uses.append(
                {
                    "id": tool_use_id,
                    "name": response_string(
                        block.get("name"), "content tool_use.name", empty=False
                    ),
                    "input": tool_input,
                }
            )
        elif block_type == "thinking":
            response_string(block.get("thinking", ""), "content thinking")
            response_string(block.get("signature", ""), "content signature", empty=False)
        elif block_type == "compaction" and len(content) == 1:
            response_string(block.get("signature"), "compaction signature", empty=False)
            text_parts.append(
                response_string(block.get("content"), "compaction content", empty=False)
            )
            summary_block = block
        elif block_type == "redacted_thinking":
            response_string(block.get("data", ""), "content redacted_thinking data", empty=False)
    usage = billed_usage(data.get("usage"))
    transformations = data.get("input_transformations") or []
    edits = (data.get("context_management") or {}).get("applied_edits", [])
    return provider_types.ProviderResponse(
        text="\n\n".join(text_parts),
        tool_uses=tuple(tool_uses),
        stop_reason=response_string(data.get("stop_reason"), "stop_reason", empty=False),
        input_tokens=usage_count(usage, "input_tokens"),
        output_tokens=usage_count(usage, "output_tokens"),
        cache_read_tokens=usage_count(usage, "cache_read_input_tokens"),
        cache_write_tokens=usage_count(usage, "cache_creation_input_tokens"),
        raw=data,
        summary_block=summary_block,
        thinking_blocks_dropped=sum(
            item.get("type") == "thinking_dropped" for item in transformations
        ),
        tool_results_cleared=sum(
            usage_count(edit, "cleared_tool_uses")
            for edit in edits
            if edit.get("type") == "clear_tool_uses_20250919"
        ),
    )
