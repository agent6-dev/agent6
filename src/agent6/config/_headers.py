# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[providers.<name>].extra_headers` grammar: literal headers and request-fact templates.

A literal string rides every API request. A template table names the request classes it
reaches and renders from a fixed fact vocabulary; nothing in it reads the environment, a
file, a secret or the prompt. Every rule here is checked at config load, before any
request exists.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import Literal

import pydantic

from agent6.config import _base

HeaderTarget = Literal["exchange", "inference", "models", "oauth", "usage"]
BooleanFact = Literal["has_images", "has_tool_results", "is_first_turn", "is_subagent"]

# The facts every request class has.
FACTS_EVERYWHERE: frozenset[str] = frozenset({"agent6_version", "request_id"})
# The facts a model call adds; a listing or a credential request has none of them.
FACTS_INFERENCE: frozenset[str] = frozenset(
    {
        "has_images",
        "has_tool_results",
        "initiator",
        "is_first_turn",
        "is_subagent",
        "model",
        "session_id",
    }
)
# The request classes a literal string reaches; a credential request takes templates only.
API_TARGETS: frozenset[str] = frozenset({"inference", "models", "usage"})
# Headers the transport owns: the credential, the framing, the adapter's content negotiation.
PROTECTED_HEADERS: frozenset[str] = frozenset(
    {
        "accept",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "proxy-authorization",
        "transfer-encoding",
        "x-api-key",
    }
)
_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_FACT_NAME = re.compile(r"[a-z][a-z0-9_]*")
# A header value agent6 sends: visible ASCII and space; a control byte or a newline refuses.
VALUE_OK = re.compile(r"[\x20-\x7e]*")


@dataclasses.dataclass(frozen=True, slots=True)
class Segment:
    """One piece of a parsed template.

    Attributes:
        kind: `text` for literal text, `fact` for a placeholder.
        value: The text, or the fact's name.
    """

    kind: Literal["fact", "text"]
    value: str


def parse_template(template: str) -> tuple[Segment, ...]:
    """Split a template into literal text and fact names.

    `{name}` is a fact; `{{` and `}}` are literal braces.

    Args:
        template: The template text.

    Returns:
        The segments in order.

    Raises:
        ValueError: An unbalanced brace, or anything inside the braces but a plain name (an
            attribute, an index, a format specification, a conversion, an expression).
    """
    out: list[Segment] = []
    text: list[str] = []
    i = 0
    while i < len(template):
        ch = template[i]
        if ch == "{":
            if template.startswith("{{", i):
                text.append("{")
                i += 2
                continue
            end = template.find("}", i)
            if end < 0:
                raise ValueError(f"template {template!r} opens a brace it never closes")
            name = template[i + 1 : end]
            if not _FACT_NAME.fullmatch(name):
                raise ValueError(
                    f"template {template!r}: a placeholder holds a fact name and nothing"
                    " else (no attribute, index, format or expression)"
                )
            if text:
                out.append(Segment("text", "".join(text)))
                text = []
            out.append(Segment("fact", name))
            i = end + 1
            continue
        if ch == "}":
            if template.startswith("}}", i):
                text.append("}")
                i += 2
                continue
            raise ValueError(f"template {template!r} closes a brace it never opened")
        text.append(ch)
        i += 1
    if text:
        out.append(Segment("text", "".join(text)))
    return tuple(out)


class HeaderTemplate(pydantic.BaseModel):
    """A header rendered from request facts, for the request classes it names."""

    model_config = _base.MODEL_CONFIG

    template: str = pydantic.Field(
        description=(
            "literal text with `{fact}` placeholders (`{{` and `}}` are literal braces); the"
            " facts: `request_id`, `agent6_version` on every request, and on a model call"
            " `model`, `session_id`, `initiator`, `is_first_turn`, `is_subagent`,"
            " `has_tool_results`, `has_images`"
        ),
    )
    targets: tuple[HeaderTarget, ...] = pydantic.Field(
        default=("inference",),
        min_length=1,
        strict=False,  # the list-to-tuple conversion, as `_base.StrTuple`
        description=(
            "the request classes the header reaches: `inference` (model calls), `models` (the"
            " listing), `usage` (the plan reading), `oauth` (device, token and revocation"
            " requests), `exchange` (the configured second exchange)"
        ),
    )
    when: BooleanFact | None = pydantic.Field(
        default=None,
        description="a Boolean fact; the header is sent only while it is true (model calls only)",
    )

    @pydantic.model_validator(mode="after")
    def _facts_exist_for_every_target(self) -> HeaderTemplate:
        """Refuse a template its targets cannot render.

        Returns:
            The model unchanged.

        Raises:
            ValueError: A malformed template, an unknown fact, a model-call fact on a
                credential or listing target, a `when` off the inference target, or a
                repeated target.
        """
        facts = self.facts()
        unknown = sorted(facts - FACTS_EVERYWHERE - FACTS_INFERENCE)
        if unknown:
            raise ValueError(f"template names no such fact: {', '.join(unknown)}")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets repeats a request class")
        outside = sorted(t for t in self.targets if t != "inference")
        call_facts = sorted(facts & FACTS_INFERENCE)
        if outside and call_facts:
            raise ValueError(
                f"{', '.join(call_facts)}: a model-call fact; a template reaching"
                f" {', '.join(outside)} may use {', '.join(sorted(FACTS_EVERYWHERE))} only"
            )
        if self.when is not None and outside:
            raise ValueError("`when` gates on a model-call fact, so targets must be inference only")
        return self

    def facts(self) -> frozenset[str]:
        """Return the fact names the template renders.

        Returns:
            The names.

        Raises:
            ValueError: The template does not parse.
        """
        return frozenset(seg.value for seg in parse_template(self.template) if seg.kind == "fact")


def extra_headers_refusal(
    headers: Mapping[str, str | HeaderTemplate], *, reserved: frozenset[str] = frozenset()
) -> str:
    """Return why a header map cannot be sent, or "".

    Args:
        headers: The configured map.
        reserved: Headers the entry's profile owns beyond `PROTECTED_HEADERS`.

    Returns:
        The refusal: an invalid name, a protected or reserved header, two spellings of one
        header, or a literal value that cannot ride a header; "" when the map is acceptable.
    """
    seen: dict[str, str] = {}
    for name, value in headers.items():
        low = name.lower()
        if not _TOKEN.fullmatch(name):
            return f"extra_headers: {name!r} is not a valid header name"
        if low in PROTECTED_HEADERS or low in reserved:
            return f"extra_headers may not set {name}: the transport owns that header"
        if low in seen:
            return f"extra_headers names {seen[low]!r} and {name!r}, the same header twice"
        seen[low] = name
        if isinstance(value, str) and not VALUE_OK.fullmatch(value):
            return (
                f"extra_headers.{name}: the value holds a control character, newline or"
                " non-ASCII byte"
            )
    return ""
