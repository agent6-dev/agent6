# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a provider's `extra_headers` for one request, from the facts that request has.

The application supplies a `RequestContext` per model call; the adapter adds what its
serialized body holds; the transport mints a request id per attempt. A literal header
reaches the API classes only; a template reaches the classes it names. Every rendered
value is checked before it is sent.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agent6 import __version__
from agent6.config import _headers as header_config
from agent6.providers import types


@dataclasses.dataclass(frozen=True, slots=True)
class RequestContext:
    """What the application knows about a model call before it is serialized.

    Attributes:
        session_id: The opaque id of the conversation the call belongs to; "" names the
            initiator only, and the application fills the rest from its conversation state.
        is_first_turn: No call of this conversation has completed yet.
        is_subagent: The call belongs to a fan-out lane.
        initiator: `user` when the latest input is the operator's (the task, a steer, an
            answer), else `agent`.
        observe: Receives each HTTP attempt's facts; None observes nothing.
    """

    session_id: str
    is_first_turn: bool
    is_subagent: bool
    initiator: str
    observe: Callable[[RequestFacts], None] | None = None

    @classmethod
    def initiated_by(cls, initiator: str) -> RequestContext:
        """Return the partial context a caller states the initiator with.

        Args:
            initiator: `user` or `agent`.

        Returns:
            A context with no session id; the application fills it in.
        """
        return cls(session_id="", is_first_turn=False, is_subagent=False, initiator=initiator)


@dataclasses.dataclass(frozen=True, slots=True)
class RequestFacts:
    """The facts one HTTP request renders headers from.

    Attributes:
        target: The request class: inference, models, usage, oauth or exchange.
        request_id: The attempt's fresh id; "" until `attempt` mints one.
        model: The model the request names; "" off a model call.
        has_tool_results: The serialized request carries tool results.
        has_images: The serialized request carries image input.
        context: The application's context; None off a model call.
    """

    target: str
    request_id: str = ""
    model: str = ""
    has_tool_results: bool = False
    has_images: bool = False
    context: RequestContext | None = None

    def attempt(self) -> RequestFacts:
        """Return these facts with a fresh request id for one HTTP attempt."""
        return dataclasses.replace(self, request_id=uuid.uuid4().hex)

    def value(self, fact: str) -> str:
        """Return a fact as header text; a Boolean renders `true` or `false`.

        Args:
            fact: The fact's name.

        Returns:
            The text.

        Raises:
            KeyError: This request class lacks the fact.
        """
        if fact == "agent6_version":
            return __version__
        if fact == "request_id":
            return self.request_id
        if self.context is None or self.target != "inference":
            raise KeyError(fact)
        values = {
            "has_images": _flag(self.has_images),
            "has_tool_results": _flag(self.has_tool_results),
            "initiator": self.context.initiator,
            "is_first_turn": _flag(self.context.is_first_turn),
            "is_subagent": _flag(self.context.is_subagent),
            "model": self.model,
            "session_id": self.context.session_id,
        }
        return values[fact]


def _flag(value: bool) -> str:
    """Return a Boolean fact's header text."""
    return "true" if value else "false"


def call_facts(
    model: str, body_facts: tuple[bool, bool], context: RequestContext | None
) -> RequestFacts:
    """Return a model call's facts, its request id still to mint.

    Args:
        model: The model the request names.
        body_facts: (has_tool_results, has_images), read off the serialized body.
        context: The application's context; None outside a conversation.

    Returns:
        The facts.
    """
    has_tool_results, has_images = body_facts
    return RequestFacts(
        target="inference",
        model=model,
        has_tool_results=has_tool_results,
        has_images=has_images,
        context=context,
    )


def plain_facts(target: str) -> RequestFacts:
    """Return a listing or credential request's facts: a fresh request id and the version.

    Args:
        target: The request class.

    Returns:
        The facts.
    """
    return RequestFacts(target=target).attempt()


@dataclasses.dataclass(frozen=True, slots=True)
class HeaderSpec:
    """One compiled template header.

    Attributes:
        name: The header name, lowercased.
        segments: The parsed template.
        targets: The request classes it reaches.
        when: The Boolean fact gating it; None sends it always.
    """

    name: str
    segments: tuple[header_config.Segment, ...]
    targets: frozenset[str]
    when: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class HeaderSet:
    """A provider entry's extra headers, compiled once.

    Attributes:
        literal: The literal headers, names lowercased; they reach the API classes.
        templates: The template headers.
    """

    literal: tuple[tuple[str, str], ...] = ()
    templates: tuple[HeaderSpec, ...] = ()

    @classmethod
    def from_config(cls, headers: Mapping[str, str | header_config.HeaderTemplate]) -> HeaderSet:
        """Compile a validated `extra_headers` map.

        Args:
            headers: The configured map.

        Returns:
            The set, in the map's order.
        """
        literal: list[tuple[str, str]] = []
        templates: list[HeaderSpec] = []
        for name, value in headers.items():
            if isinstance(value, str):
                literal.append((name.lower(), value))
            else:
                templates.append(
                    HeaderSpec(
                        name=name.lower(),
                        segments=header_config.parse_template(value.template),
                        targets=frozenset(value.targets),
                        when=value.when,
                    )
                )
        return cls(literal=tuple(literal), templates=tuple(templates))

    def render(self, facts: RequestFacts) -> dict[str, str]:
        """Return the headers addressed to one request, rendered and checked.

        Args:
            facts: The request's facts, its request id minted.

        Returns:
            The headers, names lowercased; a header needing a fact this request lacks (a
            conversation fact on a call outside a conversation) is left out.

        Raises:
            ProviderError: A rendered value holds a control character or a newline.
        """
        out: dict[str, str] = {}
        if facts.target in header_config.API_TARGETS:
            out.update(self.literal)
        for spec in self.templates:
            if facts.target not in spec.targets:
                continue
            try:
                if spec.when is not None and facts.value(spec.when) != "true":
                    continue
                value = "".join(
                    seg.value if seg.kind == "text" else facts.value(seg.value)
                    for seg in spec.segments
                )
            except KeyError:
                continue
            if not header_config.VALUE_OK.fullmatch(value):
                raise types.ProviderError(
                    f"extra_headers.{spec.name} rendered a value that cannot ride an HTTP header"
                )
            out[spec.name] = value
        return out

    def renderer(self, target: str) -> Callable[[], dict[str, str]]:
        """Return a callable rendering this set for a credential request class per call.

        Args:
            target: `oauth` or `exchange`.

        Returns:
            The callable; each call mints a request id.
        """
        return lambda: self.render(plain_facts(target))


def _parts(content: Any) -> list[dict[str, Any]]:
    """Return a message's content as its list of typed parts; a string has none."""
    return [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []


def _has_type(parts: list[dict[str, Any]], names: frozenset[str]) -> bool:
    """Return whether any part, or a part nested in a tool result, has one of the types."""
    for part in parts:
        if part.get("type") in names:
            return True
        if part.get("type") == "tool_result" and _has_type(_parts(part.get("content")), names):
            return True
    return False


def anthropic_body_facts(body: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (has_tool_results, has_images) for a Messages API body."""
    messages = [m for m in body.get("messages", ()) if isinstance(m, dict)]
    parts = [part for m in messages for part in _parts(m.get("content"))]
    return _has_type(parts, frozenset({"tool_result"})), _has_type(parts, frozenset({"image"}))


def openai_body_facts(body: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (has_tool_results, has_images) for a Chat Completions body."""
    messages = [m for m in body.get("messages", ()) if isinstance(m, dict)]
    parts = [part for m in messages for part in _parts(m.get("content"))]
    tool = any(m.get("role") == "tool" for m in messages)
    return tool, _has_type(parts, frozenset({"image_url", "input_image"}))


def responses_body_facts(body: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (has_tool_results, has_images) for a Responses API body."""
    items = [item for item in body.get("input", ()) if isinstance(item, dict)]
    parts = [part for item in items for part in _parts(item.get("content"))]
    tool = any(item.get("type") == "function_call_output" for item in items)
    return tool, _has_type(parts, frozenset({"input_image"}))
