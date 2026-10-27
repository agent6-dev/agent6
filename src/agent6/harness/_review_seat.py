# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run a review seat and parse its verdict."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pydantic

from agent6 import review_records
from agent6.harness import _llm_json, _panel
from agent6.prompts import review
from agent6.providers import (
    Provider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    output_cap_truncated,
)
from agent6.tools import results

# A read-only dispatch callable for repo-scope seats: (tool_name, input) -> result.
ReviewDispatch = Callable[[str, dict[str, Any]], results.ToolResult]


def build_user_message(ctx: _panel.ReviewContext) -> str:
    """Return the task, verify result, prior findings and diff for one seat."""
    parts: list[str] = [f"TASK:\n{ctx.task.strip()[:4000]}"]
    parts.append(_panel.rules_message(ctx.rules))
    if ctx.verify_ok is None:
        parts.append("VERIFY: not run.")
    else:
        status = "PASSED" if ctx.verify_ok else "FAILED"
        out = ctx.verify_output.strip()[-2000:]
        parts.append(f"VERIFY: {status}\n{out}" if out else f"VERIFY: {status}")
    if ctx.prior_findings:
        already = "; ".join(f"{f.location} {f.category}" for f in ctx.prior_findings[:20])
        parts.append(f"ALREADY RAISED (do not repeat): {already}")
    parts.append(f"DIFF:\n{ctx.diff}")
    return "\n\n".join(parts)


def coerce_findings(raw: object) -> tuple[review_records.Finding, ...]:
    """Return findings with normalized categories, severities and single-line text."""
    out: list[review_records.Finding] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "other"))
        if category not in _panel.ALL_CATEGORIES:
            category = "other"
        severity = str(item.get("severity", "warn"))
        if severity not in ("block", "warn", "nit"):
            severity = "warn"
        out.append(
            review_records.Finding(
                category=category,
                severity=severity,  # type: ignore[arg-type]
                location=review_records.ReviewLocation.model_validate(item.get("location")),
                title=" ".join(str(item.get("title", "")).split())[:200],
                detail=" ".join(str(item.get("detail", "")).split())[:1000],
            )
        )
    return tuple(out)


def no_verdict_error(resp: ProviderResponse) -> str:
    """Return why a seat produced no verdict JSON: the output cap, no content, or junk.

    Args:
        resp: The seat's response.

    Returns:
        The error text for the abstaining verdict.
    """
    if output_cap_truncated(resp):
        detail = (
            "before emitting any content (likely all reasoning)"
            if not resp.text.strip()
            else "mid-answer, before the verdict JSON completed"
        )
        return (
            f"output hit the cap {detail}"
            f" (stop_reason={resp.stop_reason}, {resp.output_tokens} output tokens);"
            " raise max_tokens or use a model with more output headroom"
        )
    if not resp.text.strip():
        # The reasoning produced tells a starved reasoner from an upstream error.
        thought = sum(
            len(str(block.get("thinking") or ""))
            for block in (resp.raw.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "thinking"
        )
        channel = f", {thought:,} chars of it in the reasoning channel" if thought else ""
        return (
            f"the reviewer returned no content (stop_reason={resp.stop_reason},"
            f" {resp.output_tokens} output tokens{channel})"
        )
    return "unparseable reviewer output"


def structured_review(
    provider: Provider,
    ctx: _panel.ReviewContext,
    *,
    seat_index: int,
    focus: str,
    model_route: str,
    max_tokens: int = 1500,
    temperature: float | None = 0.0,
) -> review_records.ReviewVerdict:
    """Run one diff-scope seat.

    Args:
        provider: The seat's provider.
        ctx: The review context, with the seat's focus.
        seat_index: The position in the panel roster.
        focus: The seat's focus.
        model_route: The provider/model route, for the verdict.
        max_tokens: The output cap of the call.
        temperature: The sampling temperature.

    Returns:
        The seat's verdict; a provider error or junk output abstains with `error` set.
    """
    system = review.REVIEW_SYSTEM_PROMPT.format(focus=ctx.focus or "general correctness")
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": build_user_message(ctx)}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ProviderError as exc:
        return review_records.ReviewVerdict(
            seat_index=seat_index,
            focus=focus,
            model_route=model_route,
            verdict="abstain",
            error=f"provider: {exc}",
        )
    obj = _llm_json.extract_json(resp.text, prefer=("verdict", "findings"))
    if obj is None:
        return review_records.ReviewVerdict(
            seat_index=seat_index,
            focus=focus,
            model_route=model_route,
            verdict="abstain",
            error=no_verdict_error(resp),
        )
    return verdict_from_obj(obj, seat_index, focus, model_route)


def verdict_from_obj(
    obj: dict[str, Any], seat_index: int, focus: str, model_route: str
) -> review_records.ReviewVerdict:
    """Return the parsed verdict, or an abstention for an invalid reply."""
    raw_verdict = obj.get("verdict")
    if not isinstance(raw_verdict, str) or raw_verdict.lower() not in ("pass", "block"):
        return review_records.ReviewVerdict(
            seat_index=seat_index,
            focus=focus,
            model_route=model_route,
            verdict="abstain",
            error="invalid reviewer verdict",
        )
    try:
        reply = review_records.SeatReply(
            verdict="block" if raw_verdict.lower() == "block" else "pass",
            findings=coerce_findings(obj.get("findings")),
            summary=str(obj.get("summary", "")).strip()[:300],
        )
    except pydantic.ValidationError as exc:
        return review_records.ReviewVerdict(
            seat_index=seat_index,
            focus=focus,
            model_route=model_route,
            verdict="abstain",
            error=f"invalid reviewer findings: {exc}",
        )
    return review_records.ReviewVerdict(
        seat_index=seat_index,
        focus=focus,
        model_route=model_route,
        verdict=reply.verdict,
        findings=reply.findings,
        summary=reply.summary,
    )


def repo_review(
    provider: Provider,
    ctx: _panel.ReviewContext,
    *,
    seat_index: int,
    focus: str,
    model_route: str,
    tools: list[ToolDefinition],
    dispatch: ReviewDispatch,
    max_turns: int,
    max_tokens: int = 2000,
    temperature: float | None = 0.0,
) -> review_records.ReviewVerdict:
    """Run one repo-scope seat with at most `max_turns` provider calls.

    Args:
        provider: The seat's provider.
        ctx: The review context, with the seat's focus.
        seat_index: The position in the panel roster.
        focus: The seat's focus.
        model_route: The provider/model route, for the verdict.
        tools: The read-only tools offered.
        dispatch: The dispatch that refuses every other tool.
        max_turns: The provider calls allowed.
        max_tokens: The output cap of each call.
        temperature: The sampling temperature on every call.

    Returns:
        The seat's verdict; a provider error or no valid verdict abstains with `error` set.
    """
    system = review.REPO_REVIEW_SYSTEM_PROMPT.format(
        focus=ctx.focus or "general correctness", max_turns=max_turns
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": build_user_message(ctx)}]
    for i in range(max_turns):
        try:
            resp = provider.call(
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except ProviderError as exc:
            return review_records.ReviewVerdict(
                seat_index=seat_index,
                focus=focus,
                model_route=model_route,
                verdict="abstain",
                error=f"provider: {exc}",
            )
        messages.append({"role": "assistant", "content": resp.raw.get("content") or []})
        if not resp.tool_uses:
            obj = _llm_json.extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is None:
                return review_records.ReviewVerdict(
                    seat_index=seat_index,
                    focus=focus,
                    model_route=model_route,
                    verdict="abstain",
                    error=no_verdict_error(resp),
                )
            return verdict_from_obj(obj, seat_index, focus, model_route)
        # On the last turn a verdict beside tool calls counts; without one no tool runs.
        if i == max_turns - 1:
            obj = _llm_json.extract_json(resp.text, prefer=("verdict", "findings"))
            if obj is not None and ("verdict" in obj or "findings" in obj):
                return verdict_from_obj(obj, seat_index, focus, model_route)
            break
        tool_results: list[dict[str, Any]] = []
        for tu in resp.tool_uses:
            name = tu.get("name", "")
            tu_id = tu.get("id", "")
            try:
                out = dispatch(name, tu.get("input", {}) or {})
                content = json.dumps(out.to_wire(), ensure_ascii=False)[:8000]
            except Exception as exc:
                content = f"error: {exc}"[:2000]
            tool_results.append({"type": "tool_result", "tool_use_id": tu_id, "content": content})
        messages.append({"role": "user", "content": tool_results})
    return review_records.ReviewVerdict(
        seat_index=seat_index,
        focus=focus,
        model_route=model_route,
        verdict="abstain",
        error=f"repo: no verdict within {max_turns} calls",
    )
