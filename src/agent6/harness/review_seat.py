# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prepare review messages and parse bound lane completions."""

from __future__ import annotations

import dataclasses
from typing import Any

import pydantic

from agent6 import review_records
from agent6.harness import llm_json, review_panel
from agent6.sessions import completion
from agent6.vcs import git_files


def build_user_message(ctx: review_panel.ReviewContext) -> str:
    """Return the task, verify result, prior findings and diff for one seat."""
    parts: list[str] = [f"TASK:\n{ctx.task.strip()[:4000]}"]
    parts.append(review_panel.rules_message(ctx.rules))
    if ctx.verification is None:
        parts.append("VERIFY: not run.")
    else:
        status = ctx.verification.upper()
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
        if category not in review_panel.ALL_CATEGORIES:
            category = "other"
        severity = str(item.get("severity", "warn"))
        if severity not in ("block", "warn", "nit"):
            severity = "warn"
        location = item.get("location")
        if isinstance(location, dict) and isinstance(location.get("path"), str):
            # A seat may cite the path as the diff prints it, in git's quoting; a name that
            # merely starts and ends with a quote holds no escape and stays as cited.
            cited = location["path"]
            if "\\" in cited:
                location = {**location, "path": git_files.unquote_git_path(cited)}
        out.append(
            review_records.Finding(
                category=category,
                severity=severity,  # type: ignore[arg-type]
                location=review_records.ReviewLocation.model_validate(location),
                title=" ".join(str(item.get("title", "")).split())[:200],
                detail=" ".join(str(item.get("detail", "")).split())[:1000],
            )
        )
    return tuple(out)


def no_verdict_error(resp: completion.CompletionRecord) -> str:
    """Return why a seat produced no verdict JSON: the output cap, no content, or junk.

    Args:
        resp: The seat's response.

    Returns:
        The error text for the abstaining verdict.
    """
    if resp.response_stop_reason.strip().lower() in ("length", "max_tokens"):
        detail = (
            "before emitting any content (likely all reasoning)"
            if not resp.text.strip()
            else "mid-answer, before the verdict JSON completed"
        )
        return (
            f"output hit the cap {detail}"
            f" (stop_reason={resp.response_stop_reason},"
            f" {resp.response_output_tokens} output tokens);"
            " raise max_tokens or use a model with more output headroom"
        )
    if not resp.text.strip():
        # The reasoning produced tells a starved reasoner from an upstream error.
        thought = resp.reasoning_chars
        channel = f", {thought:,} chars of it in the reasoning channel" if thought else ""
        return (
            f"the reviewer returned no content (stop_reason={resp.response_stop_reason},"
            f" {resp.response_output_tokens} output tokens{channel})"
        )
    return "unparseable reviewer output"


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
    risk = obj.get("risk")
    risk = risk if isinstance(risk, str) else ""
    try:
        reply = review_records.SeatReply(
            verdict="block" if raw_verdict.lower() == "block" else "pass",
            findings=coerce_findings(obj.get("findings")),
            summary=str(obj.get("summary", "")).strip()[:300],
            risk=risk[:1000],
            risk_truncated=len(risk) > 1000,
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
        risk=reply.risk,
        risk_truncated=reply.risk_truncated,
    )


def verdict_from_completion(
    record: completion.CompletionRecord | None,
    *,
    binding: review_records.ReviewBinding,
    session_id: str,
    focus: str,
    model_route: str,
    scope: str,
    max_calls: int,
    error: str | None = None,
) -> review_records.ReviewVerdict:
    """Validate provenance before parsing any model text.

    Args:
        record: The imported bounded-conversation artifact, never a worker answer.
        binding: The parent's panel, seat and target binding.
        session_id: The child assigned by the parent dispatcher.
        focus: The parent's seat focus.
        model_route: The parent's resolved route.
        scope: The seat's diff or repo scope.
        max_calls: The request's call ceiling.
        error: A dispatch, timeout or artifact-read failure.

    Returns:
        A verdict or an abstention carrying the actual failure diagnostic.
    """
    if error is None:
        if record is None:
            error = "missing completion artifact"
        elif record.review_binding != binding or record.session_id != session_id:
            error = "completion binding does not match the parent panel, seat, target or session"
        elif record.calls > max_calls:
            error = "completion exceeded its bound call allowance"
        elif record.stop_reason not in ("answered", "calls_exhausted"):
            error = "; ".join(record.diagnostics) or record.text or record.stop_reason
    if error is not None:
        return review_records.ReviewVerdict(
            binding.seat_index, focus, model_route, "abstain", error=error, session_id=session_id
        )
    assert record is not None
    obj = llm_json.extract_json(record.text, prefer=("verdict", "findings"))
    if obj is None or (
        scope == "repo" and record.tool_uses and not ({"verdict", "findings"} & obj.keys())
    ):
        error = (
            f"repo: no verdict within {max_calls} calls"
            if scope == "repo" and record.tool_uses
            else no_verdict_error(record)
        )
        return review_records.ReviewVerdict(
            binding.seat_index, focus, model_route, "abstain", error=error, session_id=session_id
        )
    return dataclasses.replace(
        verdict_from_obj(obj, binding.seat_index, focus, model_route), session_id=session_id
    )
