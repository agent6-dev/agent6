# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Review-panel seat call + sequential orchestration.

A *seat* is one adversarial reviewer: a single grounded LLM call over the diff
(+ verify result) that returns a structured ``ReviewVerdict``. ``run_panel`` runs
the seats and folds them with the pure ``aggregate_verdicts`` (in ``_panel``).
The grounding that actually prevents false-blocks is enforced in the aggregator,
not here; this module just asks each model for findings in a parseable shape.

Network calls live here (each seat takes an injected ``Provider``); the pure
grounding/aggregation stays in ``_panel`` so it is testable without the network.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from agent6.providers import Provider, ProviderError, ToolDefinition
from agent6.workflows._panel import (
    ALL_CATEGORIES,
    Decision,
    Finding,
    PanelResult,
    ReviewContext,
    ReviewVerdict,
    aggregate_verdicts,
)

# A read-only dispatch callable for explore-tier seats: (tool_name, input) -> result.
ReviewDispatch = Callable[[str, dict[str, Any]], Any]

# Original wording (no third-party prompt text). Grounding is ALSO enforced
# mechanically downstream (aggregate_verdicts), so this prompt is guidance, not
# the safety boundary.
REVIEW_SYSTEM_PROMPT = """You are one reviewer on an adversarial code-review panel.
You are shown a DIFF the worker just produced, the task, and (if available) the
result of the project's verify/test command. Your assigned stance: {persona}.

If verify PASSED, the change is presumed correct. Raise a BLOCK only for a
concrete, test-independent defect you can NAME and CITE at a line in the diff:
  - security: an introduced vulnerability (injection, path traversal, secret
    leak, unsafe deserialization, weakened authn/authz)
  - sandbox-bypass: weakens or escapes the sandbox/jail
  - off-topic-edit: edits unrelated to the task, or deletion of unrelated code
  - data-loss: destroys user data or irreversibly drops state
  - verify-uncovered-correctness: a correctness bug the verify command provably
    does NOT exercise (only meaningful when verify passed)
Everything else -- style, naming, missing tests, "could be cleaner",
over-engineering, speculation -- is at most a "warn" or "nit", NEVER a block.

Rules:
  - Cite every finding at a `path:line` that appears in the DIFF. Uncited or
    out-of-diff findings are ignored by the aggregator.
  - Do not block on taste, and do not invent problems to look useful. If the
    diff is fine, return verdict "pass" with an empty findings list.

Categories: the five block-eligible ones above, or one of
test-gap / style / over-eng / other (these can only be warn/nit).

Output STRICT JSON and nothing else (no prose, no markdown fence):
{{"verdict": "pass" | "block",
  "summary": "<one line>",
  "findings": [
    {{"category": "<one of the categories listed above>",
      "severity": "block|warn|nit",
      "file_line": "path:line",
      "title": "<short>",
      "detail": "<why, terse>"}}
  ]}}"""


@dataclass(frozen=True, slots=True)
class Seat:
    """One panel seat: a persona stance bound to a provider/model.

    ``tier`` is "diff" (a single grounded call over the diff) or "explore" (a
    read-only tool-using mini-loop that investigates the broader repo first)."""

    persona: str
    model: str
    provider: Provider
    tier: str = "diff"


def parse_seat_spec(spec: str) -> tuple[str, str, str]:
    """Parse a ``review_seats`` entry into ``(persona, provider, model)``.

    ``"security@openrouter/moonshotai/kimi-k2"`` -> ``("security", "openrouter",
    "moonshotai/kimi-k2")``; ``"security"`` (no ``@``) -> ``("security", "", "")``
    (route via the reviewer role); ``"@anthropic/claude-opus-4-8"`` ->
    ``("", "anthropic", "claude-opus-4-8")``. The model may itself contain ``/``
    (only the first ``/`` after ``@`` splits provider from model)."""
    persona, sep, route = spec.partition("@")
    if not sep:
        return (spec.strip(), "", "")
    provider, _, model = route.partition("/")
    return (persona.strip(), provider.strip(), model.strip())


def _build_user_message(ctx: ReviewContext) -> str:
    parts: list[str] = [f"TASK:\n{ctx.task.strip()[:4000]}"]
    if ctx.agents_md.strip():
        parts.append(f"AGENTS.md:\n{ctx.agents_md.strip()[:8000]}")
    if ctx.verify_ok is None:
        parts.append("VERIFY: none configured for this run.")
    else:
        status = "PASSED" if ctx.verify_ok else "FAILED"
        out = ctx.verify_output.strip()[-2000:]
        parts.append(f"VERIFY: {status}\n{out}" if out else f"VERIFY: {status}")
    if ctx.prior_findings:
        already = "; ".join(f"{f.file_line} {f.category}" for f in ctx.prior_findings[:20])
        parts.append(f"ALREADY RAISED (do not repeat): {already}")
    parts.append(f"DIFF:\n{ctx.diff[:60_000]}")
    return "\n\n".join(parts)


def _balanced_objects(text: str) -> list[str]:
    """Every TOP-LEVEL balanced ``{...}`` span in *text* (brace depth, honoring
    string literals + escapes), so prose / markdown fences / a stray pre-amble
    object don't truncate or mis-capture the real verdict."""
    spans: list[str] = []
    depth = 0
    start: int | None = None
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
    return spans


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse the reviewer's reply as the verdict object, tolerating fences/prose
    and a stray object before the real one: prefer the LAST balanced object that
    carries a ``verdict``/``findings`` key, else the last parseable dict."""
    objs: list[dict[str, Any]] = []
    for span in _balanced_objects(text):
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    for obj in reversed(objs):
        if "verdict" in obj or "findings" in obj:
            return obj
    return objs[-1] if objs else None


def _coerce_findings(raw: object) -> tuple[Finding, ...]:
    out: list[Finding] = []
    if not isinstance(raw, list):
        return ()
    for item in raw:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "other"))
        if category not in ALL_CATEGORIES:
            category = "other"
        severity = str(item.get("severity", "warn"))
        if severity not in ("block", "warn", "nit"):
            severity = "warn"
        out.append(
            Finding(
                category=category,
                severity=severity,  # type: ignore[arg-type]
                file_line=str(item.get("file_line", "")).strip(),
                title=str(item.get("title", "")).strip()[:200],
                detail=str(item.get("detail", "")).strip()[:1000],
            )
        )
    return tuple(out)


def structured_review(
    provider: Provider, ctx: ReviewContext, *, seat: str, model: str, max_tokens: int = 1500
) -> ReviewVerdict:
    """Run one seat. Returns a ReviewVerdict; any failure (provider error, junk
    output) yields an ABSTAINING verdict (``error`` set) -- never a false pass."""
    system = REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": _build_user_message(ctx)}],
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        return ReviewVerdict(seat=seat, model=model, verdict="pass", error=f"provider: {exc}")
    obj = _extract_json(resp.text)
    if obj is None:
        return ReviewVerdict(
            seat=seat, model=model, verdict="pass", error="unparseable reviewer output"
        )
    return _verdict_from_obj(obj, seat, model)


def _verdict_from_obj(obj: dict[str, Any], seat: str, model: str) -> ReviewVerdict:
    findings = _coerce_findings(obj.get("findings"))
    verdict = "block" if str(obj.get("verdict", "")).lower() == "block" else "pass"
    return ReviewVerdict(
        seat=seat,
        model=model,
        verdict=verdict,
        findings=findings,
        summary=str(obj.get("summary", "")).strip()[:300],
    )


EXPLORE_REVIEW_SYSTEM_PROMPT = (
    REVIEW_SYSTEM_PROMPT
    + """

You ALSO have read-only tools (read_file, grep, outline, list_dir,
find_definition, find_references) to INVESTIGATE the broader repo before judging.
When the diff changes a function/class signature, public API, return type, or a
shared constant, USE find_references / grep to find existing callers/usages and
check they still work.

A diff that BREAKS an existing caller or usage you find elsewhere (e.g. it
changed `f(x)` to `f(x, y)` but `f(a)` is still called in another file) is a
real `verify-uncovered-correctness` defect of THIS diff -- the verify command
passed only because it didn't exercise that path. Report it as a BLOCK, but cite
it at the `path:line` IN THE DIFF that caused the break (the changed signature),
and name the broken caller (file:line) in the `detail`. Do NOT cite the finding
at the other file's line -- only diff lines gate.

Investigate first; when done, reply with ONLY the JSON verdict and no tool calls."""
)


def explore_review(
    provider: Provider,
    ctx: ReviewContext,
    *,
    seat: str,
    model: str,
    tools: list[ToolDefinition],
    dispatch: ReviewDispatch,
    max_iters: int = 6,
    max_tokens: int = 2000,
    deadline_s: float = 90.0,
) -> ReviewVerdict:
    """A read-only tool-using reviewer: a bounded mini-loop where the seat may
    call read-only tools to investigate the repo, then emits a ReviewVerdict.
    Tools are an explicit read-only allowlist enforced by the caller's dispatch;
    any failure (provider error, deadline, no verdict within max_iters) ABSTAINS."""
    system = EXPLORE_REVIEW_SYSTEM_PROMPT.format(persona=ctx.persona or "general correctness")
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_message(ctx)}]
    start = time.monotonic()
    for i in range(max_iters):
        if time.monotonic() - start > deadline_s:
            return ReviewVerdict(
                seat=seat, model=model, verdict="pass", error="explore: deadline exceeded"
            )
        try:
            resp = provider.call(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except ProviderError as exc:
            return ReviewVerdict(seat=seat, model=model, verdict="pass", error=f"provider: {exc}")
        messages.append({"role": "assistant", "content": resp.raw.get("content") or []})
        if not resp.tool_uses:
            obj = _extract_json(resp.text)
            if obj is None:
                return ReviewVerdict(
                    seat=seat, model=model, verdict="pass", error="unparseable reviewer output"
                )
            return _verdict_from_obj(obj, seat, model)
        # On the last allowed iteration, a verdict emitted ALONGSIDE tool calls
        # still counts (don't waste the investigation by abstaining). With no
        # verdict, skip the dispatches: no model call follows to consume their
        # results, so executing them only spends tool time on an abstention.
        if i == max_iters - 1:
            obj = _extract_json(resp.text)
            if obj is not None and ("verdict" in obj or "findings" in obj):
                return _verdict_from_obj(obj, seat, model)
            break
        tool_results: list[dict[str, Any]] = []
        for tu in resp.tool_uses:
            name = tu.get("name", "")
            tu_id = tu.get("id", "")
            try:
                out = dispatch(name, tu.get("input", {}) or {})
                content = json.dumps(out, ensure_ascii=False)[:8000]
            except Exception as exc:
                content = f"error: {exc}"[:2000]
            tool_results.append({"type": "tool_result", "tool_use_id": tu_id, "content": content})
        messages.append({"role": "user", "content": tool_results})
    return ReviewVerdict(
        seat=seat, model=model, verdict="pass", error="explore: no verdict within max_iters"
    )


def run_panel(
    seats: list[Seat],
    ctx: ReviewContext,
    *,
    decision: Decision,
    quorum: int,
    panel_id: str,
    concurrency: int = 1,
    tools: list[ToolDefinition] | None = None,
    dispatch: ReviewDispatch | None = None,
) -> PanelResult:
    """Run every seat and aggregate. Each seat sees the same context with its own
    persona substituted. With ``concurrency > 1`` the seat calls run on a thread
    pool (the shared budget tracker + transcript sink are both lock-protected, and
    each seat has its own provider); results stay in seat order, so the merged
    verdict is deterministic regardless of how the calls interleave."""

    def _run(s: Seat) -> ReviewVerdict:
        seat_ctx = replace(ctx, persona=s.persona)
        if s.tier == "explore" and tools is not None and dispatch is not None:
            return explore_review(
                s.provider, seat_ctx, seat=s.persona, model=s.model, tools=tools, dispatch=dispatch
            )
        return structured_review(s.provider, seat_ctx, seat=s.persona, model=s.model)

    if concurrency > 1 and len(seats) > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(seats))) as pool:
            verdicts = list(pool.map(_run, seats))  # map preserves input order
    else:
        verdicts = [_run(s) for s in seats]
    return aggregate_verdicts(verdicts, ctx, decision=decision, quorum=quorum, panel_id=panel_id)


__all__ = [
    "REVIEW_SYSTEM_PROMPT",
    "ReviewDispatch",
    "Seat",
    "explore_review",
    "parse_seat_spec",
    "run_panel",
    "structured_review",
]
