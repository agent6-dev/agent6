# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structured compare judge over parallel-run candidates.

One LLM call ranks N candidate lane runs -- same task, independent diffs --
best first, with a rationale. Mirrors `workflows/_review.structured_review`'s
request/parse shape (strict JSON, tolerant of fences/prose), but unlike a
review seat's silent abstain, a compare needs one authoritative order: it
retries once on a malformed reply (unparseable JSON, a provider error, or a
ranking that doesn't name exactly the candidate run_ids) and raises
`JudgeError` on the second failure. `mechanical_ranking` is the
network-free fallback callers use when no reviewer model is configured or the
judge call raises.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent6.prompts.judge import JUDGE_SYSTEM_PROMPT
from agent6.providers import Provider, ProviderError


class JudgeError(Exception):
    """The compare judge could not produce a valid verdict."""


class CandidateBrief(BaseModel):
    """One candidate lane run shown to the judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    task: str
    diff: str
    verify_ok: bool | None
    cost_usd: float


class CompareVerdict(BaseModel):
    """The judge's ranking of candidates, best first, plus its rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ranking: tuple[str, ...]
    rationale: str


# Per-candidate diff cap in the judge prompt. Oversized diffs are truncated and
# marked (the prompt tells the judge to read every diff, so a silent cut would
# make the transcript and the judge dishonest about what was compared).
_DIFF_CAP = 60_000


def _build_user_message(candidates: list[CandidateBrief]) -> str:
    parts = [f"Comparing {len(candidates)} candidates for the same task."]
    for c in candidates:
        verify = "PASSED" if c.verify_ok else "FAILED" if c.verify_ok is False else "not run"
        diff = c.diff[:_DIFF_CAP]
        if len(c.diff) > _DIFF_CAP:
            diff += "\n[diff truncated]"
        parts.append(
            f"--- CANDIDATE {c.run_id} ---\n"
            f"TASK:\n{c.task.strip()[:4000]}\n"
            f"VERIFY: {verify}\n"
            f"COST: ${c.cost_usd:.4f}\n"
            f"DIFF:\n{diff}"
        )
    return "\n\n".join(parts)


def _balanced_objects(text: str) -> list[str]:
    """Every top-level balanced ``{...}`` span in *text* (brace depth, honoring
    string literals + escapes), tolerating prose/markdown fences around the
    real verdict object. Mirrors `_review._balanced_objects`."""
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
    """Parse the judge's reply as the verdict object: prefer the LAST balanced
    object carrying a ``ranking`` key, else the last parseable dict."""
    objs: list[dict[str, Any]] = []
    for span in _balanced_objects(text):
        try:
            obj = json.loads(span)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            objs.append(obj)
    for obj in reversed(objs):
        if "ranking" in obj:
            return obj
    return objs[-1] if objs else None


def _parse_verdict(obj: dict[str, Any], run_ids: set[str]) -> CompareVerdict | None:
    """None if ``ranking`` isn't a list of strings naming exactly `run_ids`."""
    ranking_raw = obj.get("ranking")
    if not isinstance(ranking_raw, list) or not all(isinstance(r, str) for r in ranking_raw):
        return None
    ranking = tuple(ranking_raw)
    if len(ranking) != len(run_ids) or set(ranking) != run_ids:
        return None
    rationale = str(obj.get("rationale", "")).strip()[:2000]
    return CompareVerdict(ranking=ranking, rationale=rationale)


def compare(
    provider: Provider, model: str, candidates: list[CandidateBrief], *, max_tokens: int = 1500
) -> CompareVerdict:
    """One structured call to *model* via *provider*: rank *candidates* best
    first with a rationale.

    Retries once on a failed attempt (provider error, unparseable JSON, or a
    ranking that doesn't name exactly the candidate run_ids); raises
    `JudgeError` on the second failure -- never a guessed order.
    """
    if not candidates:
        raise JudgeError("compare called with no candidates")
    run_ids = {c.run_id for c in candidates}
    user = _build_user_message(candidates)
    last_err = ""
    for _attempt in range(2):
        try:
            resp = provider.call(
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
        except ProviderError as exc:
            last_err = f"provider ({model}): {exc}"
            continue
        obj = _extract_json(resp.text)
        if obj is None:
            last_err = f"unparseable judge output ({model})"
            continue
        verdict = _parse_verdict(obj, run_ids)
        if verdict is None:
            last_err = f"judge ranking did not name exactly the candidate run_ids ({model})"
            continue
        return verdict
    raise JudgeError(last_err)


def mechanical_ranking(candidates: list[CandidateBrief]) -> tuple[str, ...]:
    """Deterministic fallback ranking: verify-pass first, then lower cost.
    Stable within ties (Python's sort is stable, so equal candidates keep
    their input order)."""
    ranked = sorted(candidates, key=lambda c: (c.verify_ok is not True, c.cost_usd))
    return tuple(c.run_id for c in ranked)


__all__ = [
    "CandidateBrief",
    "CompareVerdict",
    "JudgeError",
    "compare",
    "mechanical_ranking",
]
