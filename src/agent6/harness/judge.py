# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rank the candidate lanes of a parallel run.

One provider call ranks the lanes (same task, independent diffs) best first with a rationale.
A compare needs one authoritative order, so a malformed reply is retried once and the second
failure raises `JudgeError`; `mechanical_ranking` is the network-free fallback.
"""

from __future__ import annotations

from typing import Any

import pydantic

from agent6 import budget
from agent6.harness import _llm_json
from agent6.prompts import judge
from agent6.providers import Provider, ProviderError


class JudgeError(Exception):
    """The compare judge could not produce a valid verdict."""


class CandidateBrief(pydantic.BaseModel):
    """One candidate lane run shown to the judge."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    session_id: str
    task: str
    diff: str
    verify_ok: bool | None
    cost_usd: float


class CompareVerdict(pydantic.BaseModel):
    """The judge's ranking of candidates, best first, plus its rationale."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    ranking: tuple[str, ...]
    rationale: str


# Per-candidate diff cap in the judge prompt; a cut is marked, since the judge reads every diff.
_DIFF_CAP = 60_000


def _build_user_message(candidates: list[CandidateBrief]) -> str:
    parts = [f"Comparing {len(candidates)} candidates for the same task."]
    for c in candidates:
        verify = "PASSED" if c.verify_ok else "FAILED" if c.verify_ok is False else "not run"
        diff = c.diff[:_DIFF_CAP]
        if len(c.diff) > _DIFF_CAP:
            diff += "\n[diff truncated]"
        parts.append(
            f"--- CANDIDATE {c.session_id} ---\n"
            f"TASK:\n{c.task.strip()[:4000]}\n"
            f"VERIFY: {verify}\n"
            f"COST: ${c.cost_usd:.4f}\n"
            f"DIFF:\n{diff}"
        )
    return "\n\n".join(parts)


def _parse_verdict(obj: dict[str, Any], session_ids: set[str]) -> CompareVerdict | None:
    """Return the verdict, or None unless `ranking` names exactly `session_ids`."""
    ranking_raw = obj.get("ranking")
    if not isinstance(ranking_raw, list) or not all(isinstance(r, str) for r in ranking_raw):
        return None
    ranking = tuple(ranking_raw)
    if len(ranking) != len(session_ids) or set(ranking) != session_ids:
        return None
    rationale = str(obj.get("rationale", "")).strip()[:2000]
    return CompareVerdict(ranking=ranking, rationale=rationale)


def compare(
    provider: Provider, model: str, candidates: list[CandidateBrief], *, max_tokens: int = 1500
) -> CompareVerdict:
    """Rank the candidates best first with a rationale, in one structured call.

    A failed attempt (a provider error, a spent judge budget, unparseable JSON, a ranking that
    does not name exactly the candidate session_ids) is retried once.

    Args:
        provider: The judge's provider.
        model: The judge model, named in errors.
        candidates: The lanes to rank.
        max_tokens: The output cap of the call.

    Returns:
        The judge's verdict.

    Raises:
        JudgeError: On no candidates, or on the second failed attempt.
    """
    if not candidates:
        raise JudgeError("compare called with no candidates")
    session_ids = {c.session_id for c in candidates}
    user = _build_user_message(candidates)
    last_err = ""
    for _attempt in range(2):
        try:
            resp = provider.call(
                system=judge.JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
            )
        except ProviderError as exc:
            last_err = f"provider ({model}): {exc}"
            continue
        except budget.BudgetExceededError as exc:
            # A spent budget is a judge failure like any other: the caller degrades to mechanical.
            last_err = f"judge budget exhausted ({model}): {exc}"
            continue
        obj = _llm_json.extract_json(resp.text, prefer=("ranking",))
        if obj is None:
            last_err = f"unparseable judge output ({model})"
            continue
        verdict = _parse_verdict(obj, session_ids)
        if verdict is None:
            last_err = f"judge ranking did not name exactly the candidate session_ids ({model})"
            continue
        return verdict
    raise JudgeError(last_err)


def mechanical_ranking(candidates: list[CandidateBrief]) -> tuple[str, ...]:
    """Return the candidates ranked verify-pass first, then by lower cost, stable within ties."""
    ranked = sorted(candidates, key=lambda c: (c.verify_ok is not True, c.cost_usd))
    return tuple(c.session_id for c in ranked)


__all__ = [
    "CandidateBrief",
    "CompareVerdict",
    "JudgeError",
    "compare",
    "mechanical_ranking",
]
