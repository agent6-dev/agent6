"""Ask for repeat IDs after a panel has decided its verdict."""

from __future__ import annotations

import json

import pydantic

from agent6 import budget, providers, review_records
from agent6.prompts import review


class _Edge(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)

    id: str
    repeat_of: str
    reason: str = ""


def merge(
    provider: providers.Provider, observations: tuple[review_records.FindingObservation, ...]
) -> review_records.RepeatMerge:
    """Return a panel's groups, or the reason its original findings stand."""
    by_id = {o.finding_id: o for o in observations}
    rows = [{"id": o.finding_id, **o.finding.model_dump(mode="json")} for o in observations]
    try:
        response = provider.call(
            system=review.REPEAT_PROMPT,
            messages=[{"role": "user", "content": json.dumps(rows, ensure_ascii=False)}],
            tools=None,
        )
    except (providers.ProviderError, budget.BudgetExceededError) as exc:
        return review_records.RepeatMerge(skipped=str(exc) or "reviewer call failed")
    try:
        if response.tool_uses:
            return review_records.RepeatMerge(skipped="unexpected tool call")
        edges = pydantic.TypeAdapter(list[_Edge]).validate_json(response.text)
        return _groups(by_id, edges)
    except pydantic.ValidationError:
        return review_records.RepeatMerge(skipped="invalid repeat list")
    except ValueError as exc:
        return review_records.RepeatMerge(skipped=str(exc))


def _groups(
    by_id: dict[str, review_records.FindingObservation], edges: list[_Edge]
) -> review_records.RepeatMerge:
    """Flatten repeat chains and count each reviewer once per kept finding.

    Args:
        by_id: The current panel's observations.
        edges: The repeat IDs proposed by the reviewer.

    Returns:
        Every group, singletons included.

    Raises:
        ValueError: An edge is invalid or the graph has a cycle.
    """
    parents: dict[str, str] = {}
    for edge in edges:
        if edge.id not in by_id or edge.repeat_of not in by_id or edge.id in parents:
            raise ValueError("unknown or repeated finding ID")
        if (
            by_id[edge.id].finding.location.path != by_id[edge.repeat_of].finding.location.path
            and not edge.reason.strip()
        ):
            raise ValueError("cross-file repeat without a reason")
        parents[edge.id] = edge.repeat_of
    groups: dict[str, list[str]] = {fid: [] for fid in by_id if fid not in parents}
    for fid in parents:
        root = fid
        seen: set[str] = set()
        while root in parents:
            if root in seen:
                raise ValueError("cycle in repeat IDs")
            seen.add(root)
            root = parents[root]
        groups[root].append(fid)
    return review_records.RepeatMerge(
        groups=tuple(
            review_records.RepeatGroup(
                kept=root,
                folded=tuple(folded),
                reviewers=len({i for fid in (root, *folded) for i in by_id[fid].reviewer_indexes}),
            )
            for root, folded in groups.items()
        )
    )
