"""Score the reviewer's first reply, before panel grounding or voting."""

from __future__ import annotations

import collections
import dataclasses
import re
import sys
from typing import Any

from agent6 import llm_json, review_records
from agent6.harness.review import verdict
from agent6.providers import provider_types

ARMS = ("recorded", "tree")
_VERDICTS = ("pass", "block")

# A lexical measure, not a judge of the claimed access. Keep the match for inspection.
_READER = re.compile(
    r"\b(?:reader|attacker|consumer):[ \t]*(?!(?:unknown|unspecified|none)\b)"
    r"(?:`[^`\n]+`|[\w][\w /.-]{0,60})"
    r"|\b(?:un(?:authenticated|privileged|trusted)|remote|local|another|other|downstream|API) "
    r"(?:clients?|users?|process(?:es)?|tenants?|workers?|callers?|readers?|attackers?)\b"
    r"|\b(?:operators?|administrators?|reviewer children)\b",
    re.IGNORECASE,
)


def reader_evidence(text: str) -> str | None:
    """Return the phrase naming a reader under the bench rule, if one matched."""
    match = _READER.search(text)
    return match[0].strip() if match else None


@dataclasses.dataclass(frozen=True, slots=True)
class Reply:
    """Hold a verdict and its blocks, or the reason the call has no verdict."""

    outcome: str
    blocks: tuple[review_records.Finding, ...] = ()
    tools_requested: tuple[str, ...] = ()
    error: str = ""

    def fields(self) -> dict[str, Any]:
        """Return the reply fields of the per-call JSON record."""
        blocks = []
        for block in self.blocks:
            evidence = reader_evidence(f"{block.title}\n{block.detail}")
            blocks.append(
                {
                    **block.model_dump(),
                    "names_reader": evidence is not None,
                    "reader_evidence": evidence,
                }
            )
        return {
            "outcome": self.outcome,
            "verdict": self.outcome if self.outcome in _VERDICTS else None,
            "tools_requested": self.tools_requested,
            "blocks": blocks,
            "error": self.error or None,
        }


def score(response: provider_types.ProviderResponse) -> Reply:
    """Classify a single reply without answering any tool request."""
    if response.tool_uses or response.refused or response.stop_reason in ("tool_use", "tool_calls"):
        return Reply(
            "asked for a tool", tools_requested=tuple(t["name"] for t in response.tool_uses)
        )
    if provider_types.output_cap_truncated(response):
        return Reply("truncated")
    obj = llm_json.extract_json(response.text, prefer=("verdict", "findings"))
    if obj is None:
        return Reply("invalid reply", error="no verdict JSON")
    parsed = verdict.verdict_from_obj(obj, 0, "", "")
    if parsed.error:
        return Reply("invalid reply", error=parsed.error)
    return Reply(parsed.verdict, tuple(f for f in parsed.findings if f.severity == "block"))


type Results = dict[str, list[dict[str, Reply]]]


def _counts(items: collections.Counter[str]) -> str:
    return ", ".join(f"{key}={n}" for key, n in items.most_common()) or "-"


def print_table(results: Results) -> None:
    """Print arm counts and compare blocks only within pairs with two verdicts."""
    print("review\tarm\toutcomes\tblock categories\tname a reader/blocks", file=sys.stderr)
    for session, pairs in results.items():
        for arm in ARMS:
            outcomes = collections.Counter(p[arm].outcome for p in pairs)
            blocks = [b for p in pairs for b in p[arm].blocks]
            categories = collections.Counter(sorted(b.category for b in blocks))
            named = sum(reader_evidence(f"{b.title}\n{b.detail}") is not None for b in blocks)
            print(
                session,
                arm,
                _counts(outcomes),
                _counts(categories),
                f"{named}/{len(blocks)}",
                sep="\t",
                file=sys.stderr,
            )

    print("review\tcompared pairs\tverdict changes\tblocks kept\tblocks lost", file=sys.stderr)
    comparable_reviews = changed_reviews = kept_blocks = lost_blocks = 0
    for session, pairs in results.items():
        compared = changes = kept = lost = 0
        for pair in pairs:
            old, new = pair["recorded"], pair["tree"]
            if old.outcome not in _VERDICTS or new.outcome not in _VERDICTS:
                continue
            compared += 1
            changes += old.outcome != new.outcome
            before, after = (
                collections.Counter((b.category, b.location) for b in r.blocks) for r in (old, new)
            )
            kept += (before & after).total()
            lost += (before - after).total()
        print(session, f"{compared}/{len(pairs)}", changes, kept, lost, sep="\t", file=sys.stderr)
        comparable_reviews += compared > 0
        changed_reviews += changes > 0
        kept_blocks += kept
        lost_blocks += lost
    print(
        f"Reviews whose verdict changed: {changed_reviews}/{comparable_reviews};"
        f" blocks kept: {kept_blocks}; blocks lost: {lost_blocks}",
        file=sys.stderr,
    )
