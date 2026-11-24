"""Keep one panel's repeat decision out of the next review."""

from __future__ import annotations

import dataclasses
import json
from unittest import mock

import pytest

from agent6 import review_records
from agent6.harness.review import repeats
from tests import review_env


def observation(
    fid: str, title: str, index: int, path: str = "retry.py"
) -> review_records.FindingObservation:
    return review_records.FindingObservation(
        fid,
        "p",
        review_records.Finding(
            category="retry",
            severity="block",
            title=title,
            location=review_records.ReviewLocation(path=path, side="head"),
        ),
        (index,),
    )


_LOOP = observation("loop", "the retry loop never stops", 0)
_LOCK = observation("lock", "the retry sleeps while it holds the lock", 1)


def _merge(
    answer: str, observations: tuple[review_records.FindingObservation, ...]
) -> review_records.RepeatMerge:
    provider = mock.MagicMock()
    provider.call.return_value.text = answer
    result = repeats.merge(review_env.ResponseAdapter(provider), observations)
    provider.call.assert_called_once()
    args = provider.call.call_args.kwargs
    assert args["tools"] is None
    assert [row["id"] for row in json.loads(args["messages"][0]["content"])] == [
        o.finding_id for o in observations
    ]
    return result


def test_repeat_count_counts_reviewers_not_reports() -> None:
    result = _merge(
        '[{"id":"lock","repeat_of":"loop"}]',
        (_LOOP, _LOCK, dataclasses.replace(_LOCK, finding_id="another")),
    )
    assert [(g.kept, g.folded, g.reviewers) for g in result.groups] == [
        ("loop", ("lock",), 2),
        ("another", (), 1),
    ]


@pytest.mark.parametrize(
    "answer",
    [
        "not JSON",
        '[{"id":"unknown","repeat_of":"lock"}]',
        '[{"id":"lock","repeat_of":"loop"},{"id":"loop","repeat_of":"lock"}]',
        '[{"id":"lock","repeat_of":"loop"}]',
        '[{"id":"loop","repeat_of":"loop"}]',
        '[{"id":"loop","repeat_of":"lock","extra":2}]',
    ],
)
def test_invalid_repeat_answer_skips_the_whole_merge(answer: str) -> None:
    result = _merge(answer, (_LOOP, observation("lock", "lock held", 1, "other.py")))
    assert result.skipped and not result.groups
