from __future__ import annotations

import pytest

from agent6 import directive


@pytest.mark.parametrize("target", ["", "feature-x", "A..B", "A...B", "SHA^..SHA"])
def test_review_keeps_the_target_as_typed(target: str) -> None:
    """The argument is the reviewed side, not a base or a lane specification."""
    assert directive.parse_review(f"/review {target}") == target
    assert directive.steer_problem(f"/review {target}") is None


@pytest.mark.parametrize("text", ["please /review main", "/reviewer main", "ordinary text"])
def test_review_requires_a_leading_exact_token(text: str) -> None:
    """Mentions and prefix lookalikes remain ordinary text."""
    assert directive.parse_review(text) is None


def test_review_accepts_case_and_outer_whitespace() -> None:
    """Directive spelling is case-insensitive; ref spelling is not."""
    assert directive.parse_review("  /Review Feature-X \n") == "Feature-X"


@pytest.mark.parametrize("target", ["--stat", "main HEAD", "A....B", "A..B..C"])
def test_review_refuses_flags_and_multiple_targets(target: str) -> None:
    """Malformed review syntax refuses before an execution starts."""
    text = f"/review {target}"
    with pytest.raises(directive.DirectiveError, match="/review accepts"):
        directive.parse_review(text)
    assert "/review accepts" in (directive.steer_problem(text) or "")
