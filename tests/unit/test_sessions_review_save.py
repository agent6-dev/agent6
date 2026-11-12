from __future__ import annotations

import pathlib
from unittest import mock

from agent6.app import review_recap


def test_two_reviews_in_one_second_keep_both(tmp_path: pathlib.Path) -> None:
    """Concurrent saves claim unique names without replacing an existing review."""
    first = review_recap.save_review(tmp_path, label="a", body="one")
    second = review_recap.save_review(tmp_path, label="b", body="two")
    assert first != second
    taken = tmp_path / "20260101T000000Z-review.md"
    taken.write_text("# review: theirs\n\nkept\n", encoding="utf-8")
    with mock.patch("agent6.app.review_recap.time.strftime", return_value="20260101T000000Z"):
        mine = review_recap.save_review(tmp_path, label="c", body="three")
    assert mine == tmp_path / "20260101T000000Z-2-review.md"
    assert taken.read_text(encoding="utf-8") == "# review: theirs\n\nkept\n"
    assert {p.read_text(encoding="utf-8") for p in (first, second)} == {
        "# review: a\n\none\n",
        "# review: b\n\ntwo\n",
    }
