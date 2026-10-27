from __future__ import annotations

from agent6 import evidence


def test_result_label_has_one_shape_per_surface() -> None:
    """Two surfaces rewrote "exit=" to "exit " by string replace; the label takes its joiner."""
    assert evidence.result_label(1, "failed") == "exit 1"
    assert evidence.result_label(1, "failed", joiner="=") == "exit=1"
    assert evidence.result_label(None, "denied", "not available") == "denied: not available"
