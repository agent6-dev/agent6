from __future__ import annotations

import pytest

from agent6.ui.cli import main


def test_stop_help_names_the_lanes_wait_and_conditional_resume(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop help names the wait for a run's lanes and promises no run a resume point it lacks."""
    with pytest.raises(SystemExit) as caught:
        main(["stop", "--help"])
    assert caught.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "5 s (30 s while it drains lanes)" in text
    assert "A run `agent6 resume` accepts stays resumable." in text
