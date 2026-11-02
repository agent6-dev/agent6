from __future__ import annotations

import pytest

from agent6.ui.cli import main


def test_stop_help_names_fanout_wait_and_conditional_resume(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop help names the coordinator timeout and does not promise every run a resume point."""
    with pytest.raises(SystemExit) as caught:
        main(["stop", "--help"])
    assert caught.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "5 s (30 s for a fan-out)" in text
    assert "A run `agent6 resume` accepts stays resumable." in text
