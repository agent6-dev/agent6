# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One bucket per mode, and every list of buckets agreeing with the record."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.sessions.layout import HUB_BUCKETS, SESSION_BUCKETS
from agent6.types import SESSION_KINDS, session_kind


def test_every_mode_bucket_is_a_known_bucket() -> None:
    """A mode whose bucket no listing scans is invisible: it writes a session dir
    nothing can find. `agent` alone has none -- its state lives inside a machine
    instance."""
    for kind in SESSION_KINDS.values():
        if kind.bucket is None:
            assert kind.name == "agent", f"{kind.name} silently has no bucket"
            continue
        assert kind.bucket in SESSION_BUCKETS, f"{kind.name} writes to an unlisted bucket"


def test_a_plan_does_not_live_in_the_runs_bucket() -> None:
    """Bucket == mode. A plan sharing runs/ was the one session whose directory
    disagreed with its own manifest."""
    assert session_kind("plan").bucket == "plans"
    assert session_kind("run").bucket == "runs"
    assert session_kind("ask").bucket == "asks"


def test_machine_authoring_names_the_bucket_it_actually_writes() -> None:
    """`machine create` writes machine-drafts/; the record said runs/. It never
    broke only because machine sessions do not reach the run lifecycle."""
    assert session_kind("machine").bucket == "machine-drafts"


def test_hub_buckets_are_session_buckets_without_the_drafts() -> None:
    assert set(HUB_BUCKETS) < set(SESSION_BUCKETS)
    assert set(SESSION_BUCKETS) - set(HUB_BUCKETS) == {"machine-drafts"}


@pytest.mark.parametrize("bucket", ["runs", "plans", "asks"])
def test_bare_resume_finds_the_newest_session_in_every_resumable_bucket(
    tmp_path: Path, bucket: str
) -> None:
    """Splitting plans/ out of runs/ must not hide a plan from bare `resume`:
    before the split the newest-run scan saw plans because they shared runs/.
    A machine draft is deliberately absent -- `machine` is not resumable."""
    from agent6.app.resume import resumable_bucket_dirs
    from agent6.viewmodel import newest_session_dir

    session = tmp_path / bucket / "brave-oak-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text('{"type": "session.start"}\n', encoding="utf-8")
    (tmp_path / "machine-drafts" / "quiet-fox-BBBBBB").mkdir(parents=True)

    found = newest_session_dir(resumable_bucket_dirs(tmp_path))
    assert found == session


@pytest.mark.parametrize("bucket", HUB_BUCKETS)
def test_every_hub_lists_every_hub_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bucket: str,
) -> None:
    """A bucket a hub does not scan is a session the operator cannot see. Each
    surface carried its own `("runs", "asks")` tuple, so adding plans/ left
    `agent6 sessions` printing "no sessions yet" over a real plan."""
    from agent6.config.layer import resolved_state_dir
    from agent6.ui.cli import main
    from agent6.ui.tui.home import _list_sessions as tui_list  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.web import model as web_model

    monkeypatch.chdir(tmp_path)
    state = resolved_state_dir(tmp_path)
    session = state / bucket / "brave-oak-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "run", "user_task": "t"}\n', encoding="utf-8"
    )

    assert main(["sessions", "list"]) == 0
    assert "brave-oak-AAAAAA" in capsys.readouterr().out
    assert [p.name for p in tui_list(state)] == ["brave-oak-AAAAAA"]
    assert [s["id"] for s in web_model.hub_payload(tmp_path)["sessions"]] == ["brave-oak-AAAAAA"]
