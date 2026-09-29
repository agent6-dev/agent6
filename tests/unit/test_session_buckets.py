# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One bucket per mode, named after it, under one `sessions/` root."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds, paths
from agent6.sessions import layout as sessions_layout


def test_a_bucket_is_the_mode_plus_s() -> None:
    """The bucket is derived rather than stored.

    A record cannot disagree with where its sessions actually go.
    """
    assert [kinds.session_bucket(name) for name in ("run", "plan", "ask", "machine")] == [
        "runs",
        "plans",
        "asks",
        "machines",
    ]


def test_an_unknown_mode_has_no_bucket() -> None:
    with pytest.raises(kinds.UnknownSessionKindError):
        kinds.session_bucket("nonsense")


def test_an_agent_execution_has_no_sessions_bucket() -> None:
    """A machine's agent states live inside their machine instance's own directory.

    Answering "agents" here minted a bucket nothing writes, so a misrouted session landed somewhere
    no listing scans instead of failing loudly at the routing bug.
    """
    with pytest.raises(kinds.UnknownSessionKindError):
        kinds.session_bucket("agent")


def test_every_mode_has_a_scanned_bucket() -> None:
    """A mode whose bucket no listing scans writes a session dir nothing can find."""
    for name in kinds.SESSION_KINDS:
        if name == "agent":
            continue  # no directory of its own (see the refusal test)
        assert kinds.session_bucket(name) in sessions_layout.SESSION_BUCKETS


def test_session_dirs_live_under_the_sessions_root() -> None:
    """Session dirs live under the sessions root.

    Nesting the buckets frees `machines/` at the top level for live machine instances, so the
    authoring sessions can be named for their mode.
    """
    layout = sessions_layout.SessionLayout(
        state_dir=pathlib.Path("/s"), session_id="brave-oak-AAAAAA", subdir="machines"
    )
    assert (
        layout.session_dir
        == pathlib.Path("/s") / sessions_layout.SESSIONS_ROOT / "machines" / "brave-oak-AAAAAA"
    )
    assert layout.session_dir != pathlib.Path("/s") / "machines" / "brave-oak-AAAAAA"


def test_bucket_dir_is_the_one_owner_of_that_arithmetic() -> None:
    assert (
        sessions_layout.bucket_dir(pathlib.Path("/s"), "runs")
        == pathlib.Path("/s") / sessions_layout.SESSIONS_ROOT / "runs"
    )


def test_hub_buckets_are_session_buckets_without_the_machine_ones() -> None:
    """A hub lists ordinary sessions; machine authoring gets its own card."""
    assert set(sessions_layout.HUB_BUCKETS) < set(sessions_layout.SESSION_BUCKETS)
    assert set(sessions_layout.SESSION_BUCKETS) - set(sessions_layout.HUB_BUCKETS) == {"machines"}


@pytest.mark.parametrize("bucket", ["runs", "plans", "asks"])
def test_bare_resume_finds_the_newest_session_in_every_resumable_bucket(
    tmp_path: pathlib.Path, bucket: str
) -> None:
    """Bare `resume` finds the newest session in every resumable bucket.

    Splitting plans/ out of runs/ must not hide a plan from bare `resume`; a machine draft is
    deliberately absent, since `machine` is not resumable.
    """
    from agent6.app import resume
    from agent6.viewmodel import newest_session_dir

    session = sessions_layout.bucket_dir(tmp_path, bucket) / "brave-oak-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text('{"type": "session.start"}\n', encoding="utf-8")
    (sessions_layout.bucket_dir(tmp_path, "machines") / "quiet-fox-BBBBBB").mkdir(parents=True)
    (
        sessions_layout.bucket_dir(tmp_path, "machines") / "quiet-fox-BBBBBB" / "logs.jsonl"
    ).write_text("{}\n", encoding="utf-8")

    found = newest_session_dir(resume.resumable_bucket_dirs(tmp_path))
    assert found == session


@pytest.mark.parametrize("bucket", sessions_layout.HUB_BUCKETS)
def test_every_hub_lists_every_hub_bucket(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bucket: str,
) -> None:
    """A bucket a hub does not scan is a session the operator cannot see.

    Each surface carried its own `("runs", "asks")` tuple, so adding plans/ left `agent6 sessions`
    printing "no sessions yet" over a real plan.
    """
    from agent6.ui.cli import main
    from agent6.ui.web import model as web_model
    from agent6.viewmodel import session_dirs

    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
    session = sessions_layout.bucket_dir(state, bucket) / "brave-oak-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "run", "user_task": "t"}\n', encoding="utf-8"
    )

    assert main(["sessions", "list"]) == 0
    assert "brave-oak-AAAAAA" in capsys.readouterr().out
    assert [p.name for p in session_dirs(state)] == ["brave-oak-AAAAAA"]
    hub_ids = [s["session_id"] for s in web_model.hub_payload(tmp_path)["sessions"]]
    assert hub_ids == ["brave-oak-AAAAAA"]


def test_a_machine_draft_does_not_collide_with_a_machine_instance(tmp_path: pathlib.Path) -> None:
    """The reason for the nesting.

    `machine create` authoring sessions are named for their mode without landing in the directory
    holding live instances.
    """
    draft = sessions_layout.bucket_dir(tmp_path, kinds.session_bucket("machine")) / "same-name"
    instance = tmp_path / "machines" / "same-name"
    draft.mkdir(parents=True)
    instance.mkdir(parents=True)
    assert draft != instance


def test_a_machine_instance_is_not_reachable_as_a_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`machines` names two things, and only the path separates them.

    A session lookup that reached the INSTANCES dir would let `sessions rm` delete a running
    machine's state, so the buckets must never resolve there.
    """
    from agent6.ui.cli import _common

    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
    (state / "machines" / "tiny").mkdir(parents=True)

    with pytest.raises(Exception, match="no session matches"):
        _common.resolve_session_layout(tmp_path, "tiny")


def test_a_machine_draft_is_reachable_as_a_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converse: an authoring draft IS a session, so an id resolves to it."""
    from agent6.ui.cli import _common

    monkeypatch.chdir(tmp_path)
    state = paths.state_dir(tmp_path)
    (sessions_layout.bucket_dir(state, "machines") / "brave-oak-AAAAAA").mkdir(parents=True)
    (sessions_layout.bucket_dir(state, "machines") / "brave-oak-AAAAAA" / "logs.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    layout = _common.resolve_session_layout(tmp_path, "brave-oak-AAAAAA")
    assert layout.subdir == "machines"
    assert layout.session_dir == sessions_layout.bucket_dir(state, "machines") / "brave-oak-AAAAAA"
