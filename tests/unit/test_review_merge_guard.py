from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
import threading
from typing import Any

import pytest

from agent6 import event_log, review_records
from agent6.app import finalize, merge, reporter, review_actions
from agent6.config import Config
from agent6.sessions import layout, lock, manifest
from agent6.ui import spawn
from agent6.ui.cli import main
from agent6.ui.tui import home
from agent6.ui.web import server
from agent6.vcs import git_ops, git_read, review
from tests import review_helpers
from tests.web import signin


def _git(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def run(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> layout.SessionLayout:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        spawn, "agent6_exe", lambda: str(pathlib.Path(sys.executable).with_name("agent6"))
    )
    _git(tmp_path, "init", "-qb", "main")
    (tmp_path / "parser.py").write_text("base\ncontext\nolder defect\nlast\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    base = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "switch", "-qc", "agent6/review-run")
    (tmp_path / "parser.py").write_text("changed\ncontext\nolder defect\nlast\n")
    _git(tmp_path, "commit", "-qam", "work")
    _git(tmp_path, "switch", "-q", "main")
    saved = review_helpers.reviewed_session(tmp_path)
    head = _git(tmp_path, "rev-parse", "agent6/review-run")
    target = review_records.ReviewTarget(
        base,
        git_read.tree_sha(tmp_path, base),
        head,
        git_read.tree_sha(tmp_path, head),
        head,
        (),
        "revisions",
    )
    panel = review_records.ReviewPanel.model_validate(
        event_log.payload(json.loads(saved.logs_path.read_text()))
    )
    panel = panel.model_copy(
        update={
            "target": target,
            "findings": tuple(
                dataclasses.replace(
                    f,
                    line_origin=review.classify_line_origin(
                        tmp_path, target, f.finding.location, hide=()
                    ),
                )
                for f in panel.findings
            ),
        }
    )
    saved.logs_path.write_text(
        json.dumps({"type": "loop.review.panel", **panel.model_dump(mode="json")}) + "\n"
    )
    saved.manifest_path.write_text(
        manifest.SessionManifest(
            session_id=saved.session_id,
            base_sha=base,
            base_branch="main",
            run_branch="agent6/review-run",
            user_task="Repair parser",
        ).model_dump_json()
    )
    return saved


def _merge(cwd: pathlib.Path, saved: layout.SessionLayout, **kwargs: Any) -> merge.MergeOutcome:
    record = manifest.read_manifest(saved.session_dir)
    return merge.execute_merge(
        cwd,
        layout=saved,
        manifest=record,
        run_branch="agent6/review-run",
        target="main",
        base_sha=record.base_sha,
        method="ff_only",
        message=None,
        cfg=Config(),
        identity=git_read.CommitIdentity(),
        **kwargs,
    )


def _resolve_all(saved: layout.SessionLayout) -> None:
    for finding_id in ("f1", "f2"):
        review_actions.act(
            saved.session_dir,
            finding_id,
            review_records.FindingAction(
                request_id=finding_id,
                expected_version=review_actions.read_review(saved.session_dir).version,
                action="resolve",
                resolution=review_records.FindingResolution(
                    disposition="intentional", reason="The caller forbids empty input"
                ),
            ),
            source="cli",
        )


@pytest.mark.parametrize("surface", ["cli", "web", "tui", "auto"])
def test_every_merge_path_refuses_open_blocks_even_when_panel_did_not_reject(
    tmp_path: pathlib.Path,
    run: layout.SessionLayout,
    surface: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An HTTP POST and both terminal actions reach the same guard as auto-merge."""
    base = _git(tmp_path, "rev-parse", "main")
    view = review_actions.read_review(run.session_dir)
    assert all(f.line_origin.kind == "pre_existing" for f in view.findings)
    if surface == "cli":
        ok = main(["sessions", "merge", run.session_id]) == 0
        captured = capsys.readouterr()
        text = captured.out + captured.err
    elif surface == "tui":
        ok, text = home._run_merge_cli(tmp_path, run.session_id)  # pyright: ignore[reportPrivateUsage]
    elif surface == "auto":
        finalize.finalize_auto_merge(
            tmp_path, layout=run, cfg=Config(), reporter=reporter.STDIO_REPORTER
        )
        ok = _git(tmp_path, "rev-parse", "main") != base
        text = capsys.readouterr().err
    else:
        srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", secret=signin.SECRET)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        conn = signin.SignedIn(srv.server_address[1])
        try:
            conn.request("POST", f"/api/session/{run.session_id}/merge", "{}")
            response = conn.getresponse()
            ok, text = response.status == 200, response.read().decode()
        finally:
            conn.close()
            srv.shutdown()
            srv.server_close()
            thread.join()
    assert not ok, text
    assert "f1" in text and "f2" in text and "resolve" in text.lower()
    assert _git(tmp_path, "rev-parse", "main") == base
    assert manifest.read_manifest(run.session_dir).merged is None


def test_unreadable_structured_evidence_refuses_with_its_reason(
    tmp_path: pathlib.Path, run: layout.SessionLayout
) -> None:
    run.logs_path.write_text(json.dumps({"type": "loop.review.panel", "panel_id": "broken"}))
    result = _merge(tmp_path, run)
    assert result.status == "error"
    assert "review" in result.error and "target" in result.error
    assert manifest.read_manifest(run.session_dir).merged is None


@pytest.mark.parametrize("evidence", ["missing", "legacy", "resolved"])
def test_unknown_review_or_resolved_findings_do_not_require_a_clean_panel(
    tmp_path: pathlib.Path, run: layout.SessionLayout, evidence: str
) -> None:
    if evidence == "missing":
        run.logs_path.unlink()
    elif evidence == "legacy":
        run.logs_path.write_text('{"type":"loop.review.panel","blocked":true,"n_block":2}\n')
    else:
        _resolve_all(run)
    assert _merge(tmp_path, run).status == "merged"


def test_merge_excludes_another_session_writer(
    tmp_path: pathlib.Path, run: layout.SessionLayout
) -> None:
    _resolve_all(run)
    fd = lock.acquire_single_writer(run.session_dir)
    assert fd is not None
    try:
        result = _merge(tmp_path, run)
        assert result.status == "error" and "writer" in result.error
    finally:
        lock.release_single_writer(fd)


def test_merge_rechecks_dispositions_after_taking_the_writer_lock(
    tmp_path: pathlib.Path, run: layout.SessionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_all(run)
    acquire = lock.acquire_single_writer

    def reopen_then_acquire(session_dir: pathlib.Path) -> int | None:
        event_log.EventSink(run.logs_path).emit(
            "loop.review.finding.reopened",
            finding_id="f1",
            request_id="reopen",
            reason="Still reproducible",
            source="cli",
            version=4,
        )
        return acquire(session_dir)

    monkeypatch.setattr(lock, "acquire_single_writer", reopen_then_acquire)
    result = _merge(tmp_path, run)
    assert result.status == "error" and "f1" in result.error


def test_auto_merge_keeps_its_worker_lock_until_landing_finishes(
    tmp_path: pathlib.Path, run: layout.SessionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_all(run)
    fd = lock.acquire_single_writer(run.session_dir)
    assert fd is not None
    plumb = git_ops.plumb_merge

    def merge_while_locked(*args: Any, **kwargs: Any) -> git_ops.MergeResult:
        with pytest.raises(review_actions.ReviewActionError, match="writer is busy"):
            review_actions.act(
                run.session_dir,
                "f1",
                review_records.FindingAction(
                    request_id="reopen", expected_version=3, action="reopen", reason="Reproduced"
                ),
                source="web",
            )
        return plumb(*args, **kwargs)

    monkeypatch.setattr(git_ops, "plumb_merge", merge_while_locked)
    try:
        finalize.finalize_auto_merge(
            tmp_path,
            layout=run,
            cfg=Config(),
            reporter=reporter.STDIO_REPORTER,
            worker_lock_fd=fd,
        )
        assert manifest.read_manifest(run.session_dir).merged is not None
        assert lock.acquire_single_writer(run.session_dir) is None
    finally:
        lock.release_single_writer(fd)


def test_moving_source_ref_cannot_land_or_stamp_an_unchecked_tip(
    tmp_path: pathlib.Path, run: layout.SessionLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resolve_all(run)
    checked = _git(tmp_path, "rev-parse", "agent6/review-run")
    later = _git(tmp_path, "commit-tree", "main^{tree}", "-p", checked, "-m", "unchecked")
    plumb = git_ops.plumb_merge

    def move_then_merge(*args: Any, **kwargs: Any) -> git_ops.MergeResult:
        _git(tmp_path, "update-ref", "refs/heads/agent6/review-run", later)
        return plumb(*args, **kwargs)

    monkeypatch.setattr(git_ops, "plumb_merge", move_then_merge)
    assert _merge(tmp_path, run).status == "merged"
    assert _git(tmp_path, "rev-parse", "main") == checked
    stamp = manifest.read_manifest(run.session_dir).merged
    assert stamp is not None and stamp.tip == checked
