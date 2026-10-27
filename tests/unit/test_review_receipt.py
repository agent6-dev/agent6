from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Literal
from unittest import mock

import pytest

from agent6 import evidence, paths
from agent6.app import merge
from agent6.config import Config
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest
from agent6.vcs import git_read
from tests.unit import test_cli_sessions_merge as merge_tests


@pytest.mark.parametrize("method", ["ff_only", "merge", "squash"])
def test_merge_records_bounds_for_reversal(
    tmp_path: pathlib.Path, method: Literal["ff_only", "merge", "squash"]
) -> None:
    """All landing methods retain the target's old tip and the actual method."""
    base = merge_tests._setup_run(
        tmp_path, "receipt", commits=[("a", "one", "one"), ("b", "two", "two")]
    )
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="receipt"
    )
    outcome = merge.execute_merge(
        tmp_path,
        layout=layout,
        manifest=manifest.read_manifest(layout.session_dir),
        run_branch="agent6/receipt",
        base_sha=manifest.read_manifest(layout.session_dir).base_sha,
        target="main",
        method=method,
        cfg=Config(),
        identity=git_read.CommitIdentity(),
        message="land",
    )
    assert outcome.status == "merged"
    stamp = manifest.read_manifest(layout.session_dir).merged
    assert stamp is not None
    assert stamp.model_dump().get("method") == method
    assert stamp.model_dump().get("target_before") == base


def test_historical_stamp_keeps_unknown_bounds(tmp_path: pathlib.Path) -> None:
    """A legacy stamp never guesses a method or starting target."""
    stamp = manifest.MergeStamp.model_validate({"sha": "a" * 40, "into": "main"})
    assert "method" in stamp.model_dump()
    assert stamp.model_dump()["method"] is None
    assert stamp.model_dump()["target_before"] is None


def test_failed_stamp_does_not_claim_the_merge_failed(tmp_path: pathlib.Path) -> None:
    """A manifest write failure leaves a successful merge explicitly unstamped."""
    merge_tests._setup_run(tmp_path, "receipt", commits=[("a", "one", "one")])
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="receipt"
    )
    with mock.patch("agent6.app.stamps.write_manifest", side_effect=OSError("disk full")):
        outcome = merge.execute_merge(
            tmp_path,
            layout=layout,
            manifest=manifest.read_manifest(layout.session_dir),
            run_branch="agent6/receipt",
            base_sha=manifest.read_manifest(layout.session_dir).base_sha,
            target="main",
            method="merge",
            cfg=Config(),
            identity=git_read.CommitIdentity(),
            message="land",
        )
    assert outcome.status == "merged" and outcome.stamp_error == "disk full"
    assert manifest.read_manifest(layout.session_dir).merged is None


@pytest.mark.parametrize("method", ["ff_only", "merge", "squash"])
def test_reversal_uses_objects_even_after_source_ref_pruning(
    tmp_path: pathlib.Path, method: str
) -> None:
    """Reversal depends on recorded object ids, never a surviving source ref."""
    from agent6.viewmodel import receipt

    base = merge_tests._setup_run(
        tmp_path, "receipt", commits=[("a", "one", "one"), ("b", "two", "two")]
    )
    source = merge_tests._git(tmp_path, "rev-parse", "agent6/receipt")
    from agent6.vcs import git_ops

    landed = git_ops.plumb_merge(tmp_path, "main", source, method=method, message="land").merged_sha
    merge_tests._git(tmp_path, "update-ref", "-d", "refs/heads/agent6/receipt")
    stamp = manifest.MergeStamp.model_validate(
        {"method": method, "sha": landed, "target_before": base, "tip": source}
    )
    plan = receipt.revert_plan(tmp_path, stamp)
    assert plan.kind == method
    assert plan.mainline == (1 if method == "merge" else None)
    assert plan.commits[0] == landed
    assert len(plan.commits) == (2 if method == "ff_only" else 1)


def test_reversal_refuses_missing_objects_and_unverified_parents(tmp_path: pathlib.Path) -> None:
    """Missing or contradictory merge facts never produce executable-looking guidance."""
    from agent6.viewmodel import receipt

    base = merge_tests._setup_run(tmp_path, "receipt", commits=[("a", "one", "one")])
    for stamp in (
        manifest.MergeStamp(sha="f" * 40, method="squash", target_before=base),
        manifest.MergeStamp(sha=base, method="merge", target_before="e" * 40),
        manifest.MergeStamp(sha=base),
    ):
        plan = receipt.revert_plan(tmp_path, stamp)
        assert plan.kind == "manual" and plan.commits == ()
        assert "cannot determine a safe revert" in plan.explanation
    absent = receipt.revert_plan(tmp_path, None)
    assert absent.kind == "not_recorded"
    assert "no merge recorded" in absent.explanation
    assert "no merge occurred" not in absent.explanation


def test_fast_forward_with_merges_requires_a_manual_plan(tmp_path: pathlib.Path) -> None:
    """Fast-forward reversal never guesses a mainline for a landed merge."""
    from agent6.vcs import git_ops
    from agent6.viewmodel import receipt

    base = merge_tests._setup_run(tmp_path, "receipt", commits=[("a", "one", "one")])
    landed = git_ops.plumb_merge(
        tmp_path, "main", "agent6/receipt", method="merge", message="land"
    ).merged_sha
    stamp = manifest.MergeStamp(sha=landed, method="ff_only", target_before=base)
    plan = receipt.revert_plan(tmp_path, stamp)
    assert plan.kind == "manual" and plan.mainline is None and not plan.commits


@pytest.mark.parametrize(
    "before,after,end,word",
    [
        ("a", "a", "a", "Final tree passed"),
        ("a", "a", "b", "Final tree not certified"),
        ("a", "b", "b", "Final tree not certified"),
    ],
)
def test_receipt_binds_pass_to_observed_tree(
    tmp_path: pathlib.Path, before: str, after: str, end: str, word: str
) -> None:
    """A receipt cannot promote an old pass or a mutating check into final-tree evidence."""
    import json

    from agent6.viewmodel import receipt

    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "manifest.json").write_text("{}")
    rows = [
        {"type": "session.start", "execution_id": "exec"},
        {
            "type": "verify.end",
            "verify_id": "v",
            "execution_id": "exec",
            "cmd": ["true"],
            "tree_before": before,
            "tree_after": after,
            "scoped": False,
            "selected_paths": [],
            "outcome": "passed",
            "exit_code": 0,
            "duration_s": 1,
            "timeout_s": 10,
            "stdout_tail": "pass",
            "stderr_tail": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
        },
        {"type": "session.end", "tree": end, "final_verify_id": "v", "verified": "passed"},
    ]
    (saved / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    view = receipt.read_receipt(saved, tmp_path)
    assert word in receipt.render(view)
    assert view.verifications[0].verify_id == "v"
    assert view.execution_id == "exec"


def test_legacy_receipt_marks_unknown_capture_and_no_gate(tmp_path: pathlib.Path) -> None:
    """Historical tails never imply complete output, and no gate never implies a pass."""
    import json

    from agent6.viewmodel import receipt

    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "verify.end", "cmd": ["true"], "exit_code": 0, "stdout_tail": "ok"})
        + "\n"
    )
    text = receipt.render(receipt.read_receipt(tmp_path, tmp_path))
    assert "capture completeness unknown" in text
    assert "2000" in text
    assert "Final tree not certified" in text
    (tmp_path / "logs.jsonl").write_text("")
    assert "No verification recorded" in receipt.render(receipt.read_receipt(tmp_path, tmp_path))


def test_receipt_cli_is_read_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The receipt command reads evidence without needing a configured provider."""
    import json

    from agent6.ui.cli import main

    merge_tests._setup_run(tmp_path, "receipt", commits=[])
    monkeypatch.chdir(tmp_path)
    assert main(["sessions", "receipt", "receipt", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["session_id"] == "receipt"
    assert data["verifications"] == []


def test_receipt_compares_recorded_panel_with_end_tree(tmp_path: pathlib.Path) -> None:
    """A finished reviewed session compares immutable tree content without a checkout."""
    from agent6 import event_log
    from agent6.viewmodel import receipt
    from tests import review_helpers

    saved = review_helpers.reviewed_session(tmp_path)
    sink = event_log.EventSink(saved.logs_path)
    sink.emit("session.end", tree="ht")
    view = receipt.read_receipt(saved.session_dir, tmp_path)
    assert view.review.freshness == "current"
    sink.emit("session.end", tree="other")
    assert receipt.read_receipt(saved.session_dir, tmp_path).review.freshness == "stale"


def test_historical_baseline_marks_only_its_observation(tmp_path: pathlib.Path) -> None:
    """An id-less baseline event never relabels every earlier verification as baseline."""
    from agent6 import event_log
    from agent6.viewmodel import receipt

    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit("verify.end", cmd=["true"], exit_code=0)
    sink.emit("verify.end", cmd=["false"], exit_code=1)
    sink.emit("loop.baseline", ok=False)
    view = receipt.read_receipt(tmp_path, tmp_path)
    assert [record.baseline for record in view.verifications] == [False, True]


@pytest.mark.parametrize("method", ["ff_only", "merge", "squash", None])
def test_revert_request_carries_recorded_facts(tmp_path: pathlib.Path, method: str | None) -> None:
    """A new worker gets recorded facts without needing masked session files."""
    from agent6.app import stamps
    from agent6.vcs import git_ops
    from agent6.viewmodel import receipt

    base = merge_tests._setup_run(tmp_path, "receipt", commits=[("a", "one", "one")])
    saved = sessions_layout.SessionLayout(paths.state_dir(tmp_path), "receipt")
    tip = merge_tests._git(tmp_path, "rev-parse", "agent6/receipt")
    landed = git_ops.plumb_merge(tmp_path, "main", tip, method=method or "merge", message="land")
    stamped = manifest.read_manifest(saved.session_dir).model_copy(
        update={
            "merged": manifest.MergeStamp.model_validate(
                {
                    "sha": landed.merged_sha,
                    "method": method,
                    "target_before": base,
                    "tip": tip,
                    "into": "main",
                }
            )
        }
    )
    stamps.write_manifest(saved.manifest_path, stamped)
    view = receipt.read_receipt(saved.session_dir, tmp_path)
    task = receipt.revert_request(view)
    assert base in task and landed.merged_sha in task and tip in task
    assert "main" in task and (method or "not recorded") in task
    if method:
        assert "Commits to revert in order: " + ", ".join(view.revert.commits) in task
    else:
        assert "No safe plan determined" in task
    if method == "merge":
        assert "Mainline: 1" in task
    assert "Inspect the receipt" not in task and "isolated run" not in task


def _record(**changes: Any) -> evidence.VerificationEvidence:
    return dataclasses.replace(
        evidence.VerificationEvidence(
            verify_id="v",
            execution_id=None,
            cmd=("true",),
            tree_before=None,
            tree_after=None,
            scoped=False,
            selected_paths=(),
            outcome="denied",
            exit_code=None,
            duration_s=0.0,
            timeout_s=0.0,
            stdout_tail="",
            stderr_tail="",
            stdout_truncated=False,
            stderr_truncated=False,
            reason="not available",
        ),
        **changes,
    )


def test_summary_states_the_outcome_once() -> None:
    """A denied attempt's summary read "denied, denied"; the outcome is said once."""
    from agent6.viewmodel import receipt

    assert receipt.verification_summary(_record()).startswith("v: denied, full gate\n")
    passed = _record(outcome="passed", exit_code=0)
    assert receipt.verification_summary(passed).startswith("v: passed, exit 0, full gate\n")


def test_receipt_uses_one_shared_scan_without_per_line_adapters(tmp_path: pathlib.Path) -> None:
    """Receipt reads share the listing scan and do not build schemas per journal line."""
    import json

    import pydantic

    from agent6.viewmodel import _session_log, receipt

    row = json.dumps({"type": "verify.end", "verify_id": "v", "cmd": ["true"], "exit_code": 0})
    (tmp_path / "logs.jsonl").write_text((row + "\n") * 2000)
    with (
        mock.patch.object(
            _session_log, "scan_session_log", wraps=_session_log.scan_session_log
        ) as scan,
        mock.patch.object(pydantic, "TypeAdapter", wraps=pydantic.TypeAdapter) as adapter,
    ):
        receipt.read_receipt(tmp_path, tmp_path)
    assert adapter.call_count == 0
    scan.assert_called_once_with(tmp_path / "logs.jsonl")


def test_torn_live_journal_keeps_evidence_but_cannot_certify(tmp_path: pathlib.Path) -> None:
    """A torn multibyte tail keeps earlier output without certifying a tree."""
    from agent6 import event_log
    from agent6.viewmodel import _session_log, receipt

    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit(
        "verify.end",
        verify_id="v",
        cmd=["true"],
        exit_code=0,
        tree_before="tree",
        tree_after="tree",
        stdout_tail="the retained output",
    )
    sink.emit("session.end", final_verify_id="v", tree="tree")
    with sink.path.open("ab") as stream:
        stream.write(b'{"type":"loop.resume.start","user_task":"\xe2\x82')
    view = receipt.read_receipt(tmp_path, tmp_path)
    assert len(view.verifications) == 1
    assert view.verifications[0].stdout_tail == "the retained output"
    assert "Final tree not certified" in receipt.render(view)
    assert view.review == _session_log.scan_session_log(sink.path).review
    assert view.evidence_notes
