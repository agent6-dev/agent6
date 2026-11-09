from __future__ import annotations

import concurrent.futures
import contextlib
import json
import pathlib
import subprocess
import time
from unittest import mock

import pytest

from agent6.harness import loop, run_chain
from agent6.sessions import event_log, layout
from agent6.tools import tool_errors
from agent6.viewmodel import receipt
from tests.unit import test_execution_evidence, test_harness

saved = test_execution_evidence.saved


def test_lost_launcher_records_no_success_or_settlement(
    saved: layout.SessionLayout, jail_bin: pathlib.Path
) -> None:
    """A killed supervisor cannot supply a verification exit or confirm job cleanup."""
    tools = test_execution_evidence._tools(saved)
    dispatcher = tools.dispatcher
    access = dispatcher._command_access
    assert access is not None and access.session is not None
    access.session._proc.kill()
    access.session._proc.wait(timeout=5)
    try:
        with pytest.raises(tool_errors.ToolError):
            dispatcher.run_verify()
    finally:
        dispatcher.close()
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    assert view.verifications[-1].outcome == "unavailable"
    assert view.verifications[-1].exit_code is None
    assert view.verifications[-1].target_id == view.targets[-1].target_id
    assert view.jobs[-1].outcome == "unknown"
    assert view.jobs[-1].reason


def test_launcher_loss_during_verification_keeps_the_outcome_unknown(
    saved: layout.SessionLayout, tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """An accepted verifier's exit cannot replace the lost supervisor's response."""
    ready, release = tmp_path / "verifier.ready", tmp_path / "verifier.release"
    tools = test_execution_evidence._tools(
        saved,
        verify_command=(
            "/bin/sh",
            "-c",
            'printf ready > "$1"; while [ ! -f "$2" ]; do sleep 0.02; done; exit 0',
            "verifier",
            str(ready),
            str(release),
        ),
    )
    dispatcher = tools.dispatcher
    access = dispatcher._command_access
    assert access is not None and access.session is not None
    # Close the command channel before joining a timed-out worker.
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
        contextlib.closing(dispatcher),
    ):
        verification = executor.submit(dispatcher.run_verify)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "the supervisor never started the verifier"
            assert not verification.done()
            access.session._proc.kill()
            access.session._proc.wait(timeout=5)
            release.touch()
            with pytest.raises(tool_errors.ToolError):
                verification.result(timeout=10)
        finally:
            release.touch()
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    attempt = view.verifications[-1]
    assert attempt.outcome == "unavailable"
    assert attempt.exit_code is None
    assert attempt.target_id == view.targets[-1].target_id
    assert not attempt.certifies(attempt.tree_before)
    assert view.jobs[-1].outcome == "unknown"
    assert view.jobs[-1].reason


def test_launcher_loss_at_finish_replaces_the_passing_baseline(
    saved: layout.SessionLayout, tmp_path: pathlib.Path, jail_bin: pathlib.Path
) -> None:
    """The terminal event and receipt reference the lost final gate, not its passing baseline."""
    ready, release = tmp_path / "verifier.ready", tmp_path / "verifier.release"
    tools = test_execution_evidence._tools(
        saved,
        verify_command=(
            "/bin/sh",
            "-c",
            'if [ ! -f changed.txt ]; then exit 0; fi; printf ready > "$1"; '
            'while [ ! -f "$2" ]; do sleep 0.02; done; exit 0',
            "verifier",
            str(ready),
            str(release),
        ),
        verify_retries=0,
    )
    dispatcher = tools.dispatcher
    access = dispatcher._command_access
    assert access is not None and access.session is not None
    provider = mock.Mock()
    provider.call.side_effect = [
        test_harness._tool_resp("apply_edit", {"path": "changed.txt", "new_string": "changed\n"}),
        test_harness._tool_resp("finish_session", {"summary": "checked"}, tool_id="done"),
    ]
    harness = loop.Harness(
        chain=run_chain.RunChain(
            pathlib.Path.cwd(),
            per_turn=False,
            base_sha=subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        ),
        config=tools.cfg,
        provider=provider,
        dispatcher=dispatcher,
        events=event_log.EventSink(saved.logs_path),
        max_turns=2,
    )
    # Close the command channel before joining a timed-out worker.
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
        contextlib.closing(dispatcher),
    ):
        execution = executor.submit(harness.run, "Edit and check this tree")
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "the supervisor never started the final gate"
            assert not execution.done()
            access.session._proc.kill()
            access.session._proc.wait(timeout=5)
            release.touch()
            result = execution.result(timeout=10)
        finally:
            release.touch()
    rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
    assert [row["why"] for row in rows if row["type"] == "loop.verify_harness"] == [
        "start",
        "finish",
    ]
    (terminal,) = [row for row in rows if row["type"] == "session.stop"]
    view = receipt.read_receipt(saved.session_dir, pathlib.Path.cwd())
    baseline, final = view.verifications
    assert baseline.baseline and baseline.outcome == "passed" and baseline.exit_code == 0
    assert baseline.tree_before == baseline.tree_after and baseline.tree_after
    assert final.outcome == "unavailable" and final.exit_code is None
    assert final.tree_before == final.tree_after == terminal["tree"] == view.end_tree
    assert view.end_tree and view.end_tree != baseline.tree_after
    assert final.verify_id and final.verify_id != baseline.verify_id
    assert final.target_id == baseline.target_id == view.targets[-1].target_id
    assert result.reason == "finish_session"
    assert result.verification == terminal["verification"] == "unverified"
    assert result.final_verify_id == terminal["final_verify_id"] == final.verify_id
    assert view.final_verify_id == final.verify_id
    assert view.jobs[-1].outcome == "unknown" and view.jobs[-1].reason
    assert f"Final tree not certified: {view.end_tree}" in receipt.render(view)
    assert "Final tree passed" not in receipt.render(view)
