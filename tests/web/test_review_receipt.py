from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths
from agent6.sessions import event_log, layout
from tests.web import test_client_review, test_review_evidence, test_web_browser, test_web_git

if TYPE_CHECKING:
    from playwright.sync_api import Page

evidence = test_review_evidence.evidence
open_page = test_web_browser.open_page


def test_receipt_route_reads_evidence(evidence: tuple[int, dict[str, Any]]) -> None:
    """The receipt route reads a saved session without a provider or a gate run."""
    port, _ = evidence
    code, payload = test_web_git._get(port, "/api/session/review-run/receipt")
    assert code == 200, payload
    assert payload["session_id"] == "review-run"
    assert payload["revert"]["kind"] == "not_recorded"
    assert payload["review"]["panels"]
    assert payload["text"].startswith("Receipt review-run\n")


def test_receipt_route_and_card_state_the_output_cap_once(
    tmp_path: pathlib.Path, evidence: tuple[int, dict[str, Any]]
) -> None:
    """The web card retains truncation notes without repeating the cap for every attempt."""
    saved = layout.SessionLayout(paths.state_dir(tmp_path), "review-run")
    sink = event_log.EventSink(saved.logs_path)
    for cut in (False, True):
        sink.emit(
            "verify.end",
            cmd=["pytest"],
            exit_code=0,
            stdout_tail="out",
            stderr_tail="",
            stdout_truncated=cut,
            stderr_truncated=False,
            scoped=False,
            selected_paths=[],
            outcome="passed",
            duration_s=0.2,
            timeout_s=60.0,
            baseline=False,
            reason="",
        )
    port, _ = evidence
    code, payload = test_web_git._get(port, "/api/session/review-run/receipt")
    assert code == 200
    assert "Testing (output cap: 2000 characters per stream)\n" in payload["text"]
    test_client_review._run(
        f"const receipt = {json.dumps(payload)};"
        + """
const host = el('div');
drawReceipt(host, receipt);
assert.equal(host.textContent.split('2000 characters').length - 1, 1);
assert.equal(host.textContent.split('stdout truncated').length - 1, 1);
assert.ok(!host.textContent.includes('complete capture'));
"""
    )


@pytest.mark.browser
def test_receipt_renders_job_settlement_separately_from_environment_lifecycle(
    tmp_path: pathlib.Path,
    evidence: tuple[int, dict[str, Any]],
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The browser shows settled jobs without implying that their environment stopped."""
    from playwright.sync_api import expect

    saved = layout.SessionLayout(paths.state_dir(tmp_path), "review-run")
    events = event_log.EventSink(saved.logs_path)
    events.emit(
        "execution.jobs",
        execution_id="execution",
        target_id="attached",
        outcome="settled",
        reason="",
    )
    events.emit(
        "execution.environment",
        execution_id="execution",
        target_id="attached",
        outcome="left_running",
        reason="ownership unproved; stop by hand",
    )
    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{saved.session_id}")
    page.get_by_role("button", name="Receipt", exact=True).click()
    summary = page.locator(".receipt > pre").first
    expect(summary).to_be_visible()
    lines = summary.inner_text().splitlines()
    assert "Run jobs settled (target attached)" in lines
    assert "Environment left running (target attached): ownership unproved; stop by hand" in lines
    assert not errors, errors


def test_receipt_client_uses_literal_risk_and_existing_new_work() -> None:
    """The receipt text and output bodies render literally; the revert button pre-fills new work."""
    test_client_review._run("""
const host = el('div'); globalThis.location = {hash:''};
const r = {session_id:'run',
 text:'Receipt run\\n\\nTesting\\n\\nRisk\\n<script>raw risk</script>\\n\\nHow to revert',
 revert_request:'Inspect the recorded run before reverting',
 verifications:[{verify_id:'v', stdout_tail:'<b>out</b>', stderr_tail:''}]};
drawReceipt(host, r);
assert.ok(host.textContent.includes('<script>raw risk</script>'));
assert.ok(host.textContent.includes('<b>out</b>'));
assert.ok(host.textContent.includes('Testing') && host.textContent.includes('How to revert'));
host.querySelector('button').onclick();
assert.equal(window.pendingNewWork, r.revert_request);
assert.equal(location.hash, '#/');
assert.equal(posts.length, 0);
""")
