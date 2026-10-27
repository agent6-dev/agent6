from __future__ import annotations

from typing import Any

from tests.web import test_client_review, test_review_evidence, test_web_git

evidence = test_review_evidence.evidence


def test_receipt_route_reads_evidence(evidence: tuple[int, dict[str, Any]]) -> None:
    """The receipt route reads a saved session without a provider or a gate run."""
    port, _ = evidence
    code, payload = test_web_git._get(port, "/api/session/review-run/receipt")
    assert code == 200, payload
    assert payload["session_id"] == "review-run"
    assert payload["revert"]["kind"] == "not_recorded"
    assert payload["review"]["panels"]
    assert payload["text"].startswith("Receipt review-run\n")


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
assert.equal(window.receiptTask, r.revert_request);
assert.equal(location.hash, '#/');
assert.equal(posts.length, 0);
""")
