from __future__ import annotations

import os
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths, review_records
from agent6.app.review import actions
from agent6.sessions import event_log, layout, process
from agent6.ui import keymap
from tests.web import (
    geometry,
    test_files_preview_browser,
    test_findings,
    test_review_evidence,
    test_web_browser,
)

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
evidence = test_review_evidence.evidence
OpenPage = Callable[[int], tuple["Page", list[str]]]


def test_reviewed_source_shares_the_heading_gutter(
    open_page: OpenPage, evidence: tuple[int, dict[str, Any]]
) -> None:
    """The reviewed bytes and their heading start at the same page inset."""
    port, revs = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/{revs['base']}/old.txt?panel=panel&side=base")
    heading, source = geometry.boxes(
        page.get_by_role("heading", name="base · old.txt"), page.get_by_text("Reviewed tree:")
    )
    assert heading["x"] == pytest.approx(source["x"], abs=0.5)
    assert not errors


def test_resolution_shows_the_disposition_labels_and_names_the_reason(
    open_page: OpenPage, evidence: tuple[int, dict[str, Any]]
) -> None:
    """Read a disposition by its label and see the Reason label after typing."""
    from playwright import sync_api

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    card = page.locator('.card-findings [data-finding="rename"]')
    card.get_by_role("button", name="Resolve", exact=True).click()
    options = card.get_by_role("combobox", name="Disposition").get_by_role("option")
    assert [(o.inner_text(), o.get_attribute("value")) for o in options.all()] == list(
        keymap.REVIEW_DISPOSITIONS
    )
    card.get_by_label("Disposition").select_option(label="Won't fix")
    reason = card.get_by_role("textbox", name=keymap.REVIEW_TEXT["reason"], exact=True)
    reason.fill("The old format remains supported.")
    sync_api.expect(card.get_by_text(keymap.REVIEW_TEXT["reason"], exact=True)).to_be_visible()
    card.get_by_role("button", name="Submit resolution").click()
    page.get_by_label("Finding status").select_option("resolved")
    sync_api.expect(card).to_contain_text("Won't fix: The old format remains supported.")
    assert not errors


def test_another_resolution_preserves_the_reason_and_uses_the_new_version(
    open_page: OpenPage, evidence: tuple[int, dict[str, Any]]
) -> None:
    """A review frame for a different finding must leave the open form ready to submit."""
    from playwright import sync_api

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    findings = page.locator(".card-findings")
    card = findings.locator('[data-finding="rename"]')
    card.get_by_role("button", name="Resolve", exact=True).click()
    reason = card.get_by_role("textbox", name="Reason")
    reason.fill("Keep this format.")
    code, review = test_findings._request(port)
    assert code == 200
    code, result = test_findings._request(
        port,
        "/delete/resolve",
        dict(
            expected_version=review["version"],
            request_id="elsewhere",
            resolution=dict(disposition="fixed", reason="Checked the deletion."),
        ),
    )
    assert code == 200, result
    sync_api.expect(findings).to_contain_text("3 block")
    sync_api.expect(reason).to_be_focused()
    sync_api.expect(reason).to_have_value("Keep this format.")
    card.get_by_role("button", name="Submit resolution").click()
    sync_api.expect(findings).to_contain_text("2 block")
    assert not errors


def test_a_worker_refusal_allows_a_fresh_resolution(
    open_page: OpenPage, evidence: tuple[int, dict[str, Any]], tmp_path: pathlib.Path
) -> None:
    """An unacknowledged request keeps its id, but the worker remembers a refused id forever."""
    from playwright import sync_api

    port, _ = evidence
    run = layout.SessionLayout(paths.state_dir(tmp_path), "review-run")
    version = actions.read_review(run.session_dir).version
    process.write_worker_pid(run.session_dir, os.getpid())
    first = review_records.FindingAction(
        request_id="0",  # Sorts before the browser's request in the worker queue.
        expected_version=version,
        action="resolve",
        resolution=review_records.FindingResolution(disposition="fixed", reason="Checked."),
    )
    assert actions.act(run.session_dir, "delete", first, source="cli", wait_s=0).status == "queued"
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    findings = page.locator(".card-findings")
    card = findings.locator('[data-finding="rename"]')
    card.get_by_role("button", name="Resolve", exact=True).click()
    card.get_by_role("textbox", name="Reason").fill("The renamed file works.")
    submit = card.get_by_role("button", name="Submit resolution")
    url = "**/findings/rename/resolve"
    with page.expect_response(url) as queued:
        submit.click()
    assert queued.value.status == 202
    sync_api.expect(submit).to_be_enabled()
    actions.drain(event_log.EventSink(run.logs_path), execution_id="worker")
    process.clear_worker_pid(run.session_dir)
    sync_api.expect(findings.locator(".finding-counts")).to_contain_text("3 block")
    with page.expect_response(url) as refused:
        submit.click()
    assert refused.value.status == 409
    assert refused.value.request.post_data_json == queued.value.request.post_data_json
    sync_api.expect(submit).to_be_enabled()
    with page.expect_response(url) as retried:
        submit.click()
    assert retried.value.request.post_data_json["request_id"] != queued.value.json()["request_id"]
    assert retried.value.request.post_data_json["expected_version"] == version + 1
    assert retried.value.status == 200, retried.value.json()
    sync_api.expect(findings).to_contain_text("2 block")
    assert not [e for e in errors if not e.startswith("Failed to load resource")]
