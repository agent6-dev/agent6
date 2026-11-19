"""The session page's drawer cards driven in a browser: findings, receipt, log, tasks, diff."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6 import paths
from agent6.sessions import event_log, journal, layout
from tests import review_helpers
from tests.web import session_logs, test_review_evidence, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
evidence = test_review_evidence.evidence


def test_a_live_frame_leaves_the_cards_a_reader_is_using(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A frame rebuilds only the cards it changes, and the event log keeps the reader's place.

    Every frame rebuilt every card: the event log snapped to its end, and a selection in the
    run card or a node in the tool calls went with the old nodes.
    """
    from playwright.sync_api import expect

    rows: list[dict[str, Any]] = []
    for i in range(60):
        rows.append({"type": "tool.call", "name": "read_file", "args_preview": {"path": f"f{i}"}})
        rows.append({"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"})
    session = session_logs.write_run(tmp_path, "live-AAA111", rows, live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/live-AAA111")
    lines = page.locator(".card-log .log > div")
    expect(lines).to_have_count(121)
    page.evaluate(
        """() => {
        document.querySelector('.card-log .card-body').scrollTop = 0;
        getSelection().selectAllChildren(document.querySelectorAll('.card-head .kv .v')[1]);
        document.querySelector('.card-tools .call').dataset.kept = 'yes';
    }"""
    )
    with (session / "logs.jsonl").open("a") as log:
        log.write(json.dumps({"type": "budget.update", "usd_total": 0.5}) + "\n")
    expect(lines).to_have_count(122)
    kept = page.evaluate(
        """() => [document.querySelector('.card-log .card-body').scrollTop,
        getSelection().toString(),
        document.querySelectorAll('.card-tools .call[data-kept]').length]"""
    )
    assert kept == [0, "live-AAA111", 1]
    assert not errors


def test_a_wrapped_event_log_row_hangs_under_its_event(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A wrapped row's later lines start under its event name, clear of the time column.

    They started at the row's left edge, under the times, and read as rows of their own.
    """
    from playwright.sync_api import expect

    argv = ["sh", "-c", "echo " + "word " * 40]
    call = {"type": "tool.call", "ts": "2026-11-13T10:00:00.000000+00:00", "name": "run_command"}
    session_logs.write_run(tmp_path, "wrap-AAA111", [{**call, "args_preview": {"argv": argv}}])
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/wrap-AAA111")
    row = page.locator(".card-log .log > div").filter(has_text="run_command")
    expect(row).to_be_visible()
    # The event name starts at the row's 15th character, after the time and two spaces.
    lines, last, event = row.evaluate(
        """e => { const text = e.firstChild, r = document.createRange();
        r.selectNodeContents(text); const lines = [...r.getClientRects()];
        r.setStart(text, 14); r.setEnd(text, 15);
        return [lines.length, lines.at(-1).left, r.getBoundingClientRect().left]; }"""
    )
    assert lines > 1
    assert last == pytest.approx(event, abs=0.5)  # layout rounds to 1/64 px
    assert not errors


def test_the_findings_card_spaces_its_parts_and_labels_its_filters(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """The findings card's parts stand 6px apart, its facts are rows, its selects take the skin.

    The counts touched the Receipt button, Resume to fix all open touched the review and merge
    lines it sat between, and the filters and the disposition were unlabelled native selects.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    body = page.locator(".card-findings .card-body")
    expect(body.locator(".kv .k")).to_have_text(["review", "merge"])
    expect(body.locator(".pickers .pick > span")).to_have_text(["status", "severity"])
    gaps = body.evaluate(
        """b => { const parts = [...b.children].filter(part => part.offsetParent);
        return parts.slice(1).map((part, i) => Math.round(
            part.getBoundingClientRect().top - parts[i].getBoundingClientRect().bottom)); }"""
    )
    assert set(gaps) == {6}
    body.locator(".finding").first.get_by_role("button", name="Resolve", exact=True).click()
    selects = body.locator("select:visible")
    expect(selects).to_have_count(3)
    assert selects.evaluate_all("all => all.map(s => s.className)") == ["field"] * 3
    assert not errors


@pytest.mark.parametrize("reviewed", [False, True])
def test_a_drafts_findings_card_waits_for_a_review(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    *,
    reviewed: bool,
) -> None:
    """A machine draft shows its Findings card once a review ran; a run shows its card always.

    Every draft showed the card with no panel, no finding and no Receipt: nothing to read or do.
    """
    from playwright.sync_api import expect

    start = {"type": "session.start", "mode": "machine", "user_task": "Author a triage machine"}
    call = {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}}
    review = review_helpers.panel_rows(warn=1) if reviewed else []
    draft = paths.state_dir(tmp_path) / "sessions" / "machines" / "brave-otter"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [start, call, *review]))
    session_logs.write_run(tmp_path, "plain-AAA111", [])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/draft/brave-otter")
    expect(page.locator(".card-conv .ci").first).to_be_visible()
    card = page.locator(".card-findings")
    if reviewed:
        expect(card.locator(".finding")).to_have_count(1)
        expect(card).to_be_visible()
    else:
        expect(card).to_be_hidden()
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Widgets", exact=True).click()
    entry = page.locator(".wmenu").get_by_role("button", name="Findings", exact=True)
    if reviewed:
        expect(entry).to_be_visible()
    else:
        expect(entry).to_be_hidden()
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/plain-AAA111")
    expect(page.locator(".card-findings").get_by_role("button", name="Receipt")).to_be_visible()
    assert not errors


def test_a_task_title_wraps_and_a_skipped_task_strikes_only_its_title(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A long task title wraps under itself, and a skipped task's title is struck, not its glyph.

    A long title was cut at the drawer's edge with no way to read the rest, and the strike ran
    through the indent and the glyph while the title stayed plain.
    """
    from playwright.sync_api import expect

    title = "Handle a quoted field that spans a CRLF and keeps a doubled quote " * 2
    nodes = {
        "t1": {"title": title, "parent_id": None, "status": "skipped", "children": []},
    }
    session_logs.write_run(
        tmp_path, "tasks-AAA111", [{"type": "graph.update", "nodes": nodes, "cursor": None}]
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/tasks-AAA111")
    node = page.locator(".card-tasks .tree > div").filter(has_text="Handle a quoted field")
    expect(node).to_be_visible()
    fits, lines = node.evaluate(
        "n => [n.scrollWidth <= n.clientWidth, Math.round(n.offsetHeight / 19.5)]"
    )
    struck = node.evaluate(
        """n => [...n.querySelectorAll('span')].map(s =>
        [s.textContent.trim().slice(0, 8), getComputedStyle(s).textDecorationLine])"""
    )
    assert fits
    assert lines > 1
    assert ["\u2013", "none"] in struck  # the skipped glyph
    assert ["Handle a", "line-through"] in struck
    assert not errors


def test_a_budget_bar_stands_apart_from_its_label_and_the_next_line(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A budget bar stands 4px under its label and 10px over the line under it.

    The bar touched its label above and the tokens line below.
    """
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "budget-AAA111", [{"type": "budget.update", "usd_total": 0.5}])
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/budget-AAA111")
    bar = page.locator(".card-budget .bar")
    expect(bar).to_be_visible()
    gaps = bar.evaluate(
        """b => { const r = e => e.getBoundingClientRect();
        const label = b.previousElementSibling, next = b.parentElement.nextElementSibling;
        return [r(b).top - r(label).bottom, r(next).top - r(b).bottom]; }"""
    )
    assert gaps == [4, 10]
    assert not errors


def test_the_details_cards_and_the_conversation_set_text_in_the_dense_size(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The findings' counts and the budget's lines stood in 14px beside 13px rows.

    The conversation sat between the scale's sizes, at 12.5px.
    """
    from playwright.sync_api import expect

    rows = [
        {"type": "budget.update", "usd_total": 0.5},
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
    ]
    session_logs.write_run(tmp_path, "sizes-AAA111", rows)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/sizes-AAA111")
    expect(page.locator(".card-budget .bar")).to_be_visible()
    expect(page.locator(".conv .ci").first).to_be_visible()
    sizes = page.evaluate(
        """() => { const sizes = {};
        for (const body of document.querySelectorAll('.drawer .card-body')) {
          const walk = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
          while (walk.nextNode()) {
            const text = walk.currentNode, e = text.parentElement;
            if (!text.data.trim() || !e.getClientRects().length) continue;
            (sizes[getComputedStyle(e).fontSize] ||= []).push(text.data.trim());
          }
        }
        return sizes; }"""
    )
    assert set(sizes) <= {"12px", "13px"}, sizes
    assert any(text.startswith("usage: ") for text in sizes.get("13px", [])), sizes
    conv = page.locator(".conv .ci").first.evaluate("e => getComputedStyle(e).fontSize")
    assert conv == "13px"
    assert not errors


def test_the_latest_commit_card_shows_no_error_under_the_logs_diff(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """With no commit picked, a pruned commit leaves the log's diff and no error under it.

    The card painted the diff the log recorded, then added "no diff for <sha>" under it.
    """
    from playwright.sync_api import expect

    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    sha = "5" * 40
    rows = [
        {"type": "checkpoint.committed", "turn": 1, "sha": sha, "subject": "change"},
        {"type": "diff.updated", "patch": patch, "sha": sha},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    session = session_logs.write_run(tmp_path, "pruned-AAA111", rows)
    (session / "manifest.json").write_text(json.dumps({"version": 3, "mode": "run"}))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    with page.expect_response(lambda response: "/diff?sha=" in response.url) as answer:
        page.goto(f"http://127.0.0.1:{port}/#/session/pruned-AAA111")
    refusal = answer.value.json()["error"]
    card = page.locator(".card-diff .card-body")
    expect(card).to_contain_text("new")
    page.wait_for_timeout(300)  # the refusal's paint
    assert refusal.startswith("no diff for 5555")
    assert refusal not in card.inner_text()
    assert [error for error in errors if "Failed to load resource" not in error] == []


def test_a_finding_card_on_its_session_links_away_and_sizes_its_submit(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """A finding on its own session's page has no link back to that page, and Submit fits its label.

    Each card linked "Session findings" to the page it was on, and the Resolve form's submit
    button stretched across the card.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    finding = page.locator(".card-findings .finding").first
    expect(finding.get_by_role("link", name="Reviewed comparison")).to_be_visible()
    expect(finding.get_by_role("link", name="Session findings")).to_have_count(0)
    finding.get_by_role("button", name="Resolve", exact=True).click()
    form = finding.locator(".finding-form")
    submit = form.get_by_role("button", name="Submit resolution")
    expect(submit).to_be_visible()
    widths = [form.bounding_box(), submit.bounding_box()]
    assert widths[0] and widths[1]
    assert widths[1]["width"] < widths[0]["width"] / 2
    assert not errors


def test_the_receipt_hangs_its_wrapped_lines_in_a_box_and_folds_only_output(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
    tmp_path: pathlib.Path,
) -> None:
    """The receipt's lines sit in a bordered 12px box, a wrapped line hanging under its start.

    The receipt was 14px text with no box, its wrapped lines restarting at column 0, and a
    check's empty stream still had a fold that opened onto nothing.
    """
    from playwright.sync_api import expect

    sink = event_log.EventSink(
        layout.SessionLayout(paths.state_dir(tmp_path), "review-run").logs_path
    )
    sink.emit(
        journal.VerifyEnd(
            command=["pytest"],
            exit_code=1,
            stdout_tail="",
            stderr_tail="1 failed",
            stdout_truncated=False,
            stderr_truncated=False,
            scoped=False,
            selected_paths=[],
            outcome="failed",
            duration_s=0.2,
            timeout_s=60.0,
            baseline=False,
            reason="",
        )
    )
    port, _ = evidence
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    page.get_by_role("button", name="Receipt", exact=True).click()
    text = page.locator(".receipt > pre")
    expect(text).to_contain_text("Receipt review-run")
    size, border = text.evaluate(
        "e => [getComputedStyle(e).fontSize, getComputedStyle(e).borderTopWidth]"
    )
    hangs = text.evaluate(
        """e => [...e.children].map(line => { const r = document.createRange();
        r.selectNodeContents(line); const rects = [...r.getClientRects()];
        const tops = [...new Set(rects.map(x => Math.round(x.top)))];
        const left = top => Math.min(...rects.filter(x => Math.round(x.top) === top)
            .map(x => x.left));
        return tops.length > 1 ? left(tops[1]) - left(tops[0]) : null; })
        .filter(gap => gap !== null)"""
    )
    expect(page.locator(".receipt summary")).to_have_count(1)
    assert (size, border) == ("12px", "1px")
    assert hangs
    assert all(gap == pytest.approx(2 * 7.2, abs=1.5) for gap in hangs)  # 2ch of 12px mono
    assert page.locator(".receipt summary").inner_text().endswith("stderr")
    assert not errors


def test_the_receipt_says_it_is_loading_and_its_button_hides_it_again(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Receipt shows a loading line while it fetches, and a second press hides the receipt.

    The card showed nothing until the receipt landed, seconds on a busy machine, and a second
    press fetched and drew it again. Disabling the button while it fetched dropped the keyboard
    focus, so Enter could not hide the receipt.
    """
    from playwright.sync_api import Route, expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)  # answered once the loading line is seen

    page.route("**/receipt", hold)
    button = page.get_by_role("button", name="Receipt", exact=True)
    button.focus()
    page.keyboard.press("Enter")
    expect(page.locator(".receipt")).to_have_text("loading the receipt…")
    expect(button).to_be_focused()
    while not held:
        page.wait_for_timeout(50)
    held[0].continue_()
    expect(page.locator(".receipt > pre")).to_contain_text("Receipt review-run")
    expect(button).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Enter")
    expect(page.locator(".receipt")).to_be_hidden()
    expect(button).to_have_attribute("aria-expanded", "false")
    assert not errors


def test_a_button_keeps_the_keyboard_focus_while_its_request_is_in_flight(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Submit resolution and Review run dropped the keyboard focus to the page while they posted.

    Each was disabled for its request, as Receipt was. Each still reads disabled while it posts.
    """
    from playwright.sync_api import Route, expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)  # failed once the focus is checked

    page.route("**/findings/*/resolve", hold)
    page.route("**/review-run/review", hold)
    card = page.locator(".card-findings .finding").first
    card.get_by_role("button", name="Resolve", exact=True).click()
    card.get_by_role("textbox", name="Reason").fill("the test fixture says so")
    submit = card.get_by_role("button", name="Submit resolution", exact=True)
    review = page.get_by_role("button", name="Review run", exact=True)
    for button in (submit, review):
        button.focus()
        page.keyboard.press("Enter")
        while not held:
            page.wait_for_timeout(50)
        expect(button).to_be_focused()
        expect(button).to_be_disabled()
        expect(button).to_have_css("opacity", "0.45")
        page.keyboard.press("Enter")  # a press meanwhile sends nothing
        page.wait_for_timeout(200)
        assert len(held) == 1
        held.pop().abort()
        expect(page.locator(".toast.bad").last).to_be_visible()
        expect(button).to_be_enabled()
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_a_resolved_finding_hands_the_focus_on(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """The Findings card's repaint after Submit resolution dropped the focus to the page."""
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    card = page.locator(".card-findings")
    finding = card.locator(".finding").first
    finding.get_by_role("button", name="Resolve", exact=True).click()
    finding.get_by_role("textbox", name="Reason").fill("the test fixture says so")
    submit = finding.get_by_role("button", name="Submit resolution", exact=True)
    pressed = submit.element_handle()
    submit.focus()
    page.keyboard.press("Enter")
    page.wait_for_function("b => !b.isConnected", arg=pressed)  # the card repainted
    expect(card.get_by_role("button", name="Receipt", exact=True)).to_be_focused()
    assert not errors


def test_a_streamed_review_keeps_the_focus_on_receipt(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """A frame with a new review repainted the Findings card: Receipt, put back, lost the focus."""
    from playwright.sync_api import Route, expect

    port, _ = evidence
    page, errors = open_page(port)
    streams: list[Route] = []

    def hold(route: Route) -> None:
        streams.append(route)

    page.route("**/review-run/events", hold)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    card = page.locator(".card-findings")
    receipt = card.get_by_role("button", name="Receipt", exact=True)
    painted = card.locator(".finding").first.element_handle()
    receipt.focus()
    snap = page.evaluate("getJSON('/api/session/review-run')")
    snap["review"]["version"] += 1
    while not streams:
        page.wait_for_timeout(50)
    frame = "data: " + json.dumps(snap) + "\n\n"
    streams[0].fulfill(headers={"Content-Type": "text/event-stream"}, body=frame)
    page.wait_for_function("f => !f.isConnected", arg=painted)  # the card repainted
    expect(receipt).to_be_focused()
    assert not errors


def test_a_picked_commit_says_its_diff_is_loading(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A picked commit's diff shows a loading line until it lands, then the outcome alone.

    The diff went blank until the picked commit's diff arrived.
    """
    from playwright.sync_api import Route, expect

    patch = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    rows = [
        {"type": "checkpoint.committed", "turn": 1, "sha": "1" * 40, "subject": "first"},
        {"type": "checkpoint.committed", "turn": 2, "sha": "2" * 40, "subject": "second"},
        {"type": "diff.updated", "patch": patch, "sha": "2" * 40},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    session = session_logs.write_run(tmp_path, "picked-AAA111", rows)
    (session / "manifest.json").write_text(json.dumps({"version": 3, "mode": "run"}))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/picked-AAA111")
    expect(page.locator(".card-diff .card-body")).to_contain_text("new")
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)  # answered once the loading line is seen

    page.route(f"**/diff?sha={'1' * 40}*", hold)
    page.locator(".card-diff select").select_option("1" * 40)
    body = page.locator(".card-diff .card-body > div").last
    expect(body).to_have_text("loading the diff…")
    while not held:
        page.wait_for_timeout(50)
    held[0].continue_()
    expect(body).to_have_text(f"no diff for {'1' * 12} (pruned, or not a commit of this run)")
    assert [error for error in errors if "Failed to load resource" not in error] == []


def test_a_control_tabbed_to_under_the_drawers_fold_shows_its_whole_ring(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Tab scrolls each drawer control into view with its focus ring clear of the drawer's edge.

    The drawer scrolled a focused control flush with its bottom edge and cut the ring's outer 4px.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 600})
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    expect(page.locator(".card-findings .finding").first).to_be_attached()
    page.get_by_role("button", name="Details", exact=True).focus()
    edges = []
    for _ in range(40):
        page.keyboard.press("Tab")
        edge = page.evaluate(
            """() => { const a = document.activeElement, d = a.closest('.drawer');
            if (!d) return null;
            const r = a.getBoundingClientRect(), s = getComputedStyle(a);
            const ring = parseFloat(s.outlineOffset) + parseFloat(s.outlineWidth);
            const top = d.getBoundingClientRect().top + d.clientTop;
            return [r.top - ring - top, top + d.clientHeight - r.bottom - ring, d.scrollTop]; }"""
        )
        if edge:
            edges.append(edge)
    assert max(scrolled for _, _, scrolled in edges) > 0
    assert min(min(above, below) for above, below, _ in edges) >= 0
    assert not errors


def test_details_and_resolve_say_whether_what_they_open_shows(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """Details and a finding's Resolve carry aria-expanded as the drawer or the form shows.

    Details showed the drawer's state by its colour alone, and Resolve not at all.
    """
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    details = page.get_by_role("button", name="Details", exact=True)
    drawer = page.locator(".run-app .drawer")
    expect(details).to_have_attribute("aria-expanded", "true")
    details.click()
    expect(drawer).to_be_hidden()
    expect(details).to_have_attribute("aria-expanded", "false")
    details.click()
    finding = page.locator(".card-findings .finding").first
    resolve = finding.get_by_role("button", name="Resolve", exact=True)
    expect(resolve).to_have_attribute("aria-expanded", "false")
    resolve.click()
    expect(finding.locator(".finding-form")).to_be_visible()
    expect(resolve).to_have_attribute("aria-expanded", "true")
    resolve.click()
    expect(finding.locator(".finding-form")).to_be_hidden()
    expect(resolve).to_have_attribute("aria-expanded", "false")
    assert not errors
