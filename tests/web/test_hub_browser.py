"""The Sessions and Machines lists driven in a browser: rows, their refresh and the composer."""

from __future__ import annotations

import json
import pathlib
import subprocess
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.unit import test_sessions_show_group
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
group = test_sessions_show_group.group

STOP: dict[str, object] = {
    "type": "session.stop",
    "reason": "finish_session",
    "verification": "passed",
}
# Keyboard focus on a row: its title link, or the row where the row is the control.
FOCUS_ROW = "e => (e.querySelector('a[href]') || e).focus()"


def _run(
    repo: pathlib.Path, sid: str, task: str, *events: dict[str, object], mode: str = "run"
) -> pathlib.Path:
    """Record a session in `mode` that started on `task`, then `events`; return its journal."""
    run = paths.state_dir(repo) / "sessions" / f"{mode}s" / sid
    run.mkdir(parents=True, exist_ok=True)
    log = run / "logs.jsonl"
    start = {"type": "session.start", "mode": mode, "user_task": task}
    log.write_text("".join(json.dumps(e) + "\n" for e in (start, *events)), encoding="utf-8")
    return log


def _manifest(repo: pathlib.Path, sid: str, **fields: object) -> None:
    """Write a run's manifest with `fields`."""
    run = paths.state_dir(repo) / "sessions" / "runs" / sid
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 3, "session_id": sid, "mode": "run", **fields}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _refresh(page: Page) -> None:
    """Run the hub's list refresh now, as a tab coming back into view does."""
    with page.expect_response(lambda r: r.url.endswith("/api/hub")):
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_timeout(300)


def test_a_refresh_keeps_the_focus_a_selection_and_a_press_in_the_list(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The 4 s refresh rebuilt every row: the focus fell to the page, a selection and a click died.

    A row whose data changed is painted afresh and keeps the focus on the same control.
    """
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    beta = _run(tmp_path, "beta-BBBBBB", "Beta task")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    rows = page.locator(".item")
    alpha, other = rows.filter(has_text="alpha-AAAAAA"), rows.filter(has_text="beta-BBBBBB")
    expect(other).to_contain_text("Beta task")
    focused = "() => document.activeElement.closest('.item')?.textContent || ''"
    alpha.evaluate(FOCUS_ROW)
    beta.write_text(beta.read_text() + json.dumps(STOP) + "\n", encoding="utf-8")
    _refresh(page)
    expect(other).to_contain_text("passed")
    assert "alpha-AAAAAA" in page.evaluate(focused)
    # The focused row itself changes: its fresh paint takes the focus.
    other.evaluate(FOCUS_ROW)
    spent = {"type": "budget.update", "usd_total": 1.5}
    beta.write_text(beta.read_text() + json.dumps(spent) + "\n", encoding="utf-8")
    _refresh(page)
    expect(other).to_contain_text("$1.50")
    assert "beta-BBBBBB" in page.evaluate(focused)
    # A selection waits out the refresh.
    page.evaluate("""() => { const r = document.createRange();
      r.selectNodeContents([...document.querySelectorAll('.item .sub')].at(-1));
      getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    selected = page.evaluate("String(getSelection())")
    _run(tmp_path, "gamma-CCCCCC", "Gamma task")
    _refresh(page)
    assert page.evaluate("String(getSelection())") == selected
    # A press held across a refresh still opens its row.
    box = alpha.bounding_box()
    assert box
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    _refresh(page)
    page.mouse.up()
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/session/alpha-AAAAAA")
    assert not errors


@pytest.fixture
def fan_port(group: pathlib.Path) -> Iterator[int]:
    """A server over a fan-out group, `fan`, with two children."""
    yield from test_web_browser._serve(group)  # pyright: ignore[reportPrivateUsage]


def test_a_drag_that_selects_text_in_a_row_keeps_the_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A drag that selected a row's id opened the session, so no id in the list could be copied."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    sub = page.locator(".item .sub").first
    expect(sub).to_contain_text("alpha-AAAAAA")
    box = sub.bounding_box()
    assert box
    page.mouse.move(box["x"] + 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + 160, box["y"] + box["height"] / 2, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    assert page.evaluate("String(getSelection())")
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/")
    page.mouse.click(box["x"] + 40, box["y"] + box["height"] / 2)  # a plain click still opens it
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/session/alpha-AAAAAA")
    assert not errors


def test_a_groups_children_are_rows_of_their_own(
    open_page: Callable[[int], tuple[Page, list[str]]], fan_port: int
) -> None:
    """Expanded children sat inside the group's text column, nested in its row.

    The group's status floated to the middle of the group, hovering a child lit the group, and
    Enter on a child opened the group. Each row is now a title link beside its controls, never a
    button holding buttons.
    """
    from playwright.sync_api import expect

    page, errors = open_page(fan_port)
    page.goto(f"http://127.0.0.1:{fan_port}/#/")
    parent = page.locator(".item").filter(has_text="fan · ")
    expect(parent).to_have_count(1)
    expect(page.locator(".list [role=button]")).to_have_count(0)
    expect(parent.get_by_role("link")).to_have_attribute("href", "#/session/fan")
    parent.get_by_role("button", name="children: 2 · show").click()
    child = page.locator(".item").filter(has_text="fan-l1")
    expect(child).to_be_visible()
    above, under = parent.bounding_box(), child.bounding_box()
    assert above and under
    assert above["y"] + above["height"] <= under["y"], (above, under)
    child.hover()
    lit = "e => getComputedStyle(e).backgroundColor"
    assert parent.evaluate(lit) != child.evaluate(lit)
    child.get_by_role("link").focus()
    page.keyboard.press("Enter")
    expect(page).to_have_url(f"http://127.0.0.1:{fan_port}/#/session/fan-l1")
    assert not errors


def test_a_rows_status_leads_its_sub_line_and_its_text_keeps_the_width(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A row's status and buttons sat beside its text, so a phone left the text a sliver.

    The dots stood wherever each label's length put them, and a squeezed sub line broke inside
    an id. The status now leads the first sub line and the buttons wrap under the text.
    """
    from playwright.sync_api import expect

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "master")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.invalid")
    git("commit", "-q", "--allow-empty", "-m", "base")
    git("branch", "agent6/review-BBBBBB")
    reason = "the checkout has uncommitted changes"
    _manifest(tmp_path, "parked-AAAAAA", parked_task="Bump textual", parked_reason=reason)
    _manifest(tmp_path, "review-BBBBBB", run_branch="agent6/review-BBBBBB")
    _run(tmp_path, "review-BBBBBB", "Review me", STOP)
    _run(tmp_path, "short-CCCCCC", "Short", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    rows = page.locator(".item")
    expect(rows).to_have_count(3)
    expect(rows.filter(has_text="parked-AAAAAA")).to_contain_text(reason)
    expect(rows.get_by_role("button", name="Review")).to_have_count(1)
    lefts = "ps => ps.map(p => Math.round(p.getBoundingClientRect().left))"
    assert len(set(page.locator(".item .pill").evaluate_all(lefts))) == 1
    page.set_viewport_size({"width": 390, "height": 844})
    text = "rows => rows.map(r => r.querySelector('.grow').getBoundingClientRect().width)"
    assert min(rows.evaluate_all(text)) >= 300
    # Each id sits on one line: the range over it draws a single box.
    whole = """rows => rows.map(r => {
      const sub = r.querySelector('.sub'), id = r.querySelector('a.title').hash.split('/').pop();
      const walk = document.createTreeWalker(sub, NodeFilter.SHOW_TEXT);
      for (let n = walk.nextNode(); n; n = walk.nextNode()) {
        const at = n.data.indexOf(id);
        if (at < 0) continue;
        const range = document.createRange();
        range.setStart(n, at); range.setEnd(n, at + id.length);
        return [...range.getClientRects()].filter(b => b.width).length;
      }
      return 0;
    })"""
    assert rows.evaluate_all(whole) == [1, 1, 1]
    assert not errors


def test_space_opens_a_focused_row_as_enter_does(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Space on a row's title link scrolled the page, though `?` lists Space as opening a row."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".item a.title").first.focus()
    page.keyboard.press(" ")
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/session/alpha-AAAAAA")
    assert not errors


def test_a_repeated_start_or_prune_sends_one_request(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Enter twice while a Start was in flight started two paid sessions; Prune sent twice too."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    for url in ("**/api/new", "**/api/sessions/prune"):
        page.route(url, hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    task = page.locator(".dock textarea")
    task.fill("probe")
    task.press("Enter")
    task.press("Enter")
    page.get_by_role("button", name="Prune merged runs", exact=True).dblclick()
    page.wait_for_timeout(500)
    assert [r.request.url.rsplit("/", 1)[1] for r in held] == ["new", "prune"]
    for route in held:
        route.fulfill(json={"ok": False, "error": "held"})
    expect(page.locator(".toast")).to_have_count(2)
    expect(page.get_by_role("button", name="Start", exact=True)).to_be_enabled()
    assert not errors


def test_the_folds_say_whether_they_are_open_and_keep_the_focus(
    open_page: Callable[[int], tuple[Page, list[str]]], fan_port: int
) -> None:
    """The children toggle was a 40px button with no open state, and Enter on it lost the focus.

    The advanced fold was an 18px line on the entry's top edge, and its spend cap stood taller
    than the plan dropdown beside it.
    """
    from playwright.sync_api import expect

    page, errors = open_page(fan_port)
    page.goto(f"http://127.0.0.1:{fan_port}/#/")
    height = "e => e.getBoundingClientRect().height"
    show = page.get_by_role("button", name="children: 2 · show")
    expect(show).to_have_attribute("aria-expanded", "false")
    show.focus()
    page.keyboard.press("Enter")
    hide = page.get_by_role("button", name="children: 2 · hide")
    expect(hide).to_be_focused()
    expect(hide).to_have_attribute("aria-expanded", "true")
    assert 24 <= hide.evaluate(height) < 32
    advanced = page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True)
    expect(advanced).to_have_attribute("aria-expanded", "false")
    advanced.click()
    expect(advanced).to_have_attribute("aria-expanded", "true")
    fields = page.locator(".dock select[title^='seed the run'], .dock input[type=number]")
    expect(fields).to_have_count(2)
    boxes = fields.evaluate_all("es => es.map(e => [e.getBoundingClientRect().y, e.offsetHeight])")
    assert boxes[0] == boxes[1], boxes
    assert not errors


def test_a_disabled_picker_reads_disabled_with_its_label(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A plan greys worktree and from plan: Firefox drew both, and both labels, as enabled."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".dock select").first.select_option("plan")
    worktree = page.locator(".dock input[type=checkbox]")
    expect(worktree).to_be_disabled()
    opacity = "es => es.map(e => +getComputedStyle(e).opacity)"
    labels = page.locator(".dock .pick", has=page.locator(":disabled")).locator("> span")
    expect(labels).to_have_text(["worktree", "from plan"])
    assert max(worktree.evaluate_all(opacity) + labels.evaluate_all(opacity)) < 1
    assert not errors


def test_a_suggestion_popup_opens_on_its_entry_at_the_entrys_width(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The directive popup opened 10px left of the entry and over the pickers, cut at 480px."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    entry.fill("/")
    popup = page.locator(".ac-pop")
    expect(popup).to_contain_text("/parallel")
    above, under = popup.bounding_box(), entry.bounding_box()
    assert above and under
    assert above["x"] == under["x"], (above, under)
    assert 0 <= under["y"] - (above["y"] + above["height"]) <= 8, (above, under)
    whole = "rows => rows.every(r => r.scrollWidth <= r.clientWidth)"
    assert popup.locator(".ac-item").evaluate_all(whole)
    assert not errors


def test_a_failed_first_load_offers_retry_and_recovers_on_its_own(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A failed first load showed one red line and stayed so after the server answered again."""
    from playwright.sync_api import Route, expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)

    def fail(route: Route) -> None:
        route.fulfill(status=500, json={"error": "hub listing failed: probe"})

    page.route("**/api/hub", fail)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("main")).to_contain_text("hub listing failed: probe")
    expect(page.get_by_role("button", name="Retry")).to_be_visible()
    page.unroute("**/api/hub", fail)
    expect(page.locator(".item")).to_contain_text("Alpha task", timeout=10000)
    errors[:] = [e for e in errors if "500" not in e]  # the failed loads the test answered
    assert not errors


@pytest.mark.parametrize(("width", "height"), [(1280, 900), (390, 844)])
def test_the_hubs_text_starts_at_one_left_edge(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
    height: int,
) -> None:
    """The toolbar's text, the rows' text and the composer started at three left edges."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".item")).to_have_count(1)
    left = """sel => {
      const e = document.querySelector(sel), range = document.createRange();
      const text = [...e.childNodes].find(n => n.nodeType === 3 && n.data.trim());
      if (!text) return Math.round(e.getBoundingClientRect().left);
      range.selectNodeContents(text);
      return Math.round(range.getBoundingClientRect().left);
    }"""
    places = [".toolbar button", ".item a.title", ".dock .pick > span", ".dock textarea"]
    assert len({page.evaluate(left, sel) for sel in places}) == 1
    assert not errors


def test_a_selection_holds_back_only_its_own_row(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A text selection anywhere in the list held back the whole refresh, so other rows went stale.

    The row holding the selection keeps its paint until the selection goes.
    """
    from playwright.sync_api import expect

    alpha = _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    beta = _run(tmp_path, "beta-BBBBBB", "Beta task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    rows = page.locator(".item")
    first, second = rows.filter(has_text="alpha-AAAAAA"), rows.filter(has_text="beta-BBBBBB")
    expect(second).to_contain_text("Beta task")
    second.locator(".sub").first.evaluate(
        """e => { const r = document.createRange(); r.selectNodeContents(e);
        getSelection().removeAllRanges(); getSelection().addRange(r); }"""
    )
    for log, usd in ((alpha, 1.5), (beta, 2.5)):
        spent = {"type": "budget.update", "usd_total": usd}
        log.write_text(log.read_text() + json.dumps(spent) + "\n", encoding="utf-8")
    _refresh(page)
    expect(first).to_contain_text("$1.50")
    expect(second).not_to_contain_text("$2.50")
    assert "beta-BBBBBB" in page.evaluate("String(getSelection())")
    page.evaluate("getSelection().removeAllRanges()")
    _refresh(page)
    expect(second).to_contain_text("$2.50")
    assert not errors


def test_a_picker_is_as_wide_as_its_chosen_option(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The 'from plan' dropdown took its longest plan's width, 460px, to show 'none'."""
    from playwright.sync_api import expect

    task = "Plan moving the argument parsing onto subcommand modules, one module per verb"
    _run(tmp_path, "plan-AAAAAA", task, STOP, mode="plan")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True).click()
    seed = page.locator(".dock select[title^='seed the run']")
    expect(seed.locator("option")).to_have_count(2)
    width = "e => e.getBoundingClientRect().width"
    resting = seed.evaluate(width)
    seed.select_option("plan-AAAAAA")
    assert resting < 100 < seed.evaluate(width), resting
    assert not errors


def test_a_model_list_that_fails_to_load_leaves_the_config_default(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A failed /api/routes left the model dropdown reading "…", as if it were still resolving."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.route("**/api/routes*", lambda route: route.abort())
    page.goto(f"http://127.0.0.1:{port}/#/")
    model = page.locator(".dock select[title^='the model']")
    expect(model).to_be_enabled()
    expect(model.locator("option")).to_have_text(["config default"])
    expect(page.locator(".toast.bad")).to_contain_text("could not list the models: ")
    errors[:] = [e for e in errors if "ERR_FAILED" not in e]  # the request the test aborted
    assert not errors


def test_the_model_picker_says_it_is_resolving_until_its_routes_arrive(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The model dropdown painted empty and 40px wide until /api/routes answered."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/routes*", hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    model = page.locator(".dock select[title^='the model']")
    expect(model.locator("option")).to_have_text(["…"])
    expect(model).to_be_disabled()
    for route in held:
        route.continue_()
    expect(model).to_be_enabled()
    expect(model.locator("option").first).to_contain_text("(")
    assert not errors


def test_start_and_create_machine_wait_for_something_to_send(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Start on an empty entry and Create machine with no description read enabled, did nothing."""
    from playwright.sync_api import expect

    _run(tmp_path, "plan-AAAAAA", "Plan it", STOP, mode="plan")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    start = page.get_by_role("button", name="Start", exact=True)
    entry = page.locator(".dock textarea")
    expect(start).to_be_disabled()
    entry.fill("probe")
    expect(start).to_be_enabled()
    entry.fill("  ")
    expect(start).to_be_disabled()
    page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True).click()
    page.locator(".dock select[title^='seed the run']").select_option("plan-AAAAAA")
    expect(start).to_be_enabled()  # a run from a plan needs no task
    page.locator(".dock select").first.select_option("plan")
    expect(start).to_be_disabled()
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    create = page.get_by_role("button", name="Create machine")
    expect(create).to_be_disabled()
    page.locator(".dock textarea").fill("watch CI")
    expect(create).to_be_enabled()
    assert not errors


def test_review_fills_the_entry_on_screen_and_focuses_it(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Review fetched and repainted the whole hub to fill the entry, then left the focus nowhere."""
    from playwright.sync_api import Request, expect

    for args in (
        ("init", "-q", "-b", "master"),
        ("config", "user.name", "t"),
        ("config", "user.email", "t@example.invalid"),
        ("commit", "-q", "--allow-empty", "-m", "base"),
        ("branch", "agent6/review-BBBBBB"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    _manifest(tmp_path, "review-BBBBBB", run_branch="agent6/review-BBBBBB")
    _run(tmp_path, "review-BBBBBB", "Review me", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    entry.fill("a draft")
    page.evaluate("clearInterval(hubTimer)")  # only the click may fetch the hub
    fetched: list[str] = []

    def hub(request: Request) -> None:
        if request.url.endswith("/api/hub"):
            fetched.append(request.url)

    page.on("request", hub)
    page.get_by_role("button", name="Review").click()
    expect(entry).to_have_value("/review agent6/review-BBBBBB")
    expect(entry).to_be_focused()
    assert not fetched
    assert not errors


def test_a_missing_worktree_is_named_once_in_its_row(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A run whose worktree is gone printed the worktree's full path twice in its row."""
    from playwright.sync_api import expect

    gone = tmp_path / "worktrees" / "gone-AAAAAA"
    _manifest(tmp_path, "gone-AAAAAA", worktree=str(gone))
    _run(tmp_path, "gone-AAAAAA", "Gone task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    row = page.locator(".item")
    expect(row).to_contain_text("not found; resume needs it")
    assert row.inner_text().count(str(gone)) == 1
    assert not errors
