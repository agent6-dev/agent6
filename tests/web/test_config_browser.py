"""The Config page in a browser: the settings grid, its bar, its editors and their states."""

from __future__ import annotations

import contextlib
import pathlib
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.ui.web import server as web_server
from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page, Route

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port

READY = '[data-key="agent6.config_version"]'


@pytest.fixture
def flag_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server started with `--config FILE`: the leaf that file sets reads `flag (FILE)`."""
    flag = tmp_path / "operator-projects" / "a-monorepo-checkout" / "review-heavy-overrides.toml"
    flag.parent.mkdir(parents=True)
    flag.write_text('[sandbox]\nnetwork = "host"\n', encoding="utf-8")
    srv = web_server.WebServer(("127.0.0.1", 0), tmp_path, "", (flag,), secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


def _config(page: Page, port: int, width: int = 1280) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.locator(READY).wait_for()


def _global_config(text: str) -> pathlib.Path:
    """Write the global config file the server reads."""
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(text, encoding="utf-8")
    return config


@pytest.mark.parametrize("width", [1024, 390])
def test_the_config_bar_keeps_every_button_on_the_card(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """The filter and eight buttons shared one line that never wrapped.

    Below 1440px the last buttons sat past the card's edge, cut off, and the filter shrank to a
    sliver too narrow to show what was typed.
    """
    page, errors = open_page(port)
    _config(page, port, width)
    bar = page.locator(".cfg-bar")
    card = bar.evaluate("e => e.closest('.card').getBoundingClientRect().right")
    rights = bar.locator("button").evaluate_all(
        "buttons => buttons.map(b => b.getBoundingClientRect().right)"
    )
    assert len(rights) == 8
    assert max(rights) <= card
    assert bar.locator("input.filter").evaluate("e => e.offsetWidth") == bar.evaluate(
        "e => e.clientWidth"
    )
    assert not errors


# Per visible row: where its value starts, where its source ends, and whether the source overflows.
COLUMNS = """() => [...document.querySelectorAll('.cfg tr[data-key]')].filter(r => r.offsetParent)
  .map(r => [Math.round(r.cells[1].getBoundingClientRect().left),
             Math.round(r.cells[2].getBoundingClientRect().right),
             r.cells[2].scrollWidth > r.cells[2].clientWidth])"""


@pytest.mark.parametrize("width", [1440, 1024, 390])
def test_every_config_section_lines_its_columns_up(
    open_page: Callable[[int], tuple[Page, list[str]]], flag_port: int, width: int
) -> None:
    """Each section's table sized its columns from its own rows.

    The value and source columns started at a different x in every section and moved again
    when Modified only hid rows; a `--config FILE` source never wrapped, so it widened its
    column on a desktop and pushed a phone's rows past the card.
    """
    page, errors = open_page(flag_port)
    _config(page, flag_port, width)
    assert page.locator('[data-key="sandbox.network"] .source').inner_text().startswith("flag (")
    rows = page.evaluate(COLUMNS)
    columns = {(value, source) for value, source, _ in rows}
    assert len(columns) == 1
    assert not any(overflow for *_, overflow in rows)
    page.get_by_role("button", name="Modified only").click()
    modified = page.evaluate(COLUMNS)
    assert 0 < len(modified) < len(rows)
    assert {(value, source) for value, source, _ in modified} == columns
    assert page.evaluate("document.scrollingElement.scrollWidth") == width
    assert not errors


# The character before each line break in every visible settings cell.
BREAKS = r"""() => [...document.querySelectorAll('.cfg td')].filter(td => td.offsetParent)
.flatMap(td => {
  const walker = document.createTreeWalker(td, NodeFilter.SHOW_TEXT);
  let text = '', top = null; const before = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    for (let i = 0; i < node.length; i++) {
      const range = document.createRange(); range.setStart(node, i); range.setEnd(node, i + 1);
      const box = [...range.getClientRects()].find(rect => rect.width > 0);
      if (box && top !== null && box.top > top + 2) before.push(text.slice(-1));
      if (box) top = box.top;
      text += node.data[i];
    }
  }
  return before;
})"""


def test_a_long_config_key_or_value_wraps_between_its_parts(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Keys broke anywhere (`providers.company-gateway-eu-west-produ` over `ction.base_url`).

    Chromium also split a URL value inside a word; every break now falls after a dot, a slash,
    a hyphen or a space.
    """
    _global_config(
        "[providers.company-gateway-eu-west-production]\n"
        'backend = "openai"\n'
        'base_url = "https://llm-gateway.internal.example.com/openrouter/api/v1/chat/completions"\n'
    )
    page, errors = open_page(port)
    _config(page, port, 1024)
    before = page.evaluate(BREAKS)
    assert len(before) >= 12
    assert set(before) <= set("./- ")
    assert not errors


def test_a_config_row_lights_under_the_pointer(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A row opens its editor on a click, yet only the cursor changed under the pointer."""
    page, errors = open_page(port)
    _config(page, port)
    row = page.locator('[data-key="web.port"]')
    rest = row.evaluate("e => getComputedStyle(e).backgroundColor")
    row.hover()
    assert row.evaluate("e => getComputedStyle(e).backgroundColor") != rest
    assert not errors


def test_an_unset_effort_opens_its_editor_on_the_value_its_row_shows(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """An effort no layer sets matched none of its choices, so its editor opened on a blank select.

    The row read `"off"  (adaptive)` meanwhile.
    """
    _global_config(
        '[providers.anthropic]\nbackend = "anthropic"\n\n'
        '[models.worker]\nmodel = "anthropic/claude-sonnet-4-6"\n'
    )
    page, errors = open_page(port)
    _config(page, port)
    row = page.locator('[data-key="models.worker.effort"]')
    shown = " ".join(row.locator(".val").inner_text().split())
    assert shown == '"off" (adaptive)'
    row.click()
    select = page.get_by_role("dialog").locator("select.field")
    assert select.evaluate("e => e.selectedIndex") == 0
    option = select.locator("option").first
    assert option.inner_text() == shown
    assert option.is_disabled()
    assert not errors


def test_an_add_dialog_shows_each_field_description_whole_under_the_field(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Add provider and Add MCP server held each field's description in its placeholder.

    The placeholder was cut at the field's edge with its backticks raw, and went once typing
    started.
    """
    page, errors = open_page(port)
    _config(page, port)
    page.get_by_role("button", name="Add provider…").click()
    dialog = page.get_by_role("dialog")
    field = dialog.locator('input[name="base_url"]')
    field.fill("https://llm.example.com/v1")
    assert field.get_attribute("placeholder") is None
    described = dialog.locator(f"#{field.get_attribute('aria-describedby')}")
    assert described.is_visible()
    assert "https://api.anthropic.com/v1" in described.locator("code").all_inner_texts()
    assert "`" not in described.inner_text()
    assert described.evaluate("e => e.scrollWidth <= e.clientWidth")
    assert not errors


def test_a_long_list_value_shows_whole_in_its_editor(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A list, table or string was edited in a one-line input that showed only a long value's tail.

    Enter still saves; Shift+Enter breaks the line.
    """
    command = (
        '["sh", "-c", "uv run ruff check && uv run ruff format --check && uv run pyright'
        ' && uv run tach check && uv run pytest"]'
    )
    _global_config(f"[harness.verify]\ncommand = {command}\n")
    page, errors = open_page(port)
    _config(page, port)
    page.locator('[data-key="harness.verify.command"]').click()
    dialog = page.get_by_role("dialog")
    field = dialog.locator(".field")
    assert field.input_value() == command
    hidden = "e => [e.scrollWidth - e.clientWidth, e.scrollHeight - e.clientHeight, e.scrollLeft]"
    assert field.evaluate(hidden) == [0, 0, 0]
    field.press("Shift+Enter")
    assert field.input_value() == command + "\n"
    assert dialog.is_visible()
    field.press("Enter")
    dialog.wait_for(state="hidden")
    assert not errors


def test_the_reviewer_roster_lists_each_reviewer_with_its_remove_at_the_row_end(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The roster's rows touched, and each Remove followed its label.

    The roster ran into the field under it.
    """
    _global_config('[review]\nreviewers = [{ focus = "security" }, { focus = "correctness" }]\n')
    page, errors = open_page(port)
    _config(page, port)
    page.locator('[data-key="review.reviewers"]').click()
    dialog = page.get_by_role("dialog")
    rows = dialog.get_by_role("button", name="Remove reviewer").evaluate_all(
        """buttons => buttons.map(b => {
          const row = b.parentElement.getBoundingClientRect();
          return [row.top, row.bottom, b.getBoundingClientRect().right];
        })"""
    )
    assert len(rows) == 2
    (first_top, first_bottom, first_right), (second_top, second_bottom, second_right) = rows
    assert second_top - first_bottom >= 4
    assert first_right == second_right
    assert dialog.locator(".field").evaluate("e => e.getBoundingClientRect().top") >= (
        second_bottom + 8
    )
    assert first_top < second_top
    assert not errors


def test_a_save_keeps_the_folded_sections_and_the_edited_row_in_view(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A save re-rendered the page with every folded section open again.

    The edited row then sat far below the restored scroll position, off the screen.
    """
    page, errors = open_page(port)
    _config(page, port)
    for name in ("git", "harness"):
        page.locator("details.cfg-section > summary").get_by_text(name, exact=True).click()
    row = page.locator('[data-key="web.port"]')
    row.scroll_into_view_if_needed()
    row.click()
    field = page.get_by_role("dialog").locator(".field")
    field.fill("8799")
    field.press("Enter")
    page.wait_for_function(
        """() => document.querySelector('[data-key="web.port"] .val')?.textContent === '8799'"""
    )
    folded = page.locator("details.cfg-section:not([open]) > summary").all_inner_texts()
    assert folded == ["git", "harness"]
    row = page.locator('[data-key="web.port"]')
    assert row.evaluate("e => e.contains(document.activeElement)")
    box = row.bounding_box()
    assert box is not None
    assert box["y"] >= 0
    assert box["y"] + box["height"] <= 900
    assert not errors


def test_the_filter_opens_folded_matches_and_says_when_nothing_matches(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A match inside a folded section stayed hidden, and no match at all left a blank page.

    Clearing the filter brings the folds back.
    """
    page, errors = open_page(port)
    _config(page, port)
    page.locator("details.cfg-section > summary").get_by_text("sandbox", exact=True).click()
    search = page.locator("input.filter")
    search.fill("extra_read")
    assert page.locator('[data-key="sandbox.extra_read_paths"]').is_visible()
    none = page.locator(".cfg-none")
    assert none.inner_text() == ""
    search.fill("zzzz")
    assert none.inner_text() == "no setting matches"
    assert none.get_attribute("role") == "status"
    search.fill("")
    folded = page.locator("details.cfg-section:not([open]) > summary").all_inner_texts()
    assert folded == ["sandbox"]
    assert not errors


def test_the_model_suggestions_leave_every_button_in_reach(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The model suggestions floated over the editor's checkbox, Save and Cancel.

    A press where Save sat picked a model, and Enter in the field only closed the list.
    """
    config = _global_config(
        '[providers.a]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[models.worker]\nmodel = "a/old"\n'
    )
    page, errors = open_page(port)
    values = [f"a/model-{n}" for n in range(20)]
    page.route("**/api/config/suggest/**", lambda route: route.fulfill(json={"values": values}))
    _config(page, port)
    page.locator('[data-key="models.worker.model"]').click()
    dialog = page.get_by_role("dialog")
    field = dialog.get_by_role("combobox")
    field.click()
    field.fill("a/model-1")
    dialog.get_by_role("option").first.wait_for()
    reached = dialog.locator("button, input[type=checkbox]").evaluate_all(
        """controls => controls.map(c => {
          const r = c.getBoundingClientRect();
          return document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2) === c;
        })"""
    )
    assert len(reached) >= 3
    assert all(reached)
    field.fill("a/custom")
    field.press("Enter")
    dialog.wait_for(state="hidden")
    assert 'model = "a/custom"' in config.read_text(encoding="utf-8")
    assert not errors


# Per cell of one row: the bottom of its first character and how many lines the cell takes.
FIRST_LINES = """row => [...row.cells].map(td => {
  const first = document.createTreeWalker(td, NodeFilter.SHOW_TEXT).nextNode();
  const glyph = document.createRange(); glyph.setStart(first, 0); glyph.setEnd(first, 1);
  const whole = document.createRange(); whole.selectNodeContents(td);
  const lines = [...whole.getClientRects()].filter(r => r.width > 0);
  const bottoms = lines.map(r => Math.round(r.bottom));
  return [glyph.getBoundingClientRect().bottom, new Set(bottoms).size];
})"""


def test_a_wrapped_value_keeps_its_key_and_source_on_its_first_line(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A value that wrapped centred its key and source on its middle line."""
    _global_config(
        '[sandbox]\nextra_read_paths = ["/home/operator/.cache/uv",'
        ' "/home/operator/.local/share/uv/python",'
        ' "/opt/toolchains/rust-1.91.0-x86_64-unknown-linux-gnu"]\n'
    )
    page, errors = open_page(port)
    _config(page, port, 1024)
    (key, _), (value, lines), (source, _) = page.locator(
        '[data-key="sandbox.extra_read_paths"]'
    ).evaluate(FIRST_LINES)
    assert lines >= 3
    assert abs(key - value) <= 2
    assert abs(source - value) <= 2
    assert not errors


def test_the_code_views_rules_never_reach_the_config_source_column(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The source cell carried the Files code view's `.src` class, and so its type."""
    page, errors = open_page(port)
    _config(page, port)
    page.add_style_tag(content=".src { letter-spacing: 9px; }")
    source = page.locator('[data-key="agent6.preset"] td').last
    assert source.evaluate("e => getComputedStyle(e).letterSpacing") == "normal"
    assert source.evaluate("e => getComputedStyle(e).fontSize") == "12px"
    assert not errors


STYLE = "(e, name) => getComputedStyle(e).getPropertyValue(name)"


def test_a_config_section_name_heads_its_muted_column_labels(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A section's name was muted under each browser's own marker, its column labels bold.

    A phone stacked those labels into two more bold lines per section.
    """
    page, errors = open_page(port)
    _config(page, port)
    summary = page.locator("details.cfg-section > summary").first
    row = page.locator('[data-key="agent6.config_version"] td')
    assert summary.evaluate(STYLE, "color") == row.first.evaluate(STYLE, "color")
    assert int(summary.evaluate(STYLE, "font-weight")) >= 600
    assert summary.evaluate(STYLE, "list-style-type") == "none"
    assert page.locator(".cfg th").first.evaluate(STYLE, "color") == row.last.evaluate(
        STYLE, "color"
    )
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.locator(".cfg thead").first.evaluate("e => e.offsetHeight") <= 1
    assert not errors


def test_the_command_palette_offers_the_config_bar_actions(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The palette listed none of the config page's actions: they sat outside a toolbar."""
    page, errors = open_page(port)
    _config(page, port)
    page.keyboard.press("Control+k")
    page.get_by_role("dialog").locator("input").fill(">add provider")
    assert page.get_by_role("dialog").locator(".ac-item").all_inner_texts() == [
        "Add provider… · this page"
    ]
    page.keyboard.press("Enter")
    assert page.get_by_role("dialog").locator("h2").inner_text() == "Add provider"
    assert not errors


def test_a_long_preset_name_wraps_in_the_line_under_the_bar(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """On a phone a long preset name ran past the card's edge, cut off."""
    name = "security_correctness_and_documentation_review_panel"
    _global_config(f'[agent6]\npreset = "{name}"\n\n[presets.{name}]\n')
    page, errors = open_page(port)
    _config(page, port, 390)
    note = page.locator(".cfg-bar + .notif-banner")
    assert note.inner_text() == f"preset {name}: saves write [presets.{name}]"
    assert note.evaluate("e => e.scrollWidth <= e.clientWidth")
    assert not errors


def test_a_selected_preset_says_where_saves_go(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Only the Preset… button's hover text said a selected preset takes every save."""
    _global_config('[agent6]\npreset = "fast"\n\n[presets.fast]\n')
    page, errors = open_page(port)
    _config(page, port)
    note = page.locator(".cfg-bar + .notif-banner")
    assert note.inner_text() == "preset fast: saves write [presets.fast]"
    assert not errors


ERR = """() => {
  const probe = document.body.appendChild(document.createElement('p')); probe.className = 'err';
  const color = getComputedStyle(probe).color; probe.remove(); return color;
}"""


def test_every_config_remove_reads_as_destructive(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Remove preset, Remove provider and a reviewer's Remove looked like every other button.

    The editor's Remove provider sat between Add provider… and Cancel.
    """
    _global_config(
        '[providers.openrouter]\nbackend = "openai"\nbase_url = "https://openrouter.ai/api/v1"\n\n'
        '[review]\nreviewers = [{ focus = "security" }]\n'
    )
    page, errors = open_page(port)
    _config(page, port)
    red = page.evaluate(ERR)
    assert page.get_by_role("button", name="Remove preset").evaluate(STYLE, "color") == red
    page.locator('[data-key="providers.openrouter.base_url"]').click()
    row = page.get_by_role("dialog").locator(".form-row")
    remove = row.locator("button").last
    assert remove.inner_text() == "Remove provider"
    assert remove.evaluate(STYLE, "color") == red
    end = "e => e.parentElement.getBoundingClientRect().right - e.getBoundingClientRect().right"
    assert remove.evaluate(end) == 0
    page.keyboard.press("Escape")
    page.locator('[data-key="review.reviewers"]').click()
    reviewer = page.get_by_role("dialog").get_by_role("button", name="Remove reviewer")
    assert reviewer.evaluate(STYLE, "color") == red
    assert not errors


def test_a_config_row_keeps_its_cells_for_a_screen_reader(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Each row was one button named `edit <key>`: its value and source cells read as nothing."""
    page, errors = open_page(port)
    _config(page, port)
    cdp = page.context.new_cdp_session(page)
    nodes = cdp.send("Accessibility.getFullAXTree")["nodes"]
    named = {(n.get("role", {}).get("value"), n.get("name", {}).get("value")) for n in nodes}
    assert ("button", "edit agent6.preset") in named
    assert ("cell", '"none"') in named
    assert not errors


def test_a_config_rows_edit_button_carries_the_settings_description(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The row's edit button, which takes the focus, had no description: the row's title had it."""
    page, errors = open_page(port)
    _config(page, port)
    title = page.locator('[data-key="budget.max_usd"]').get_attribute("title")
    assert title
    assert title.startswith("cap on the metered spend of one run")
    cdp = page.context.new_cdp_session(page)
    nodes = cdp.send("Accessibility.getFullAXTree")["nodes"]
    described = {
        n["name"]["value"]: n.get("description", {}).get("value")
        for n in nodes
        if n.get("role", {}).get("value") == "button" and "name" in n
    }
    assert described["edit budget.max_usd"] == " ".join(title.split())
    assert not errors


def test_a_config_value_reads_whole_to_a_screen_reader(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Chrome read each <wbr> in a value as a space: `"https:/ / openrouter.ai/ api/ v1"`."""
    _global_config(
        '[providers.openrouter]\nbackend = "openai"\nbase_url = "https://openrouter.ai/api/v1"\n'
    )
    page, errors = open_page(port)
    _config(page, port)
    cdp = page.context.new_cdp_session(page)
    nodes = cdp.send("Accessibility.getFullAXTree")["nodes"]
    cells = {n["name"]["value"] for n in nodes if n.get("role", {}).get("value") == "cell"}
    assert '"https://openrouter.ai/api/v1"' in cells
    assert not errors


def test_a_drag_across_a_config_value_selects_it_without_opening_the_editor(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A row opened its editor on any click, so a drag never left the value selected to copy.

    A plain click, and Enter on the row's key, still open the editor.
    """
    _global_config(
        '[providers.openrouter]\nbackend = "openai"\nbase_url = "https://openrouter.ai/api/v1"\n'
    )
    page, errors = open_page(port)
    _config(page, port)
    row = page.locator('[data-key="providers.openrouter.base_url"]')
    text = row.locator(".val").evaluate(
        """e => { const r = document.createRange(); r.selectNodeContents(e);
        const b = r.getBoundingClientRect(); return [b.left, b.right, b.top + b.height / 2]; }"""
    )
    left, right, middle = text
    page.mouse.move(left + 1, middle)
    page.mouse.down()
    page.mouse.move(right - 1, middle, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    assert page.locator("dialog[open]").count() == 0
    assert "openrouter.ai/api/v1" in page.evaluate("String(getSelection())")
    row.click()
    dialog = page.get_by_role("dialog")
    assert dialog.locator("h2").inner_text() == "providers.openrouter.base_url"
    page.keyboard.press("Escape")
    dialog.wait_for(state="detached")
    row.get_by_role("button").press("Enter")
    assert dialog.locator("h2").inner_text() == "providers.openrouter.base_url"
    assert not errors


def test_a_held_enter_on_a_config_row_saves_nothing(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A held Enter opened a row's editor, then its first repeat pressed Save.

    The unchanged default became an explicit setting in the global config.
    """
    page, errors = open_page(port)
    posts: list[str] = []

    def refuse(route: Route) -> None:
        if route.request.method != "POST":
            route.fallback()
            return
        posts.append(route.request.post_data or "")
        route.fulfill(status=400, json={"error": "refused by the test"})

    page.route("**/api/config**", refuse)
    _config(page, port)
    page.locator('[data-key="web.port"]').get_by_role("button").focus()
    page.keyboard.down("Enter")
    dialog = page.get_by_role("dialog")
    dialog.wait_for()
    page.keyboard.down("Enter")  # the held key's first repeat
    page.wait_for_timeout(300)
    page.keyboard.up("Enter")
    assert posts == []
    page.keyboard.press("Enter")
    dialog.get_by_role("alert").get_by_text("refused").wait_for()
    assert len(posts) == 1
    assert not [error for error in errors if "400 (Bad Request)" not in error]


def test_enter_in_any_config_dialog_field_presses_its_add_or_save(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Enter in Add provider, Add MCP server, Add skill or Add reviewer did nothing.

    Add preset and the editors took it; Enter on a model suggestion picks it and sends nothing.
    """
    from playwright.sync_api import TimeoutError as Unanswered

    _global_config(
        '[providers.a]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[models.worker]\nmodel = "a/old"\n'
    )
    page, errors = open_page(port)
    posts: list[str] = []

    def refuse(route: Route) -> None:
        if route.request.method != "POST":
            route.fallback()
            return
        posts.append(route.request.url)
        route.fulfill(status=400, json={"error": "refused by the test"})

    page.route("**/api/config**", refuse)
    page.route("**/api/config/suggest/**", lambda route: route.fulfill(json={"values": ["a/m"]}))
    _config(page, port)
    sent = {}
    opens = {name: page.get_by_role("button", name=name) for name in ADD_DIALOGS} | {
        key: page.locator(f'[data-key="{key}"]') for key in EDITORS
    }
    for name, opener in opens.items():
        opener.click()
        dialog = page.get_by_role("dialog")
        field = dialog.locator("input:not([type=checkbox]), select, textarea").first
        if name in ADD_DIALOGS:
            field.fill("x")
        before = len(posts)
        field.press("Enter")
        with contextlib.suppress(Unanswered):
            dialog.get_by_role("alert").get_by_text("refused").wait_for(timeout=3000)
        sent[name] = len(posts) - before
        page.keyboard.press("Escape")
        dialog.wait_for(state="detached")
    assert sent == dict.fromkeys(opens, 1)
    page.locator('[data-key="models.worker.model"]').click()
    model = page.get_by_role("dialog").get_by_role("combobox")
    model.press("ArrowDown")
    model.press("Enter")
    assert model.input_value() == "a/m"
    assert len(posts) == len(opens)
    assert not [error for error in errors if "400 (Bad Request)" not in error]


ADD_DIALOGS = ("Add provider…", "Add MCP server…", "Add skill…", "Add reviewer…", "Add preset…")
EDITORS = ("budget.max_usd", "sandbox.isolation", "harness.verify.command", "models.worker.model")


def test_a_config_editor_names_its_value_field_by_the_key(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The editor's value field had no name: a screen reader read a bare `10.0`."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    _config(page, port)
    for key in ("budget.max_usd", "sandbox.isolation", "harness.verify.command"):
        page.locator(f'[data-key="{key}"]').click()
        dialog = page.get_by_role("dialog", name=key)
        field = dialog.locator("input:not([type=checkbox]), select, textarea").first
        expect(field).to_have_accessible_name(key)
        page.keyboard.press("Escape")
        dialog.wait_for(state="detached")
    assert not errors


def test_a_choice_editor_shows_the_chosen_meaning_whole_on_a_phone(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Each option read `value: meaning`, cut at the select's edge on a phone."""
    page, errors = open_page(port)
    _config(page, port, 390)
    page.locator('[data-key="sandbox.isolation"]').click()
    dialog = page.get_by_role("dialog")
    select = dialog.locator("select")
    assert select.locator("option").all_inner_texts() == [
        "auto",
        "full",
        "partial",
        "none",
        "container",
    ]
    meaning = dialog.locator(f"#{select.get_attribute('aria-describedby')}")
    assert meaning.inner_text() == "Strongest supported isolation, warning below full"
    assert meaning.evaluate("e => e.scrollWidth <= e.clientWidth")
    select.select_option("none")
    assert meaning.inner_text() == "Run commands without confinement"
    assert not errors


FONT = "e => { const s = getComputedStyle(e); return [s.fontFamily, s.fontSize]; }"


def test_a_config_editor_shows_its_value_in_the_tables_type(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A number, a choice or a model id switched from the table's monospace to the page's sans."""
    _global_config(
        '[providers.a]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[models.worker]\nmodel = "a/old"\n'
    )
    page, errors = open_page(port)
    _config(page, port)
    table = page.locator('[data-key="budget.max_usd"] .val').evaluate(FONT)
    fonts = {}
    for key in (
        "budget.max_usd",
        "sandbox.isolation",
        "harness.verify.command",
        "models.worker.model",
    ):
        page.locator(f'[data-key="{key}"]').click()
        dialog = page.get_by_role("dialog")
        fonts[key] = dialog.locator("input:not([type=checkbox]), select, textarea").first.evaluate(
            FONT
        )
        page.keyboard.press("Escape")
        dialog.wait_for(state="detached")
    assert fonts == dict.fromkeys(fonts, table)
    assert not errors


def test_only_the_arrow_keys_model_suggestion_reads_as_active(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The option under the pointer wore the active option's accent, so two read as chosen."""
    _global_config(
        '[providers.a]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[models.worker]\nmodel = "a/old"\n'
    )
    page, errors = open_page(port)
    values = [f"a/model-{n}" for n in range(6)]
    page.route("**/api/config/suggest/**", lambda route: route.fulfill(json={"values": values}))
    _config(page, port)
    page.locator('[data-key="models.worker.model"]').click()
    dialog = page.get_by_role("dialog")
    field = dialog.get_by_role("combobox")
    field.fill("")
    field.press("ArrowDown")
    field.press("ArrowDown")
    options = dialog.get_by_role("option")
    options.nth(4).hover()
    colors = options.evaluate_all("os => os.map(o => getComputedStyle(o).color)")
    accent = options.nth(1).evaluate(STYLE, "color")
    assert [n for n, color in enumerate(colors) if color == accent] == [1]
    lit = options.evaluate_all("os => os.map(o => getComputedStyle(o).backgroundColor)")
    assert lit[4] == lit[1] != lit[0]
    assert not errors
