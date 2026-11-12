# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The rendered web UI driven in a browser: the forms and views the HTTP tests cannot reach.

Marked `browser`: Playwright's Chromium runs them, and they skip where it is not installed.
"""

from __future__ import annotations

import json
import pathlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.ui.web import server as web_server
from tests.unit.test_git_history import _git, build_history, make_dirty
from tests.web import signin

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """One Chromium for the module."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


def _serve(cwd: pathlib.Path) -> Iterator[int]:
    srv = web_server.WebServer(("127.0.0.1", 0), cwd, "", secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server on an ephemeral port, serving from an empty directory."""
    yield from _serve(tmp_path)


@pytest.fixture
def git_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server reading a repository with a merge, a rename, a binary file and tags."""
    build_history(tmp_path)
    yield from _serve(tmp_path)


@pytest.fixture
def dirty_git_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server reading that repository with five uncommitted files, the PNG among them."""
    build_history(tmp_path)
    make_dirty(tmp_path)
    yield from _serve(tmp_path)


@pytest.fixture
def files_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """The dirty repository with an untracked Python file under `src/`."""
    build_history(tmp_path)
    make_dirty(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tool.py").write_text('def f(x):\n    return "hi"\n', encoding="utf-8")
    yield from _serve(tmp_path)


@pytest.fixture
def open_page(browser: Browser) -> Iterator[Callable[[int], tuple[Page, list[str]]]]:
    """Open a signed-in desktop page on a server's port, collecting console and page errors."""
    contexts: list[BrowserContext] = []

    def _open(port: int) -> tuple[Page, list[str]]:
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        contexts.append(context)
        token = json.dumps(signin.token(port))
        context.add_init_script(f"localStorage.setItem('a6-token', {token})")
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        return page, errors

    try:
        yield _open
    finally:
        for context in contexts:
            context.close()


def test_resume_composer_keeps_the_recorded_model(
    port: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The resume composer names the recorded model and offers the current config route."""
    from playwright.sync_api import expect

    config = tmp_path / "config" / "agent6"
    config.mkdir(parents=True)
    (config / "config.toml").write_text(
        '[providers.o]\nbackend = "openai"\nbase_url = "https://x/v1"\n'
        '[models.worker]\nmodel = "o/configured"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config.parent))
    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "recorded1"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session.name,
                "mode": "run",
                "parked_task": "continue",
                "models": {"driver": {"provider": "o", "model": "recorded"}},
            }
        ),
        encoding="utf-8",
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    model = page.locator(".composer select").nth(1)
    expect(model).to_be_visible()
    expect(model.locator("option:checked")).to_have_text("o/recorded (as recorded)")
    expect(model.locator('option[value="o/configured"]')).to_have_count(1)
    expect(model).to_have_value("")
    model.select_option("o/configured")
    expect(model).to_have_value("o/configured")
    assert not errors


@pytest.mark.parametrize("unreadable", [False, True])
def test_unreviewed_run_names_the_absent_panel(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    unreadable: bool,
) -> None:
    """An unreviewed run says no panel ran unless its journal cannot be read."""
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "unreviewed"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "No review"})
        + "\n"
        + (
            json.dumps({"type": "loop.review.panel", "panel_id": "broken", "target": {}}) + "\n"
            if unreadable
            else ""
        )
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    expected = "Review: unknown · freshness: unknown" if unreadable else "Review: no panel ran"
    expect(page.get_by_text(expected, exact=True)).to_have_count(1)
    if not unreadable:
        expect(page.get_by_text("Review: unknown · freshness: unknown", exact=True)).to_have_count(
            0
        )
    assert not errors


def test_session_details_label_the_lifecycle_status(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The session details label the lifecycle result status, not machine state."""
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "status-run-AAA111"
    session.mkdir(parents=True)
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "Show the status"},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    label = page.locator(".kv .k").filter(has_text=re.compile(r"^status$"))
    expect(label).to_be_visible()
    expect(label.locator("+ .v")).to_have_text("passed")
    expect(page.locator(".kv .k").filter(has_text=re.compile(r"^state$"))).to_have_count(0)
    assert not errors


def test_config_model_editor_saves_one_qualified_leaf(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Model editor has one route field and saves only model, preserving sampling."""
    import tomllib

    from playwright.sync_api import expect

    from agent6.models import cache

    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.p]\nbackend = "openai"\n[models.worker]\n'
        'model = "p/old"\neffort = "high"\ntemperature = 0.4\n'
    )

    def listing(*_args: object, **_kwargs: object) -> list[str]:
        return ["vendor/new"]

    monkeypatch.setattr(cache, "list_models", listing)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name="edit models.worker.model", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog.locator("input:not([type=checkbox])")).to_have_count(1)
    expect(dialog.locator("select")).to_have_count(0)
    field = dialog.get_by_role("combobox")
    expect(field).to_have_value("p/old")
    field.fill("p/vendor")
    page.get_by_role("option", name="p/vendor/new", exact=True).click()
    # The response waits for the config CLI; the dialog closes before the row refresh.
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/api/config")
    ) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()
    assert saved.value.request.post_data_json == {
        "key": "models.worker.model",
        "value": '"p/vendor/new"',
        "repo": False,
    }
    expect(page.get_by_role("button", name="edit models.worker.model", exact=True)).to_contain_text(
        "p/vendor/new"
    )
    expect(dialog).to_have_count(0)
    assert tomllib.loads(config.read_text())["models"]["worker"] == {
        "model": "p/vendor/new",
        "effort": "high",
        "temperature": 0.4,
    }
    assert not errors


def test_expanded_task_keeps_the_full_operator_words(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A task headline expands to every line, including literal markup and its long tail."""
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "whole-task-AAA111"
    session.mkdir(parents=True)
    words = "Keep <file>this example</file>. " + "full words " * 600 + "\nLAST PROMPT LINE"
    steer = "Continue " + "steer words " * 600 + "\nLAST STEER LINE"
    rows = [
        {"type": "session.start", "mode": "run", "user_task": words},
        {"type": "loop.steer.injected", "text": steer},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    item = page.locator(".ci").filter(has=page.locator(".s-task")).first
    expect(item).to_have_attribute("title", "click to expand")
    assert "LAST PROMPT LINE" not in item.inner_text()
    item.click()
    expect(item).to_contain_text("LAST PROMPT LINE")
    assert words in item.inner_text().replace("\n  ", "\n")
    item = page.locator(".ci").filter(has=page.locator(".s-operator")).first
    expect(item).to_have_attribute("title", "click to expand")
    item.click()
    expect(item).to_contain_text("LAST STEER LINE")
    assert steer in item.inner_text().replace("\n  ", "\n")
    assert not errors


def test_verification_waits_runs_and_finishes_in_one_transcript_item(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """A live run blocked on baseline approval shows its verification before any model call."""
    import os

    from playwright.sync_api import expect

    from agent6.sessions import process
    from tests.unit import test_transcript_verify

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "verify-wait-AAA111"
    session.mkdir(parents=True)
    process.write_worker_pid(session, os.getpid())
    rows = test_transcript_verify.attempt()
    log = session / "logs.jsonl"
    log.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "Check the baseline"})
        + "\n"
        + "".join(json.dumps(row) + "\n" for row in rows[:2])
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    item = page.locator(".ci").filter(has_text="verify:")
    expect(item).to_have_count(1)
    expect(item).to_contain_text("verify: waiting for approval")
    for row, text in [
        (rows[2], "verify: running `pytest -q`"),
        (rows[3], "verify: ✓ pass · 0.2s"),
    ]:
        with log.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        expect(item).to_have_count(1)
        expect(item).to_contain_text(text)
    assert not errors


@pytest.mark.parametrize("expanded", [False, True])
def test_every_tool_expands_to_whole_arguments(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    *,
    expanded: bool,
) -> None:
    """Long commands and no-argument calls expand and collapse at either global detail level."""
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "whole-tool-AAA111"
    session.mkdir(parents=True)
    words = "script " * 900 + "LAST SCRIPT BYTE"
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "Show commands"},
        {"type": "tool.call", "name": "run_command", "args_preview": {"argv": ["sh", "-c", words]}},
        {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"},
        {"type": "tool.call", "name": "run_verify_command", "args_preview": {}},
        {"type": "tool.result", "name": "run_verify_command", "ok": True, "summary": "exit 0"},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    level = "expanded" if expanded else "collapsed"
    page.add_init_script(f"localStorage.setItem('a6-detail', '{level}')")
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    command = page.locator(".ci").filter(has=page.locator(".s-call")).first
    expect(command).to_have_attribute(
        "title", "click to collapse" if expanded else "click to expand"
    )
    if expanded:
        command.click()
    assert "LAST SCRIPT BYTE" not in command.inner_text()
    command.click()
    expect(command).to_contain_text("LAST SCRIPT BYTE")
    assert words in command.inner_text()
    command.click()
    assert "LAST SCRIPT BYTE" not in command.inner_text()
    verify = page.locator(".ci").filter(has=page.locator(".s-verify")).first
    expect(verify).to_have_class(re.compile(r"\bexp\b"))
    if expanded:
        verify.click()
    verify.click()
    expect(verify).to_contain_text("arguments: (none)")
    assert not errors


@pytest.mark.parametrize("asks", [False, True])
def test_removal_controls_reach_the_cli(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
    asks: bool,
) -> None:
    """Delete and Clear saved asks invoke the accepted CLI verb and remove the saved record."""
    from playwright.sync_api import expect

    session_id = "remove-run-AAAAAA"
    mode = "ask" if asks else "run"
    bucket = paths.state_dir(tmp_path) / "sessions" / ("asks" if asks else "runs")
    session = bucket / session_id
    session.mkdir(parents=True)
    rows = [
        {"type": "session.start", "mode": mode, "user_task": "Remove this record"},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(git_port)
    route = "#/" if asks else f"#/session/{session_id}"
    page.goto(f"http://127.0.0.1:{git_port}/{route}")
    button = page.get_by_role("button", name="Clear saved asks" if asks else "Delete", exact=True)
    expect(button).to_be_enabled()
    page.once("dialog", lambda dialog: dialog.accept())
    endpoint = "/api/sessions/rm_asks" if asks else f"/api/session/{session_id}/rm"
    with page.expect_response(lambda response: response.url.endswith(endpoint)) as response:
        button.click()
    assert response.value.ok, response.value.text()
    assert not session.exists()
    assert not errors


def test_the_provider_form_writes_the_chosen_deployment(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The rendered add-provider form posts the deployment the operator picked.

    The HTTP tests post hand-written payloads; this drives the form the browser builds from
    the entry-table registry, so a field the form drops or misnames shows here.
    """
    cfg_path = paths.global_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        '[providers.az]\nbackend = "openai"\ndeployment = "direct"\n'
        'base_url = "https://res.openai.azure.com"\n'
        'extra_query = { "api-version" = "2024-10-21" }\n',
        encoding="utf-8",
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name=re.compile(r"^Add provider", re.I)).click()
    dialog = page.get_by_role("dialog")
    dialog.locator("input[name='name']").fill("az")
    dialog.locator("select[name='backend']").select_option("openai")
    dialog.locator("select[name='deployment']").select_option("azure")
    dialog.get_by_role("button", name="Add", exact=True).click()
    page.wait_for_function("() => !document.querySelector('[role=dialog]')")
    assert errors == [], errors
    cfg = cfg_path.read_text(encoding="utf-8")
    assert 'deployment = "azure"' in cfg and "api-version" in cfg, cfg


@pytest.mark.parametrize("action", ["Save", "Unset"])
def test_config_save_keeps_filter_scroll_and_row_focus(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, action: str
) -> None:
    """A config refresh keeps the operator's filtered rows and reading position."""
    from playwright.sync_api import expect

    cfg = paths.global_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('[sandbox]\nrun_commands = "allow"\n')
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    search = page.get_by_placeholder("filter settings…")
    search.fill("sandbox.")
    rows = page.locator(".cfg tr:visible .key")
    before = rows.all_text_contents()
    row = page.get_by_role("button", name="edit sandbox.run_commands", exact=True)
    row.focus()
    scroll = page.evaluate("window.scrollY")
    row.press("Enter")
    dialog = page.get_by_role("dialog")
    dialog.locator("select.field").select_option("deny")
    # Allow the CLI round-trip to finish before checking the refreshed page.
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/api/config")
    ) as saved:
        dialog.get_by_role("button", name=action, exact=True).click()
    assert saved.value.ok, saved.value.text()
    expect(row.locator(".val")).to_have_text('"deny"' if action == "Save" else '"ask"')
    expect(dialog).to_have_count(0)
    expect(search).to_have_value("sandbox.")
    assert rows.all_text_contents() == before
    expect(row).to_be_focused()
    assert page.evaluate("window.scrollY") == scroll
    assert not errors


def test_a_late_config_save_does_not_repaint_another_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A save response cannot replace the page opened while the write was pending."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name="edit web.port", exact=True).click()
    held: list[Route] = []

    def hold(route: Route) -> None:
        if route.request.method == "POST":
            held.append(route)
        else:
            route.continue_()

    page.route("**/api/config", hold)
    page.get_by_role("dialog").get_by_role("button", name="Save", exact=True).click()
    page.wait_for_timeout(100)
    assert len(held) == 1
    page.evaluate("location.hash = '#/machines'")
    expect(page.locator(".cfg")).to_have_count(0)
    with page.expect_response(lambda r: r.url.endswith("/api/config")):
        held[0].continue_()
    page.wait_for_timeout(300)
    assert page.locator(".cfg").count() == 0
    assert not errors


def test_modified_only_includes_explicit_defaults_and_combines_with_search(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Modified means set by a layer, even when the value equals its default."""
    from playwright.sync_api import expect

    cfg = paths.global_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("[web]\nport = 7658\n")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    toggle = page.get_by_role("button", name="Modified only", exact=True)
    toggle.click()
    page.get_by_placeholder("filter settings…").fill("web.")
    expect(page.locator(".cfg tr:visible .key")).to_have_text(["web.port"])
    page.get_by_role("button", name="edit web.port", exact=True).click()
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/api/config")
    ) as saved:
        page.get_by_role("dialog").get_by_role("button", name="Unset", exact=True).click()
    assert saved.value.ok, saved.value.text()
    expect(page.get_by_role("dialog")).to_have_count(0)
    expect(page.locator(".cfg tr:visible .key")).to_have_count(0)
    expect(toggle).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_placeholder("filter settings…")).to_have_value("web.")
    assert not errors


def test_the_model_editor_selects_qualified_suggestions_without_saving(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Selecting a qualified suggestion never submits the editor."""
    from playwright.sync_api import expect

    cfg = paths.global_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[providers.a]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[providers.b]\nbackend = "openai"\nbase_url = "http://localhost:2"\n'
        '[models.worker]\nmodel = "a/old"\neffort = "high"\n'
        "temperature = 0.4\n"
    )
    page, errors = open_page(port)
    page.route(
        "**/api/config/suggest/**",
        lambda route: route.fulfill(json={"values": ["b/alpha", "b/beta"]}),
    )
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name="edit models.worker.model", exact=True).click()
    dialog = page.get_by_role("dialog")
    model = dialog.get_by_role("combobox", name="Model", exact=True)
    expect(model).to_have_value("a/old")
    model.focus()
    expect(dialog.get_by_role("listbox")).to_be_visible()
    model.press("ArrowDown")
    model.press("ArrowDown")
    model.press("ArrowUp")
    model.press("Enter")
    expect(model).to_have_value("b/alpha")
    expect(dialog).to_be_visible()
    assert 'model = "a/old"' in cfg.read_text()
    model.fill("b/newer-than-catalog")
    expect(dialog.get_by_role("status")).to_contain_text("No matches")
    model.press("Escape")
    expect(dialog.get_by_role("listbox")).to_be_hidden()
    expect(dialog).to_be_visible()
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/api/config")
    ) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()
    expect(page.get_by_role("button", name="edit models.worker.model", exact=True)).to_contain_text(
        "newer-than-catalog"
    )
    expect(dialog).to_have_count(0)
    text = cfg.read_text()
    assert (
        'model = "b/newer-than-catalog"' in text
        and 'effort = "high"' in text
        and "temperature = 0.4" in text
    )
    assert not errors


def test_model_catalog_ignores_a_closed_editors_response(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A closed editor's listing cannot overwrite a new editor's empty catalog."""
    from playwright.sync_api import Route, expect

    cfg = paths.global_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        '[providers.a]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[providers.b]\nbackend = "openai"\nbase_url = "http://localhost:2"\n'
        '[models.worker]\nmodel = "a/old"\n'
    )
    page, errors = open_page(port)
    held: list[Route] = []

    def listing(route: Route) -> None:
        if not held:
            held.append(route)
        else:
            route.fulfill(json={"values": []})

    page.route("**/api/config/suggest/**", listing)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name="edit models.worker.model", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("status")).to_contain_text("Loading models")
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    page.get_by_role("button", name="edit models.worker.model", exact=True).click()
    dialog = page.get_by_role("dialog")
    model = dialog.get_by_role("combobox", name="Model", exact=True)
    model.focus()
    expect(dialog.get_by_role("status")).to_contain_text("Catalog empty or unavailable")
    assert len(held) == 1
    held[0].fulfill(json={"values": ["stale-model"]})
    page.wait_for_timeout(100)
    expect(dialog.get_by_role("option", name="stale-model", exact=True)).to_have_count(0)
    model.fill("b/custom")
    with page.expect_response(
        lambda response: response.request.method == "POST" and response.url.endswith("/api/config")
    ) as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, saved.value.text()
    expect(dialog).to_have_count(0)
    assert 'model = "b/custom"' in cfg.read_text()
    assert not errors


def test_a_files_diff_fills_the_git_tab(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """A click on a changed file opens its patch over the whole tab; the back button returns.

    The graph and the side panel hide while the diff shows; the HTTP tests see only the JSON.
    """
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    assert page.locator(".gc-wt").count() == 0  # a clean tree has no working-tree row
    page.locator(".gc-row").first.click()
    page.locator(".gf").first.click()
    page.wait_for_selector(".git-diff .diff")
    body = page.locator(".git-body")
    assert "diffing" in (body.get_attribute("class") or "")
    assert not page.locator(".git-graph").is_visible()
    assert page.locator(".git-diff-pos").inner_text().startswith("1 / ")
    page.locator(".git-diff-head button").first.click()  # the "files" back button
    assert "diffing" not in (body.get_attribute("class") or "")
    assert page.locator(".git-graph").is_visible()
    assert errors == [], errors


def test_the_working_tree_row_opens_its_uncommitted_files(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """A dirty tree tops the graph with a row that opens its files like a commit's."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    row = page.locator(".gc-wt")
    row.wait_for()
    assert "Working tree" in row.inner_text() and "5 uncommitted files" in row.inner_text()
    row.click()
    page.wait_for_selector(".git-side .gf")
    assert page.locator(".git-side .gf").count() == 5
    page.locator(".git-side .gf", has_text="new.txt").click()
    page.wait_for_selector(".git-diff .diff")
    assert "+hello" in page.locator(".git-diff .diff").inner_text()
    assert errors == [], errors


def test_an_images_diff_shows_it_before_and_after(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """A changed PNG shows both versions above git's binary line, each with its dimensions."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.locator(".git-side .gf", has_text="logo.png").click()
    page.wait_for_function("() => document.querySelectorAll('.git-img img').length === 2")
    page.wait_for_function("() => document.body.innerText.includes('2\u00d72')")
    text = page.locator(".git-img").inner_text()
    assert "before" in text and "1\u00d71" in text and "after" in text and "2\u00d72" in text
    assert "Binary files" in page.locator(".git-diff .diff").inner_text()
    assert errors == [], errors


def test_a_changed_file_list_reads_as_a_tree_that_folds(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """The list | tree toggle groups the files under their folders.

    A folder folds on a click, the choice is kept in the browser, and a file in the tree
    opens its diff as in the list.
    """
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.wait_for_selector(".git-side .gf")
    page.locator(".git-side .gview button", has_text="tree").click()
    folder = page.locator(".git-side .gtree-dir", has_text="dir")
    folder.wait_for()
    assert "1 file" in folder.inner_text()
    leaf = page.locator(".git-side [data-path='dir/g.txt']")
    assert leaf.is_visible()
    folder.click()
    assert not leaf.is_visible()
    folder.click()
    leaf.click()
    page.wait_for_selector(".git-diff .diff")
    assert "rename from dir/f.txt" in page.locator(".git-diff .diff").inner_text()
    assert page.evaluate("localStorage.getItem('a6-files-view')") == "tree"
    assert errors == [], errors


def test_the_diff_view_switches_between_inline_split_old_and_new(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """Every rendered line carries its numbers; split pairs the sides; new marks the changes.

    The choice is kept in the browser.
    """
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.locator(".git-side [data-path='b.txt']").click()
    page.wait_for_selector(".git-diff .diff")
    assert page.locator(".git-diff .diff .dl.add[data-new='4']").count() == 1
    assert page.locator(".git-diff .diff .dl[data-old='3'][data-new='3']").count() == 1
    page.locator(".git-diff .gview button", has_text="split").click()
    page.wait_for_selector(".git-diff .ds-row")
    added = page.locator(".git-diff .ds-row[data-new='4']")
    assert added.count() == 1
    assert added.locator(".ds-cell").nth(0).inner_text().strip() == ""
    assert "four" in added.locator(".ds-cell").nth(1).inner_text()
    ctx = page.locator(".git-diff .ds-row[data-old='3']")
    assert "three" in ctx.locator(".ds-cell").nth(0).inner_text()
    assert "three" in ctx.locator(".ds-cell").nth(1).inner_text()
    # The gutter is drawn by the stylesheet, so a blank line still has a height.
    drawn = page.evaluate(
        "getComputedStyle(document.querySelector(\".ds-row[data-old='3'] .ds-cell .g\"),"
        " '::before').content"
    )
    assert drawn == '"3"', drawn
    page.locator(".git-diff .gview button", has_text="new").click()
    page.wait_for_selector(".git-diff .src .sl[data-line='4']")
    assert page.locator(".git-diff .src .sl").count() == 4
    assert "changed" in (
        page.locator(".git-diff .src .sl[data-line='4']").get_attribute("class") or ""
    )
    assert "changed" not in (
        page.locator(".git-diff .src .sl[data-line='1']").get_attribute("class") or ""
    )
    assert page.evaluate("localStorage.getItem('a6-diff-mode')") == "new"
    page.locator(".git-diff .gview button", has_text="old").click()
    page.wait_for_selector(".git-diff .src .sl[data-line='3']")
    assert page.locator(".git-diff .src .sl").count() == 3
    assert errors == [], errors


@pytest.mark.parametrize("theme,width", [("dark", 1280), ("light", 390)])
def test_files_blame_groups_lines_and_loads_a_commit_card(
    open_page: Callable[[int], tuple[Page, list[str]]],
    dirty_git_port: int,
    theme: str,
    width: int,
) -> None:
    """A group's continuation has only a bar; its label links to the historical file."""
    from playwright.sync_api import expect

    page, errors = open_page(dirty_git_port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/files/WORKTREE/b.txt")
    page.evaluate(
        "theme => document.documentElement.classList.toggle('light', theme === 'light')", theme
    )
    page.locator(".files-blame").click()
    page.wait_for_selector(".sl[data-line='4'] .blame-cell")
    assert page.locator(".blame-label").count() == 3
    assert page.locator(".sl[data-line='2'] .blame-cell").inner_text() == ""
    assert page.locator(".sl[data-line='2'] .blame-cell").evaluate(
        "e => parseFloat(getComputedStyle(e).borderLeftWidth) > 0"
    )
    bars = page.locator(".sl .blame-cell").evaluate_all(
        "cells => cells.slice(0, 2).map(c => c.getBoundingClientRect().left)"
    )
    assert bars[0] == bars[1]
    link = page.locator(".sl[data-line='1'] .blame-commit")
    assert (link.get_attribute("href") or "").endswith("?path=a.txt&line=1")
    assert page.locator(".sl[data-line='4'] a").count() == 0
    link.focus()
    # Playwright's own retry: the page's animation frames do not run in this headless browser.
    expect(page.locator(".blame-card .blame-title")).to_have_text("root commit")
    assert page.locator(".blame-card:visible").inner_text().endswith("root commit")
    assert "a.txt" in page.locator(".blame-card:visible").inner_text()
    # The card sits over the next lines' gutter; leaving the link closes it after its grace.
    link.blur()
    expect(page.locator(".blame-card")).to_be_hidden()
    page.locator(".sl[data-line='2'] .blame-cell").hover()
    assert "blame-hover" in (
        page.locator(".sl[data-line='1'] .blame-cell").get_attribute("class") or ""
    )
    assert errors == [], errors


def test_blame_opens_the_lines_commit_in_git(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """A working-tree blame marks uncommitted lines and opens a committed line's panel."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/files/WORKTREE/b.txt")
    page.locator(".files-blame").click()
    page.wait_for_selector(".sl[data-line='4'] .blame-cell")
    assert "not committed" in page.locator(".sl[data-line='4'] .blame-cell").inner_text()
    assert page.locator(".files-view .sl").count() == 4
    page.locator(".sl[data-line='1'] .blame-commit").click()
    page.wait_for_selector(".git-diff .dl[data-new='1']")
    assert "a.txt" in page.locator(".git-diff-head").inner_text()
    assert "path=a.txt" in page.url
    page.reload()
    page.wait_for_selector(".git-diff .dl[data-new='1']")
    page.locator(".git-diff-head").get_by_role("button", name="\u2039 files", exact=True).click()
    assert "root commit" in page.locator(".git-side .gd-subject").inner_text()
    assert page.locator(".gc-row.sel").count() == 1
    assert "path=" not in page.url
    assert errors == [], errors


@pytest.mark.parametrize("deleted", [False, True])
def test_diff_opens_files_at_the_correct_side(
    open_page: Callable[[int], tuple[Page, list[str]]],
    git_port: int,
    deleted: bool,
) -> None:
    """A deletion disables the head action and offers the base file separately."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    path = "d.txt" if deleted else "dir/f.txt"
    page.goto(f"http://127.0.0.1:{git_port}/#/git/compare/v1..main?path={path}")
    if deleted:
        button = page.get_by_role("button", name="Open in Files", exact=True)
        expect(button).to_be_disabled()
        expect(button).to_have_attribute("title", "Deleted: no file at the head revision")
        page.get_by_role("link", name="Open base file in Files", exact=True).click()
        expect(page.locator(".files-text")).to_contain_text("doomed")
    else:
        page.get_by_role("link", name="Open in Files", exact=True).click()
        expect(page.locator(".files-head")).to_contain_text(path)
        expect(page.locator(".files-text")).to_contain_text("nested")
    assert errors == [], errors


@pytest.mark.parametrize("mode", ["inline", "split"])
def test_diff_full_context_shows_lines_outside_the_hunk(
    open_page: Callable[[int], tuple[Page, list[str]]],
    dirty_git_port: int,
    tmp_path: pathlib.Path,
    mode: str,
) -> None:
    """The full-context control reveals omitted lines without switching to a side view."""
    from playwright.sync_api import expect

    text = "".join(f"line {i}\n" for i in range(60))
    (tmp_path / "long.txt").write_text(text)
    _git(tmp_path, "add", "long.txt")
    _git(tmp_path, "commit", "-m", "long file")
    (tmp_path / "long.txt").write_text(text.replace("line 30\n", "changed\n"))
    page, errors = open_page(dirty_git_port)
    page.add_init_script(f"localStorage.setItem('a6-diff-mode', '{mode}')")
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/compare/HEAD..WORKTREE?path=long.txt")
    expect(page.locator(".git-diff-body")).to_contain_text("changed")
    assert page.locator(".git-diff [data-new='1']").count() == 0
    page.get_by_role("button", name="Show full context", exact=True).click()
    expect(page.locator(".git-diff [data-new='1']")).to_contain_text("line 0")
    expect(page.locator(".git-diff [data-new='60']")).to_contain_text("line 59")
    page.get_by_role("button", name="Show hunks only", exact=True).click()
    expect(page.locator(".git-diff [data-new='1']")).to_have_count(0)
    page.get_by_role("link", name="Open in Files", exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{dirty_git_port}/#/files/WORKTREE/long.txt")
    assert errors == [], errors


@pytest.mark.parametrize("mode", ["inline", "split", "old", "new"])
def test_diff_blame_attributes_equal_text_to_each_sides_history(
    open_page: Callable[[int], tuple[Page, list[str]]],
    git_port: int,
    tmp_path: pathlib.Path,
    mode: str,
) -> None:
    """Restoring equal context text gives it a distinct head-side commit."""
    from playwright.sync_api import expect

    target = tmp_path / "history.txt"
    target.write_text("old\nsame\n")
    _git(tmp_path, "add", "history.txt")
    _git(tmp_path, "commit", "-m", "base attribution")
    base = _git(tmp_path, "rev-parse", "HEAD")
    target.write_text("temporary\ndifferent\n")
    _git(tmp_path, "commit", "-am", "intermediate attribution")
    target.write_text("new\nsame\n")
    _git(tmp_path, "commit", "-am", "head attribution")
    head = _git(tmp_path, "rev-parse", "HEAD")
    page, errors = open_page(git_port)
    page.add_init_script(f"localStorage.setItem('a6-diff-mode', '{mode}')")
    page.goto(f"http://127.0.0.1:{git_port}/#/git/compare/{base}..{head}?path=history.txt")
    page.locator(".diff-blame").click()
    cells = page.locator(".git-diff .blame-cell")
    expect(cells).to_have_count(4 if mode == "split" else 3 if mode == "inline" else 2)
    left = page.locator('.blame-cell[data-side="base"]')
    right = page.locator('.blame-cell[data-side="head"]')
    expect(left).to_have_count(1 if mode == "inline" else 2 if mode in {"split", "old"} else 0)
    expect(right).to_have_count(0 if mode == "old" else 2)
    for cell in left.all():
        expect(cell).to_have_attribute("data-sha", base)
    for cell in right.all():
        expect(cell).to_have_attribute("data-sha", head)
    if mode == "split":
        expect(page.locator('.ds-row[data-old="2"] .blame-cell').nth(0)).to_have_attribute(
            "data-sha", base
        )
        expect(page.locator('.ds-row[data-new="2"] .blame-cell').nth(1)).to_have_attribute(
            "data-sha", head
        )
    page.locator(".diff-blame").click()
    expect(cells).to_have_count(0)
    assert errors == [], errors


def test_diff_blame_refuses_a_worktree_file_changed_after_display(
    open_page: Callable[[int], tuple[Page, list[str]]],
    dirty_git_port: int,
    tmp_path: pathlib.Path,
) -> None:
    """Blame from later working-tree bytes cannot annotate an earlier patch."""
    from playwright.sync_api import expect

    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/compare/HEAD..WORKTREE?path=b.txt")
    expect(page.locator(".git-diff-body")).to_contain_text("four")
    (tmp_path / "b.txt").write_text("changed after display\n")
    page.locator(".diff-blame").click()
    expect(page.locator(".blame-note")).to_contain_text("Blame unavailable: head file changed")
    expect(page.locator(".blame-cell")).to_have_count(0)
    assert errors == [], errors


def test_full_context_displays_a_pure_rename_without_hunks(
    open_page: Callable[[int], tuple[Page, list[str]]],
    dirty_git_port: int,
) -> None:
    """A pure rename opens its unchanged file instead of an empty full-context patch."""
    from playwright.sync_api import expect

    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/compare/HEAD..WORKTREE?path=dir/g.txt")
    page.locator(".diff-context").click()
    expect(page.locator(".git-diff-body")).to_contain_text("no text hunks")
    expect(page.locator('.git-diff [data-new="1"]')).to_contain_text("nested")
    assert errors == [], errors


def test_git_and_files_deep_links_survive_reload_and_back(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Encoded file paths, commit and compare hashes select their panes on entry and Back."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    page.wait_for_selector(".files-view .sl[data-line='3']")
    page.reload()
    page.wait_for_selector(".files-view .sl[data-line='3']")
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/feature")
    page.wait_for_selector(".git-side .gd-subject")
    assert "feature: rename" in page.locator(".git-side .gd-subject").inner_text()
    assert page.locator(".gc-row.sel").count() == 1
    page.locator(".gc-row").first.click()
    page.wait_for_function(
        "() => document.querySelector('.gd-subject')?.textContent.includes('merge feature')"
    )
    page.go_back()
    page.wait_for_function(
        "() => document.querySelector('.gd-subject')?.textContent.includes('feature: rename')"
    )
    page.goto(f"http://127.0.0.1:{git_port}/#/git/compare/v1..v2")
    page.wait_for_selector(".gcmp-result .gf")
    assert page.locator(".gcmp-result .gf").count() == 5
    assert errors == [], errors


def test_files_marks_worktree_states_and_sums_folders(
    open_page: Callable[[int], tuple[Page, list[str]]], files_port: int
) -> None:
    """Working-tree marks distinguish tracked, untracked and modified files, not commits."""
    page, errors = open_page(files_port)
    page.goto(f"http://127.0.0.1:{files_port}/#/files")
    page.wait_for_selector(".files-tree [data-path='b.txt']")
    assert page.locator(".files-tree .gf-st").count() == 0
    page.locator(".files-bar .gpick input").click()
    page.locator(".gpick-list .ac-item", has_text="working tree").click()
    page.wait_for_selector(".files-tree [data-path='src/tool.py']")
    for path, mark in (
        ("b.txt", "M"),
        ("img.bin", "H"),
        ("new.txt", "A"),
        ("src/tool.py", "A"),
        ("src", "A"),
        ("dir", "M"),
    ):
        assert page.locator(f".files-tree [data-path='{path}'] .gf-st").inner_text() == mark
    assert "1 untracked" in (
        page.locator(".files-tree [data-path='src'] .gf-st").get_attribute("title") or ""
    )
    page.locator(".files-bar .gpick input").fill("HEAD")
    page.locator(".files-bar .gpick input").press("Enter")
    page.wait_for_selector(".files-tree [data-path='dir/f.txt']")
    assert page.locator(".files-tree .gf-st").count() == 0
    assert errors == [], errors


@pytest.mark.needs_grammar(language="python")
def test_the_files_tab_opens_a_file_highlighted(
    open_page: Callable[[int], tuple[Page, list[str]]], files_port: int
) -> None:
    """The Files tab lists the working tree's files as a tree and shows one with its spans.

    HEAD is the revision to start; the picker's "working tree" lists the untracked file
    too, and the filter narrows the tree.
    """
    page, errors = open_page(files_port)
    page.goto(f"http://127.0.0.1:{files_port}/#/files")
    page.wait_for_selector(".files-tree [data-path='b.txt']")
    assert page.locator(".files-tree [data-path='src/tool.py']").count() == 0  # not at HEAD
    page.locator(".files-bar .gpick input").click()
    page.locator(".gpick-list .ac-item", has_text="working tree").click()
    page.wait_for_selector(".files-tree [data-path='src/tool.py']")
    assert page.locator(".files-tree .gtree-dir[data-path='src']").count() == 1
    page.locator(".files-bar input.filter").fill("tool")
    assert page.locator(".files-tree [data-path='b.txt']").count() == 0
    page.locator(".files-tree [data-path='src/tool.py']").click()
    page.wait_for_selector(".files-view .src .sl[data-line='2']")
    first = page.locator(".files-view .src .sl[data-line='1']")
    assert "def" in first.inner_text() and first.locator(".hl-keyword").count() >= 1
    assert page.locator(".files-view .src .sl").count() == 2
    assert "python" in page.locator(".files-view").inner_text()
    assert errors == [], errors


def test_back_from_a_compare_link_returns_to_the_graph(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Back from a comparison lands on the bare Git tab; the restore pushed its hash again."""
    page, errors = open_page(git_port)
    base = f"http://127.0.0.1:{git_port}/"
    page.goto(base + "#/git")
    page.wait_for_selector(".gc-row")
    page.goto(base + "#/git/compare/v1..v2")
    page.wait_for_selector(".gcmp-result .gf")
    page.go_back()
    page.wait_for_selector(".git-side .empty", timeout=3000)
    assert page.evaluate("location.hash") == "#/git"
    assert page.locator(".gcmp-result").count() == 0
    assert errors == [], errors


def test_the_worktree_pane_has_a_hash_that_survives_reload(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """The working-tree row publishes `#/git/worktree`, and a reload reopens that pane."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.wait_for_selector(".git-side .gd-subject")
    assert page.evaluate("location.hash") == "#/git/worktree"
    page.reload()
    page.wait_for_selector(".git-side .gd-subject")
    assert "Uncommitted changes" in page.locator(".git-side .gd-subject").inner_text()
    assert errors == [], errors


def test_lane_details_read_the_coordinators_rationale(
    port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The lane header reads the group record rather than a copied lane rationale."""
    from playwright.sync_api import expect

    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    for sid, stamp in {
        "fan": {
            "comparison": {
                "group": "fan",
                "ranking": ["lane"],
                "ranked_by": "judge",
                "rationale": "covers the edge case",
                "judge_cost_usd": 0.0123,
            }
        },
        "lane": {"compare": {"group": "fan", "rank": 1, "of": 1, "winner": True}},
    }.items():
        session = runs / sid
        session.mkdir(parents=True)
        (session / "manifest.json").write_text(
            json.dumps({"mode": "run", **stamp}), encoding="utf-8"
        )
        (session / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "Compare lanes"})
            + "\n",
            encoding="utf-8",
        )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/lane")
    expect(page.get_by_text("judge: covers the edge case", exact=True)).to_be_visible()
    expect(page.get_by_text("rank 1/1 · winner · judge ($0.01)", exact=True)).to_be_visible()
    assert not errors
