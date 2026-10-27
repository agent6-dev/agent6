# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render snapshot: the composed web page is pinned byte-for-byte.

`PAGE_HTML` splices the `client.js` / `styles.css` resources into the HTML
template at import time; pinning its sha256 makes any byte drift in the page or
its assets a deliberate, visible change reviewed alongside the edit that moved
it.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from importlib import resources

import pytest

from agent6.ui.web import page

# sha256 of PAGE_HTML; an edit to page.py, client.js or styles.css moves it in the same commit.
PAGE_SHA256 = "e5ae6a443e12a4c8a94e69b604252ddbc1ab787c99f244fbe70606451274a1cd"


def test_rendered_page_bytes_are_pinned() -> None:
    got = hashlib.sha256(page.PAGE_HTML.encode("utf-8")).hexdigest()
    assert got == PAGE_SHA256, (
        f"page bytes changed (sha256 {got}); if intended, update PAGE_SHA256 in this test"
    )


def test_page_assets_load_non_empty() -> None:
    # An asset missing from the wheel would otherwise surface only at the release build.

    web = resources.files("agent6.ui.web")
    for name in (*page._CLIENT_FILES, "styles.css"):
        assert web.joinpath(name).read_text(encoding="utf-8").strip(), f"{name} is empty"


def test_the_plan_bar_meters_this_executions_points() -> None:
    """The run view's plan bar subtracts the points earlier executions banked.

    The wire's `plan_consumed` sums every execution, and the sum against the cap a resume
    re-armed would read a resumed run as over it.
    """
    client = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "(b.plan_consumed || 0) - (b.plan_prior_executions || 0)" in client
    assert "executionPt / planCap" in client


def test_the_sessions_card_folds_a_fan_outs_lanes() -> None:
    """The hub renders nested lane rows under their fan-out behind a `lanes: N` line."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    assert "const lanes = r.lanes || [];" in client
    assert "lanes: ${lanes.length}" in client and "expandedFanouts" in client
    # Enter toggles the row (its key handler does not swallow it), and a lane row is a button.
    assert "toggle.onkeydown = (e) => e.stopPropagation();" in client
    assert "actionable(li, " in client


def test_a_session_row_shows_its_mode() -> None:
    """The Sessions page lists each session's mode, as docs/web.md promises."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    paint = client[client.index("function paintSession") : client.index("function sessionsCard")]
    assert "esc(r.mode)" in paint


def test_new_work_route_refresh_clears_and_ignores_stale_models() -> None:
    """A pending route request cannot submit the previous pair's model or override a newer one."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    refresh = client[client.index("function newWorkDock") : client.index("// The create-machine")]
    request = refresh.index("const request = ++routeRequest;")
    cleared = refresh.index("model.value = '';")
    awaited = refresh.index("await getJSON('/api/routes")
    stale_guard = refresh.index("if (request !== routeRequest) return;")
    populated = refresh.index("el('option', null, d.default_label)")
    assert request < cleared < awaited < stale_guard < populated


def test_the_pickers_sit_in_a_row_above_each_composer() -> None:
    """New work and resume put their dropdowns in one labelled row above the text."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    dock = client[client.index("function newWorkDock") : client.index("// The create-machine")]
    assert "row.appendChild(task); row.appendChild(go);" in dock
    picks = dock.index("pickerRow([['mode', mode], ['preset', preset], ['model', model]])")
    assert picks < dock.index("root.appendChild(row);")
    composer = client[client.index("function makeComposer") :]
    assert "pickerRow([['continue under preset', preset], ['model', model]])" in composer
    assert "root.appendChild(presetRow); root.appendChild(ta);" in composer


def test_the_policy_line_breaks_only_between_its_settings() -> None:
    """The run card wrapped `run_commands` on one line and `ask` on the next at 300 px."""
    run = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "s.policy.split(' · ')" in run
    assert "pol.appendChild(el('span', 'nowrap', part));" in run
    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    assert ".nowrap { white-space: nowrap; }" in css


def test_the_commit_step_row_is_a_picker_row() -> None:
    """The Latest commit card's dropdown shares the picker rows' style on the dark page."""
    run = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "const nav = el('div', 'row pickers');" in run
    assert "const sel = el('select', 'field');" in run


def test_a_tool_call_clips_its_args_instead_of_breaking_words() -> None:
    """Each call in the details drawer is its name over one clipped line of args.

    The full text is on hover; a narrow args column broke a path mid-word.
    """
    run = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "const calls = el('div', 'calls');" in run and "const a = el('div', 'args');" in run
    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    rule = css[css.index(".calls .args {") :]
    rule = rule[: rule.index("}")]
    assert "white-space: nowrap" in rule and "text-overflow: ellipsis" in rule
    assert "break-word" not in rule


def test_native_controls_take_the_page_theme() -> None:
    """The page declares its colour scheme per theme, so checkboxes and dropdowns follow it."""
    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    root, light = css.index(":root {"), css.index(":root.light {")
    assert "color-scheme: dark;" in css[root:light]
    assert "color-scheme: light;" in css[light : css.index("}", light)]
    assert "input[type=checkbox] { accent-color: var(--accent); }" in css


def test_the_resume_row_asks_what_a_bare_resume_runs_under() -> None:
    """The resume row's first options name what a resume without flags runs under.

    Asked when the row appears and on a preset pick; a late answer never replaces a newer one.
    """
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    composer = client[client.index("function makeComposer") :]
    assert "'/resume_defaults?preset=' + encodeURIComponent(preset.value)" in composer
    assert "if (request !== labelRequest) return;" in composer
    assert "preset.onchange = relabel;" in composer
    assert "if (finished && !rowShown) relabel();" in composer


def test_parallel_model_completion_handles_each_whole_fragment() -> None:
    """Model completion replaces a comma fragment before `--`, never task text after it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise the web composer")
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    suggest = client[
        client.index("function attachParallelSuggest") : client.index("// The new-work composer")
    ]
    scenario = r"""
const assert = require('node:assert/strict');
const getJSON = async () => ({values: ['o/alpha', 'o/beta']});
function el(tag, cls, text = '') {
  return {
    text, children: [], removed: false,
    set textContent(value) { this.text = value; this.children = []; },
    appendChild(child) { this.children.push(child); },
    remove() { this.removed = true; },
  };
}
async function check(text, wanted, caret = text.length, replacement = null) {
  const listeners = {};
  const task = {
    value: text, selectionStart: caret,
    addEventListener(name, fn) { listeners[name] = fn; },
    setSelectionRange(start) { this.selectionStart = start; },
    focus() {},
  };
  const root = el('div');
  attachParallelSuggest(task, root);
  listeners.input();
  await Promise.resolve();
  const rows = root.children.filter(box => !box.removed).flatMap(box => box.children);
  assert.deepEqual(rows.map(row => row.text), wanted, text);
  if (replacement !== null) {
    rows[0].onmousedown({preventDefault() {}});
    assert.equal(task.value, replacement);
  }
}
(async () => {
  await check('/parallel o/alpha, ', ['o/alpha', 'o/beta']);
  await check('/parallel o/alpha,\n o/b', ['o/beta']);
  await check('/parallel o/beta -- o/', []);
  await check('/parallel o/beta --', []);
  await check('/parallel -- ', []);
  await check('/parallel 2 ', []);
  await check('/parallel 2 -- task /parallel o/beta', ['o/beta']);
  const text = '/parallel 2 -- task /parallel o/alpha, o/beTAIL -- task';
  await check(text, ['o/beta'], text.indexOf('TAIL'), text.replace('o/beTAIL', 'o/beta'));
  const spaced = '/parallel o/beta -- task';
  await check(spaced, ['o/beta'], spaced.indexOf(' --') + 1, spaced);
  await check('  /PARALLEL o/b', ['o/beta']);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", suggest + scenario], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reviewer_controls_post_form_fields_and_position() -> None:
    """The roster controls send five fields on add and a zero-based position on removal."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise the web config editor")
    source = (
        resources.files("agent6.ui.web").joinpath("client_config.js").read_text(encoding="utf-8")
    )
    setup = r"""
const assert = require('node:assert/strict');
function el(tag, cls, text = '') {
  return {
    tag, cls, text, children: [], style: {}, value: '', checked: false, isConnected: true,
    set textContent(value) { this.text = value; this.children = []; },
    set innerHTML(value) { this.children = []; },
    appendChild(child) { this.children.push(child); child.parent = this; return child; },
    insertBefore(child, before) {
      this.children.splice(this.children.indexOf(before), 0, child); child.parent = this;
    },
    setAttribute(key, value) { this[key] = value; },
    remove() {
      this.isConnected = false; this.parent.children = this.parent.children.filter(c => c !== this);
    },
    focus() { document.activeElement = this; },
  };
}
const document = {
  body: el('body'), activeElement: null, addEventListener() {}, removeEventListener() {},
};
const view = el('div'); document.body.appendChild(view);
const boot = () => {}, setCrumb = () => {}, esc = String, toast = () => {};
const actionable = (node, action) => { node.onclick = action; };
let routeGen = 0, activeOverlayClose = null;
const setting = {
  type: 'list', source: 'repo', input: '[]', display: 'tests, tests',
  value: [{focus: 'tests'}, {focus: 'tests'}],
};
const getJSON = async path => path === '/api/config/entry_tables' ? {tables: []}
  : path === '/api/config/reviewer_choices'
  ? {effort: ['off', 'low', 'medium', 'high', 'xhigh', 'max']}
  : path === '/api/config' ? {'review.reviewers': setting} : {values: []};
const posts = [];
const postJSON = async (path, body) => { posts.push({path, body}); return {ok: true}; };
let confirmed = false;
const confirm = () => confirmed;
function nodes(root = document.body) {
  return [root, ...root.children.flatMap(child => nodes(child))];
}
function button(text) { return nodes().find(n => n.tag === 'button' && n.text === text); }
function field(id) { return nodes().find(n => n.id === id); }
"""
    scenario = r"""
(async () => {
  await renderConfig();
  assert.ok(button('Add reviewer…'), 'the config bar offers Add reviewer');
  await button('Add reviewer…').onclick();
  field('reviewer-focus').value = 'security';
  field('reviewer-provider').value = 'p';
  field('reviewer-model').value = 'm';
  assert.deepEqual(field('reviewer-effort').children.map(n => n.value),
    ['', 'off', 'low', 'medium', 'high', 'xhigh', 'max']);
  field('reviewer-effort').value = 'off';
  field('reviewer-temperature').value = '0';
  field('reviewer-repo').checked = true;
  await button('Add').onclick();
  assert.deepEqual(posts.shift(), {path: '/api/config/reviewer', body: {
    focus: 'security', provider: 'p', model: 'm', effort: 'off', temperature: '0', repo: true,
  }});
  editConfig('review.reviewers', setting);
  const removes = nodes().filter(n => n.tag === 'button' && n.text === 'Remove');
  assert.equal(removes.length, 2);
  assert.ok(nodes().some(n => n.text === '1: tests'));
  assert.ok(nodes().some(n => n.text === '2: tests'));
  await removes[1].onclick();
  assert.equal(posts.length, 0, 'cancel does not write');
  confirmed = true;
  await removes[1].onclick();
  assert.deepEqual(posts.shift(), {
    path: '/api/config/reviewer/remove', body: {index: 1, repo: true},
  });
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run(
        [node, "-e", setup + source + scenario], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_config_editor_keeps_a_list_as_toml_on_an_untouched_save() -> None:
    """The edit field holds the server's round-trippable TOML value.

    Joining a list with commas turned an untouched Save into a rejected string.
    """
    config = resources.files("agent6.ui.web").joinpath("client_config.js")
    text = config.read_text(encoding="utf-8")
    assert "const cur = s.input;" in text
    assert "s.value.join(',')" not in text


def test_the_config_editor_sends_a_string_leaf_as_a_toml_string() -> None:
    """A str leaf is posted quoted, so a value that parses as another TOML type stays a string."""
    config = resources.files("agent6.ui.web").joinpath("client_config.js")
    assert "s.type === 'str' ? JSON.stringify(field.value) : field.value" in config.read_text(
        encoding="utf-8"
    )


def test_the_empty_machines_card_says_what_the_tui_says() -> None:
    """The web card says "no machines yet", as the TUI's machines screen does."""
    assert "'no machines yet'" in page.CLIENT_JS and "no machine instances" not in page.CLIENT_JS


def test_the_hub_keeps_its_maintenance_actions_behind_one_control() -> None:
    """The danger actions open from one "more…" disclosure, as the TUI's File menu holds them."""
    assert "el('details', 'more')" in page.CLIENT_JS
    for label in ("Prune merged runs", "Prune merged runs, squash-merged too", "Clear saved asks"):
        assert f"action('{label}'" in page.CLIENT_JS
    assert "also squash-merged branches" not in page.CLIENT_JS


def test_the_phone_widget_menu_can_show_the_run_review() -> None:
    """The review card has a menu entry, so a review that lands on a phone is visible."""
    client = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "['review', 'Run review']" in client
    assert "setW('review')" in client
