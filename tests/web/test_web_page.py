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
from importlib import resources

from agent6.ui.web.page import CLIENT_JS, PAGE_HTML

# sha256 of PAGE_HTML.encode("utf-8"). An edit to page.py, client.js, or
# styles.css moves it; update it in the same commit as that edit.
PAGE_SHA256 = "b34b21a632efb2c86c3c9eed7b17dceaee216d7c0b22fcbcceb5077332c4dc87"


def test_rendered_page_bytes_are_pinned() -> None:
    got = hashlib.sha256(PAGE_HTML.encode("utf-8")).hexdigest()
    assert got == PAGE_SHA256, (
        f"page bytes changed (sha256 {got}); if intended, update PAGE_SHA256 in this test"
    )


def test_page_assets_load_non_empty() -> None:
    # Guards a packaging regression (an asset missing from the wheel) that the
    # build-time wheel check would otherwise catch only at release.
    from agent6.ui.web.page import _CLIENT_FILES  # pyright: ignore[reportPrivateUsage]

    web = resources.files("agent6.ui.web")
    for name in (*_CLIENT_FILES, "styles.css"):
        assert web.joinpath(name).read_text(encoding="utf-8").strip(), f"{name} is empty"


def test_the_sessions_card_folds_a_fan_outs_lanes() -> None:
    """The hub renders the server's nested lane rows (`row_json`'s `lanes`)
    under their fan-out behind a `lanes: N` line, never as top-level rows."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    assert "const lanes = r.lanes || [];" in client
    assert "lanes: ${lanes.length}" in client and "expandedFanouts" in client
    # Enter on the toggle toggles (the row's key handler does not swallow it),
    # and a lane row is a keyboard-reachable button like every other row.
    assert "toggle.onkeydown = (e) => e.stopPropagation();" in client
    assert "actionable(li, " in client


def test_a_session_row_shows_its_mode() -> None:
    """The Sessions page lists each session's mode, as docs/web.md promises."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    paint = client[client.index("function paintSession") : client.index("function sessionsCard")]
    assert "esc(r.mode)" in paint


def test_new_work_route_refresh_clears_and_ignores_stale_models() -> None:
    """A mode or preset change cannot submit the previous pair's model while
    its route request is pending, and late older responses cannot replace the
    newest pair's choices."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    refresh = client[client.index("function newWorkDock") : client.index("// The create-machine")]
    request = refresh.index("const request = ++routeRequest;")
    cleared = refresh.index("model.value = '';")
    awaited = refresh.index("await getJSON('/api/routes")
    stale_guard = refresh.index("if (request !== routeRequest) return;")
    populated = refresh.index("el('option', null, d.default_label)")
    assert request < cleared < awaited < stale_guard < populated


def test_the_pickers_sit_in_a_row_above_each_composer() -> None:
    """New work and resume both put their dropdowns in one labelled row above
    the text (beside it, the model dropdown squeezed the task box)."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    dock = client[client.index("function newWorkDock") : client.index("// The create-machine")]
    assert "row.appendChild(task); row.appendChild(go);" in dock
    picks = dock.index("pickerRow([['mode', mode], ['preset', preset], ['model', model]])")
    assert picks < dock.index("root.appendChild(row);")
    composer = client[client.index("function makeComposer") :]
    assert "pickerRow([['continue under preset', preset], ['model', model]])" in composer
    assert "root.appendChild(presetRow); root.appendChild(ta);" in composer


def test_the_commit_step_row_is_a_picker_row() -> None:
    """The Latest commit card's dropdown and checkbox share the picker rows'
    style, so neither shows the browser's light default on the dark page."""
    run = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "const nav = el('div', 'row pickers');" in run
    assert "const sel = el('select', 'field');" in run
    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    assert ".pickers input[type=checkbox] {" in css and "accent-color: var(--accent)" in css


def test_the_resume_row_asks_what_a_bare_resume_runs_under() -> None:
    """The resume row's first options name what a resume without flags runs
    under: asked when the row appears and on a preset pick, and a late answer
    never replaces a newer one."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    composer = client[client.index("function makeComposer") :]
    assert "'/resume_defaults?preset=' + encodeURIComponent(preset.value)" in composer
    assert "if (request !== labelRequest) return;" in composer
    assert "preset.onchange = relabel;" in composer
    assert "if (finished && !rowShown) relabel();" in composer


def test_parallel_model_completion_handles_each_whole_fragment() -> None:
    """A repeated `/parallel` segment completes too, replacing the whole
    comma-delimited fragment when the caret sits in its middle."""
    client = resources.files("agent6.ui.web").joinpath("client.js").read_text(encoding="utf-8")
    suggest = client[
        client.index("function attachParallelSuggest") : client.index("// The new-work composer")
    ]
    assert "v.matchAll(/(^|\\s)\\/parallel(?=\\s|$)/g)" in suggest
    assert "while (fragEnd < end && v[fragEnd] !== ',') fragEnd++;" in suggest


def test_add_provider_does_not_keep_another_names_autofilled_url() -> None:
    config = resources.files("agent6.ui.web").joinpath("client_config.js")
    text = config.read_text(encoding="utf-8")
    prefill = text[text.index("name.oninput") : text.index("const repoRow")]
    assert "baseUrl.value === autofilled" in prefill
    assert "autofilled = '';" in prefill


def test_the_config_editor_keeps_a_list_as_toml_on_an_untouched_save() -> None:
    """The edit field must contain the server's round-trippable TOML value;
    joining a list with commas turned an untouched Save into a rejected string."""
    config = resources.files("agent6.ui.web").joinpath("client_config.js")
    text = config.read_text(encoding="utf-8")
    assert "const cur = s.input;" in text
    assert "s.value.join(',')" not in text


def test_the_config_editor_sends_a_string_leaf_as_a_toml_string() -> None:
    """A str leaf is posted quoted, as the TUI's editor sends it, so a value
    that parses as another TOML type (`true`, `42`, `[a]`) stays a string."""
    config = resources.files("agent6.ui.web").joinpath("client_config.js")
    assert "s.type === 'str' ? JSON.stringify(field.value) : field.value" in config.read_text(
        encoding="utf-8"
    )


def test_the_empty_machines_card_says_what_the_tui_says() -> None:
    """The web card read "no machine instances" where the TUI's machines screen
    says "no machines yet"."""
    assert "'no machines yet'" in CLIENT_JS and "no machine instances" not in CLIENT_JS


def test_the_hub_keeps_its_maintenance_actions_behind_one_control() -> None:
    """Two danger buttons and a checkbox sat under every session list; the
    actions the TUI's File menu holds open from one "more…" disclosure."""
    assert "el('details', 'more')" in CLIENT_JS
    for label in ("Prune merged runs", "Prune merged runs, squash-merged too", "Clear saved asks"):
        assert f"action('{label}'" in CLIENT_JS
    assert "also squash-merged branches" not in CLIENT_JS
