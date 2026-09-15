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

from agent6.ui.web.page import PAGE_HTML

# sha256 of PAGE_HTML.encode("utf-8"). An edit to page.py, client.js, or
# styles.css moves it; update it in the same commit as that edit.
PAGE_SHA256 = "c88db1025f7e548ec3896041249871c20c2e6114a81c89307e356f709677b717"


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
    populated = refresh.index("model.value = d.default || '';")
    assert request < cleared < awaited < stale_guard < populated


def test_the_config_editor_sends_a_string_leaf_as_a_toml_string() -> None:
    """A str leaf is posted quoted, as the TUI's editor sends it, so a value
    that parses as another TOML type (`true`, `42`, `[a]`) stays a string."""
    config = resources.files("agent6.ui.web").joinpath("client_config.js")
    assert "s.type === 'str' ? JSON.stringify(field.value) : field.value" in config.read_text(
        encoding="utf-8"
    )
