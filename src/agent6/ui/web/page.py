# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the web page: HTML, CSS and vanilla JS, served as one string at `GET /`.

The page renders the wire form the JSON and SSE endpoints emit; the domain logic
stays in the Python read side. The `client*.js` and `styles*.css` files beside this
module are read once at import time and spliced in, so `PAGE_HTML` is a constant
the server serves from memory and tests assert against.
"""

from __future__ import annotations

from importlib import resources

_ASSETS = resources.files(__package__)
# One script block, concatenated in declaration order: core and hub, layout, run, machine, config,
# the source views the Git and Files tabs share, git, files.
_CLIENT_FILES = (
    "client.js",
    "client_layout.js",
    "client_run.js",
    "client_machine.js",
    "client_config.js",
    "client_source.js",
    "client_review.js",
    "client_git_diff.js",
    "client_git.js",
    "client_files.js",
)
CLIENT_JS = "".join(_ASSETS.joinpath(name).read_text(encoding="utf-8") for name in _CLIENT_FILES)
# One style block, concatenated in declaration order: the shared base (tokens, shell, controls,
# composers), then each page's own rules, which may override the base.
_STYLE_FILES = (
    "styles.css",
    "styles_run.css",
    "styles_review.css",
    "styles_config.css",
    "styles_source.css",
    "styles_git.css",
    "styles_files.css",
    "styles_machine.css",
)
STYLES_CSS = "".join(_ASSETS.joinpath(name).read_text(encoding="utf-8") for name in _STYLE_FILES)

_NAV_ITEMS = (
    (
        "hub",
        "#/",
        "Sessions",
        '<path d="M4 5.5h10a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2H9l-3.5 3v-3H4'
        'a2 2 0 0 1-2-2v-4a2 2 0 0 1 2-2z"/>'
        '<path d="M18 9.5h2a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-.5v3l-3.5-3H12"/>',
    ),
    (
        "git",
        "#/git",
        "Git",
        '<circle cx="6.5" cy="5" r="2.2"/><circle cx="6.5" cy="19" r="2.2"/>'
        '<circle cx="17.5" cy="8" r="2.2"/><path d="M6.5 7.2v9.6"/>'
        '<path d="M17.5 10.2c0 3.6-4 4.6-7 5.6-1.6.5-3 1.1-4 2"/>',
    ),
    (
        "files",
        "#/files",
        "Files",
        '<rect x="5" y="3" width="14" height="18" rx="1.5"/><path d="M5 12h14"/>'
        '<path d="M10.5 7.5h3"/><path d="M10.5 16.5h3"/>',
    ),
    (
        "machines",
        "#/machines",
        "State machines",
        '<circle cx="6.5" cy="7" r="2.5"/><circle cx="17.5" cy="7" r="2.5"/>'
        '<circle cx="12" cy="17.5" r="2.5"/><path d="M9 7h6"/>'
        '<path d="M8 9.2l2.8 5.9"/><path d="M16 9.2l-2.8 5.9"/>',
    ),
    (
        "config",
        "#/config",
        "Config",
        '<path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h16"/>'
        '<circle cx="9" cy="7" r="1.8" fill="currentColor" stroke="none"/>'
        '<circle cx="15" cy="12" r="1.8" fill="currentColor" stroke="none"/>'
        '<circle cx="7" cy="17" r="1.8" fill="currentColor" stroke="none"/>',
    ),
)


def _icon(drawing: str) -> str:
    """Return a 24-unit line icon drawn in the text colour, hidden from assistive technology."""
    return (
        '<svg class="ic" viewBox="0 0 24 24" stroke-width="1.75" stroke-linecap="round" '
        'stroke-linejoin="round" fill="none" stroke="currentColor" aria-hidden="true">'
        f"{drawing}</svg>"
    )


_NAV_LINKS = "\n".join(
    f'<a href="{href}" data-tab="{tab}" title="{label}" aria-label="{label}">'
    f"{_icon(drawing)}<span>{label}</span></a>"
    for tab, href, label, drawing in _NAV_ITEMS
)
# A circle half filled; client.js names each button by the theme it switches to.
_THEME_ICON = _icon(
    '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor"/>'
)
# Two chevrons pointing left: a press folds the rail to its icons, then hides it.
_COLLAPSE_ICON = _icon('<path d="M11.5 18l-6-6 6-6"/><path d="M18.5 18l-6-6 6-6"/>')
# Three lines: the header's button that brings a hidden rail back.
_MENU_ICON = _icon('<path d="M4 6.5h16"/><path d="M4 12h16"/><path d="M4 17.5h16"/>')

# A hash-routed page: live views open an event stream, static views fetch a snapshot.
PAGE_HTML = (
    r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f6f8fa" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0e1116" media="(prefers-color-scheme: dark)">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<title>agent6</title>
<script>try { const c = document.documentElement.classList, t = localStorage.getItem("a6-theme"), r = localStorage.getItem("a6-rail"); if (t === "light" || t === "dark") c.add(t); if (r === "min" || r === "hidden") c.add("rail-" + r); } catch (_) {}</script>
<style>
"""
    + STYLES_CSS
    + r"""</style>
</head>
<body>
<aside class="rail">
  <a class="rail-brand" href="#/" aria-label="agent6 home"><img src="/favicon.svg" width="24" height="24" alt=""><b>agent6</b></a>
  <nav class="rail-nav">
"""
    + _NAV_LINKS
    + r"""
  </nav>
  <div class="rail-foot">
    <button type="button" class="rail-row theme-toggle">"""
    + _THEME_ICON
    + r"""<span class="rail-label">Theme</span></button>
    <button type="button" class="rail-row" id="rail-toggle">"""
    + _COLLAPSE_ICON
    + r"""<span class="rail-label">Collapse</span></button>
  </div>
</aside>
<div class="content">
<header>
  <button type="button" id="rail-show">"""
    + _MENU_ICON
    + r"""</button>
  <a class="brand" href="#/" aria-label="agent6 home"><img src="/favicon.svg" width="22" height="22" alt=""><b>agent6</b></a>
  <span class="crumb" id="crumb"></span>
  <button type="button" class="theme-toggle">"""
    + _THEME_ICON
    + r"""</button>
</header>

<main id="view"><div class="empty">loading…</div></main>
</div>

<nav class="tabs">
"""
    + _NAV_LINKS
    + r"""
</nav>
<div id="toasts" role="status" aria-live="polite" aria-atomic="false"></div>

<script>
"""
    + CLIENT_JS
    + r"""</script>
</body>
</html>
"""
)


# The PWA manifest; start_url "." keeps it relative to the mount (a served path prefix differs).
MANIFEST_JSON = r"""{
  "name": "agent6",
  "short_name": "agent6",
  "start_url": ".",
  "scope": ".",
  "display": "standalone",
  "background_color": "#0e1116",
  "theme_color": "#0e1116",
  "icons": [
    { "src": "icon.svg", "type": "image/svg+xml", "sizes": "any", "purpose": "any maskable" }
  ]
}
"""

# Required with the manifest for installability; a network passthrough, no caching, no push.
SERVICE_WORKER_JS = r"""self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});
"""

# docs/assets/favicon.svg verbatim (a test pins the two equal), so the tab matches the docs site.
FAVICON_SVG = _ASSETS.joinpath("favicon.svg").read_text(encoding="utf-8")
# The PWA tile: the same mark inside the safe-area inset a maskable icon needs.
ICON_SVG = _ASSETS.joinpath("icon.svg").read_text(encoding="utf-8")
