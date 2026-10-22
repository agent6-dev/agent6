# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the web page: HTML, CSS and vanilla JS, served as one string at `GET /`.

The page renders the wire form the JSON and SSE endpoints emit; the domain logic
stays in the Python read side. The `client*.js` files and `styles.css` beside this
module are read once at import time and spliced in, so `PAGE_HTML` is a constant
the server serves from memory and tests assert against.
"""

from __future__ import annotations

from importlib import resources

_ASSETS = resources.files(__package__)
# One script block, concatenated in declaration order: core and hub, run, machine, config,
# the source views the Git and Files tabs share, git, files.
_CLIENT_FILES = (
    "client.js",
    "client_run.js",
    "client_machine.js",
    "client_config.js",
    "client_source.js",
    "client_git.js",
    "client_files.js",
)
CLIENT_JS = "".join(_ASSETS.joinpath(name).read_text(encoding="utf-8") for name in _CLIENT_FILES)
STYLES_CSS = _ASSETS.joinpath("styles.css").read_text(encoding="utf-8")

# A hash-routed page: live views open an event stream, static views fetch a snapshot.
PAGE_HTML = (
    r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e1116">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<title>agent6</title>
<style>
"""
    + STYLES_CSS
    + r"""</style>
</head>
<body>
<aside class="rail">
  <a class="rail-brand" href="#/" aria-label="agent6 home"><img src="/icon.svg" width="24" height="24" alt=""><b>agent6</b></a>
  <nav class="rail-nav">
    <a href="#/" data-tab="hub" title="Sessions"><span class="ic">▤</span><span>Sessions</span></a>
    <a href="#/machines" data-tab="machines" title="Machines"><span class="ic">◈</span><span>Machines</span></a>
    <a href="#/config" data-tab="config" title="Config"><span class="ic">⚙</span><span>Config</span></a>
    <a href="#/git" data-tab="git" title="Git"><span class="ic">⎇</span><span>Git</span></a>
    <a href="#/files" data-tab="files" title="Files"><span class="ic">▣</span><span>Files</span></a>
  </nav>
  <span class="rail-gap"></span>
  <button onclick="toggleTheme()" title="theme">◐<span class="rail-label"> theme</span></button>
  <button id="rail-toggle" onclick="toggleRail()" title="collapse the sidebar"><span id="rail-arrow">«</span><span class="rail-label"> collapse</span></button>
</aside>
<div class="content">
<header>
  <span class="brand" onclick="location.hash='#/'"><b>agent6</b></span>
  <span class="crumb" id="crumb"></span>
  <span class="spacer"></span>
  <button onclick="toggleTheme()" title="theme">◐</button>
</header>

<main id="view"><div class="empty">loading…</div></main>
</div>

<nav class="tabs">
  <a href="#/" data-tab="hub"><span class="ic">▤</span>Sessions</a>
  <a href="#/machines" data-tab="machines"><span class="ic">◈</span>Machines</a>
  <a href="#/config" data-tab="config"><span class="ic">⚙</span>Config</a>
  <a href="#/git" data-tab="git"><span class="ic">⎇</span>Git</a>
  <a href="#/files" data-tab="files"><span class="ic">▣</span>Files</a>
</nav>

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
