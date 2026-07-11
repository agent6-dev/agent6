# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web-UI page: HTML + CSS + vanilla JS, served as one string.

Served verbatim by web.server at `GET /`. It renders the wire form the JSON / SSE
endpoints emit (the same shape as `agent6 watch --json`); it is a thin renderer,
so all domain logic stays in the Python read-side.

Kept as a module-level constant so the server has nothing to read from disk and
tests can assert against it directly.
"""

from __future__ import annotations

# The page is a hash-routed SPA: #/ hub, #/run/<id>, #/machine/<name>,
# #/transcript/<id>, #/config. Live views open an EventSource against the
# matching /events endpoint; static views fetch a snapshot. Writes are small JSON
# POSTs (new work / steer / approve / answer / merge / prune / config set /
# machine create+run) to the typed endpoints, never arbitrary execution.
PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e1116">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<title>agent6</title>
<style>
:root {
  --bg: #0e1116; --surface: #161b22; --surface2: #1c2230; --border: #2a3140;
  --text: #d7dde5; --muted: #8b95a5; --accent: #6ea8fe; --accent2: #b48ead;
  --ok: #4ec9a5; --warn: #e2c08d; --err: #f07178; --nav-h: 56px;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
:root.light {
  --bg: #f6f8fa; --surface: #ffffff; --surface2: #eef1f5; --border: #d5dae1;
  --text: #1b2028; --muted: #5a6472; --accent: #2d6fe0; --accent2: #8250df;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, sans-serif;
  -webkit-text-size-adjust: 100%;
}
a { color: var(--accent); text-decoration: none; }
code, pre, .mono { font-family: var(--mono); }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; }

header {
  position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 12px;
  padding: 10px 16px; background: var(--surface); border-bottom: 1px solid var(--border);
  padding-top: calc(10px + env(safe-area-inset-top));
}
header .brand { font-weight: 700; letter-spacing: .3px; }
header .brand b { color: var(--accent); }
header .spacer { flex: 1; }
header .crumb { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
button, .btn {
  font: inherit; color: var(--text); background: var(--surface2); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 12px; cursor: pointer; min-height: 40px;
}
button:hover, .btn:hover { border-color: var(--accent); }
button:active { transform: translateY(1px); }
button:disabled { opacity: .45; cursor: not-allowed; }
button:disabled:hover { border-color: var(--border); }

nav.tabs {
  display: none; position: fixed; bottom: 0; left: 0; right: 0; z-index: 20;
  background: var(--surface); border-top: 1px solid var(--border);
  height: calc(var(--nav-h) + env(safe-area-inset-bottom));
  padding-bottom: env(safe-area-inset-bottom);
}
nav.tabs a {
  flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
  color: var(--muted); font-size: 11px; gap: 2px; padding: 6px 0;
}
nav.tabs a.active { color: var(--accent); }
nav.tabs a .ic { font-size: 18px; line-height: 1; }

main { padding: 16px; max-width: 1200px; margin: 0 auto; }
.grid { display: grid; gap: 14px; }
@media (min-width: 900px) { .grid.cols2 { grid-template-columns: 1fr 1fr; } }

.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px; overflow: hidden;
}
.card h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); }
.card.scroll { max-height: 420px; overflow: auto; }

.row { display: flex; gap: 10px; align-items: center; }
.wrap { flex-wrap: wrap; }
.muted { color: var(--muted); }
.pill { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); }
.pill.ok { color: var(--ok); border-color: var(--ok); }
.pill.done { color: var(--accent); border-color: var(--accent); }
.pill.running { color: var(--warn); border-color: var(--warn); }
.pill.stale, .pill.failed { color: var(--err); border-color: var(--err); }

.list { display: flex; flex-direction: column; gap: 8px; }
.item {
  display: flex; gap: 10px; align-items: center; padding: 12px; border-radius: 10px;
  background: var(--surface2); border: 1px solid transparent; cursor: pointer;
}
.item:hover { border-color: var(--accent); }
.item .grow { flex: 1; min-width: 0; }
.item .title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.item .sub { font-size: 12px; color: var(--muted); }

.kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 13px; }
.kv .k { color: var(--muted); }

.bar { height: 8px; background: var(--surface2); border-radius: 999px; overflow: hidden; }
.bar > span { display: block; height: 100%; background: var(--accent); }
.bar.warn > span { background: var(--warn); }

.tree { font-family: var(--mono); font-size: 13px; }
.tree .node { padding: 1px 0; white-space: pre; overflow: hidden; text-overflow: ellipsis; }
.tree .cursor { color: var(--accent); font-weight: 700; }
.st-passed { color: var(--ok); } .st-failed { color: var(--err); }
.st-in_progress { color: var(--warn); } .st-pending { color: var(--muted); }
.st-skipped, .st-obsolete { color: var(--muted); text-decoration: line-through; }

table.tools { width: 100%; border-collapse: collapse; font-size: 13px; }
table.tools td { padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
table.tools .name { font-family: var(--mono); white-space: nowrap; }
table.tools .args { color: var(--muted); font-family: var(--mono); word-break: break-word; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.dot.ok { background: var(--ok); } .dot.bad { background: var(--err); }

.log { font-family: var(--mono); font-size: 12px; line-height: 1.4; }
.log div { white-space: pre-wrap; word-break: break-word; }

.diff { font-family: var(--mono); font-size: 12px; }
.diff .add { color: var(--ok); } .diff .del { color: var(--err); } .diff .hunk { color: var(--accent2); }

.turn { padding: 10px 0; border-bottom: 1px solid var(--border); }
.turn .who { font-weight: 700; font-size: 12px; text-transform: uppercase; color: var(--muted); }
.turn.assistant .who { color: var(--accent); }
.turn.user .who { color: var(--ok); }
.turn.tool .who { color: var(--accent2); }
.turn .think { color: var(--muted); font-style: italic; border-left: 2px solid var(--border); padding-left: 8px; margin: 6px 0; }

.cfg { width: 100%; border-collapse: collapse; font-size: 13px; }
.cfg td, .cfg th { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
.cfg .key { font-family: var(--mono); word-break: break-all; }
.cfg .val { font-family: var(--mono); word-break: break-word; }
.cfg tr.mod .key { color: var(--accent); }
.cfg .src { color: var(--muted); white-space: nowrap; }
input.filter {
  width: 100%; font: inherit; color: var(--text); background: var(--surface2);
  border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; min-height: 42px;
}
.empty { color: var(--muted); padding: 24px; text-align: center; }
.err { color: var(--err); }

textarea.field, select.field, input.field {
  width: 100%; font: inherit; color: var(--text); background: var(--surface2);
  border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; min-height: 42px;
}
textarea.field { min-height: 72px; resize: vertical; }
.form-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
button.primary { background: var(--accent); color: #05121f; border-color: var(--accent); font-weight: 600; }
:root.light button.primary { color: #fff; }
button.danger:hover { border-color: var(--err); color: var(--err); }
.prompt-box { background: var(--surface2); border: 1px solid var(--warn); border-radius: 10px; padding: 12px; margin-bottom: 10px; }
.prompt-box .q { margin-bottom: 8px; }
.toast {
  position: fixed; left: 50%; transform: translateX(-50%); z-index: 50;
  bottom: calc(20px + env(safe-area-inset-bottom)); max-width: 90vw;
  background: var(--surface); border: 1px solid var(--accent); border-radius: 10px;
  padding: 10px 16px; box-shadow: 0 4px 16px rgba(0,0,0,.4);
}
.toast.bad { border-color: var(--err); color: var(--err); }
.notif-banner { display: flex; align-items: flex-start; gap: 10px; background: var(--surface2); border: 1px solid var(--accent); border-left-width: 4px; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
.notif-banner.warn { border-color: var(--warn); }
.notif-banner.error { border-color: var(--err); }
.notif-banner .grow { flex: 1; min-width: 0; }
.notif-banner .nb-msg { word-break: break-word; }
.notif-banner .nb-sub { font-size: 12px; color: var(--muted); }
.notif-banner .nb-x { cursor: pointer; color: var(--muted); background: none; border: none; min-height: auto; padding: 0 4px; font-size: 16px; }
.poke-box { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin-top: 10px; }

.overlay {
  position: fixed; inset: 0; z-index: 60; padding: 16px;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0,0,0,.55);
}
.overlay .card { max-height: 90vh; overflow: auto; }

@media (max-width: 780px) {
  nav.tabs { display: flex; }
  main { padding: 12px 12px calc(var(--nav-h) + 24px); }
  .card.scroll { max-height: 60vh; }
  header .desktop-only { display: none; }
}

/* --- desktop: a persistent left nav rail (>=781px); phones keep the bottom
   tab bar and the plain single-column stack, untouched. --- */
aside.rail { display: none; }
@media (min-width: 781px) {
  aside.rail {
    display: flex; flex-direction: column; gap: 4px; z-index: 30;
    position: fixed; left: 0; top: 0; width: 216px; height: 100vh; overflow: auto;
    padding: 16px 12px calc(16px + env(safe-area-inset-bottom));
    background: var(--surface); border-right: 1px solid var(--border);
  }
  aside.rail .rail-brand { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 17px; padding: 6px 8px 16px; cursor: pointer; }
  aside.rail .rail-brand b { color: var(--accent); }
  aside.rail .rail-brand img { border-radius: 6px; }
  aside.rail .rail-nav { display: flex; flex-direction: column; gap: 2px; }
  aside.rail .rail-nav a { display: flex; align-items: center; gap: 12px; padding: 10px 12px; border-radius: 8px; color: var(--muted); }
  aside.rail .rail-nav a:hover { background: var(--surface2); color: var(--text); }
  aside.rail .rail-nav a.active { background: var(--surface2); color: var(--accent); }
  aside.rail .rail-nav .ic { font-size: 16px; width: 18px; text-align: center; }
  aside.rail .rail-gap { flex: 1; }
  .content { margin-left: 216px; }
  header { justify-content: flex-start; }
  header .brand, header > button { display: none; }  /* the rail owns brand + actions */
  main { max-width: 1280px; margin: 0; padding: 20px 28px; }
}

/* --- run dashboard: a real multi-pane layout on wide screens, not a 2-col
   reflow. Reasoning / tools / log run down the main column; the task graph,
   budget, and latest commit sit in a narrower side column. --- */
@media (min-width: 1024px) {
  .run-grid {
    grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr);
    grid-template-areas: "head head" "role tasks" "tools budget" "log diff";
  }
  .run-grid .card-head { grid-area: head; }
  .run-grid .card-role { grid-area: role; }
  .run-grid .card-tasks { grid-area: tasks; }
  .run-grid .card-tools { grid-area: tools; }
  .run-grid .card-budget { grid-area: budget; }
  .run-grid .card-log { grid-area: log; }
  .run-grid .card-diff { grid-area: diff; }
}
</style>
</head>
<body>
<aside class="rail">
  <div class="rail-brand" onclick="location.hash='#/'"><img src="/icon.svg" width="24" height="24" alt=""><b>agent6</b></div>
  <nav class="rail-nav">
    <a href="#/" data-tab="hub"><span class="ic">▤</span><span>Runs</span></a>
    <a href="#/machines" data-tab="machines"><span class="ic">◈</span><span>Machines</span></a>
    <a href="#/config" data-tab="config"><span class="ic">⚙</span><span>Config</span></a>
  </nav>
  <span class="rail-gap"></span>
  <button onclick="toggleTheme()" title="theme">◐ theme</button>
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
  <a href="#/" data-tab="hub"><span class="ic">▤</span>Runs</a>
  <a href="#/machines" data-tab="machines"><span class="ic">◈</span>Machines</a>
  <a href="#/config" data-tab="config"><span class="ic">⚙</span>Config</a>
</nav>

<script>
"use strict";
const view = document.getElementById('view');
const crumb = document.getElementById('crumb');
let live = null; // the active EventSource, closed on navigation
let activeOverlayClose = null; // an open modal dialog's dismisser, closed on navigation

// --- theme -------------------------------------------------------------------
if (localStorage.getItem('a6-theme') === 'light') document.documentElement.classList.add('light');
function toggleTheme() {
  const on = document.documentElement.classList.toggle('light');
  localStorage.setItem('a6-theme', on ? 'light' : 'dark');
}

// --- PWA + notifications -----------------------------------------------------
// Install the service worker so the page is an installable PWA (manifest + SW).
// No Web Push / VAPID: OS notifications are the foreground Notification API only.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(()=>{}); });
}
// Ask for OS-notification permission on a user gesture (browsers block passive
// requests). A granted permission lets machine.notify/end pop a desktop/PWA
// notification even when the tab is backgrounded (on desktop).
function enableNotifications() {
  if (!('Notification' in window)) { toast('notifications not supported', true); return; }
  Notification.requestPermission().then(p => toast(p === 'granted' ? 'notifications on' : 'notifications ' + p));
}
// Fire an OS notification when permitted; always safe (never throws into a repaint).
function osNotify(title, body) {
  try { if ('Notification' in window && Notification.permission === 'granted') new Notification(title, { body: body || '', icon: '/icon.svg' }); } catch (_) {}
}

// --- helpers -----------------------------------------------------------------
const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const esc = s => (s == null ? '' : String(s));
async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error((await r.json().catch(()=>({error:r.statusText}))).error || r.statusText); return r.json(); }
async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) throw new Error(data.error || r.statusText);
  return data;
}
function toast(msg, bad) { const t = el('div', 'toast' + (bad ? ' bad' : ''), msg); document.body.appendChild(t); setTimeout(() => t.remove(), 4000); }
function fmtUsd(u) { return u ? '$' + Number(u).toFixed(4) : '$0'; }
function when(ts) { if (!ts) return ''; const d = new Date(ts * 1000); return d.toLocaleString(); }
function setCrumb(t) { crumb.textContent = t || ''; }
function closeLive() { if (live) { live.close(); live = null; } }
function closeOverlay() { if (activeOverlayClose) activeOverlayClose(); }
function pill(status) { const p = el('span', 'pill ' + esc(status), esc(status)); return p; }

function setTab(name) {
  document.querySelectorAll('nav.tabs a, aside.rail .rail-nav a').forEach(a => a.classList.toggle('active', a.dataset.tab === name));
}

// --- router ------------------------------------------------------------------
async function route() {
  closeLive();
  closeOverlay();
  const h = location.hash.replace(/^#/, '') || '/';
  const parts = h.split('/').filter(Boolean); // e.g. ['run','abc']
  try {
    if (parts.length === 0) { setTab('hub'); await renderHub(); }
    else if (parts[0] === 'machines') { setTab('machines'); await renderHub('machines'); }
    else if (parts[0] === 'config') { setTab('config'); await renderConfig(); }
    else if (parts[0] === 'run' && parts[1]) { setTab('hub'); renderRun(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'transcript' && parts[1]) { setTab('hub'); await renderTranscript(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'machine' && parts[1]) { setTab('machines'); renderMachine(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'draft' && parts[1]) { const n = decodeURIComponent(parts[1]); setTab('machines'); renderRun(n, { base: '/api/draft/' + encodeURIComponent(n), readOnly: true, title: 'Machine draft', crumb: 'draft ' + n }); }
    else { view.innerHTML = ''; view.appendChild(el('div', 'empty', 'not found')); }
  } catch (e) {
    view.innerHTML = '';
    view.appendChild(el('div', 'empty err', 'error: ' + e.message));
  }
}
window.addEventListener('hashchange', route);

// --- hub ---------------------------------------------------------------------
function newWorkCard() {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'New work'));
  const task = el('textarea', 'field'); task.placeholder = 'task / question…';
  card.appendChild(task);
  const row = el('div', 'form-row');
  const mode = el('select', 'field'); mode.style.flex = '0 0 auto'; mode.style.width = 'auto';
  for (const m of ['run', 'plan', 'ask']) { const o = el('option', null, m); o.value = m; mode.appendChild(o); }
  row.appendChild(mode);
  const go = el('button', 'primary', 'Start');
  go.onclick = async () => {
    if (!task.value.trim()) return;
    go.disabled = true;
    try { const d = await postJSON('/api/new', { mode: mode.value, task: task.value }); if (d.run_id) location.hash = '#/run/' + encodeURIComponent(d.run_id); }
    catch (e) { toast(e.message, true); go.disabled = false; }
  };
  row.appendChild(go);
  card.appendChild(row);
  return card;
}

function machineControls() {
  const wrap = el('div');
  const ct = el('textarea', 'field'); ct.placeholder = 'describe a machine to create…'; ct.style.minHeight = '52px';
  wrap.appendChild(ct);
  const row = el('div', 'form-row');
  const cbtn = el('button', null, 'Create machine');
  cbtn.onclick = async () => {
    if (!ct.value.trim()) return; cbtn.disabled = true;
    try { const d = await postJSON('/api/machine/create', { task: ct.value }); ct.value=''; if (d.draft) location.hash = '#/draft/' + encodeURIComponent(d.draft); }
    catch (e) { toast(e.message, true); cbtn.disabled = false; }
  };
  row.appendChild(cbtn);
  wrap.appendChild(row);
  return wrap;
}

async function renderHub(focus) {
  setCrumb('');
  const data = await getJSON('/api/hub');
  view.innerHTML = '';
  const runsCard = el('div', 'card');
  runsCard.appendChild(el('h2', null, 'Runs'));
  const runsList = el('div', 'list');
  if (!data.runs.length) runsList.appendChild(el('div', 'empty', 'no runs yet'));
  for (const r of data.runs) {
    const it = el('div', 'item');
    it.onclick = () => location.hash = '#/run/' + encodeURIComponent(r.id);
    const g = el('div', 'grow');
    g.appendChild(el('div', 'title', r.task || '(no task)'));
    g.appendChild(el('div', 'sub', `${esc(r.mode)} · ${r.id.slice(0,12)} · ${when(r.mtime)} · ${fmtUsd(r.usd)}`));
    it.appendChild(g);
    it.appendChild(pill(r.status));
    runsList.appendChild(it);
  }
  runsCard.appendChild(runsList);
  const prune = el('button', 'danger'); prune.textContent = 'Prune merged runs'; prune.style.marginTop = '10px';
  prune.onclick = async () => { try { const d = await postJSON('/api/runs/prune', {}); toast(d.message || 'pruned'); route(); } catch (e) { toast(e.message, true); } };
  runsCard.appendChild(prune);

  const mCard = el('div', 'card');
  mCard.appendChild(el('h2', null, 'Machines'));
  const mList = el('div', 'list');
  if (!data.machines.length) mList.appendChild(el('div', 'empty', 'no machine instances'));
  for (const m of data.machines) {
    const it = el('div', 'item');
    it.onclick = () => location.hash = '#/machine/' + encodeURIComponent(m.name);
    const g = el('div', 'grow');
    g.appendChild(el('div', 'title', m.machine || m.name));
    g.appendChild(el('div', 'sub', `${m.name} · at ${esc(m.current || '?')} · ${when(m.mtime)}`));
    it.appendChild(g);
    it.appendChild(pill(m.status));
    mList.appendChild(it);
  }
  mCard.appendChild(mList);
  // Authored machine files: run one, or view its structure.
  if ((data.machine_files||[]).length) {
    mCard.appendChild(el('div', 'sub muted', 'run a machine:'));
    const frow = el('div', 'form-row');
    for (const mf of data.machine_files) {
      const b = el('button', null, '▶ ' + mf.name);
      b.onclick = async () => { try { await postJSON('/api/machine/run', { file: mf.path }); toast('started ' + mf.name); setTimeout(route, 800); } catch (e) { toast(e.message, true); } };
      frow.appendChild(b);
    }
    mCard.appendChild(frow);
  }
  mCard.appendChild(machineControls());

  const nCard = newWorkCard();
  const grid = el('div', 'grid cols2');
  // On the Machines tab (phone), lead with machines; else lead with new-work + runs.
  if (focus === 'machines') { grid.appendChild(mCard); grid.appendChild(runsCard); }
  else { grid.appendChild(nCard); grid.appendChild(runsCard); grid.appendChild(mCard); }
  view.appendChild(grid);
}

// --- run dashboard -----------------------------------------------------------
// A multi-line steer dialog (browser prompt() is single-line). onResult(text|null):
// the instruction to send (may be multi-line), or null to cancel. Steering never
// stops the run -- that is the separate Stop button.
function steerDialog(title, onResult) {
  const back = el('div'); back.className = 'overlay';
  const box = el('div', 'card'); box.style.width = 'min(680px, 92vw)';
  box.appendChild(el('h2', null, title));
  const ta = el('textarea', 'field'); ta.placeholder = 'instruction (blank = continue)';
  ta.style.minHeight = '120px'; box.appendChild(ta);
  const row = el('div', 'form-row');
  const send = el('button', 'primary', 'Send'), cont = el('button', null, 'Continue'), cancel = el('button', null, 'Cancel');
  row.appendChild(send); row.appendChild(cont); row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back); ta.focus();
  const close = (r) => { activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey); onResult(r); };
  activeOverlayClose = () => close(null); // navigating away dismisses it (no orphaned overlay/listener)
  function onKey(e) { if (e.key === 'Escape') close(null); }
  document.addEventListener('keydown', onKey);
  send.onclick = () => close(ta.value); cont.onclick = () => close(''); cancel.onclick = () => close(null);
  back.onclick = (e) => { if (e.target === back) close(null); };
}

async function stopRun(base, label) {
  if (!confirm('Stop ' + label + '? It ends now and can be resumed later.')) return;
  try { await postJSON(base + '/steer', { text: 'abort' }); toast('stopping…'); } catch (e) { toast(e.message, true); }
}

// opts: { base, readOnly, title } — a draft (machine-create authoring log) is
// watched read-only against /api/draft/<name>; a run is driveable at /api/run/<id>.
function renderRun(id, opts) {
  opts = opts || {};
  const base = opts.base || ('/api/run/' + encodeURIComponent(id));
  const readOnly = !!opts.readOnly;
  setCrumb(opts.crumb || id.slice(0, 16));
  view.innerHTML = '';
  const prompts = el('div'); view.appendChild(prompts); // approval/question boxes surface here
  const grid = el('div', 'grid run-grid');
  const cards = { _id: id, _prompts: prompts, _readOnly: readOnly };
  const mk = (key, title, cls) => { const c = el('div', 'card card-' + key + ' ' + (cls||'')); c.appendChild(el('h2', null, title)); const body = el('div'); c.appendChild(body); cards[key] = body; grid.appendChild(c); return body; };
  mk('head', opts.title || 'Run');
  mk('budget', 'Budget');
  mk('tasks', 'Task graph', 'scroll');
  mk('role', 'Reasoning', 'scroll');
  mk('tools', 'Tool calls', 'scroll');
  mk('log', 'Event log', 'scroll');
  mk('diff', 'Latest commit', 'scroll');
  if (!readOnly) {  // controls at the TOP so Steer/Stop are reachable without scrolling
    const actions = el('div', 'row wrap'); actions.style.marginBottom = '14px';
    const steerBtn = el('button', null, '↪ Steer');
    steerBtn.onclick = () => steerDialog('Steer this run', async (text) => {
      if (text === null) return;
      try { await postJSON('/api/run/' + encodeURIComponent(id) + '/steer', { text }); toast('steer sent'); } catch (e) { toast(e.message, true); }
    });
    const stopBtn = el('button', 'danger', '■ Stop');
    stopBtn.onclick = () => stopRun('/api/run/' + encodeURIComponent(id), 'the run');
    const mergeBtn = el('button', null, '⑃ Merge');
    mergeBtn.onclick = async () => { try { const d = await postJSON('/api/run/' + encodeURIComponent(id) + '/merge', {}); toast(d.message || 'merged'); } catch (e) { toast(e.message, true); } };
    const tbtn = el('button', null, 'Transcript →');
    tbtn.onclick = () => location.hash = '#/transcript/' + encodeURIComponent(id);
    actions.appendChild(steerBtn); actions.appendChild(stopBtn); actions.appendChild(mergeBtn); actions.appendChild(tbtn);
    cards._steer = steerBtn; cards._stop = stopBtn; // paintRun disables these once the run is finished
    view.appendChild(actions);
  }
  view.appendChild(grid);

  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    paintRun(cards, s);
    if (s.finished) closeLive(); // run is done; stop the stream so it doesn't reconnect
  };
  live.onerror = () => { /* EventSource auto-retries a live run; leave last paint up */ };
}

// Render the run's unanswered approval / ask_user prompts as actionable boxes.
// Reconcile by id: keep existing boxes so a repaint (any SSE frame) never wipes a
// half-typed free-text answer or drops focus; only add new prompts and remove
// resolved ones.
function paintPrompts(cards, s) {
  const host = cards._prompts;
  // base is the POST prefix: runs use /api/run/<id>, machines /api/machine/<name>.
  const base = cards._base || ('/api/run/' + encodeURIComponent(cards._id));
  // For a machine, the per-state dir the reasoning (and its prompts) came from.
  // Prompt ids reset per state (approval-1 in every state), so the answer must
  // carry it AND the box key must include it: when the machine advances to a new
  // state, the key changes so the stale box is rebuilt rather than reused with a
  // now-wrong prompt still showing.
  const state = cards._state || '';
  const pfx = state ? state + ':' : '';
  const extra = state ? { state } : {};
  const build = {};
  for (const ap of (s.pending_approvals || [])) {
    if (ap.answered) continue;
    build[pfx + 'ap:' + ap.id] = () => {
      const box = el('div', 'prompt-box');
      box.appendChild(el('div', 'q', ap.prompt || 'Approve this action?'));
      const row = el('div', 'form-row');
      const yes = el('button', 'primary', 'Approve');
      const sess = el('button', 'primary', 'Allow session');
      const no = el('button', 'danger', 'Deny');
      const send = (ok, session) => async () => { try { await postJSON(base + '/approve', { id: ap.id, approved: ok, session: !!session, ...extra }); } catch (e) { toast(e.message, true); } };
      yes.onclick = send(true, false); sess.onclick = send(true, true); no.onclick = send(false, false);
      row.appendChild(yes); row.appendChild(sess); row.appendChild(no); box.appendChild(row);
      return box;
    };
  }
  for (const q of (s.pending_questions || [])) {
    if (q.answered) continue;
    build[pfx + 'q:' + q.id] = () => {
      const box = el('div', 'prompt-box');
      box.appendChild(el('div', 'q', q.question || 'The agent asked a question'));
      const row = el('div', 'form-row');
      for (const opt of (q.options || [])) {
        const b = el('button', null, opt);
        b.onclick = async () => { try { await postJSON(base + '/answer', { id: q.id, answer: opt, ...extra }); } catch (e) { toast(e.message, true); } };
        row.appendChild(b);
      }
      const inp = el('input', 'field'); inp.placeholder = 'or type an answer…'; inp.style.flex = '1';
      const send = el('button', 'primary', 'Send');
      send.onclick = async () => { try { await postJSON(base + '/answer', { id: q.id, answer: inp.value, ...extra }); } catch (e) { toast(e.message, true); } };
      row.appendChild(inp); row.appendChild(send); box.appendChild(row);
      return box;
    };
  }
  const want = new Set(Object.keys(build));
  for (const child of Array.from(host.children)) {
    if (!want.has(child.dataset.key)) child.remove(); // resolved / gone
  }
  const present = new Set(Array.from(host.children).map(c => c.dataset.key));
  for (const key of want) {
    if (present.has(key)) continue; // leave the live box (input + focus) intact
    const box = build[key](); box.dataset.key = key; host.appendChild(box);
  }
}

function paintRun(cards, s) {
  if (!cards._readOnly) paintPrompts(cards, s);
  // Steer/Stop only mean something on a live run; a finished run ignores the bridge.
  if (cards._steer) { cards._steer.disabled = s.finished; cards._stop.disabled = s.finished; }
  // header
  cards.head.innerHTML = '';
  const kv = el('div', 'kv');
  const add = (k, v) => { kv.appendChild(el('div', 'k', k)); kv.appendChild(el('div', 'v', v)); };
  add('task', s.user_task || '(none)');
  add('id', s.run_id || '');
  add('state', s.status_label || (s.finished ? 'finished' : 'running'));
  cards.head.appendChild(kv);
  if (s.last_role) {
    const r = s.last_role;
    cards.head.appendChild(el('div', 'sub muted', `${esc(r.role)} / ${esc(r.model)}${r.in_flight ? ' …' : ''}`));
  }

  // budget
  const b = s.budget || {};
  cards.budget.innerHTML = '';
  const inFrac = b.input_cap ? Math.min(1, b.input_total / b.input_cap) : 0;
  const outFrac = b.output_cap ? Math.min(1, b.output_total / b.output_cap) : 0;
  const barRow = (label, frac, tot, cap) => {
    const w = el('div'); w.appendChild(el('div', 'sub muted', `${label}: ${tot}${cap ? ' / ' + cap : ''}`));
    const bar = el('div', 'bar' + (frac > 0.85 ? ' warn' : '')); const sp = el('span'); sp.style.width = (frac*100)+'%'; bar.appendChild(sp); w.appendChild(bar); return w;
  };
  cards.budget.appendChild(barRow('input tokens', inFrac, b.input_total||0, b.input_cap||0));
  cards.budget.appendChild(barRow('output tokens', outFrac, b.output_total||0, b.output_cap||0));
  cards.budget.appendChild(el('div', 'sub muted', 'cost: ' + fmtUsd(b.usd_total) + (b.usd_partial ? ' (partial)' : '')));

  // task tree
  cards.tasks.innerHTML = '';
  const tree = el('div', 'tree');
  if (!(s.tasks||[]).length) tree.appendChild(el('div', 'muted', 'no task graph yet'));
  for (const t of s.tasks || []) {
    const line = el('div', 'node' + (t.is_cursor ? ' cursor' : ''));
    const glyph = { passed:'✓', failed:'✗', in_progress:'▸', pending:'·', skipped:'–', obsolete:'×' }[t.status] || '·';
    line.appendChild(el('span', 'st-' + t.status, '  '.repeat(t.depth) + glyph + ' '));
    line.appendChild(document.createTextNode(t.title));
    tree.appendChild(line);
  }
  cards.tasks.appendChild(tree);

  // reasoning (thinking + streamed text)
  cards.role.innerHTML = '';
  if (s.last_role && (s.last_role.streamed_thinking || s.last_role.streamed_text)) {
    if (s.last_role.streamed_thinking) { const t = el('pre', 'think'); t.textContent = s.last_role.streamed_thinking; cards.role.appendChild(t); }
    if (s.last_role.streamed_text) { const t = el('pre'); t.textContent = s.last_role.streamed_text; cards.role.appendChild(t); }
  } else { cards.role.appendChild(el('div', 'muted', 'waiting for the model…')); }
  cards.role.scrollTop = cards.role.scrollHeight;

  // tools
  cards.tools.innerHTML = '';
  const tbl = el('table', 'tools');
  for (const tc of (s.tool_calls||[]).slice(-30)) {
    const tr = el('tr');
    const d = el('td'); d.appendChild(el('span', 'dot ' + (tc.ok === null ? '' : tc.ok ? 'ok' : 'bad'))); tr.appendChild(d);
    tr.appendChild(el('td', 'name', tc.name));
    const a = el('td', 'args'); a.textContent = tc.args_preview + (tc.result_summary ? '  → ' + tc.result_summary : ''); tr.appendChild(a);
    tbl.appendChild(tr);
  }
  if (!(s.tool_calls||[]).length) cards.tools.appendChild(el('div', 'muted', 'no tool calls yet'));
  else cards.tools.appendChild(tbl);

  // log
  cards.log.innerHTML = '';
  const log = el('div', 'log');
  for (const line of (s.log_tail||[]).slice(-200)) log.appendChild(el('div', null, line));
  cards.log.appendChild(log);
  cards.log.scrollTop = cards.log.scrollHeight;

  // diff
  cards.diff.innerHTML = '';
  if (s.latest_diff) cards.diff.appendChild(renderDiff(s.latest_diff));
  else cards.diff.appendChild(el('div', 'muted', 'no commit yet'));
}

function renderDiff(text) {
  const box = el('pre', 'diff');
  for (const line of text.split('\n')) {
    let cls = null;
    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'del';
    else if (line.startsWith('@@')) cls = 'hunk';
    const span = el('span', cls); span.textContent = line + '\n'; box.appendChild(span);
  }
  return box;
}

// --- transcript --------------------------------------------------------------
async function renderTranscript(id) {
  setCrumb('transcript ' + id.slice(0, 12));
  view.innerHTML = '';
  const base = '/api/run/' + encodeURIComponent(id);
  const card = el('div', 'card scroll'); card.style.maxHeight = '78vh';
  card.appendChild(el('h2', null, 'Transcript'));
  const body = el('div'); card.appendChild(body); view.appendChild(card);

  const paint = async () => {
    if (!card.isConnected) return; // navigated away: don't fetch or paint stale
    let data; try { data = await getJSON(base + '/transcript'); } catch (_) { return; }
    if (!card.isConnected) return;
    const atBottom = card.scrollTop + card.clientHeight >= card.scrollHeight - 40;
    body.innerHTML = '';
    if (!data.turns.length) { body.appendChild(el('div', 'empty', 'no transcript recorded')); return; }
    for (const t of data.turns) {
      if (t.role === 'marker') { body.appendChild(el('div', 'muted', '— ' + t.text + ' —')); continue; }
      const turn = el('div', 'turn ' + t.role);
      turn.appendChild(el('div', 'who', t.role + (t.seq ? '  · seq ' + t.seq : '')));
      if (t.thinking) { const th = el('pre', 'think'); th.textContent = t.thinking; turn.appendChild(th); }
      if (t.text) { const p = el('pre'); p.textContent = t.text; turn.appendChild(p); }
      for (const [name, args] of t.tool_calls || []) turn.appendChild(el('pre', 'mono muted', '→ ' + name + '(' + args + ')'));
      body.appendChild(turn);
    }
    if (atBottom) card.scrollTop = card.scrollHeight; // follow the tail unless scrolled up
  };
  await paint();

  // Live-follow: the RunState /events stream is a change signal; re-fold the
  // transcript on it (debounced) and stop once the run finishes. route()'s
  // closeLive tears the stream down on navigation.
  let pending = false;
  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    if (!pending) { pending = true; setTimeout(() => { pending = false; paint(); }, 1200); }
    if (s.finished) { closeLive(); setTimeout(paint, 1300); } // one final fold after last writes flush
  };
}

// --- machine watch -----------------------------------------------------------
function renderMachine(name) {
  setCrumb(name);
  view.innerHTML = '';
  const base = '/api/machine/' + encodeURIComponent(name);
  // Ephemeral notification banners live here; the prompts host holds pending
  // approval/question boxes; both are APPENDED to, never wiped, so a repaint can
  // never clear a half-typed answer or the poke box below.
  const notifs = el('div'); view.appendChild(notifs);
  const prompts = el('div'); view.appendChild(prompts);
  const cards = { _prompts: prompts, _base: base };

  const controls = el('div', 'row wrap'); controls.style.marginBottom = '10px';
  const bell = el('button', null, '🔔 Notifications');
  bell.onclick = enableNotifications;
  const steerBtn = el('button', null, '↪ Steer');
  steerBtn.onclick = () => steerDialog('Steer the current agent state', async (text) => {
    if (text === null) return;
    // cards._state is set each frame to the agent state currently rendered, so
    // the steer routes to that state, not whichever is newest at click time.
    const body = cards._state ? { text, state: cards._state } : { text };
    try { await postJSON(base + '/steer', body); toast('steer sent'); } catch (e) { toast(e.message, true); }
  });
  controls.appendChild(bell); controls.appendChild(steerBtn);
  view.appendChild(controls);

  const grid = el('div', 'grid cols2');
  const structCard = el('div', 'card scroll'); structCard.appendChild(el('h2', null, 'States')); const structBody = el('div'); structCard.appendChild(structBody);
  const pathCard = el('div', 'card scroll'); pathCard.appendChild(el('h2', null, 'Path')); const pathBody = el('div'); pathCard.appendChild(pathBody);
  const reasonCard = el('div', 'card scroll'); reasonCard.appendChild(el('h2', null, 'Current state reasoning')); const reasonBody = el('div'); reasonCard.appendChild(reasonBody);
  grid.appendChild(structCard); grid.appendChild(pathCard); grid.appendChild(reasonCard);
  view.appendChild(grid);

  // The poke ("send message") box: created ONCE so its input survives repaints.
  const poke = el('div', 'poke-box');
  poke.appendChild(el('div', 'sub muted', 'Send a message to a waiting machine (poke payload):'));
  const prow = el('div', 'form-row');
  const pin = el('input', 'field'); pin.placeholder = 'message…'; pin.style.flex = '1';
  const psend = el('button', 'primary', 'Send');
  const doPoke = async () => { try { await postJSON(base + '/poke', { message: pin.value }); toast('poked'); pin.value = ''; } catch (e) { toast(e.message, true); } };
  psend.onclick = doPoke;
  pin.onkeydown = e => { if (e.key === 'Enter') doPoke(); };
  prow.appendChild(pin); prow.appendChild(psend); poke.appendChild(prow);
  view.appendChild(poke);

  // Notification de-dup across repaints: seed with history on the first frame so
  // opening a machine does not replay every past notification; banner + OS-notify
  // only genuinely new ones.
  const ctx = { notifsHost: notifs, seen: null, endedNotified: false };

  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let data; try { data = JSON.parse(ev.data); } catch (_) { return; }
    paintMachine(structBody, pathBody, reasonBody, cards, ctx, data);
    if (data.machine && data.machine.ended) closeLive(); // machine done; stop the stream
  };
}

function machineNotify(ctx, m) {
  const notes = m.notifications || [];
  const keyOf = n => (n.ts || '') + '|' + (n.state || '') + '|' + (n.message || '');
  if (ctx.seen === null) {
    // First frame: seed history (notifications AND an already-ended machine)
    // silently, so opening a finished machine does not replay past notifications
    // or fire a spurious "ended" banner/OS-notify. Only events that happen while
    // watching fire.
    ctx.seen = new Set(notes.map(keyOf));
    if (m.ended) ctx.endedNotified = true;
    return;
  }
  for (const n of notes) {
    const k = keyOf(n);
    if (ctx.seen.has(k)) continue;
    ctx.seen.add(k);
    const banner = el('div', 'notif-banner ' + esc(n.level || 'info'));
    const g = el('div', 'grow');
    g.appendChild(el('div', 'nb-msg', n.message || ''));
    g.appendChild(el('div', 'nb-sub', `${esc(m.machine || '')} · ${esc(n.state || '')}`));
    const x = el('button', 'nb-x', '×'); x.onclick = () => banner.remove();
    banner.appendChild(g); banner.appendChild(x);
    ctx.notifsHost.appendChild(banner);
    osNotify('agent6: ' + (m.machine || 'machine'), n.message || '');
  }
}

function paintMachine(structBody, pathBody, reasonBody, cards, ctx, data) {
  if (data.error) { structBody.innerHTML=''; structBody.appendChild(el('div', 'err', data.error)); return; }
  const m = data.machine || {};
  // Pending approval/question/steer come from the current agent state's RunState.
  // Track which per-state dir this frame rendered so prompt answers + steer route
  // to that exact state (ids reset per state; the machine may advance meanwhile).
  cards._state = (data.reasoning || {}).state_dir || '';
  paintPrompts(cards, data.reasoning || {});
  machineNotify(ctx, m);
  if (m.ended && !ctx.endedNotified) {
    ctx.endedNotified = true;
    const banner = el('div', 'notif-banner ' + (m.ended.status === 'ok' ? 'info' : 'error'));
    banner.appendChild(el('div', 'grow', `${esc(m.machine || '')} ended: ${esc(m.ended.status)} (${esc(m.ended.reason)})`));
    const x = el('button', 'nb-x', '×'); x.onclick = () => banner.remove();
    banner.appendChild(x); ctx.notifsHost.appendChild(banner);
    osNotify('agent6: ' + (m.machine || 'machine') + ' ' + m.ended.status, m.ended.reason || '');
  }
  structBody.innerHTML = '';
  structBody.appendChild(el('div', 'sub muted', `${esc(m.machine)} v${esc(m.version)} · current: ${esc(m.current)}`));
  const tree = el('div', 'tree');
  for (const st of m.states || []) {
    const line = el('div', 'node' + (st.is_current ? ' cursor' : ''));
    const glyph = st.is_current ? '▸' : (st.is_visited ? '·' : ' ');
    line.textContent = `${glyph} ${st.name}  (${st.kind})`;
    tree.appendChild(line);
  }
  structBody.appendChild(tree);

  pathBody.innerHTML = '';
  const path = el('div', 'tree');
  for (const t of m.transitions || []) path.appendChild(el('div', 'node', `${t.seq}. ${t.state} —${t.label}→ ${t.goto}`));
  if (!(m.transitions||[]).length) path.appendChild(el('div', 'muted', 'no transitions yet'));
  pathBody.appendChild(path);
  if (m.ended) pathBody.appendChild(el('div', 'sub muted', `ended: ${m.ended.status} (${m.ended.reason}) at ${m.ended.state}`));

  reasonBody.innerHTML = '';
  const r = data.reasoning || {};
  if (r.last_role && (r.last_role.streamed_thinking || r.last_role.streamed_text)) {
    if (r.last_role.streamed_thinking) { const t = el('pre', 'think'); t.textContent = r.last_role.streamed_thinking; reasonBody.appendChild(t); }
    if (r.last_role.streamed_text) { const t = el('pre'); t.textContent = r.last_role.streamed_text; reasonBody.appendChild(t); }
  } else if ((r.log_tail||[]).length) {
    const log = el('div', 'log'); for (const line of r.log_tail.slice(-60)) log.appendChild(el('div', null, line)); reasonBody.appendChild(log);
  } else { reasonBody.appendChild(el('div', 'muted', 'no agent state running')); }
  reasonBody.scrollTop = reasonBody.scrollHeight;
}

// --- config ------------------------------------------------------------------
async function renderConfig() {
  setCrumb('config');
  const data = await getJSON('/api/config');
  view.innerHTML = '';
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Config'));
  const filter = el('input', 'filter'); filter.placeholder = 'filter keys…'; filter.type = 'search';
  card.appendChild(filter);
  const tbl = el('table', 'cfg');
  const head = el('tr'); ['key','value','source'].forEach(h => head.appendChild(el('th', null, h))); tbl.appendChild(head);
  const keys = Object.keys(data).sort();
  const rows = [];
  for (const k of keys) {
    const s = data[k];
    const tr = el('tr', s.modified ? 'mod' : '');
    tr.appendChild(el('td', 'key', k));
    const shown = s.adaptive ? (esc(s.effective) + '  (adaptive)') : fmtVal(s.value);
    tr.appendChild(el('td', 'val', shown));
    tr.appendChild(el('td', 'src', esc(s.source)));
    tr.title = 'click to edit';
    tr.style.cursor = 'pointer';
    tr.onclick = () => editConfig(k, s);
    tbl.appendChild(tr); rows.push([k.toLowerCase(), tr]);
  }
  card.appendChild(tbl);
  filter.oninput = () => { const q = filter.value.toLowerCase(); for (const [key, tr] of rows) tr.style.display = key.includes(q) ? '' : 'none'; };
  view.appendChild(card);
}
async function editConfig(key, s) {
  const cur = s.value === null || s.value === undefined ? '' : (Array.isArray(s.value) ? s.value.join(',') : String(s.value));
  const choicesHint = s.choices ? ' (one of: ' + s.choices.join(', ') + ')' : '';
  const value = prompt('Set ' + key + choicesHint + ':', cur);
  if (value === null) return;
  try { const d = await postJSON('/api/config', { key, value }); toast(d.message || 'set ' + key); renderConfig(); }
  catch (e) { toast(e.message, true); }
}
function fmtVal(v) { if (v === null || v === undefined) return '—'; if (Array.isArray(v)) return '[' + v.join(', ') + ']'; return String(v); }

route();
</script>
</body>
</html>
"""


# The PWA manifest: makes the page installable (phone home-screen, desktop app).
# start_url "." keeps it relative to wherever the server is mounted (behind
# `tailscale serve` the path prefix may differ).
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

# A minimal service worker: required (with the manifest) for installability. It is
# a network passthrough, no caching, no Web Push / VAPID (OS notifications are the
# foreground Notification API only, fired from the page).
SERVICE_WORKER_JS = r"""self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});
"""

# The app icon: the docs site's hex-framed snowflake (docs/assets/favicon.svg),
# centred on a full-bleed dark backdrop so it stays "maskable"-safe. Self-contained
# SVG, no raster asset to ship.
ICON_SVG = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs><linearGradient id="g" x1="6" y1="4" x2="42" y2="44" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#7aa2f7"/><stop offset="1" stop-color="#06f5f3"/></linearGradient></defs>
  <rect width="512" height="512" rx="96" fill="#0e1116"/>
  <g transform="translate(96 96) scale(6.6667)">
    <path d="M24 3.5 41.7 13.75 V34.25 L24 44.5 6.3 34.25 V13.75 Z" fill="#161618"/>
    <path d="M24 3.5 41.7 13.75 V34.25 L24 44.5 6.3 34.25 V13.75 Z" stroke="url(#g)" stroke-width="2.4" stroke-linejoin="round" fill="none"/>
    <g stroke="url(#g)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" fill="none">
      <line x1="24" y1="24" x2="24" y2="10"/>
      <line x1="24" y1="15.32" x2="19.25" y2="12.35"/>
      <line x1="24" y1="15.32" x2="28.75" y2="12.35"/>
      <line x1="24" y1="24" x2="11.88" y2="17"/>
      <line x1="16.48" y1="19.66" x2="11.54" y2="22.29"/>
      <line x1="16.48" y1="19.66" x2="16.29" y2="14.06"/>
      <line x1="24" y1="24" x2="11.88" y2="31"/>
      <line x1="16.48" y1="28.34" x2="16.29" y2="33.94"/>
      <line x1="16.48" y1="28.34" x2="11.54" y2="25.71"/>
      <line x1="24" y1="24" x2="24" y2="38"/>
      <line x1="24" y1="32.68" x2="28.75" y2="35.65"/>
      <line x1="24" y1="32.68" x2="19.25" y2="35.65"/>
      <line x1="24" y1="24" x2="36.12" y2="31"/>
      <line x1="31.52" y1="28.34" x2="36.46" y2="25.71"/>
      <line x1="31.52" y1="28.34" x2="31.71" y2="33.94"/>
      <line x1="24" y1="24" x2="36.12" y2="17"/>
      <line x1="31.52" y1="19.66" x2="31.71" y2="14.06"/>
      <line x1="31.52" y1="19.66" x2="36.46" y2="22.29"/>
    </g>
  </g>
</svg>
"""
