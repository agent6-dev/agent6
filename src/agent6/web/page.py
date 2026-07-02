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

@media (max-width: 780px) {
  nav.tabs { display: flex; }
  main { padding: 12px 12px calc(var(--nav-h) + 24px); }
  .card.scroll { max-height: 60vh; }
  header .desktop-only { display: none; }
}
</style>
</head>
<body>
<header>
  <span class="brand" onclick="location.hash='#/'"><b>agent6</b></span>
  <span class="crumb" id="crumb"></span>
  <span class="spacer"></span>
  <button class="desktop-only" onclick="location.hash='#/config'">config</button>
  <button onclick="toggleTheme()" title="theme">◐</button>
</header>

<main id="view"><div class="empty">loading…</div></main>

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

// --- theme -------------------------------------------------------------------
if (localStorage.getItem('a6-theme') === 'light') document.documentElement.classList.add('light');
function toggleTheme() {
  const on = document.documentElement.classList.toggle('light');
  localStorage.setItem('a6-theme', on ? 'light' : 'dark');
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
function pill(status) { const p = el('span', 'pill ' + esc(status), esc(status)); return p; }

function setTab(name) {
  document.querySelectorAll('nav.tabs a').forEach(a => a.classList.toggle('active', a.dataset.tab === name));
}

// --- router ------------------------------------------------------------------
async function route() {
  closeLive();
  const h = location.hash.replace(/^#/, '') || '/';
  const parts = h.split('/').filter(Boolean); // e.g. ['run','abc']
  try {
    if (parts.length === 0) { setTab('hub'); await renderHub(); }
    else if (parts[0] === 'machines') { setTab('machines'); await renderHub('machines'); }
    else if (parts[0] === 'config') { setTab('config'); await renderConfig(); }
    else if (parts[0] === 'run' && parts[1]) { setTab('hub'); renderRun(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'transcript' && parts[1]) { setTab('hub'); await renderTranscript(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'machine' && parts[1]) { setTab('machines'); renderMachine(decodeURIComponent(parts[1])); }
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
    try { const d = await postJSON('/api/machine/create', { task: ct.value }); toast('creating machine (draft ' + (d.draft||'?') + ')'); ct.value=''; }
    catch (e) { toast(e.message, true); }
    cbtn.disabled = false;
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
function renderRun(id) {
  setCrumb(id.slice(0, 16));
  view.innerHTML = '';
  const prompts = el('div'); view.appendChild(prompts); // approval/question boxes surface here
  const grid = el('div', 'grid cols2');
  const cards = { _id: id, _prompts: prompts };
  const mk = (key, title, cls) => { const c = el('div', 'card ' + (cls||'')); c.appendChild(el('h2', null, title)); const body = el('div'); c.appendChild(body); cards[key] = body; grid.appendChild(c); return body; };
  mk('head', 'Run');
  mk('budget', 'Budget');
  mk('tasks', 'Task graph', 'scroll');
  mk('role', 'Reasoning', 'scroll');
  mk('tools', 'Tool calls', 'scroll');
  mk('log', 'Event log', 'scroll');
  mk('diff', 'Latest commit', 'scroll');
  view.appendChild(grid);

  const actions = el('div', 'row wrap'); actions.style.marginTop = '14px';
  const steerBtn = el('button', null, '↪ Steer');
  steerBtn.onclick = async () => {
    const text = prompt('Steer instruction (blank = continue, "abort" = stop):', '');
    if (text === null) return;
    try { await postJSON('/api/run/' + encodeURIComponent(id) + '/steer', { text }); toast('steer sent'); } catch (e) { toast(e.message, true); }
  };
  const mergeBtn = el('button', null, '⑃ Merge');
  mergeBtn.onclick = async () => { try { const d = await postJSON('/api/run/' + encodeURIComponent(id) + '/merge', {}); toast(d.message || 'merged'); } catch (e) { toast(e.message, true); } };
  const tbtn = el('button', null, 'Transcript →');
  tbtn.onclick = () => location.hash = '#/transcript/' + encodeURIComponent(id);
  actions.appendChild(steerBtn); actions.appendChild(mergeBtn); actions.appendChild(tbtn);
  view.appendChild(actions);

  live = new EventSource('/api/run/' + encodeURIComponent(id) + '/events');
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    paintRun(cards, s);
    if (s.finished) closeLive(); // run is done; stop the stream so it doesn't reconnect
  };
  live.onerror = () => { /* EventSource auto-retries a live run; leave last paint up */ };
}

// Render the run's unanswered approval / ask_user prompts as actionable boxes.
function paintPrompts(cards, s) {
  const host = cards._prompts, id = cards._id;
  host.innerHTML = '';
  for (const ap of (s.pending_approvals || [])) {
    if (ap.answered) continue;
    const box = el('div', 'prompt-box');
    box.appendChild(el('div', 'q', ap.prompt || 'Approve this action?'));
    const row = el('div', 'form-row');
    const yes = el('button', 'primary', 'Approve');
    const no = el('button', 'danger', 'Deny');
    const send = ok => async () => { try { await postJSON('/api/run/' + encodeURIComponent(id) + '/approve', { id: ap.id, approved: ok }); } catch (e) { toast(e.message, true); } };
    yes.onclick = send(true); no.onclick = send(false);
    row.appendChild(yes); row.appendChild(no); box.appendChild(row); host.appendChild(box);
  }
  for (const q of (s.pending_questions || [])) {
    if (q.answered) continue;
    const box = el('div', 'prompt-box');
    box.appendChild(el('div', 'q', q.question || 'The agent asked a question'));
    const row = el('div', 'form-row');
    for (const opt of (q.options || [])) {
      const b = el('button', null, opt);
      b.onclick = async () => { try { await postJSON('/api/run/' + encodeURIComponent(id) + '/answer', { id: q.id, answer: opt }); } catch (e) { toast(e.message, true); } };
      row.appendChild(b);
    }
    const inp = el('input', 'field'); inp.placeholder = 'or type an answer…'; inp.style.flex = '1';
    const send = el('button', 'primary', 'Send');
    send.onclick = async () => { try { await postJSON('/api/run/' + encodeURIComponent(id) + '/answer', { id: q.id, answer: inp.value }); } catch (e) { toast(e.message, true); } };
    row.appendChild(inp); row.appendChild(send); box.appendChild(row); host.appendChild(box);
  }
}

function paintRun(cards, s) {
  paintPrompts(cards, s);
  // header
  cards.head.innerHTML = '';
  const kv = el('div', 'kv');
  const add = (k, v) => { kv.appendChild(el('div', 'k', k)); kv.appendChild(el('div', 'v', v)); };
  add('task', s.user_task || '(none)');
  add('id', s.run_id || '');
  add('state', s.finished ? (s.all_passed ? 'finished · all passed' : 'finished') : 'running');
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
  const data = await getJSON('/api/run/' + encodeURIComponent(id) + '/transcript');
  view.innerHTML = '';
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Transcript'));
  if (!data.turns.length) { card.appendChild(el('div', 'empty', 'no transcript recorded')); view.appendChild(card); return; }
  for (const t of data.turns) {
    if (t.role === 'marker') { card.appendChild(el('div', 'muted', '— ' + t.text + ' —')); continue; }
    const turn = el('div', 'turn ' + t.role);
    turn.appendChild(el('div', 'who', t.role + (t.seq ? '  · seq ' + t.seq : '')));
    if (t.thinking) { const th = el('pre', 'think'); th.textContent = t.thinking; turn.appendChild(th); }
    if (t.text) { const p = el('pre'); p.textContent = t.text; turn.appendChild(p); }
    for (const [name, args] of t.tool_calls || []) turn.appendChild(el('pre', 'mono muted', '→ ' + name + '(' + args + ')'));
    card.appendChild(turn);
  }
  view.appendChild(card);
}

// --- machine watch -----------------------------------------------------------
function renderMachine(name) {
  setCrumb(name);
  view.innerHTML = '';
  const grid = el('div', 'grid cols2');
  const structCard = el('div', 'card scroll'); structCard.appendChild(el('h2', null, 'States')); const structBody = el('div'); structCard.appendChild(structBody);
  const pathCard = el('div', 'card scroll'); pathCard.appendChild(el('h2', null, 'Path')); const pathBody = el('div'); pathCard.appendChild(pathBody);
  const reasonCard = el('div', 'card scroll'); reasonCard.appendChild(el('h2', null, 'Current state reasoning')); const reasonBody = el('div'); reasonCard.appendChild(reasonBody);
  grid.appendChild(structCard); grid.appendChild(pathCard); grid.appendChild(reasonCard);
  view.appendChild(grid);

  live = new EventSource('/api/machine/' + encodeURIComponent(name) + '/events');
  live.onmessage = ev => { try { paintMachine(structBody, pathBody, reasonBody, JSON.parse(ev.data)); } catch (_) {} };
}

function paintMachine(structBody, pathBody, reasonBody, data) {
  if (data.error) { structBody.innerHTML=''; structBody.appendChild(el('div', 'err', data.error)); return; }
  const m = data.machine || {};
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
