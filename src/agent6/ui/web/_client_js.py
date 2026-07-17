# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Client JS for the web-UI page: the `<script>` block `page.PAGE_HTML` embeds.

The whole SPA: router, view renderers (hub / run / conversation / machine /
config), and the shared widgets (composer, conversation fold, drawer). Split
out of page.py so the HTML/CSS composition stays readable. Byte-identical to
the block page.py used to embed inline; page.py splices it back into
PAGE_HTML unchanged.
"""

from __future__ import annotations

CLIENT_JS = r""""use strict";
const view = document.getElementById('view');
const crumb = document.getElementById('crumb');
let live = null; // the active EventSource, closed on navigation
// Live heartbeat for a run that is active but silent (thinking / resuming): a
// 1s ticker updates "#hb-line" with a spinner + elapsed so it reads as alive,
// not hung. hbState is refreshed by each paintRun; hbTimer runs while on a run.
let hbState = { active: false, role: 'worker', last: 0, spin: 0 };
let hbTimer = null;
const HB_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
function hbTick() {
  const line = document.getElementById('hb-line');
  if (!line || !hbState.active) return;
  const secs = Math.floor((Date.now() - hbState.last) / 1000);
  const glyph = HB_FRAMES[hbState.spin % HB_FRAMES.length];
  line.textContent = `${glyph} ${hbState.role} working… ${secs}s`;
}
let activeOverlayClose = null; // an open modal dialog's dismisser, closed on navigation

// --- theme -------------------------------------------------------------------
if (localStorage.getItem('a6-theme') === 'light') document.documentElement.classList.add('light');
function toggleTheme() {
  const on = document.documentElement.classList.toggle('light');
  localStorage.setItem('a6-theme', on ? 'light' : 'dark');
}

// --- nav rail collapse ---------------------------------------------------------
if (localStorage.getItem('a6-rail') === 'min') document.documentElement.classList.add('rail-min');
function railArrow() {
  const a = document.getElementById('rail-arrow');
  if (a) a.textContent = document.documentElement.classList.contains('rail-min') ? '»' : '«';
}
function toggleRail() {
  const on = document.documentElement.classList.toggle('rail-min');
  localStorage.setItem('a6-rail', on ? 'min' : '');
  railArrow();
}
railArrow(); // reflect the persisted state on load

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
// Mirrors viewmodel/format.py format_cost precision (cents >= $1, else 4dp); keep in sync.
function fmtUsd(u) { if (!u) return '$0'; return u >= 0.995 ? '$' + Number(u).toFixed(2) : '$' + Number(u).toFixed(4); }
function when(ts) { if (!ts) return ''; const d = new Date(ts * 1000); return d.toLocaleString(); }
function setCrumb(t) { crumb.textContent = t || ''; }
function closeLive() {
  if (live) { live.close(); live = null; }
  if (hbTimer) { clearInterval(hbTimer); hbTimer = null; }
  hbState.active = false;
}
function closeOverlay() { if (activeOverlayClose) activeOverlayClose(); }
function pill(status, label) { const p = el('span', 'pill ' + esc(status), esc(label || status)); return p; }

function setTab(name) {
  document.querySelectorAll('nav.tabs a, aside.rail .rail-nav a').forEach(a => a.classList.toggle('active', a.dataset.tab === name));
}

// --- router ------------------------------------------------------------------
let booted = false; // the one-shot deep-link to `agent6 web <target>` ran
async function route() {
  closeLive();
  closeOverlay();
  document.querySelectorAll('.wmenu-btn').forEach(b => b.remove()); // header-mounted by the run view
  const h = location.hash.replace(/^#/, '') || '/';
  // First load with no hash: honor the CLI's target (`agent6 web <run-id>`
  // opens that run; a machine name its machine). Explicit hashes win.
  if (!booted) {
    booted = true;
    if (h === '/') {
      try {
        const meta = await getJSON('/api/meta');
        if (meta.target && meta.target_kind) {
          location.hash = '#/' + meta.target_kind + '/' + encodeURIComponent(meta.target);
          return; // the hashchange re-enters route()
        }
      } catch (_) { /* no meta: fall through to the hub */ }
    }
  }
  const parts = h.split('/').filter(Boolean); // e.g. ['run','abc']
  try {
    if (parts.length === 0) { setTab('hub'); await renderHub(); }
    else if (parts[0] === 'machines') { setTab('machines'); await renderHub('machines'); }
    else if (parts[0] === 'config') { setTab('config'); await renderConfig(); }
    else if (parts[0] === 'run' && parts[1]) { setTab('hub'); await renderRun(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'conversation' && parts[1]) { setTab('hub'); await renderConversation(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'machine' && parts[1]) { setTab('machines'); await renderMachine(decodeURIComponent(parts[1])); }
    else if (parts[0] === 'draft' && parts[1]) { const n = decodeURIComponent(parts[1]); setTab('machines'); await renderRun(n, { base: '/api/draft/' + encodeURIComponent(n), readOnly: true, title: 'Machine draft', crumb: 'draft ' + n }); }
    else { view.innerHTML = ''; view.appendChild(el('div', 'empty', 'not found')); }
  } catch (e) {
    view.innerHTML = '';
    view.appendChild(el('div', 'empty err', 'error: ' + e.message));
  }
}
window.addEventListener('hashchange', route);

// --- hub ---------------------------------------------------------------------
// /parallel model-id autocomplete for the new-work composer. When the task text
// starts with `/parallel ` and the caret sits in the spec token, offer the known
// model ids (GET /api/config/suggest/parallel.models — exactly the set
// run --parallel accepts) filtered by the comma-fragment under the caret; click,
// Enter, or ↑/↓+Enter inserts it. The web analogue of the config editor's
// datalist (a <textarea> can't carry a native datalist).
function attachParallelSuggest(task, root) {
  let models = null;           // null=unfetched, []=in-flight/empty, else the list
  let box = null, items = [], active = -1;
  const ensureModels = () => {
    if (models !== null) return;
    models = [];                // sentinel: fetch at most once per composer
    getJSON('/api/config/suggest/parallel.models').then(d => { models = d.values || []; render(); }).catch(() => {});
  };
  const frag = () => {          // the comma-fragment under the caret, or null
    const v = task.value, caret = task.selectionStart;
    const m = /^\/parallel\s+/.exec(v);
    if (!m) return null;
    const start = m[0].length;
    let end = start;
    while (end < v.length && !/\s/.test(v[end])) end++;
    if (caret < start || caret > end) return null;   // caret outside the spec token
    const fragStart = start + v.slice(start, caret).lastIndexOf(',') + 1;
    return { fragStart, fragEnd: caret, text: v.slice(fragStart, caret) };
  };
  const close = () => { if (box) { box.remove(); box = null; } items = []; active = -1; };
  const insert = (model) => {
    const f = frag(); if (!f) { close(); return; }
    const v = task.value;
    task.value = v.slice(0, f.fragStart) + model + v.slice(f.fragEnd);
    const pos = f.fragStart + model.length;
    task.setSelectionRange(pos, pos); close(); task.focus();
  };
  const render = () => {
    const f = frag();
    if (!f) { close(); return; }
    ensureModels();
    const q = f.text.toLowerCase();
    const hit = m => m.toLowerCase();
    items = models.filter(m => hit(m).startsWith(q)).concat(models.filter(m => hit(m).includes(q) && !hit(m).startsWith(q))).slice(0, 8);
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) { box = el('div', 'ac-pop'); root.appendChild(box); }
    box.textContent = '';
    items.forEach((m, i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''), m);
      o.onmousedown = (e) => { e.preventDefault(); insert(m); };
      box.appendChild(o);
    });
  };
  task.addEventListener('input', () => { active = -1; render(); });
  task.addEventListener('click', render);
  task.addEventListener('blur', () => setTimeout(close, 120));
  // Returns true when the popup consumed the key (caller must not also act on it).
  return { onKeyDown(e) {
    if (!box || !items.length) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

// The new-work composer, docked at the bottom of the Runs page: task text +
// mode + Start (Enter starts, Shift+Enter newline).
function newWorkDock() {
  const root = el('div', 'composer dock dock-fixed');
  const row = el('div', 'row');
  const task = el('textarea', 'field'); task.placeholder = 'task / question…'; task.style.flex = '1';
  const mode = el('select', 'field'); mode.style.flex = '0 0 auto'; mode.style.width = 'auto';
  for (const m of ['run', 'plan', 'ask']) { const o = el('option', null, m); o.value = m; mode.appendChild(o); }
  const go = el('button', 'primary', 'Start');
  const start = async () => {
    if (!task.value.trim()) return;
    go.disabled = true;
    try { const d = await postJSON('/api/new', { mode: mode.value, task: task.value }); if (d.run_id) location.hash = '#/run/' + encodeURIComponent(d.run_id); }
    catch (e) { toast(e.message, true); go.disabled = false; }
  };
  go.onclick = start;
  const ac = attachParallelSuggest(task, root);
  task.onkeydown = (e) => {
    if (ac.onKeyDown(e)) return;   // the /parallel suggestion popup took the key
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); start(); }
  };
  row.appendChild(task); row.appendChild(mode); row.appendChild(go);
  root.appendChild(growGrip(task));
  root.appendChild(row);
  root.appendChild(el('div', 'hint', 'Enter starts the run / plan / ask · Shift+Enter newline · '
    + '/parallel [N|models] <task> fans out lanes (repeat to queue more)'));
  return root;
}

// The create-machine composer, docked at the bottom of the Machines page.
function createMachineDock() {
  const root = el('div', 'composer dock dock-fixed');
  const row = el('div', 'row');
  const ct = el('textarea', 'field'); ct.placeholder = 'describe a machine to create…'; ct.style.flex = '1';
  const cbtn = el('button', 'primary', 'Create machine');
  const create = async () => {
    if (!ct.value.trim()) return; cbtn.disabled = true;
    try { const d = await postJSON('/api/machine/create', { task: ct.value }); ct.value=''; if (d.draft) location.hash = '#/draft/' + encodeURIComponent(d.draft); }
    catch (e) { toast(e.message, true); cbtn.disabled = false; }
  };
  cbtn.onclick = create;
  ct.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); create(); } };
  row.appendChild(ct); row.appendChild(cbtn);
  root.appendChild(growGrip(ct));
  root.appendChild(row);
  root.appendChild(el('div', 'hint', 'Enter creates a machine draft from the description · Shift+Enter newline'));
  return root;
}

// A list card: h2 title + one clickable row per entry.
function listCard(title, entries, empty, paint) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, title));
  const list = el('div', 'list');
  if (!entries.length) list.appendChild(el('div', 'empty', empty));
  for (const e of entries) {
    const it = el('div', 'item');
    const g = el('div', 'grow');
    it.appendChild(g);
    paint(e, it, g);
    list.appendChild(it);
  }
  card.appendChild(list);
  return card;
}

function runsCard(runs) {
  const card = listCard('Runs', runs, 'no runs yet', (r, it, g) => {
    it.onclick = () => location.hash = '#/run/' + encodeURIComponent(r.id);
    g.appendChild(el('div', 'title', (r.winner ? '★ ' : '') + (r.task || '(no task)')));
    g.appendChild(el('div', 'sub', `${esc(r.mode)} · ${esc(r.id)} · ${when(r.mtime)} · ${fmtUsd(r.usd)}`));
    it.appendChild(pill(r.status, r.reason ? r.status + ' · ' + String(r.reason).replaceAll('_', ' ') : r.status));
  });
  const prune = el('button', 'danger'); prune.textContent = 'Prune merged runs'; prune.style.marginTop = '10px';
  prune.onclick = async () => { try { const d = await postJSON('/api/runs/prune', {}); toast(d.message || 'pruned'); route(); } catch (e) { toast(e.message, true); } };
  card.appendChild(prune);
  return card;
}

function machinesCard(machines) {
  return listCard('Machines', machines, 'no machine instances', (m, it, g) => {
    it.onclick = () => location.hash = '#/machine/' + encodeURIComponent(m.name);
    g.appendChild(el('div', 'title', m.machine || m.name));
    g.appendChild(el('div', 'sub', `${m.name} · at ${esc(m.current || '?')} · ${when(m.mtime)}`));
    it.appendChild(pill(m.status));
  });
}

function draftsCard(drafts) {
  return listCard('Machine drafts', drafts, '', (d, it, g) => {
    it.onclick = () => location.hash = '#/draft/' + encodeURIComponent(d.id);
    g.appendChild(el('div', 'title', d.task || d.id));
    g.appendChild(el('div', 'sub', `draft · ${esc(d.id)} · ${when(d.mtime)}`));
    it.appendChild(pill(d.status));
  });
}

function machineFilesCard(files) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Run a machine'));
  const frow = el('div', 'form-row');
  for (const mf of files) {
    const b = el('button', null, '▶ ' + mf.name);
    b.onclick = async () => { try { await postJSON('/api/machine/run', { file: mf.path }); toast('started ' + mf.name); setTimeout(route, 800); } catch (e) { toast(e.message, true); } };
    frow.appendChild(b);
  }
  card.appendChild(frow);
  return card;
}

async function renderHub(focus) {
  setCrumb('');
  const data = await getJSON('/api/hub');
  view.innerHTML = '';
  const machinesTab = focus === 'machines';
  // Full-width listing stack; the tab's composer docks at the bottom of the
  // viewport (new work on Runs, create-machine on Machines).
  const lists = el('div', 'grid');
  if (machinesTab) {
    if ((data.machine_files || []).length) lists.appendChild(machineFilesCard(data.machine_files));
    lists.appendChild(machinesCard(data.machines));
    if ((data.drafts || []).length) lists.appendChild(draftsCard(data.drafts));
  } else {
    lists.appendChild(runsCard(data.runs));
  }
  view.appendChild(lists);
  view.appendChild(machinesTab ? createMachineDock() : newWorkDock());
}

// --- conversation ------------------------------------------------------------
// Renders a /conversation payload: folded transcript items whose lines are
// [text, style] spans from the shared renderer (viewmodel.transcript_style), so
// the web shows exactly what the CLI stream and the TUI conversation view show.
// The detail level cycles collapsed -> expanded -> hidden (persisted); an item
// with a longer form (clipped tool output, folded thinking) expands on click.
const DETAIL_CYCLE = { collapsed: 'expanded', expanded: 'hidden', hidden: 'collapsed' };
function tailStr(s, n) { return s.length <= n ? s : '…' + s.slice(-n); }
function firstLine(s, n) { const t = String(s == null ? '' : s).split('\n')[0]; return t.length > n ? t.slice(0, n - 1) + '…' : t; }
// `box` is the scroll container, `body` the host the items render into.
function makeConv(url, box, body) {
  const conv = {
    items: [], open: new Set(),
    detail: localStorage.getItem('a6-detail') || 'collapsed',
    timer: null,
  };
  const itemsHost = el('div', 'conv');
  const liveHost = el('div', 'conv conv-live');
  liveHost.style.display = 'none';
  body.appendChild(itemsHost); body.appendChild(liveHost);
  const following = () => box.scrollTop + box.clientHeight >= box.scrollHeight - 40;

  const paintItems = () => {
    const follow = following();
    itemsHost.innerHTML = '';
    let shown = 0;
    conv.items.forEach((it, i) => {
      if (it.kind === 'thinking' && conv.detail === 'hidden' && !conv.open.has(i)) return;
      const expanded = conv.detail === 'expanded' || conv.open.has(i);
      const lines = expanded && it.full ? it.full : it.lines;
      const div = el('div', 'ci' + (it.full ? ' exp' : ''));
      if (it.full) {
        div.title = expanded ? 'click to collapse' : 'click to expand';
        div.onclick = () => { if (conv.open.has(i)) conv.open.delete(i); else conv.open.add(i); paintItems(); };
      }
      for (const line of lines) {
        const ln = el('div');
        for (const [text, style] of line) ln.appendChild(el('span', 's-' + style, text));
        if (!line.length) ln.appendChild(document.createTextNode(' '));
        div.appendChild(ln);
      }
      itemsHost.appendChild(div); shown++;
    });
    // The empty note yields to the live pane: streamed text under a
    // "no conversation yet" banner reads as a contradiction.
    if (!shown && liveHost.style.display === 'none') {
      itemsHost.appendChild(el('div', 'muted conv-empty', 'no conversation yet; it appears as the run streams'));
    }
    if (follow) box.scrollTop = box.scrollHeight;
  };

  conv.refresh = async () => {
    if (!box.isConnected) return; // navigated away: don't fetch or paint stale
    let data; try { data = await getJSON(url); } catch (_) { return; }
    if (!box.isConnected) return;
    conv.items = data.items || [];
    paintItems();
  };
  conv.poke = () => { // debounced re-fold on an SSE change signal
    if (conv.timer) return;
    conv.timer = setTimeout(() => { conv.timer = null; conv.refresh(); }, 900);
  };
  // The in-progress turn under the folded items (streamed thinking/text from
  // the RunState SSE frame): the analogue of the TUI's docked live pane. The
  // live "thinking…" marker always shows; the reasoning text itself streams
  // only at the expanded detail level (same rule as the TUI).
  conv.setLive = (s) => {
    const r = s.last_role;
    const follow = following();
    liveHost.innerHTML = '';
    const note = itemsHost.querySelector('.conv-empty');
    if (s.finished || !r) { liveHost.style.display = 'none'; if (note) note.style.display = ''; return; }
    const think = r.streamed_thinking, text = r.streamed_text;
    liveHost.style.display = '';
    if (note) note.style.display = 'none'; // the live pane replaces the empty note
    if (think || text) {
      if (think) {
        const line = el('div');
        line.appendChild(el('span', 'lt', '· thinking… '));
        if (conv.detail === 'expanded') line.appendChild(el('span', 's-thinking', tailStr(think, 1600)));
        liveHost.appendChild(line);
      }
      if (text) liveHost.appendChild(el('div', null, tailStr(text, 1600)));
    } else {
      const hb = el('div', 'muted'); hb.id = 'hb-line'; liveHost.appendChild(hb); hbTick();
    }
    if (follow) box.scrollTop = box.scrollHeight;
  };
  conv.detailButton = () => {
    const b = el('button', 'mini', 'detail: ' + conv.detail);
    b.onclick = () => {
      conv.detail = DETAIL_CYCLE[conv.detail];
      localStorage.setItem('a6-detail', conv.detail);
      b.textContent = 'detail: ' + conv.detail;
      conv.open.clear();
      paintItems();
    };
    return b;
  };
  return conv;
}

// A titled conversation card, the detail toggle in its (non-scrolling) header
// and the items scrolling in .conv-box below it; used by the run view (main
// pane), the full-page view, and the machine view's current-state pane.
function convCard(url, title, cls) {
  const card = el('div', 'card conv-card ' + (cls || ''));
  const hrow = el('div', 'card-head-row');
  hrow.appendChild(el('h2', null, title));
  const box = el('div', 'conv-box');
  const body = el('div');
  box.appendChild(body);
  const conv = makeConv(url, box, body);
  hrow.appendChild(conv.detailButton());
  card.appendChild(hrow); card.appendChild(box);
  return { card, conv, box };
}

// A horizontal grip along a docked composer's top edge: dragging it up grows
// the text entry (the native grip sits bottom-right, where the screen ends).
function growGrip(ta) {
  const g = el('div', 'grow-grip');
  g.title = 'drag to resize';
  g.onpointerdown = (e) => {
    e.preventDefault();
    g.setPointerCapture(e.pointerId);
    g.classList.add('dragging');
    const startY = e.clientY;
    const startH = ta.getBoundingClientRect().height;
    g.onpointermove = (ev) => {
      ta.style.height = Math.round(Math.max(46, Math.min(window.innerHeight * 0.5, startH + startY - ev.clientY))) + 'px';
    };
    g.onpointerup = (ev) => {
      g.releasePointerCapture(ev.pointerId);
      g.classList.remove('dragging');
      g.onpointermove = null; g.onpointerup = null;
    };
  };
  return g;
}

// The composer bar under a run's conversation. On a LIVE run Enter sends the
// text as a steer (injected at the run's next safe boundary); on a FINISHED
// run Enter resumes the run with the text as the follow-up instruction (empty
// = plain resume), then waits for the resumed worker to take over and
// re-renders. Shift+Enter inserts a newline. setState(s) keeps the mode in
// sync with each SSE frame.
function makeComposer(id) {
  const root = el('div', 'composer');
  const ta = el('textarea', 'field');
  const hint = el('div', 'hint');
  let finished = null; // unknown until the first SSE frame
  let busy = false;
  const apply = () => {
    ta.disabled = busy;
    if (busy) { hint.textContent = 'resuming…'; return; }
    if (finished) {
      ta.placeholder = 'continue the run…';
      hint.textContent = 'Enter resumes this run with the instruction (empty = just resume) · Shift+Enter newline';
    } else {
      ta.placeholder = 'steer the run…';
      hint.textContent = 'Enter sends the instruction at the run’s next safe boundary · Shift+Enter newline';
    }
  };
  const resume = async (text) => {
    busy = true; apply();
    try {
      await postJSON('/api/run/' + encodeURIComponent(id) + '/resume', { text });
      toast('resuming the run…');
      // The resume is a detached spawn: wait for it to take over (the folded
      // state un-finishes once it appends events), then re-open the view so
      // the SSE stream and controls come back live.
      for (let i = 0; i < 25; i++) {
        await new Promise(r => setTimeout(r, 1000));
        if (!root.isConnected) return; // navigated away
        let s; try { s = await getJSON('/api/run/' + encodeURIComponent(id)); } catch (_) { continue; }
        if (s && s.finished === false) { ta.value = ''; route(); return; }
      }
      toast('the resume has not started yet; check `agent6 runs`', true);
    } catch (e) { toast(e.message, true); }
    busy = false; apply();
  };
  ta.onkeydown = (e) => {
    if (e.key !== 'Enter' || e.shiftKey) return;
    e.preventDefault();
    if (finished === null || busy) return;
    const text = ta.value.trim();
    if (!finished) {
      if (!text) return;
      postJSON('/api/run/' + encodeURIComponent(id) + '/steer', { text })
        .then(() => { toast('steer sent'); ta.value = ''; })
        .catch(err => toast(err.message, true));
    } else {
      resume(text);
    }
  };
  root.appendChild(growGrip(ta)); root.appendChild(ta); root.appendChild(hint);
  root.setState = (s) => { if (!busy && typeof s.finished === 'boolean') { finished = s.finished; apply(); } };
  apply();
  return root;
}

// --- run dashboard -----------------------------------------------------------
async function stopRun(base, label) {
  if (!confirm('Stop ' + label + '? It ends now and can be resumed later.')) return;
  try { await postJSON(base + '/steer', { text: 'abort' }); toast('stopping…'); } catch (e) { toast(e.message, true); }
}

// opts: { base, readOnly, title }: a draft (machine-create authoring log) is
// watched read-only against /api/draft/<name>; a run is driveable at /api/run/<id>.
// The snapshot fetch up front is the existence probe (a bad id used to leave a
// hollow dashboard: the conversation fetch swallowed its 404 and the
// EventSource error is silent) and the first paint, so the view never flashes
// empty while waiting for the first SSE frame.
// The details drawer: the run's context widgets combined into one collapsible,
// drag-resizable panel on the left, so the conversation keeps the focus.
function drawerHandle(drawer) {
  const h = el('div', 'drawer-handle');
  h.title = 'drag to resize · double-click to reset';
  h.onpointerdown = (e) => {
    e.preventDefault();
    h.setPointerCapture(e.pointerId);
    h.classList.add('dragging');
    const startX = e.clientX;
    const startW = drawer.getBoundingClientRect().width;
    h.onpointermove = (ev) => {
      const w = Math.round(Math.max(220, Math.min(window.innerWidth * 0.6, startW + ev.clientX - startX)));
      document.documentElement.style.setProperty('--drawer-w', w + 'px');
      localStorage.setItem('a6-drawer-w', w + 'px');
    };
    h.onpointerup = (ev) => {
      h.releasePointerCapture(ev.pointerId);
      h.classList.remove('dragging');
      h.onpointermove = null; h.onpointerup = null;
    };
  };
  h.ondblclick = () => {
    document.documentElement.style.removeProperty('--drawer-w');
    localStorage.removeItem('a6-drawer-w');
  };
  return h;
}
{ const w = localStorage.getItem('a6-drawer-w'); if (w) document.documentElement.style.setProperty('--drawer-w', w); }

async function renderRun(id, opts) {
  opts = opts || {};
  const base = opts.base || ('/api/run/' + encodeURIComponent(id));
  const snap = await getJSON(base); // throws -> route() shows the error
  const readOnly = !!opts.readOnly;
  setCrumb(opts.crumb || id);
  view.innerHTML = '';
  // "paged" only acts on phones: one widget shows at a time there, picked by
  // the floating menu; the desktop drawer ignores it.
  const app = el('div', 'run-app paged');
  const prompts = el('div', 'page-pad'); app.appendChild(prompts); // approval/question boxes surface here
  const cards = { _id: id, _prompts: prompts, _readOnly: readOnly };
  const drawer = el('div', 'grid drawer');
  const mk = (key, title, cls, parent) => { const c = el('div', 'card card-' + key + ' ' + (cls||'')); c.dataset.w = key; c.appendChild(el('h2', null, title)); const body = el('div', 'card-body'); c.appendChild(body); cards[key] = body; (parent || drawer).appendChild(c); return body; };

  // Controls at the TOP so Stop stays reachable without scrolling; the Details
  // toggle folds the drawer away (persisted; default open on wide screens).
  const actions = el('div', 'row wrap page-pad'); actions.style.margin = '10px 22px';
  const dBtn = el('button', 'details-btn', 'Details'); // desktop drawer toggle; phones page widgets instead
  const applyDrawer = (open) => {
    drawer.classList.toggle('closed', !open);
    dBtn.classList.toggle('active', open);
    localStorage.setItem('a6-drawer', open ? 'open' : 'closed');
  };
  dBtn.onclick = () => applyDrawer(drawer.classList.contains('closed'));
  actions.appendChild(dBtn);
  if (!readOnly) {
    const post = (verb, okMsg) => async () => {
      try { const d = await postJSON('/api/run/' + encodeURIComponent(id) + '/' + verb, {}); toast(d.message || okMsg); }
      catch (e) { toast(e.message, true); }
    };
    const stopBtn = el('button', 'danger', '■ Stop now');
    stopBtn.onclick = () => stopRun('/api/run/' + encodeURIComponent(id), 'the run');
    const stepBtn = el('button', null, 'Stop after step');
    stepBtn.onclick = post('stop_step', 'stopping after the current step');
    const compactBtn = el('button', null, 'Compact context');
    compactBtn.onclick = post('compact', 'compaction requested');
    const mergeBtn = el('button', null, 'Merge'); // no glyph: U+2443 was tofu in common fonts
    mergeBtn.onclick = post('merge', 'merged');
    cards._merge_btn = mergeBtn; // paintRun gates it on the run actually having a branch
    for (const b of [stopBtn, stepBtn, compactBtn, mergeBtn]) actions.appendChild(b);
    cards._live_btns = [stopBtn, stepBtn, compactBtn]; // paintRun disables these once finished
  }
  app.appendChild(actions);

  mk('head', opts.title || 'Run', ''); // status/summary leads the drawer
  mk('tasks', 'Task graph', 'scroll');
  mk('budget', 'Budget', '');
  mk('tools', 'Tool calls', 'scroll');
  mk('diff', 'Latest commit', 'scroll');
  mk('log', 'Event log', 'scroll');

  const cc = convCard(base + '/conversation', 'Conversation', 'card-conv');
  cards._conv = cc.conv;
  cc.card.dataset.w = 'conv';
  const body = el('div', 'run-body');
  body.appendChild(drawer);
  body.appendChild(drawerHandle(drawer));
  body.appendChild(cc.card);
  app.appendChild(body);
  const saved = localStorage.getItem('a6-drawer');
  applyDrawer(saved ? saved === 'open' : window.innerWidth >= 1024);

  // The phone widget menu: pick which single widget the page shows.
  const entries = [['conv', 'Conversation'], ['head', 'Overview'], ['tasks', 'Task graph'],
                   ['budget', 'Budget'], ['tools', 'Tool calls'], ['diff', 'Latest commit'],
                   ['log', 'Event log']];
  const wbtn = el('button', 'wmenu-btn', '☰');
  wbtn.title = 'widgets';
  const wmenu = el('div', 'wmenu'); wmenu.style.display = 'none';
  const setW = (key) => {
    app.querySelectorAll('[data-w]').forEach(c => c.classList.toggle('w-active', c.dataset.w === key));
    wmenu.querySelectorAll('button').forEach(mb => mb.classList.toggle('w-on', mb.dataset.w === key));
  };
  for (const [key, label] of entries) {
    const mb = el('button', null, label); mb.dataset.w = key;
    mb.onclick = () => { setW(key); wmenu.style.display = 'none'; window.scrollTo(0, 0); };
    wmenu.appendChild(mb);
  }
  wbtn.onclick = () => { wmenu.style.display = wmenu.style.display === 'none' ? '' : 'none'; };
  setW('conv');
  // The button lives in the header (next to the theme toggle) so the two share
  // one row and skin; route() removes it since clearing #view won't.
  document.querySelector('header').appendChild(wbtn);
  app.appendChild(wmenu);

  if (!readOnly) {
    // The composer replaces the steer dialog: steer while live, resume when
    // done. Docked at the bottom of the view.
    const composer = makeComposer(id);
    composer.classList.add('dock');
    app.appendChild(composer);
    cards._composer = composer;
  }
  view.appendChild(app);
  paintRun(cards, snap);
  cc.conv.refresh().then(() => {
    // On a phone the page (not the box) scrolls the conversation: open at the tail.
    if (window.innerWidth < 781) window.scrollTo(0, document.body.scrollHeight);
  });

  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    paintRun(cards, s);
    hbState.spin++;
    if (s.finished) { closeLive(); setTimeout(() => cc.conv.refresh(), 900); } // one final fold after last writes flush
  };
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
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
      // One or more related questions answered together; option buttons FILL that
      // question's field, and a single Submit posts all answers (review first).
      const box = el('div', 'prompt-box');
      const qs = q.questions || [];
      const inputs = [];
      qs.forEach((sub, qi) => {
        const label = (qs.length > 1 ? (qi + 1) + '. ' : '') + (sub.question || 'The agent asked a question');
        box.appendChild(el('div', 'q', label));
        const row = el('div', 'form-row');
        const inp = el('input', 'field'); inp.placeholder = 'pick above or type an answer…'; inp.style.flex = '1';
        for (const opt of (sub.options || [])) {
          const b = el('button', null, opt);
          b.onclick = () => { inp.value = opt; };
          row.appendChild(b);
        }
        row.appendChild(inp); box.appendChild(row);
        inputs.push(inp);
      });
      const send = el('button', 'primary', qs.length > 1 ? 'Submit all' : 'Send');
      send.onclick = async () => {
        const answers = inputs.map(i => i.value.trim());
        // Guard an accidental Send: an all-empty submit would consume this
        // one-shot question and continue the run on fabricated empty input.
        if (answers.every(a => a === '')) { toast('Pick an option or type an answer first.', true); return; }
        try { await postJSON(base + '/answer', { id: q.id, answers, ...extra }); } catch (e) { toast(e.message, true); }
      };
      box.appendChild(send);
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
  // Stop/compact only mean something on a live run; a finished run ignores the
  // bridge markers. The composer flips to resume mode instead of disabling.
  if (cards._live_btns) for (const b of cards._live_btns) b.disabled = s.finished;
  // Merge needs a run branch: an ask (or a branch_per_run=false run) has none,
  // and a merged branch is gone, so clicking could only produce a git error.
  if (cards._merge_btn) {
    cards._merge_btn.disabled = !s.run_branch || !!s.merged_into;
    cards._merge_btn.title = s.merged_into ? 'already merged into ' + s.merged_into
      : s.run_branch ? 'merge ' + s.run_branch + (s.base_branch ? ' into ' + s.base_branch : '')
      : 'this run has no branch to merge';
  }
  if (cards._composer) cards._composer.setState(s);
  // header
  cards.head.innerHTML = '';
  const kv = el('div', 'kv');
  const add = (k, v) => { kv.appendChild(el('div', 'k', k)); kv.appendChild(el('div', 'v', v)); };
  add('task', s.user_task || '(none)');
  add('id', s.run_id || cards._id || ''); // older logs carry no run_id in run.start
  add('state', s.status_label || (s.finished ? 'finished' : 'running'));
  // Where the run's work lives and where Merge lands: consecutive spawns chain
  // branches, which is invisible without this line.
  if (s.run_branch) {
    add('branch', s.merged_into
      ? s.run_branch + ' (merged into ' + s.merged_into + ')'
      : s.run_branch + (s.base_branch ? ' → merges into ' + s.base_branch : ''));
  }
  // Fan-out compare outcome (stamped into a lane's manifest by --parallel's
  // auto-compare): where this lane placed and why. Absent for a non-lane run.
  if (s.compare && typeof s.compare.rank === 'number') {
    const c = s.compare;
    const bits = [];
    if (c.winner) bits.push('winner');
    if (c.ranked_by) bits.push(c.ranked_by);
    add('compare', 'rank ' + c.rank + '/' + c.of + (bits.length ? ' (' + bits.join(', ') + ')' : ''));
  }
  cards.head.appendChild(kv);
  if (s.compare && s.compare.rationale) {
    cards.head.appendChild(el('div', 'sub muted', 'judge: ' + s.compare.rationale));
  }
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
    // Mirrors viewmodel/format.py TASK_STATUS_GLYPH (JS can't import it); keep in sync.
    const glyph = { passed:'✓', failed:'✗', in_progress:'▸', pending:'·', skipped:'–', obsolete:'×' }[t.status] || '·';
    line.appendChild(el('span', 'st-' + t.status, '  '.repeat(t.depth) + glyph + ' '));
    line.appendChild(document.createTextNode(t.title));
    tree.appendChild(line);
  }
  cards.tasks.appendChild(tree);

  // conversation: the live in-progress turn paints from this frame at once (a
  // heartbeat ticks via hbTick() on a live-but-silent run so it reads as alive,
  // not hung); completed turns re-fold on a debounce.
  const streaming = s.last_role && (s.last_role.streamed_thinking || s.last_role.streamed_text);
  cards._conv.setLive(s);
  cards._conv.poke();
  hbState = {
    active: !s.finished && !!s.last_role && !streaming,
    role: (s.last_role && s.last_role.role) || 'worker',
    last: Date.now(),
    spin: 0,
  };
  hbTick();

  // tools: one clipped line per call (hover shows the full args + result; the
  // conversation carries the whole story), so a long error dump can't flood it.
  cards.tools.innerHTML = '';
  const tbl = el('table', 'tools');
  for (const tc of (s.tool_calls||[]).slice(-30)) {
    const tr = el('tr');
    const d = el('td'); d.appendChild(el('span', 'dot ' + (tc.ok === null ? '' : tc.ok ? 'ok' : 'bad'))); tr.appendChild(d);
    tr.appendChild(el('td', 'name', tc.name));
    const a = el('td', 'args');
    a.textContent = firstLine(tc.args_preview, 90) + (tc.result_summary ? '  → ' + firstLine(tc.result_summary, 90) : '');
    const extra = String(tc.result_summary || '').split('\n').length - 1;
    if (extra > 0) a.appendChild(el('span', 'more-note', ` (+${extra} more line${extra === 1 ? '' : 's'})`));
    a.title = tc.args_preview + (tc.result_summary ? '\n→ ' + tc.result_summary : '');
    tr.appendChild(a);
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

// --- conversation page ---------------------------------------------------------
// The run's conversation full-height (the same component the run view embeds),
// live-following: the RunState /events stream is the change signal; the fold
// re-fetches on it (debounced) and the stream closes once the run finishes.
async function renderConversation(id) {
  const base = '/api/run/' + encodeURIComponent(id);
  await getJSON(base); // existence probe: throws -> route() shows the error
  setCrumb('conversation ' + id);
  view.innerHTML = '';
  // The same app shell as the run view, minus the drawer: the conversation
  // fills the view full width and the composer docks at the bottom.
  const app = el('div', 'run-app');
  const cc = convCard(base + '/conversation', 'Conversation', 'card-conv');
  const body = el('div', 'run-body');
  body.appendChild(cc.card);
  app.appendChild(body);
  const composer = makeComposer(id);
  composer.classList.add('dock');
  app.appendChild(composer);
  view.appendChild(app);
  await cc.conv.refresh();
  cc.box.scrollTop = cc.box.scrollHeight; // open at the tail, like the TUI
  if (window.innerWidth < 781) window.scrollTo(0, document.body.scrollHeight); // phone: the page scrolls

  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    composer.setState(s);
    cc.conv.setLive(s);
    cc.conv.poke();
    hbState = {
      active: !s.finished && !!s.last_role && !(s.last_role.streamed_thinking || s.last_role.streamed_text),
      role: (s.last_role && s.last_role.role) || 'worker',
      last: Date.now(),
      spin: hbState.spin + 1,
    };
    hbTick();
    if (s.finished) { closeLive(); setTimeout(() => cc.conv.refresh(), 900); } // one final fold after last writes flush
  };
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
}

// --- machine watch -----------------------------------------------------------
async function renderMachine(name) {
  const base = '/api/machine/' + encodeURIComponent(name);
  // Existence + readability probe: a bad name or a corrupt machine throws here
  // and route() shows the error (the SSE error frame alone left a hollow view).
  await getJSON(base);
  setCrumb(name);
  view.innerHTML = '';
  // Ephemeral notification banners live here; the prompts host holds pending
  // approval/question boxes; both are APPENDED to, never wiped, so a repaint can
  // never clear a half-typed answer.
  const notifs = el('div', 'page-pad'); view.appendChild(notifs);
  const prompts = el('div', 'page-pad'); view.appendChild(prompts);
  const cards = { _prompts: prompts, _base: base };

  const controls = el('div', 'row wrap page-pad'); controls.style.marginBottom = '10px';
  const bell = el('button', null, '🔔 Notifications');
  bell.onclick = enableNotifications;
  controls.appendChild(bell);
  view.appendChild(controls);

  const grid = el('div', 'grid cols2');
  const structCard = el('div', 'card scroll'); structCard.appendChild(el('h2', null, 'States')); const structBody = el('div'); structCard.appendChild(structBody);
  const pathCard = el('div', 'card scroll'); pathCard.appendChild(el('h2', null, 'Path')); const pathBody = el('div'); pathCard.appendChild(pathBody);
  // The current agent state's conversation: the same folded view a run shows,
  // full-width under the states/path pair.
  const cc = convCard(base + '/conversation', 'Current state', 'span2');
  cards._conv = cc.conv;
  grid.appendChild(structCard); grid.appendChild(pathCard); grid.appendChild(cc.card);
  view.appendChild(grid);
  cc.conv.refresh();

  // The machine composer, docked at the bottom: ONE text entry with the two
  // machine verbs, matching the TUI machine watch (s = Steer, m = Message).
  // Steer injects into the current agent state at its next safe boundary
  // (blank = continue); Message is a poke payload a waiting machine's next
  // tool reads (blank = a bare wake). Created once, so the input survives
  // repaints; paintMachine gates the buttons.
  const dock = el('div', 'composer dock dock-fixed');
  const drow = el('div', 'row');
  const din = el('textarea', 'field'); din.style.flex = '1';
  din.placeholder = 'steer the agent state, or message the machine…';
  const steerBtn = el('button', 'primary', 'Steer');
  steerBtn.onclick = async () => {
    // cards._state is set each frame to the agent state currently rendered, so
    // the steer routes to that state, not whichever is newest at click time.
    const body = cards._state ? { text: din.value, state: cards._state } : { text: din.value };
    try { await postJSON(base + '/steer', body); toast('steer sent'); din.value = ''; } catch (e) { toast(e.message, true); }
  };
  const msgBtn = el('button', null, 'Message');
  msgBtn.onclick = async () => {
    try { await postJSON(base + '/poke', { message: din.value }); toast('message sent'); din.value = ''; } catch (e) { toast(e.message, true); }
  };
  drow.appendChild(din); drow.appendChild(steerBtn); drow.appendChild(msgBtn);
  dock.appendChild(growGrip(din));
  dock.appendChild(drow);
  dock.appendChild(el('div', 'hint', 'Steer injects into the current agent state · Message wakes a waiting machine (its next tool reads it)'));
  view.appendChild(dock);
  cards._steer_btn = steerBtn; cards._msg_btn = msgBtn; // paintMachine gates these

  // Notification de-dup across repaints: seed with history on the first frame so
  // opening a machine does not replay every past notification; banner + OS-notify
  // only genuinely new ones.
  const ctx = { notifsHost: notifs, seen: null, endedNotified: false };

  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let data; try { data = JSON.parse(ev.data); } catch (_) { return; }
    paintMachine(structBody, pathBody, cards, ctx, data);
    hbState.spin++;
    if (data.machine && data.machine.ended) closeLive(); // machine done; stop the stream
  };
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
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

function paintMachine(structBody, pathBody, cards, ctx, data) {
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

  // An ended machine takes no input: poking or steering it would only pretend
  // to work (nothing reads the signal), and its final state's log often has no
  // run.end, which would leave a live "thinking..." marker up forever. Steer
  // additionally needs an agent state to inject into.
  const ended = !!m.ended;
  if (cards._steer_btn) {
    cards._steer_btn.disabled = ended || !cards._state;
    cards._steer_btn.title = ended ? 'the machine has ended'
      : !cards._state ? 'no agent state is active to steer'
      : 'inject into the current agent state at its next safe boundary';
  }
  if (cards._msg_btn) {
    cards._msg_btn.disabled = ended;
    cards._msg_btn.title = ended ? 'the machine has ended'
      : 'wake a waiting machine; its next tool reads the message';
  }

  // The current state's conversation: live turn from this frame, completed
  // turns re-folded on a debounce. A live-but-silent state ticks the heartbeat.
  const r = data.reasoning || {};
  cards._conv.setLive(ended ? { finished: true } : r);
  cards._conv.poke();
  const streaming = r.last_role && (r.last_role.streamed_thinking || r.last_role.streamed_text);
  hbState = {
    active: !ended && !r.finished && !!r.last_role && !streaming,
    role: (r.last_role && r.last_role.role) || 'agent',
    last: Date.now(),
    spin: hbState.spin,
  };
  hbTick();
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
// A proper editor overlay (not a browser prompt): choices and booleans get a
// select, everything else a text field, with the default, source, and type
// shown; "set for this repo" writes the per-repo config instead of the global.
function editConfig(key, s) {
  const cur = s.value === null || s.value === undefined ? ''
    : Array.isArray(s.value) ? s.value.join(',')
    : typeof s.value === 'object' ? JSON.stringify(s.value)
    : String(s.value);
  const back = el('div', 'overlay');
  const box = el('div', 'card'); box.style.width = 'min(560px, 92vw)';
  const title = el('h2', null, key);
  title.style.textTransform = 'none'; // a config key is a case-sensitive identifier
  box.appendChild(title);
  const meta = el('div', 'sub muted');
  meta.textContent = `${esc(s.type)} · default: ${fmtVal(s.default)} · set from: ${esc(s.source)}` + (s.adaptive ? ' · adaptive' : '');
  meta.style.marginBottom = '10px';
  box.appendChild(meta);
  let field;
  const choices = s.choices || (s.type === 'bool' ? ['true', 'false'] : null);
  if (choices) {
    field = el('select', 'field');
    for (const c of choices) { const o = el('option', null, c); o.value = c; field.appendChild(o); }
    field.value = cur || String(s.default ?? '');
  } else {
    field = el('input', 'field');
    field.value = cur;
    if (s.type === 'list') field.placeholder = 'comma-separated values';
    // Dynamic suggestions (configured provider names, the provider's model
    // ids) from the same sources the TUI config page and CLI TAB completion
    // use, attached as a native datalist autocomplete.
    getJSON('/api/config/suggest/' + encodeURIComponent(key)).then(d => {
      if (!d.values || !d.values.length || !field.isConnected) return;
      const dl = el('datalist'); dl.id = 'cfg-suggest';
      for (const v of d.values) { const o = el('option'); o.value = v; dl.appendChild(o); }
      box.appendChild(dl);
      field.setAttribute('list', dl.id);
    }).catch(() => {});
  }
  box.appendChild(field);
  const repoRow = el('label', 'row'); repoRow.style.marginTop = '8px'; repoRow.style.cursor = 'pointer';
  const repoCb = el('input'); repoCb.type = 'checkbox';
  // Default the target layer to the value's ORIGIN: editing a repo-sourced value
  // must write the repo config, or the repo overlay keeps masking the edit and it
  // looks like the save vanished.
  repoCb.checked = (s.source === 'repo');
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  box.appendChild(repoRow);
  const row = el('div', 'form-row');
  const save = el('button', 'primary', 'Save'), cancel = el('button', null, 'Cancel');
  row.appendChild(save);
  // A value set from an editable layer gets an Unset: remove it from that layer
  // so it reverts to the next-lower one / the built-in default. (default-sourced
  // values have nothing to unset; flag/machine layers are not editable here.)
  let unsetBtn = null;
  if (s.source === 'repo' || s.source === 'global') {
    unsetBtn = el('button', null, 'Unset');
    unsetBtn.title = 'remove from the ' + s.source + ' config; reverts to ' + fmtVal(s.default);
    row.appendChild(unsetBtn);
  }
  row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back);
  const close = () => { activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey); };
  activeOverlayClose = close; // navigating away dismisses it
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  cancel.onclick = close;
  back.onclick = (e) => { if (e.target === back) close(); };
  field.focus();
  const submit = async () => {
    save.disabled = true;
    try {
      const d = await postJSON('/api/config', { key, value: field.value, repo: repoCb.checked });
      toast(d.message || 'set ' + key); close(); renderConfig();
    } catch (e) { toast(e.message, true); save.disabled = false; }
  };
  save.onclick = submit;
  if (unsetBtn) unsetBtn.onclick = async () => {
    unsetBtn.disabled = true;
    try {
      const d = await postJSON('/api/config', { key, unset: true, repo: s.source === 'repo' });
      toast(d.message || 'unset ' + key); close(); renderConfig();
    } catch (e) { toast(e.message, true); unsetBtn.disabled = false; }
  };
  field.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } };
}
function fmtVal(v) {
  if (v === null || v === undefined) return '—';
  if (Array.isArray(v)) return '[' + v.join(', ') + ']';
  if (typeof v === 'object') return JSON.stringify(v); // dict leaves (providers, skills.state), not "[object Object]"
  return String(v);
}

route();
"""
