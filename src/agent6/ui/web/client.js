"use strict";
const view = document.getElementById('view');
const crumb = document.getElementById('crumb');
let live = null; // the active event stream, closed on navigation
// Live heartbeat for a run that is active but silent (thinking / resuming): a
// 1s ticker updates "#hb-line" with a spinner + elapsed so it reads as alive,
// not hung. hbState is refreshed by each paintRun; hbTimer runs while on a run.
let hbState = { active: false, role: 'worker', last: 0, spin: 0 };
// A run is live per the dir-aware `live` flag the server stamps; the fold's
// `finished` stays false for a run whose worker was killed, so using it alone
// paints a ticking "working…" heartbeat under a "stale" header.
function notLive(s) { return typeof s.live === 'boolean' ? !s.live : !!s.finished; }
let hbTimer = null;
let hubTimer = null; // the hub's list refresh, cleared by closeLive()
let hubVisWake = null; // the hub's visibilitychange repaint, cleared with it
const HUB_POLL_MS = 4000;
const HB_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
function hbTick() {
  const line = document.getElementById('hb-line');
  if (!line || !hbState.active) return;
  const secs = Math.floor((Date.now() - hbState.last) / 1000);
  const glyph = HB_FRAMES[hbState.spin % HB_FRAMES.length];
  line.textContent = `${glyph} ${hbState.role} working… ${secs}s`;
}

// --- theme -------------------------------------------------------------------
// The page follows the system's scheme. The theme button flips it and stores the choice only
// while it differs from the system's, so flipping back follows the system again; <head> applies
// a stored choice before the first paint.
const systemDark = matchMedia('(prefers-color-scheme: dark)');
function themeIsDark() {
  const c = document.documentElement.classList;
  return c.contains('dark') || (!c.contains('light') && systemDark.matches);
}
function toggleTheme() {
  const next = themeIsDark() ? 'light' : 'dark';
  document.documentElement.classList.remove('light', 'dark');
  if (next === (systemDark.matches ? 'dark' : 'light')) forget('a6-theme');
  else { document.documentElement.classList.add(next); remember('a6-theme', next); }
  syncTheme();
}
// The browser's own bar takes the page colour of the theme in force: a stored choice in both
// theme-color metas, else each meta's own scheme. Each theme button is named by the theme it
// switches to.
function syncTheme() {
  const c = document.documentElement.classList;
  const forced = c.contains('dark') ? 'dark' : c.contains('light') ? 'light' : '';
  for (const m of document.querySelectorAll('meta[name=theme-color]')) {
    m.content = (forced || (m.media.includes('dark') ? 'dark' : 'light')) === 'dark' ? '#0e1116' : '#f6f8fa';
  }
  const name = themeIsDark() ? 'Switch to light theme' : 'Switch to dark theme';
  for (const b of document.querySelectorAll('.theme-toggle')) { b.setAttribute('aria-label', name); b.title = name; }
}
for (const b of document.querySelectorAll('.theme-toggle')) b.onclick = toggleTheme;
systemDark.addEventListener('change', syncTheme);
syncTheme();

// --- nav rail ------------------------------------------------------------------
// <head> applies the stored state before the first paint.
const railToggle = document.getElementById('rail-toggle');
const railShow = document.getElementById('rail-show');
const railRoom = matchMedia('(min-width: 1024px)'); // room for the labels
const RAIL_PRESS = { labels: 'Collapse the sidebar', icons: 'Hide the sidebar', hidden: 'Show the sidebar' };
function railState() {
  const c = document.documentElement.classList;
  return c.contains('rail-hidden') ? 'hidden' : c.contains('rail-min') || !railRoom.matches ? 'icons' : 'labels';
}
function syncRail() {
  const state = railState();
  for (const b of [railToggle, railShow]) {
    b.setAttribute('aria-expanded', String(state === 'labels'));
    b.setAttribute('aria-label', RAIL_PRESS[state]);
    b.title = RAIL_PRESS[state];
  }
}
function cycleRail() {
  const next = { labels: 'min', icons: 'hidden', hidden: '' }[railState()];
  const pressed = document.activeElement;
  document.documentElement.classList.remove('rail-min', 'rail-hidden');
  if (next) document.documentElement.classList.add('rail-' + next);
  remember('a6-rail', next);
  syncRail();
  // A pressed button that left with the rail hands the focus to the one that took its place.
  if (pressed === railToggle && next === 'hidden') railShow.focus();
  else if (pressed === railShow) railToggle.focus();
}
railToggle.onclick = railShow.onclick = cycleRail;
railRoom.addEventListener('change', syncRail);
syncRail();

// --- PWA + notifications -----------------------------------------------------
// Install the service worker so the page is an installable PWA (manifest + SW).
// No Web Push / VAPID: OS notifications are the foreground Notification API only.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(()=>{}); });
}
// Fire an OS notification when permitted; always safe (never throws into a repaint).
function osNotifyOn() { return 'Notification' in window && Notification.permission === 'granted'; }
function osNotify(title, body) {
  try { if (osNotifyOn()) new Notification(title, { body: body || '', icon: '/icon.svg' }); } catch (_) {}
}
// Attention on the hub: a session that starts waiting on an answer, or ends, is announced once
// as a toast and an OS notification (when granted). The first paint seeds silently, as the
// machine view does, so opening the hub replays nothing.
const LIVE_WORDS = ['running', 'starting', 'waiting'];
function seedStatuses(rows) { const m = {}; for (const r of rows || []) m[r.session_id] = r.status; return m; }
function announceSessions(prev, rows) {
  for (const r of rows || []) {
    const before = prev[r.session_id];
    if (before === undefined || before === r.status) continue;
    let title = '';
    if (r.status === 'waiting') title = `${r.session_id} needs you`;
    else if (LIVE_WORDS.includes(before) && !LIVE_WORDS.includes(r.status)) title = `${r.session_id} ${r.status_label}`;
    if (!title) continue;
    toast(title);
    osNotify('agent6: ' + title, firstLine(r.task, 120));
  }
}

// --- helpers -----------------------------------------------------------------
const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const esc = s => (s == null ? '' : String(s));
// Text whose `backticked` spans show as code, as a fragment: a refusal or a message names a
// command that way (`agent6 sessions`). A backtick with no partner on its line stays as typed.
function withCode(text) {
  const frag = document.createDocumentFragment();
  String(text).split(/`([^`\n]+)`/).forEach((part, i) => { if (part) frag.append(i % 2 ? el('code', null, part) : part); });
  return frag;
}
// The sign-in token lives in storage scoped to this origin (host and port), so a page another
// local server serves never receives it, as it would a cookie; every call sends it as a header.
const TOKEN_KEY = 'a6-token';
let loginToken = ''; // an in-memory sign-in when browser storage is unavailable
function authHeaders(extra) {
  const h = Object.assign({}, extra || {});
  const t = loginToken || recall(TOKEN_KEY, '');
  if (t) h['Authorization'] = 'Bearer ' + t;
  return h;
}
function saveToken(login) { loginToken = login.token; remember(TOKEN_KEY, loginToken); }
// Per-viewer conveniences (a view, a diff mode) kept in this browser; the page works without storage.
function recall(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch (_) { return fallback; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
function forget(key) { try { localStorage.removeItem(key); } catch (_) {} }
// A 401 is the server asking for its secret: every API call brings up the sign-in form.
function signedOut(r) {
  if (r.status === 401) {
    loginToken = '';
    try { localStorage.removeItem(TOKEN_KEY); } catch (_) {}
    showLogin();
  }
  return r;
}
// Every GET: the token on the request, a 401 to the sign-in form, the server's error text thrown
// with the response's status.
async function getResponse(url) {
  const r = signedOut(await fetch(url, { headers: authHeaders() }));
  if (r.ok) return r;
  const reason = (await r.json().catch(() => ({ error: r.statusText }))).error || r.statusText;
  throw Object.assign(new Error(reason), { status: r.status });
}
async function getJSON(url) { return (await getResponse(url)).json(); }
async function getBlob(url) { return (await getResponse(url)).blob(); }
async function postJSON(url, body) {
  const r = signedOut(await fetch(url, { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(body || {}) }));
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) throw new Error(data.error || r.statusText);
  return data;
}
// The Merge buttons' two steps. The confirm names the unmerged runs that changed the same
// paths (GET .../merge); the POST lands the branch and says what the merge said.
async function confirmMerge(r) {
  let same = '';
  try { same = (await getJSON('/api/session/' + encodeURIComponent(r.session_id) + '/merge')).same_paths; }
  catch (err) { toast(err.message, true); return false; }
  return confirmDialog('Merge run ' + r.session_id + '?', 'Lands its branch on its base with your git.merge_method, as `agent6 sessions merge` does; the checkout never moves.' + (same ? '\n\n' + same : ''), 'Merge');
}
async function postMerge(r) {
  try { const d = await postJSON('/api/session/' + encodeURIComponent(r.session_id) + '/merge', {}); toast(d.message || 'merged'); route(); return true; }
  catch (err) { toast(err.message, true); return false; }
}
// A failure or a warning holds until dismissed: its reason, often a CLI refusal, is worth reading.
function toast(msg, level) {
  const kind = level === true || level === 'error' ? 'bad' : level === 'warn' ? 'warn' : '';
  const t = el('div', kind ? 'toast ' + kind : 'toast');
  t.appendChild(el('div')).append(withCode(msg)); // one flex item beside the Dismiss button
  if (kind) {
    t.setAttribute('role', 'alert');
    t.appendChild(closeButton('Dismiss', () => t.remove()));
  } else { setTimeout(() => t.remove(), 4000); }
  document.getElementById('toasts').appendChild(t);
  return t;
}
// The crumb names the page; the browser's tab and history read the same name.
function setCrumb(t) { crumb.textContent = t || ''; document.title = t ? t + ' · agent6' : 'agent6'; }

// --- dialogs -----------------------------------------------------------------
// Every modal is a <dialog> in the card skin, named by its title, with a Close button beside it.
// showModal() makes the page behind it inert; the Close button, Esc or a click outside closes it
// (closedby="any"), the browser hands the focus back to the control that opened it, and the
// dialog leaves the page when it closes. The caller fills the body and focuses its field. One
// dialog at a time: while one is open, openDialog returns null and the caller drops its own (a
// fetch that lands after Ctrl+K opened the switcher).
let dialogCount = 0;
// A dialog's or a message's close button: the close icon in a quiet square, named by what it does
// (Close for a dialog, Dismiss for a message).
const CLOSE_ICON = '<svg class="ic" viewBox="0 0 24 24" stroke-width="1.75" stroke-linecap="round" fill="none" stroke="currentColor" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
function closeButton(label, onclick) {
  const b = el('button', 'quiet x-btn');
  b.setAttribute('aria-label', label); b.title = label;
  b.innerHTML = CLOSE_ICON;
  b.onclick = onclick;
  return b;
}
function openDialog(title, cls) {
  if (document.querySelector('dialog[open]')) return null;
  const dialog = el('dialog', cls ? 'card ' + cls : 'card');
  // A config key too long for its line breaks after a dot before it breaks inside a name.
  const [first, ...rest] = title.split(/(?<=\.)/);
  const heading = el('h2', null, first);
  for (const part of rest) { heading.appendChild(el('wbr')); heading.appendChild(el('span', null, part)); }
  heading.id = 'dialog-title-' + ++dialogCount;
  dialog.setAttribute('aria-labelledby', heading.id);
  dialog.setAttribute('closedby', 'any');
  dialog.tabIndex = -1; // no Tab stop: Firefox stops on a scrolling dialog, which draws no ring
  // A touch screen has no Esc, and nothing shows that a tap outside closes the dialog.
  const head = el('div', 'dialog-head');
  head.appendChild(heading); head.appendChild(closeButton('Close', () => dialog.close()));
  dialog.appendChild(head);
  dialog.addEventListener('close', () => dialog.remove());
  document.body.appendChild(dialog);
  dialog.showModal();
  dialog.focus(); // not its Close button, which the browser would pick; a caller's field takes it next
  return dialog;
}
// Resolves true from the action's button; false from any other close, or when a dialog is open.
async function confirmDialog(title, text, action, danger) {
  const dialog = openDialog(title, 'confirm');
  if (!dialog) return false;
  dialog.setAttribute('role', 'alertdialog');
  const consequence = dialog.appendChild(el('p'));
  consequence.id = 'dialog-text-' + dialogCount;
  consequence.appendChild(withCode(text));
  dialog.setAttribute('aria-describedby', consequence.id);
  const row = dialog.appendChild(el('div', 'form-row'));
  const yes = row.appendChild(el('button', danger ? 'danger' : 'primary', action));
  const no = row.appendChild(el('button', null, 'Cancel'));
  // A held Enter's repeats press nothing: the Enter that asked would answer too.
  dialog.onkeydown = (e) => { if (e.key === 'Enter' && e.repeat) e.preventDefault(); };
  (danger ? no : yes).focus(); // a destructive question opens on Cancel
  return new Promise((resolve) => {
    yes.onclick = () => { resolve(true); dialog.close(); };
    no.onclick = () => dialog.close();
    dialog.addEventListener('close', () => resolve(false));
  });
}

// --- sign-in -----------------------------------------------------------------
// The form takes over the view, under its own crumb and with no page current; a sign-in
// restarts the route with the token. A repeat call (a poll still in flight) keeps the form and
// what is typed in it.
function showLogin(error) {
  routeGen++; // a render in flight must not paint over the form
  closeLive();
  document.querySelector('dialog[open]')?.close();
  setTab('');
  setCrumb('Sign in');
  let form = document.getElementById('login');
  if (!form) {
    view.innerHTML = '';
    form = el('form', 'card login');
    form.id = 'login';
    form.appendChild(el('h2', null, 'Sign in'));
    form.appendChild(el('div', 'muted')).append(withCode('Type the secret `agent6 web` printed in its terminal.'));
    const row = el('div', 'form-row');
    const input = el('input', 'field');
    input.type = 'text';
    // A phone keyboard must not capitalize, correct or remember the words. A refusal marks the
    // field invalid and is read out at once; typing again clears the mark.
    const attrs = { 'aria-label': 'secret', 'aria-describedby': 'login-err', autocapitalize: 'none', autocorrect: 'off', spellcheck: 'false', autocomplete: 'off' };
    for (const name in attrs) input.setAttribute(name, attrs[name]);
    input.oninput = () => input.removeAttribute('aria-invalid');
    row.appendChild(input);
    row.appendChild(el('button', 'primary', 'Sign in'));
    form.appendChild(row);
    const refusal = form.appendChild(el('div', 'err'));
    refusal.id = 'login-err';
    refusal.setAttribute('role', 'alert');
    form.onsubmit = async (e) => {
      e.preventDefault();
      try { saveToken(await postJSON('/api/login', { secret: input.value })); route(); }
      catch (err) { showLogin(err.message); }
    };
    view.appendChild(form);
    input.focus();
  }
  form.removeAttribute('aria-busy'); // a route in flight marked the kept form as the page it replaces
  if (error) {
    document.getElementById('login-err').replaceChildren(withCode(error));
    form.querySelector('input').setAttribute('aria-invalid', 'true');
  }
}
// `agent6 web` prints its URL with `#secret=<value>`: sign in with it on load and
// drop it from the address bar.
async function boot() {
  const m = /^#secret=(.+)$/.exec(location.hash);
  if (m) {
    let refused = '';
    try { saveToken(await postJSON('/api/login', { secret: decodeURIComponent(m[1]) })); }
    catch (e) { refused = e.message; }
    history.replaceState(null, '', location.pathname + location.search);
    if (refused) { showLogin(refused); return; }
  }
  route();
}
// The live stream: a fetch carrying the token (the browser's own event-stream object
// sends no header), read as
// SSE frames. A 401 brings up the sign-in form; a dropped connection is reopened after
// a pause, so the last paint stays up meanwhile; close() ends it for good.
function openStream(url, onData) {
  const stream = {
    closed: false, ctrl: null, timer: null,
    close() { stream.closed = true; if (stream.ctrl) stream.ctrl.abort(); if (stream.timer) clearTimeout(stream.timer); },
  };
  const connect = async () => {
    stream.ctrl = new AbortController();
    let r = null;
    try { r = signedOut(await fetch(url, { headers: authHeaders(), signal: stream.ctrl.signal })); } catch (_) {}
    if (stream.closed || (r && r.status === 401)) return;
    if (r && r.ok && r.body) {
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done || stream.closed) break;
          buf += dec.decode(value, { stream: true });
          let cut;
          while ((cut = buf.indexOf('\n\n')) >= 0) {
            const frame = buf.slice(0, cut);
            buf = buf.slice(cut + 2);
            const data = frame.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
            if (data) onData(data);
          }
        }
      } catch (_) { /* aborted, or the connection dropped */ }
    }
    if (!stream.closed) stream.timer = setTimeout(connect, 1500);
  };
  connect();
  return stream;
}
function closeLive() {
  if (live) { live.close(); live = null; }
  if (hubTimer) { clearInterval(hubTimer); hubTimer = null; }
  if (hubVisWake) { document.removeEventListener('visibilitychange', hubVisWake); hubVisWake = null; }
  if (hbTimer) { clearInterval(hbTimer); hbTimer = null; }
  hbState.active = false;
}
function pill(level, label) { return el('span', 'pill ' + esc(level || 'neutral'), esc(label)); }

// One owner for anything clickable that is not a native control: a keyboard
// user gets the same activation (Tab to focus, Enter/Space to fire) and a
// screen reader gets a role + name. Native <a>/<button>/<input> never need it.
function actionable(elem, activate, label) {
  elem.setAttribute('role', 'button');
  elem.tabIndex = 0;
  if (label) elem.setAttribute('aria-label', label);
  elem.onclick = activate;
  elem.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); }
  };
}

// The open page's nav link in both bars carries aria-current, which also styles it. The crumb
// starts as that link's label; a page under it (a session, a machine) names itself before it
// loads, so a failed load stands under the name it is about.
function setTab(name) {
  let label = '';
  for (const a of document.querySelectorAll('nav.tabs a, aside.rail .rail-nav a')) {
    if (a.dataset.tab === name) { a.setAttribute('aria-current', 'page'); label = a.title; }
    else a.removeAttribute('aria-current');
  }
  setCrumb(label);
}

// --- router ------------------------------------------------------------------
let booted = false; // the one-shot deep-link to `agent6 web <target>` ran
// Navigation generation: route() is an async, directly re-entrant hashchange
// handler, and every render helper awaits a fetch before its first DOM write /
// stream assignment. A superseded render's continuation must bail at each
// await boundary or it paints the wrong view over the current one, appends a
// duplicate wmenu button, and overwrites `live`, orphaning the current view's
// stream (whose stale handler could later closeLive() the visible view when
// the old run finishes).
let routeGen = 0;
// A route that shows no page: what went wrong, a retry for a failure, and the way back.
function showProblem(heading, reason, failed) {
  view.innerHTML = '';
  const card = el('div', 'card problem');
  card.appendChild(el('h2', null, heading));
  card.appendChild(el('p', failed ? 'err' : null)).append(withCode(reason));
  const ways = el('div', 'row');
  if (failed) { const again = el('button', null, 'Try again'); again.onclick = route; ways.appendChild(again); }
  if (location.hash.replace(/^#\/?/, '')) { const back = el('a', null, 'Back to Sessions'); back.href = '#/'; ways.appendChild(back); }
  card.appendChild(ways);
  view.appendChild(card);
}
// Publish a tab selection without rebuilding its graph or file pane.
function setViewHash(hash, replace = false) { if (location.hash !== hash) history[replace ? 'replaceState' : 'pushState'](null, '', hash); }
async function route() {
  const gen = ++routeGen;
  closeLive();
  document.querySelector('dialog[open]')?.close();
  document.querySelectorAll('.wmenu-btn').forEach(b => b.remove()); // header-mounted by the run view
  // The page this route replaces is busy, and dims after a beat (styles.css) so a quick hop never
  // flickers; the next page's first paint removes it. The first load's "loading…" is no page.
  for (const old of view.children) if (!old.classList.contains('empty')) old.setAttribute('aria-busy', 'true');
  const h = location.hash.replace(/^#/, '') || '/';
  // First load with no hash: honor the CLI's target (`agent6 web <run-id>`
  // opens that run; a machine name its machine). Explicit hashes win.
  if (!booted) {
    booted = true;
    if (h === '/') {
      try {
        const meta = await getJSON('/api/meta');
        if (gen !== routeGen) return; // superseded while fetching
        if (meta.target && meta.target_kind) {
          location.hash = '#/' + meta.target_kind + '/' + encodeURIComponent(meta.target);
          return; // the hashchange re-enters route()
        }
      } catch (_) {
        if (gen !== routeGen) return; // the sign-in form or a newer route took over; else the hub
      }
    }
  }
  const [routePath, queryText = ''] = h.split('?');
  const query = new URLSearchParams(queryText);
  const parts = routePath.split('/').filter(Boolean); // e.g. ['session','abc']
  try {
    if (parts.length === 0) { setTab('hub'); await renderHub(undefined, gen); }
    else if (parts[0] === 'machines') { setTab('machines'); await renderHub('machines', gen); }
    else if (parts[0] === 'config') { setTab('config'); await renderConfig(gen); }
    else if (parts[0] === 'git') { setTab('git'); await renderGit(gen, parts.slice(1).map(decodeURIComponent), query); }
    else if (parts[0] === 'files') { setTab('files'); await renderFiles(gen, decodeURIComponent(parts[1] || ''), parts.slice(2).map(decodeURIComponent).join('/'), query); }
    else if (parts[0] === 'session' && parts[1]) { setTab('hub'); await renderRun(decodeURIComponent(parts[1]), undefined, gen); }
    else if (parts[0] === 'machine' && parts[1]) { setTab('machines'); await renderMachine(decodeURIComponent(parts[1]), gen); }
    else if (parts[0] === 'draft' && parts[1]) { const n = decodeURIComponent(parts[1]); setTab('machines'); await renderRun(n, { base: '/api/draft/' + encodeURIComponent(n), readOnly: true, title: 'Machine draft', crumb: 'draft ' + n }, gen); }
    else { setTab(''); setCrumb('Not found'); showProblem('Not found', 'No page lives at #' + h + '.'); }
  } catch (e) {
    if (gen !== routeGen) return; // a superseded render's late error must not paint
    // A 404 names something that does not exist, which a retry cannot bring back.
    if (e.status === 404) showProblem('Not found', e.message);
    else showProblem('Could not load this page', e.message, true);
  }
}
window.addEventListener('hashchange', route);

// Text and a plan handed to the new-work composer by another view (a diff
// line's quote, a receipt's revert request, a plan's Run this plan): the dock
// takes them once when it paints.
function handToNewWork(fields) {
  window.pendingNewWork = Object.assign({}, window.pendingNewWork || {}, fields);
}

// --- hub ---------------------------------------------------------------------
// A button whose action sends a request runs it once at a time, and only when `ready()` says
// there is something to send: it stays disabled while the action is in flight, so a double click
// or a repeated Enter sends one request, and while ready() is false. button.sync() reads ready()
// again; call it when what ready() reads changes.
function runOnce(button, action, ready = () => true) {
  let busy = false;
  button.sync = () => { button.disabled = busy || !ready(); };
  button.onclick = async () => {
    const focused = document.activeElement === button;
    busy = true; button.sync();
    try { await action(); } finally {
      busy = false; button.sync();
      // A dialog that closed while the button was disabled could not hand the focus back to it.
      if (focused && document.activeElement === document.body) button.focus();
    }
  };
  button.sync();
}
// Two popups share an entry: one closing leaves the references the other set.
let suggestLists = 0; // numbers each popup's id
function unpointPopup(entry, box) {
  if (entry.getAttribute('aria-controls') !== box.id) return;
  entry.removeAttribute('aria-controls'); entry.removeAttribute('aria-activedescendant');
}
// /parallel model-id autocomplete for the new-work composer. When the task text
// starts with `/parallel ` and the caret sits before `--`, offer the known
// model ids (GET /api/config/suggest/parallel.models, exactly the set
// run --parallel accepts) filtered by the comma-fragment under the caret; click,
// Enter, or ↑/↓+Enter inserts it. The web analogue of the config editor's
// datalist (a <textarea> can't carry a native datalist).
function attachParallelSuggest(task, root) {
  let models = null;           // null=unfetched, []=in-flight/empty, else the list
  let box = null, items = [], active = -1;
  task.setAttribute('aria-autocomplete', 'list');
  const ensureModels = () => {
    if (models !== null) return;
    models = [];                // sentinel: fetch at most once per composer
    getJSON('/api/config/suggest/parallel.models').then(d => { models = d.values || []; render(); }).catch(() => {});
  };
  const frag = () => {          // the comma-fragment under the caret, or null
    const v = task.value, caret = task.selectionStart;
    const markers = [...v.matchAll(/(^|\s)\/parallel(?=\s|$)/gi)].map(m => ({
      start: m.index + m[1].length,
      end: m.index + m[0].length,
    }));
    if (!markers.length || v.slice(0, markers[0].start).trim()) return null;
    const marker = markers.filter(m => m.end < caret).pop();
    if (!marker || !/\s/.test(v[marker.end])) return null;
    const segmentEnd = markers.find(m => m.start > marker.start)?.start ?? v.length;
    const separator = /(^|\s)--(?=\s|$)/.exec(v.slice(marker.end, segmentEnd));
    const end = separator ? marker.end + separator.index + separator[1].length : segmentEnd;
    let start = marker.end;
    while (start < end && /\s/.test(v[start])) start++;
    if (caret < start || caret > end) return null;
    if (/^\p{Decimal_Number}+$/u.test(v.slice(start, end).trim())) return null;
    let fragStart = start + v.slice(start, caret).lastIndexOf(',') + 1;
    let fragEnd = caret;
    while (fragEnd < end && v[fragEnd] !== ',') fragEnd++;
    while (fragStart < fragEnd && /\s/.test(v[fragStart])) fragStart++;
    while (fragEnd > fragStart && /\s/.test(v[fragEnd - 1])) fragEnd--;
    return { fragStart, fragEnd, text: v.slice(fragStart, Math.min(caret, fragEnd)) };
  };
  const close = () => {
    if (box) { unpointPopup(task, box); box.remove(); box = null; }
    items = []; active = -1;
  };
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
    if (!box) {
      box = el('div', 'ac-pop'); box.id = 'suggest-' + ++suggestLists; box.setAttribute('role', 'listbox');
      root.appendChild(box); task.setAttribute('aria-controls', box.id);
    }
    box.textContent = '';
    items.forEach((m, i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''), m);
      o.id = box.id + '-' + i; o.setAttribute('role', 'option'); o.setAttribute('aria-selected', String(i === active));
      o.onmousedown = (e) => { e.preventDefault(); insert(m); };
      box.appendChild(o);
    });
    if (active >= 0) task.setAttribute('aria-activedescendant', box.id + '-' + active);
    else task.removeAttribute('aria-activedescendant');
  };
  task.addEventListener('input', () => { active = -1; render(); });
  task.addEventListener('click', render);
  task.addEventListener('blur', () => setTimeout(close, 120));
  // Returns true when the popup consumed the key (caller must not also act on it).
  return { onKeyDown(e) {
    if (!box || !items.length || composing(e)) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

// A row of labelled dropdowns above a composer, the TUI's picker row: mode,
// preset and model over new work, preset and model over a resume.
function pickerRow(pairs) {
  const row = el('div', 'row pickers');
  for (const [text, select] of pairs) {
    const pick = el('label', 'pick'); // wraps as one piece: a label never strands from its dropdown
    pick.appendChild(el('span', 'muted', text)); pick.appendChild(select);
    row.appendChild(pick);
  }
  return row;
}
// The line repaints on a row's input event: code that sets a picker dispatches one.
function pickerSummary(rows, words) {
  const fold = el('button', 'fold picker-summary');
  const line = fold.appendChild(el('span'));
  const paint = () => { line.textContent = words().filter(Boolean).join(' · '); };
  fold.setAttribute('aria-expanded', 'false');
  fold.onclick = () => fold.setAttribute('aria-expanded', String(fold.getAttribute('aria-expanded') !== 'true'));
  for (const row of rows) row.addEventListener('input', paint);
  paint();
  return fold;
}
// A dropdown's pick as a summary line names it, without "(config default)" or "(as recorded)".
const pickedName = (select) => (select.selectedOptions[0]?.text || '').replace(/ \((config default|as recorded)\)$/, '');
// A model's pick names the model without its provider path, or "no model" where none resolves.
const pickedModel = (select) => { const m = pickedName(select).split('/').pop(); return m === 'none' ? 'no model' : m; };

// The new-work composer, docked at the bottom of the Sessions page: the mode,
// worktree, preset and model row, whose advanced fold opens a row with the plan
// to run from and a spend cap, above task text + Start (Enter starts,
// Shift+Enter newline).
// `presets` and `presetDefault` are the hub payload's list and the label of
// its first option, which keeps the config's own preset; `sessions` are the
// hub's rows, whose finished plans the fold offers. The model box lists every
// provider/model the config can run (/api/routes) under a first option that
// names the route the config resolves for the mode and preset, relabelled and
// picked again on every change of either. Each first option adds no flag;
// any other pick rides as --preset / --model / --worktree / --from / --max-usd.
function newWorkDock(presets, presetDefault, sessions) {
  const root = el('div', 'composer dock');
  const row = el('div', 'row');
  const pending = window.pendingNewWork || {}; delete window.pendingNewWork;
  const task = el('textarea', 'field'); task.placeholder = 'task / question…';
  // A typed draft survives a switch to a session and back (this browser only).
  task.value = pending.task || recall('a6-draft-new', '');
  task.oninput = () => { if (task.value) remember('a6-draft-new', task.value); else forget('a6-draft-new'); };
  const mode = el('select', 'field');
  for (const m of ['run', 'plan', 'ask']) { const o = el('option', null, m); o.value = m; mode.appendChild(o); }
  const worktree = el('input'); worktree.type = 'checkbox';
  worktree.title = 'start the run in a new git worktree of this repository (run --worktree)';
  const seed = el('select', 'field');
  seed.title = 'seed the run with a plan: its text becomes the task when you type none (run --from)';
  const noSeed = el('option', null, 'none'); noSeed.value = ''; seed.appendChild(noSeed);
  const live = new Set(['running', 'starting', 'waiting']);
  for (const r of (sessions || []).filter(r => r.mode === 'plan' && !live.has(r.status))) {
    const o = el('option', null, r.session_id + ' · ' + (r.task_line || '').split('\n')[0].slice(0, 40));
    o.value = r.session_id; seed.appendChild(o);
  }
  const cap = el('input', 'field'); cap.type = 'number'; cap.step = 'any'; cap.placeholder = 'config';
  cap.title = 'the metered spend ceiling in USD for this session (--max-usd); -1 is unlimited';
  // The last picker opens the advanced options in a row under the pickers.
  const more = pickerRow([['from plan', seed], ['max USD', cap]]);
  const advanced = el('button', 'fold', 'advanced');
  const fold = (open) => { advanced.setAttribute('aria-expanded', String(open)); more.hidden = !open; };
  advanced.onclick = () => fold(more.hidden);
  fold(false);
  if (pending.seed_from) { seed.value = pending.seed_from; fold(true); }
  // A worktree and a seed are run options: the other modes grey them.
  const followMode = () => { worktree.disabled = seed.disabled = mode.value !== 'run'; };
  const preset = el('select', 'field');
  preset.title = 'config preset for this run (a preset cannot change mid-run)';
  const dflt = el('option', null, presetDefault); dflt.value = ''; preset.appendChild(dflt);
  for (const p of (presets || [])) { const o = el('option', null, p); o.value = p; preset.appendChild(o); }
  const model = el('select', 'field');
  model.title = 'the model for this run, over every config layer; re-resolved whenever the mode or preset changes';
  // The first option names the resolved route: '…' while the first answer is on its way, and
  // 'config default' when none came.
  const resolved = el('option', null, '…'); resolved.value = ''; model.appendChild(resolved);
  let routeRequest = 0;
  const fillRoutes = async () => {
    const request = ++routeRequest;
    model.value = '';
    model.disabled = true;
    let d;
    try {
      d = await getJSON('/api/routes?mode=' + encodeURIComponent(mode.value) + '&preset=' + encodeURIComponent(preset.value));
    } catch (e) {
      if (request === routeRequest) {
        resolved.textContent = 'config default'; model.disabled = false;
        model.dispatchEvent(new Event('input', { bubbles: true }));
        toast('could not list the models: ' + e.message, true);
      }
      return;
    }
    if (request !== routeRequest) return;
    resolved.textContent = d.default_label;
    model.replaceChildren(resolved, ...(d.routes || []).map(r => new Option(r, r)));
    model.value = '';
    model.disabled = false;
    model.dispatchEvent(new Event('input', { bubbles: true })); // the summary line names the route
  };
  mode.onchange = () => { followMode(); fillRoutes(); }; preset.onchange = fillRoutes; fillRoutes(); followMode();
  const go = el('button', 'primary', 'Start');
  // Start sends a task, or runs a picked plan as written. The task stays as sent while the request
  // is in flight: a session that starts takes its text.
  runOnce(go, async () => {
    task.readOnly = true;
    try {
      const d = await postJSON('/api/new', {
        mode: mode.value, task: task.value, preset: preset.value, model: model.value,
        worktree: mode.value === 'run' && worktree.checked,
        seed_from: mode.value === 'run' ? seed.value : '',
        max_usd: cap.value.trim() ? Number(cap.value) : null,
      });
      if (d.session_id) { forget('a6-draft-new'); location.hash = '#/session/' + encodeURIComponent(d.session_id); }
    } catch (e) { toast(e.message, true); }
    task.readOnly = false;
  }, () => !!task.value.trim() || (!!seed.value && mode.value === 'run'));
  task.addEventListener('input', go.sync); mode.addEventListener('change', go.sync); seed.onchange = go.sync;
  const ac = attachParallelSuggest(task, root);
  const commands = attachCommandSuggest(task, root, () => false, ['/parallel', '/review']);
  task.onkeydown = (e) => {
    if (composing(e)) return;
    if (commands.onKeyDown(e)) return;
    if (ac.onKeyDown(e)) return;   // the /parallel suggestion popup took the key
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go.click(); }
  };
  row.appendChild(task); row.appendChild(go);
  const picks = pickerRow([['mode', mode], ['worktree', worktree], ['preset', preset], ['model', model]]);
  picks.appendChild(advanced);
  // A phone's summary line names the mode, the preset, the model, and each run option set.
  root.appendChild(pickerSummary([picks, more], () => {
    const run = mode.value === 'run';
    return [mode.value, 'preset: ' + pickedName(preset), pickedModel(model),
      run && worktree.checked && 'worktree', run && seed.value && 'from plan: ' + seed.value,
      cap.value.trim() && 'max USD: ' + cap.value.trim()];
  }));
  root.appendChild(picks);
  root.appendChild(more);
  root.appendChild(row);
  root.appendChild(el('div', 'hint', 'Enter starts the run / plan / ask (a run from a plan needs no task) · Shift+Enter newline · '
    + '/parallel [SPEC --] TASK dispatches children (SPEC: count or models; repeat to queue more) · Ctrl+K switches sessions'));
  return root;
}

// The create-machine composer, docked at the bottom of the Machines page.
function createMachineDock() {
  const root = el('div', 'composer dock');
  const row = el('div', 'row');
  const ct = el('textarea', 'field'); ct.placeholder = 'describe a machine to create…';
  const cbtn = el('button', 'primary', 'Create machine');
  runOnce(cbtn, async () => {
    try { const d = await postJSON('/api/machine/create', { task: ct.value }); ct.value=''; if (d.draft) location.hash = '#/draft/' + encodeURIComponent(d.draft); }
    catch (e) { toast(e.message, true); }
  }, () => !!ct.value.trim());
  ct.oninput = cbtn.sync;
  ct.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey && !composing(e)) { e.preventDefault(); cbtn.click(); } };
  row.appendChild(ct); row.appendChild(cbtn);
  root.appendChild(row);
  root.appendChild(el('div', 'hint', 'Enter creates a machine draft from the description · Shift+Enter newline'));
  return root;
}

// The hub's lists repaint in place on every refresh. Each row is keyed: an unchanged row keeps its
// element, and with it the focus, a text selection or a press inside it; a changed row is painted
// afresh and takes the focus over; moveBefore() puts rows in order without dropping the focus.
// On a refresh (`hold`), a changed row under a press or a text selection keeps its paint until a
// later one, so the click lands and the selection stays.
// `rows` is [{ key, data, paint }]: `data` decides whether the row changed, `paint()` builds it.
const focusables = (row) => [row, ...row.querySelectorAll('a[href], button, input, select, textarea, summary, [tabindex]')];
let refocusing = false; // a repainted row is taking the focus over, which scrolls nothing
function patchRows(list, rows, hold = false) {
  const old = new Map([...list.children].map(n => [n.rowKey, n]));
  const kept = new Set();
  let prev = null;
  for (const { key, data, paint } of rows) {
    const sig = JSON.stringify(data);
    let node = old.get(key);
    if (!node || (node.rowSig !== sig && !(hold && (node.matches(':active') || selecting(node))))) {
      const fresh = paint();
      fresh.rowKey = key; fresh.rowSig = sig;
      if (node) {
        const focused = focusables(node).indexOf(document.activeElement);
        node.replaceWith(fresh);
        if (focused >= 0) { refocusing = true; focusables(fresh)[focused]?.focus({ preventScroll: true }); refocusing = false; }
      }
      node = fresh;
    }
    kept.add(node);
    const want = prev ? prev.nextSibling : list.firstChild;
    if (node !== want) {
      if (node.parentNode === list) list.moveBefore(node, want); else list.insertBefore(node, want);
    }
    prev = node;
  }
  for (const n of [...list.children]) if (!kept.has(n)) n.remove();
}
// Whether the reader has text selected in `node`.
function selecting(node) {
  const s = getSelection();
  return !!s && !s.isCollapsed && s.containsNode(node, true);
}

// A list card: an h2 title over rows that open what their titles link to. card.show(entries, hold)
// repaints the rows in place; `rows(entry)` lists the row or rows an entry shows (patchRows).
function listCard(title, empty, rows) {
  const card = el('div', 'card');
  if (title) card.appendChild(el('h2', null, title));
  const list = el('div', 'list');
  card.appendChild(list);
  // A click anywhere on a row opens it, unless it hit a control or ended a drag: a drag selects
  // text for copying. A key press on the title link opens it too (detail 0: no pointer moved).
  let down = [0, 0];
  list.onpointerdown = (e) => { down = [e.clientX, e.clientY]; };
  list.onclick = (e) => {
    const it = e.target.closest('.item');
    if (!it) return;
    if (e.detail && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) { e.preventDefault(); return; }
    if (e.target.closest('a, button, input, select, textarea, summary, label')) return;
    const link = it.querySelector('a.title');
    if (link) location.hash = link.hash;
  };
  // Space on a row's title link opens it as Enter does, as a focused row does on every page.
  list.onkeydown = (e) => {
    if (e.key === ' ' && e.target.matches('a.title')) { e.preventDefault(); e.target.click(); }
  };
  // A row the keyboard reaches scrolls into view whole, clear of the dock (the row's
  // scroll-margin-bottom); the browser alone shows only the focused title link.
  list.addEventListener('focusin', (e) => {
    if (!refocusing && e.target.matches(':focus-visible')) e.target.closest('.item')?.scrollIntoView({ block: 'nearest' });
  });
  card.show = (entries, hold) => patchRows(list, entries.length ? entries.flatMap(rows)
    : [{ key: '', data: empty, paint: () => el('div', 'empty', empty) }], hold);
  return card;
}
// One row: `paint(it, g)` fills the row and its text column.
function itemRow(paint) {
  const it = el('div', 'item');
  const g = el('div', 'grow');
  it.appendChild(g);
  paint(it, g);
  return it;
}

// The groups whose children the operator expanded, kept across the hub's repaints.
const expandedGroups = new Set();

// A row's title is the link that opens it. A drag across it selects its text, as anywhere else in
// the row: no link drag here, and user-select in styles.css.
function titleLink(cls, text, href) {
  const title = el('a', cls, text);
  title.href = href; title.draggable = false;
  return title;
}
// A row's task: its first line is the link that opens the row, a resume's steer the line under it.
function taskTitle(parent, taskLine, href) {
  const [task, continued] = taskLine.split('\n');
  parent.appendChild(titleLink('title', task, href));
  if (continued) parent.appendChild(el('div', 'continued', continued));
}
// A row's sub line: its parts joined by ' · ', each separator kept with the part before it. A text
// part (a mode, an id, a date, a cost) stays whole on one line; a node part (the status, a name, a
// path) wraps as text.
function subLine(parts) {
  const sub = el('div', 'sub');
  parts = parts.filter(Boolean);
  parts.forEach((part, i) => {
    const f = typeof part === 'string' ? el('span', 'f', part) : el('span');
    if (typeof part !== 'string') f.append(part);
    if (i < parts.length - 1) f.append('\u00a0·');
    sub.append(...(i ? [' ', f] : [f]));
  });
  return sub;
}

// A row's status label without the mode the server folds in for a listing with no mode column
// ('plan · running'): a session row names its mode beside the status, and the Drafts card names
// its rows.
function statusOnly(r) {
  return r.status_label.startsWith(r.mode + ' · ') ? r.status_label.slice(r.mode.length + 3) : r.status_label;
}
// One session row, `depth` levels under its group's row. A group's row carries a `children: N`
// toggle that `regroup` flips; its children (`r.children`, the server's nested rows) follow it
// as rows of their own, each a level deeper.
function paintSession(r, depth, regroup) {
  return itemRow((it, g) => {
    if (depth) { it.classList.add('child'); it.style.setProperty('--depth', depth); }
    taskTitle(g, r.task_line || firstLine(r.task, 200) || '(no task)', '#/session/' + encodeURIComponent(r.session_id));
    // The status leads the first sub line, so every row's dot sits in one column: the server's
    // status and merge note, paired as the CLI and TUI status cells pair them, then the mode.
    // id_cell is the id with the winner and worktree marks the CLI and TUI id cells carry; a
    // missing worktree's note wraps. usage is blank for a clean $0.
    const status = [statusOnly(r), r.merge_note].filter(Boolean).join(' · ');
    const cell = esc(r.id_cell || r.session_id), cut = cell.indexOf(' (', r.session_id.length);
    let id = cell;
    if (cut >= 0) { id = el('span'); id.append(el('span', 'f', cell.slice(0, cut)), cell.slice(cut)); }
    g.appendChild(subLine([pill(r.status_level, status), esc(r.mode), id, esc(r.when), esc(r.usage)]));
    // Where the run works and the branch holding its commits, when it has them. A missing
    // worktree's path is in the id cell's note already.
    const where = r.worktree && !cell.includes(r.worktree);
    if (where || r.branch) g.appendChild(subLine([where && el('span', null, 'worktree ' + r.worktree), r.branch && el('span', null, 'branch ' + r.branch)]));
    // Review fills the new-work composer with `/review <branch>`; Merge lands the branch
    // as `sessions merge` does. Both need a finished run whose branch exists; Merge also
    // needs commits no merge stamped. A group has no branch of its own.
    const live = LIVE_WORDS.includes(r.status);
    if (r.branch && !live) {
      const actions = el('div', 'item-actions'); // wraps under the text as one piece
      const review = el('button', 'mini', 'Review');
      review.onclick = () => fillAsEdit(view.querySelector('.dock textarea'), '/review ' + r.branch);
      actions.appendChild(review);
      if (r.unmerged) {
        const merge = el('button', 'mini', 'Merge');
        runOnce(merge, async () => {
          if (!await confirmMerge(r)) return;
          await postMerge(r);
        });
        actions.appendChild(merge);
      }
      it.appendChild(actions);
    }
    const children = r.children || [];
    if (!children.length) return;
    const open = expandedGroups.has(r.session_id);
    const toggle = el('button', 'fold', `children: ${children.length} · ${open ? 'hide' : 'show'}`);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.onclick = regroup;
    g.appendChild(toggle);
  });
}

// The OS-notification button names the permission the browser holds: on, blocked (only the
// browser's site settings allow them again), or off until a click asks for it.
function notificationsButton(cls) {
  const b = el('button', cls);
  const supported = 'Notification' in window;
  const paint = () => {
    const p = supported ? Notification.permission : '';
    b.textContent = { granted: 'Notifications on', denied: 'Notifications blocked', default: 'Turn on notifications' }[p] || 'Notifications not supported';
    b.title = p === 'denied' ? "allow notifications for this page in the browser's site settings" : '';
    b.disabled = !supported;
  };
  // A page cannot turn notifications off, so a click while they are on says where to.
  b.onclick = async () => {
    await Notification.requestPermission();
    paint();
    // The state the label shows: headless Chromium resolves 'default' while this reads 'denied'.
    const p = Notification.permission;
    if (p === 'granted') toast("notifications are on: the browser's site settings turn them off");
    else if (p === 'denied') toast("notifications are blocked: allow them for this page in the browser's site settings", true);
  };
  paint();
  return b;
}

function sessionsCard() {
  // The page names itself in the top bar; the card is the list and its toolbar.
  let shown = [];
  const rows = (r, depth) => {
    const open = expandedGroups.has(r.session_id);
    const regroup = () => {
      if (open) expandedGroups.delete(r.session_id); else expandedGroups.add(r.session_id);
      card.show(shown);
    };
    return [{ key: r.session_id, data: [r, open], paint: () => paintSession(r, depth, regroup) },
      ...(open ? (r.children || []).flatMap(c => rows(c, depth + 1)) : [])];
  };
  const card = listCard('', 'no sessions yet (describe a task below to start one)', r => rows(r, 0));
  const show = card.show;
  card.show = (sessions, hold) => { shown = sessions; show(sessions, hold); };
  // Maintenance, the actions the TUI's File menu holds, sits in one row above the list.
  const toolbar = el('div', 'toolbar');
  // OS notifications for a session that starts waiting or ends need a granted permission.
  toolbar.appendChild(notificationsButton('mini quiet'));
  const action = (label, run) => { const b = el('button', 'mini quiet danger', label); runOnce(b, run); toolbar.appendChild(b); };
  // The CLI's --delete-squashed as its own entry: with the default squash
  // method every merged run's branch is unreachable, so a plain prune
  // reports them and deletes none.
  const prune = async (deleteSquashed) => {
    if (deleteSquashed && !await confirmDialog('Force-delete branches and chain refs recorded as squash-merged?', 'Their commits stay in the base; each deletion prints an undelete command.', 'Force-delete', true)) return;
    try { const d = await postJSON('/api/sessions/prune', { delete_squashed: deleteSquashed }); toast(d.message || 'pruned'); route(); }
    catch (e) { toast(e.message, true); }
  };
  action('Prune merged runs', () => prune(false));
  action('Prune merged runs, squash-merged too', () => prune(true));
  action('Clear saved asks', async () => {
    if (!await confirmDialog('Delete every saved ask?', 'Runs `agent6 sessions remove --asks`: removes every saved `agent6 ask` transcript for this directory. Asks run elsewhere are untouched.', 'Delete', true)) return;
    try { const d = await postJSON('/api/sessions/rm_asks', {}); toast(d.message || 'cleared'); route(); }
    catch (e) { toast(e.message, true); }
  });
  card.insertBefore(toolbar, card.firstChild);
  return card;
}

function machinesCard() {
  return listCard('Instances', 'no machines yet (describe one below to create it)', m => [{ key: m.name, data: m, paint: () => itemRow((it, g) => {
    // An instance goes by its own name, as `machine list` and the TUI list it; the sub line names
    // the machine it runs when the two differ. The reason rides the status label (failed · why).
    g.appendChild(titleLink('title name', m.name, '#/machine/' + encodeURIComponent(m.name)));
    const machine = m.machine && m.machine !== m.name && el('span', null, 'machine ' + m.machine);
    const at = m.current && el('span', null, 'at ' + m.current);
    g.appendChild(subLine([pill(m.status_level, m.status_label), machine, at, esc(m.when)]));
  }) }]);
}

function draftsCard() {
  return listCard('Drafts', '', d => [{ key: d.session_id, data: d, paint: () => itemRow((it, g) => {
    taskTitle(g, d.task_line || firstLine(d.task, 200) || d.session_id, '#/draft/' + encodeURIComponent(d.session_id));
    // The reason rides the status label (failed · provider error).
    g.appendChild(subLine([pill(d.status_level, statusOnly(d)), d.session_id, esc(d.when)]));
  }) }]);
}

function machineFilesCard() {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Run a machine'));
  const frow = el('div', 'form-row machine-files');
  const run = (mf) => {
    const b = el('button', null, '▶ ' + mf.name);
    runOnce(b, async () => {
      // A machine run spends against your provider from this one click, so
      // confirm the cost first (the deliberate-cost bar the composer's typed
      // Start and the stop dialog already hold).
      if (!await confirmDialog('Run ' + mf.name + '?', 'It starts a paid machine run now and spends against your provider.', 'Run')) return;
      try { await postJSON('/api/machine/run', { file: mf.path }); toast('started ' + mf.name); setTimeout(route, 800); } catch (e) { toast(e.message, true); }
    });
    return b;
  };
  card.appendChild(frow);
  card.show = (files, hold) => patchRows(frow, files.map(mf => ({ key: mf.path, data: mf, paint: () => run(mf) })), hold);
  return card;
}

// The docked composer's height, as --dock-h on the page for a row's scroll-margin-bottom, so a
// row scrolled into view stops clear of the dock. A dock that leaves the page reports 0.
const dockHeight = new ResizeObserver((entries) => {
  for (const e of entries) {
    const h = e.borderBoxSize[0].blockSize;
    document.documentElement.style.setProperty('--dock-h', h + 'px');
    if (!h) dockHeight.unobserve(e.target);
  }
});
async function renderHub(focus, gen) {
  const machinesTab = focus === 'machines';
  let data;
  try { data = await getJSON('/api/hub'); }
  catch (e) {
    if (gen !== undefined && gen !== routeGen) return; // superseded, or the sign-in form took over
    // A failed first load says why, offers Try again, and paints the page once the server answers.
    showProblem('Could not load this page', e.message, true);
    view.querySelector('.problem p').after(el('p', 'muted', `retrying every ${HUB_POLL_MS / 1000} s`));
    hubTimer = setInterval(() => getJSON('/api/hub').then(route, () => {}), HUB_POLL_MS);
    return;
  }
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  // Full-width listing stack; the tab's composer docks at the bottom of the
  // viewport (new work on Sessions, create-machine on Machines).
  const lists = el('div', 'grid');
  let show;
  if (machinesTab) {
    const cards = { files: machineFilesCard(), machines: machinesCard(), drafts: draftsCard() };
    show = (d, hold) => {
      const entries = { files: d.machine_files || [], machines: d.machines || [], drafts: d.drafts || [] };
      for (const k in cards) cards[k].show(entries[k], hold);
      // The machine files and drafts cards leave the page while they are empty.
      patchRows(lists, Object.keys(cards).filter(k => k === 'machines' || entries[k].length)
        .map(k => ({ key: k, data: k, paint: () => cards[k] })));
    };
  } else {
    const sessions = sessionsCard();
    lists.appendChild(sessions);
    show = (d, hold) => sessions.show(d.sessions, hold);
  }
  show(data);
  let statuses = seedStatuses(data.sessions);
  view.appendChild(lists);
  const dock = machinesTab ? createMachineDock() : newWorkDock(data.presets, data.preset_default_label, data.sessions);
  view.appendChild(dock);
  dockHeight.observe(dock);
  // Without this the hub paints once, so a child that finished, failed or
  // crashed keeps its "running" pill until a manual reload; clicking the
  // already-active tab does not re-enter route(). Refresh the lists only: a
  // whole-view repaint would discard text typed into the dock's composer.
  const refreshLists = async () => {
    // A background tab has nobody to mislead, unless it may notify: then it keeps watching.
    if (document.hidden && !osNotifyOn()) return;
    if (gen !== undefined && gen !== routeGen) return; // superseded; closeLive() clears us
    try {
      const next = await getJSON('/api/hub');
      if (gen !== undefined && gen !== routeGen) return;
      announceSessions(statuses, next.sessions);
      statuses = seedStatuses(next.sessions);
      show(next, true);
    } catch (_) { /* transient: keep the last good paint */ }
  };
  hubTimer = setInterval(refreshLists, HUB_POLL_MS);
  // Re-focusing the tab otherwise shows the pre-blur pills (the interval
  // skips hidden ticks) for up to a full poll period.
  hubVisWake = () => { refreshLists(); };
  document.addEventListener('visibilitychange', hubVisWake);
}

// --- conversation ------------------------------------------------------------
// Renders a /conversation payload: folded transcript items whose lines are
// [text, style] spans from the shared renderer (viewmodel.transcript_style), so
// the web shows exactly what the CLI stream and the TUI conversation view show.
// The detail level cycles collapsed -> expanded -> hidden (persisted); an item
// with a longer form (clipped tool output, folded thinking) expands on a click or a key.
const DETAIL_CYCLE = { collapsed: 'expanded', expanded: 'hidden', hidden: 'collapsed' };
function tailStr(s, n) { return s.length <= n ? s : '…' + s.slice(-n); }
function firstLine(s, n) { const t = String(s == null ? '' : s).split('\n')[0]; return t.length > n ? t.slice(0, n - 1) + '…' : t; }
// `box` scrolls and `body` holds the items; `empty` = { live, done } rewords the no-items note.
function makeConv(url, box, body, empty = {}) {
  const conv = {
    items: [], open: new Map(),
    detail: recall('a6-detail', '') || 'collapsed',
    timer: null,
    finished: false, // set by setLive: an ended run/machine gets past-tense empty text
  };
  const itemsHost = el('div', 'conv');
  const liveHost = el('div', 'conv conv-live');
  liveHost.style.display = 'none';
  body.appendChild(itemsHost); body.appendChild(liveHost);
  // The box follows its tail while the reader stays there: each scroll records whether they do,
  // and a resize (a composer growing as it is typed in, a narrower window) scrolls back to it.
  let atTail = true;
  box.addEventListener('scroll', () => { atTail = box.scrollTop + box.clientHeight >= box.scrollHeight - 40; });
  new ResizeObserver(() => { if (atTail) box.scrollTop = box.scrollHeight; }).observe(box);
  // A run/machine that produced no conversation: "yet ... as it streams" while it
  // can still stream, past tense once it has ended (a terminal tool-only machine).
  const emptyNote = () => conv.finished
    ? empty.done || 'this session made no conversation'
    : empty.live || 'no conversation yet; it appears as the session streams';

  const paintItems = () => {
    // A repaint keeps the keyboard on the item it was on.
    const refocus = itemsHost.contains(document.activeElement);
    itemsHost.innerHTML = '';
    let shown = 0;
    conv.items.forEach((it, i) => {
      if ((it.kind === 'thinking' || it.kind === 'tool') && conv.detail === 'hidden' && !conv.open.get(i)) return;
      const expanded = conv.open.get(i) ?? (conv.detail === 'expanded');
      const lines = expanded && it.full ? it.full : it.lines;
      const div = el('div', 'ci' + (it.full ? ' exp' : ''));
      div.dataset.i = i;
      for (const line of lines) {
        const ln = el('div');
        for (const [text, style] of line) ln.appendChild(el('span', 's-' + style, text));
        if (!line.length) ln.appendChild(document.createTextNode(' '));
        // A wrapped line continues under its own text, past its indent and its marker ("└ ").
        const hang = /^\s*(?:[\p{P}\p{S}]\s+)?/u.exec(ln.textContent)[0].length;
        if (hang) ln.style.setProperty('--hang', hang + 'ch');
        div.appendChild(ln);
      }
      if (it.full) {
        // A click on the item, or Enter or Space on its first line, opens or closes its longer form.
        const toggle = () => { conv.open.set(i, !expanded); paintItems(); };
        const head = div.firstElementChild;
        head.setAttribute('role', 'button');
        head.setAttribute('aria-expanded', String(expanded));
        head.tabIndex = -1;
        head.onfocus = () => {
          conv.stop = i;
          for (const other of itemsHost.querySelectorAll('[role=button]')) other.tabIndex = other === head ? 0 : -1;
        };
        head.onkeydown = (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); return; }
          const heads = [...itemsHost.querySelectorAll('[role=button]')], at = heads.indexOf(head);
          const next = { ArrowUp: heads[at - 1], ArrowDown: heads[at + 1], Home: heads[0], End: heads.at(-1) }[e.key];
          if (next) { e.preventDefault(); next.focus(); }
        };
        div.title = expanded ? 'click to collapse' : 'click to expand';
        div.onclick = () => {
          // A drag that selects text inside the item ends in a click: it keeps the selection.
          const picked = getSelection();
          if (!picked.isCollapsed && div.contains(picked.anchorNode)) return;
          toggle();
        };
      }
      itemsHost.appendChild(div); shown++;
    });
    // The empty note yields to the live pane: streamed text under a
    // "no conversation yet" banner reads as a contradiction.
    if (!shown && liveHost.style.display === 'none') {
      itemsHost.appendChild(el('div', 'muted conv-empty', emptyNote()));
    }
    // The conversation is one Tab stop, the item last focused or else the newest; arrows move it.
    const heads = itemsHost.querySelectorAll('[role=button]');
    const stop = itemsHost.querySelector('.ci[data-i="' + conv.stop + '"] > [role=button]') || heads[heads.length - 1];
    if (stop) { stop.tabIndex = 0; if (refocus) stop.focus({ preventScroll: true }); }
    if (atTail) box.scrollTop = box.scrollHeight;
  };

  conv.refresh = async () => {
    if (conv.closed || !box.isConnected) return; // closed or navigated away: don't fetch or paint stale
    let data; try { data = await getJSON(url); } catch (_) { return; }
    if (!box.isConnected) return;
    // A live run refreshes on every frame; a repaint of the same items would drop a selection.
    const key = JSON.stringify(data.items || []);
    if (key === conv.itemsKey) return;
    conv.itemsKey = key;
    conv.items = data.items || [];
    paintItems();
  };
  conv.poke = () => { // debounced re-fold on an SSE change signal
    if (conv.timer) return;
    conv.timer = setTimeout(() => { conv.timer = null; conv.refresh(); }, 900);
  };
  // A refresh landing after the session's removal would ask the server for a record that is gone.
  conv.close = () => { conv.closed = true; if (conv.timer) { clearTimeout(conv.timer); conv.timer = null; } };
  // The in-progress turn under the folded items (streamed thinking/text from
  // the SessionState SSE frame): the analogue of the TUI's docked live pane. The
  // live "thinking…" marker always shows; the reasoning text itself streams
  // only at the expanded detail level (same rule as the TUI).
  conv.setLive = (s) => {
    conv.finished = notLive(s); // steers emptyNote()'s tense (may run before paintItems)
    const r = s.last_model_call;
    liveHost.innerHTML = '';
    const note = itemsHost.querySelector('.conv-empty');
    if (notLive(s)) {
      // Why the run has no worker, in the read model's words; a composer that resumes it says so.
      const dead = s.dead_state || '';
      liveHost.style.display = dead ? '' : 'none';
      if (dead) liveHost.appendChild(el('div', 'muted', dead));
      if (note) { note.textContent = emptyNote(); note.style.display = dead ? 'none' : ''; } // re-tense if already painted
      return;
    }
    if (s.status === 'waiting') {
      // Blocked on the operator (the prompt card above): no thinking, no tool
      // running, so neither pulse may say so, nor "appears as the run
      // streams" over a run that has not started its first turn.
      liveHost.style.display = '';
      if (note) note.style.display = 'none';
      liveHost.appendChild(el('div', 'muted', '· ' + runState(s)));
      return;
    }
    if (!r) {
      liveHost.style.display = 'none';
      if (note) { note.textContent = emptyNote(); note.style.display = ''; }
      return;
    }
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
    if (atTail) box.scrollTop = box.scrollHeight;
  };
  conv.detailButton = () => {
    // The button holds every label and shows the current one, so it holds still as it cycles.
    const b = el('button', 'mini');
    const cycle = b.appendChild(el('span', 'cycle'));
    const labels = {};
    for (const level in DETAIL_CYCLE) labels[level] = cycle.appendChild(el('span', null, 'detail: ' + level));
    const show = () => { for (const level in labels) labels[level].classList.toggle('on', level === conv.detail); };
    show();
    b.onclick = () => {
      conv.detail = DETAIL_CYCLE[conv.detail];
      remember('a6-detail', conv.detail);
      show();
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
function convCard(url, title, cls, empty) {
  const card = el('div', 'card conv-card ' + (cls || ''));
  const hrow = el('div', 'card-head-row');
  hrow.appendChild(el('h2', null, title));
  const box = el('div', 'conv-box');
  const body = el('div');
  box.appendChild(body);
  const conv = makeConv(url, box, body, empty);
  hrow.appendChild(conv.detailButton());
  card.appendChild(hrow); card.appendChild(box);
  return { card, conv, box };
}

// The text /undo takes back, consumed by the next session view's composer.
let pendingComposerFill = '';

// Fill a composer through an edit the browser records, so the native undo
// stack (Ctrl+Z / Cmd+Z) survives programmatic fills like history recall and
// slash completion. execCommand is deprecated but remains the only widely
// implemented way to write a textarea's value as a user edit; the fallback
// keeps the fill working, minus undo.
function fillAsEdit(ta, text) {
  ta.focus();
  ta.setSelectionRange(0, ta.value.length);
  let ok = false;
  try { ok = document.execCommand('insertText', false, text); } catch (_) { ok = false; }
  if (!ok) {
    ta.value = text;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

// A lightweight lexical highlighter for an approval's command: quoted strings,
// flags, and shell operators get classes; everything else is escaped text.
function highlightCmd(text) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const re = /("(?:[^"\\]|\\.)*"|'[^']*')|(\s--?[A-Za-z][\w-]*)|(\|\||&&|\||;|>>|2>|>|<)/g;
  let out = '', last = 0, m;
  while ((m = re.exec(text)) !== null) {
    out += esc(text.slice(last, m.index));
    if (m[1]) out += '<span class="tok-str">' + esc(m[1]) + '</span>';
    else if (m[2]) out += '<span class="tok-flag">' + esc(m[2]) + '</span>';
    else out += '<span class="tok-op">' + esc(m[3]) + '</span>';
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

// The steer directives a session composer can complete, with one-line help:
// a verbatim mirror of agent6.directive.STEER_COMMANDS (drift-pinned by
// tests/web/test_steer_completion.py). The resume composer withholds
// agent6.directive.LIVE_RUN_COMMANDS (pinned there too).
const STEER_COMMANDS = [
  ['/pin', 'pin an instruction that survives compaction: /pin <text>'],
  ['/compact', 'compact the context now; /compact <focus> steers the summary'],
  ['/parallel', 'dispatch children: /parallel [SPEC --] TASK (SPEC: count or models; repeat to queue more)'],
  ['/restate', 'restate the conversation since your last message (local, no model call)'],
  ['/review', 'review changes in this session: /review [TARGET] (a ref or A..B or A...B)'],
  ['/undo', 'fork back to before your last message (the text returns to edit and resend)'],
  ['/btw', 'ask a question beside the run: /btw <question> (answers inline, later)'],
  ['/task', 'queue work into the task graph: /task <text> (worked when the queue drains)'],
  ['/standing', 'set the goal the run returns to when the queue drains: /standing <text>'],
  ['/retire', 'drop a task from the graph: /retire <task id>, the number the task tree shows'],
  ['/now', 'steer at once, aborting the call in flight: /now <text> (Ctrl+Enter on the web)'],
  ['/stop', 'stop the run now, as `agent6 stop` does (resumable)'],
  ['/shells', 'background commands this run started, and how they ended'],
];
// Slash-command completion for a session composer: while the first word is
// being typed (`/…`, no whitespace yet), the matching directives with their
// help. Same popup contract as attachParallelSuggest; Tab (or Enter/click on
// a highlighted row) completes the word plus a trailing space.
function attachCommandSuggest(ta, root, liveNow, offered = null) {
  let box = null, items = [], active = -1;
  ta.setAttribute('aria-autocomplete', 'list');
  const word = () => {
    const v = ta.value;
    return v.startsWith('/') && !/\s/.test(v) ? v : null;
  };
  const close = () => {
    if (box) { unpointPopup(ta, box); box.remove(); box = null; }
    items = []; active = -1;
  };
  const insert = (cmd) => {
    fillAsEdit(ta, cmd + ' '); // an undoable edit; its input event hands the /parallel popup over
    close();
  };
  const render = () => {
    const w = word();
    if (w === null) { close(); return; }
    items = STEER_COMMANDS.filter(([c]) => (liveNow() || (c !== '/compact' && c !== '/btw' && c !== '/now' && c !== '/stop' && c !== '/task' && c !== '/standing' && c !== '/retire')) && (!offered || offered.includes(c)) && c.startsWith(w));
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) {
      box = el('div', 'ac-pop'); box.id = 'suggest-' + ++suggestLists; box.setAttribute('role', 'listbox');
      root.appendChild(box); ta.setAttribute('aria-controls', box.id);
    }
    box.textContent = '';
    items.forEach(([c, help], i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''));
      o.id = box.id + '-' + i; o.setAttribute('role', 'option'); o.setAttribute('aria-selected', String(i === active));
      o.appendChild(el('span', null, c));
      o.appendChild(el('span', 'muted', ' · ' + help));
      o.onmousedown = (e) => { e.preventDefault(); insert(c); };
      box.appendChild(o);
    });
    if (active >= 0) ta.setAttribute('aria-activedescendant', box.id + '-' + active);
    else ta.removeAttribute('aria-activedescendant');
  };
  ta.addEventListener('input', () => { active = -1; render(); });
  ta.addEventListener('click', render);
  ta.addEventListener('blur', () => setTimeout(close, 120));
  // Returns true when the popup consumed the key (caller must not also act on it).
  return { onKeyDown(e) {
    if (!box || !items.length || composing(e)) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Tab') { e.preventDefault(); insert(items[active >= 0 ? active : 0][0]); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active][0]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

// /restate: the conversation since the operator's last message, rendered from the journal.
function openRestate(text) {
  const dialog = openDialog('Since your last message', 'wide');
  if (!dialog) return;
  dialog.appendChild(el('pre', null, text));
}

// Ctrl+R in a session composer: search this session's past messages (the task,
// then every steer, journal-read via the conversation payload, so resumes and
// steers typed on other surfaces appear). Newest first, one line
// each, repeats collapsed: the same list the CLI and TUI searches show.
// Picking fills the composer for editing (Enter keeps the highlighted match,
// or the typed text itself when nothing matches); nothing is sent.
function openHistorySearch(entries, onPick) {
  openPicker({
    title: 'Search past messages',
    placeholder: 'type to narrow…',
    rows: entries,
    limit: 8,
    typed: true,
    haystack: (t) => t.toLowerCase(),
    paint: (t, o) => { o.textContent = t; },
    pick: onPick,
  });
}

// --- keyboard -----------------------------------------------------------------
// One picker under the session switcher, the command palette and history search: a dialog with a
// filter field (a combobox) over a list box. The arrows move the highlighted option, keep it in
// view and make it the field's active descendant, which a screen reader reads; the pointer moves
// it too, so one row is ever highlighted. Enter picks it, or with `typed` the typed text when
// nothing matches. `actions(row, close)` returns a row's own buttons: a list box may hold only
// options, and a screen reader finds no button inside an option, so a picker with them is a grid,
// each row its pick's cell and a cell of its buttons. `prefix` hands the field to another picker
// when typed first.
function openPicker({ title, placeholder, rows, haystack, paint, pick, empty = '', initial = '', prefix = null, limit = 12, typed = false, actions = null }) {
  const dialog = openDialog(title, 'wide picker');
  if (!dialog) return;
  const name = dialog.getAttribute('aria-labelledby');
  const aria = (e, attrs) => { for (const k in attrs) e.setAttribute(k, attrs[k]); };
  const popup = actions ? 'grid' : 'listbox';
  const field = el('input', 'field'); field.placeholder = placeholder; field.value = initial;
  const list = el('div', 'hs-list');
  list.id = name + '-list';
  list.tabIndex = -1; // the arrows scroll it: no Tab stop
  aria(list, { role: popup, 'aria-labelledby': name });
  aria(field, { role: 'combobox', 'aria-labelledby': name, 'aria-controls': list.id, 'aria-expanded': 'true', 'aria-autocomplete': 'list', 'aria-haspopup': popup });
  const note = el('div', 'more-note');
  dialog.append(field, list, note);
  const close = () => dialog.close();
  const choose = (r) => { close(); pick(r); };
  let items = [], active = 0;
  // Highlight the active option; the pointer is over it already, so only the arrows scroll.
  const mark = (scroll) => {
    [...list.children].forEach((row, i) => {
      row.classList.toggle('on', i === active);
      row.firstChild.setAttribute('aria-selected', String(i === active));
    });
    if (!items.length) { field.removeAttribute('aria-activedescendant'); return; }
    field.setAttribute('aria-activedescendant', list.id + '-' + active);
    if (scroll) list.children[active].scrollIntoView({ block: 'nearest' });
  };
  const render = () => {
    const q = field.value.toLowerCase().trim();
    const all = rows.filter(r => !q || haystack(r).includes(q));
    items = all.slice(0, limit);
    if (active >= items.length) active = Math.max(0, items.length - 1);
    list.textContent = '';
    items.forEach((r, i) => {
      const row = el('div', 'ac-item');
      const option = el('div', 'ac-opt');
      aria(option, { id: list.id + '-' + i, role: actions ? 'gridcell' : 'option' });
      paint(r, option);
      row.appendChild(option);
      if (actions) row.setAttribute('role', 'row');
      const own = actions && actions(r, close);
      if (own) {
        own.setAttribute('role', 'gridcell');
        for (const b of own.querySelectorAll('button')) b.setAttribute('aria-describedby', option.id);
        row.appendChild(own);
      }
      row.onmousedown = (e) => { if (e.target.closest('button')) return; e.preventDefault(); choose(r); };
      row.onpointermove = () => { if (active !== i) { active = i; mark(false); } };
      list.appendChild(row);
    });
    mark(true);
    const more = all.length - items.length;
    note.textContent = !items.length ? (rows.length ? '(no match)' : empty) : more ? '… ' + more + ' more (type to narrow)' : '';
  };
  const move = (step) => { if (items.length) { active = (active + step + items.length) % items.length; mark(true); } };
  // A row's own button keeps its keys: Enter or Space presses it.
  dialog.onkeydown = (e) => {
    if (composing(e) || e.target.closest('button')) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
    else if (e.key === 'Enter') { e.preventDefault(); if (items.length || typed) choose(items.length ? items[active] : field.value); }
  };
  field.oninput = () => {
    if (prefix && field.value.startsWith(prefix.key)) { const rest = field.value.slice(prefix.key.length); close(); prefix.open(rest); return; }
    active = 0; render();
  };
  field.focus();
  render();
}

// Ctrl+K anywhere: switch to another session. The hub's rows, newest activity first,
// children included, narrowed by what is typed against the id, mode, status, task,
// branch and worktree. A row names where the run works and the ref holding its
// commits; Enter opens the highlighted one, and Review code and Merge act as the
// Sessions page's row buttons do. A draft typed in a composer waits for the way back.
// A `>` typed first hands the box to the command palette.
async function openSwitcher() {
  if (document.querySelector('dialog[open]')) return; // one dialog at a time: skip the fetch
  let data;
  try { data = await getJSON('/api/hub'); } catch (e) { toast(e.message, true); return; }
  const rows = [];
  const flatten = (r) => { rows.push(r); for (const child of (r.children || [])) flatten(child); };
  for (const r of (data.sessions || [])) flatten(r);
  openPicker({
    title: 'Switch session',
    placeholder: 'narrow, or > for commands', // fits the field on a 320px phone
    rows,
    empty: '(no sessions yet)',
    prefix: { key: '>', open: (rest) => openPalette(rest) },
    haystack: (r) => [r.session_id, r.mode, r.status_label, r.task_line || r.task || '', r.branch || '', r.worktree || ''].join(' ').toLowerCase(),
    pick: (r) => { location.hash = '#/session/' + encodeURIComponent(r.session_id); },
    // The status ends the first line, as on a Sessions row, so the ids line up.
    paint: (r, o) => {
      const head = el('div', 'row');
      head.appendChild(el('span', 'grow', r.session_id + ' · ' + r.mode + ' · ' + (r.task_line || r.task || '').split('\n')[0]));
      head.appendChild(pill(r.status_level, r.status_label));
      o.appendChild(head);
      const where = [r.worktree ? 'worktree ' + r.worktree : 'this checkout', r.branch && 'ref ' + r.branch, r.when].filter(Boolean).join(' · ');
      o.appendChild(el('div', 'muted', where));
    },
    actions: (r, close) => {
      if (!r.branch || LIVE_WORDS.includes(r.status)) return null;
      const acts = el('div', 'row');
      const review = el('button', 'mini', 'Review code');
      review.onclick = () => { close(); handToNewWork({ task: '/review ' + r.branch }); location.hash = '#/'; route(); };
      acts.appendChild(review);
      if (r.unmerged) {
        const merge = el('button', 'mini', 'Merge');
        merge.onclick = async () => {
          close(); // the confirm and a failure's toast meet the page, not the backdrop
          if (!await confirmMerge(r)) return;
          await postMerge(r);
        };
        acts.appendChild(merge);
      }
      return acts;
    },
  });
}

// `>` typed into Ctrl+K's box: the pages, the dialogs, the theme, new work, and the controls
// the current page offers, each by the name it shows.
function paletteItems() {
  const go = (hash) => () => { location.hash = hash; };
  const items = [
    { label: 'Sessions', hint: 'page', run: go('#/') },
    { label: 'Git', hint: 'page', run: go('#/git') },
    { label: 'Files', hint: 'page', run: go('#/files') },
    { label: 'State machines', hint: 'page', run: go('#/machines') },
    { label: 'Config', hint: 'page', run: go('#/config') },
    { label: 'Switch session', hint: 'Ctrl+K', run: () => openSwitcher() },
    { label: 'Keyboard shortcuts', hint: '?', run: () => openKeys() },
    { label: 'Toggle theme', hint: 'light or dark', run: () => toggleTheme() },
    { label: 'New session', hint: 'the Sessions composer', run: () => { location.hash = '#/'; setTimeout(() => view.querySelector('.dock textarea')?.focus(), 50); } },
  ];
  // The page's own controls, as the page labels them; a hidden or disabled one is not offered.
  for (const b of view.querySelectorAll('.toolbar button, .actions button')) {
    if (b.disabled || b.offsetParent === null) continue;
    items.push({ label: b.textContent.replace(/^[■▶]\s*/, ''), hint: 'this page', run: () => b.click() });
  }
  return items;
}
function openPalette(initial = '') {
  openPicker({
    title: 'Commands',
    placeholder: 'type a command…',
    rows: paletteItems(),
    empty: '(no commands)',
    initial,
    haystack: (c) => (c.label + ' ' + c.hint).toLowerCase(),
    pick: (c) => c.run(),
    paint: (c, o) => { o.appendChild(el('span', null, c.label)); o.appendChild(el('span', 'muted', ' · ' + c.hint)); },
  });
}

// Every binding the web answers, in one table: the global handler runs the rows that
// carry `run`, and the keyboard list (`?`) shows every row, grouped by where it works. The
// composer's keys are handled where the composer is built, and named here so the list is whole.
// A label spells a chord with + ('Ctrl+K') and names either key with 'or'.
// An Enter that confirms an input method's candidate (Japanese, Chinese, Korean) is part of
// typing, not a send; keyCode 229 covers engines that report the composition only that way.
function composing(e) { return e.isComposing || e.keyCode === 229; }
function inField(e) { const t = e.target; return !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)); }
const KEYS = [
  { label: 'Ctrl+K', where: 'Anywhere', does: 'switch session (> for commands)', match: (e) => e.key.toLowerCase() === 'k' && (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey, run: () => openSwitcher() },
  { label: '?', where: 'Outside a field', does: 'this list', match: (e) => e.key === '?' && !inField(e) && !e.ctrlKey && !e.metaKey && !e.altKey, run: () => openKeys() },
  { label: 'Esc', where: 'In a dialog or popup', does: 'close it' },
  { label: 'Enter', where: 'In the composer', does: 'send the instruction; resume an ended session' },
  { label: 'Shift+Enter', where: 'In the composer', does: 'newline' },
  { label: '/', where: 'In the composer', does: 'directive completion; Tab or Enter accepts' },
  { label: 'Ctrl+Enter', where: "In a session's composer", does: 'send now, interrupting the call in flight (/now)' },
  { label: 'Ctrl+R', where: "In a session's composer", does: "search this session's past messages" },
  { label: '↑ ↓', where: 'In a list', does: 'move' },
  { label: 'Home End', where: 'In a list', does: 'first, last' },
  { label: 'Enter or Space', where: 'On a focused row', does: 'open it' },
];
document.addEventListener('keydown', (e) => {
  for (const k of KEYS) if (k.run && k.match(e)) { e.preventDefault(); k.run(); return; }
});
// A key label as keycaps: each key of a chord in its own <kbd>, joined by +.
function keycaps(label) {
  const caps = el('span', 'keycaps');
  label.split(' ').forEach((word, i) => {
    if (i) caps.append(' ');
    if (word === 'or') { caps.append('or'); return; }
    word.split('+').forEach((key, j) => { if (j) caps.append('+'); caps.appendChild(el('kbd', null, key)); });
  });
  return caps;
}
function openKeys() {
  const dialog = openDialog('Keyboard shortcuts', 'wide');
  if (!dialog) return;
  const table = el('table', 'keys');
  let group = null;
  for (const k of KEYS) {
    if (!group || group.dataset.where !== k.where) {
      group = el('tbody');
      group.dataset.where = k.where;
      const th = el('th', null, k.where);
      th.colSpan = 2; th.scope = 'rowgroup';
      group.appendChild(el('tr')).appendChild(th);
      table.appendChild(group);
    }
    const tr = el('tr');
    tr.appendChild(el('td', 'key')).appendChild(keycaps(k.label));
    tr.appendChild(el('td', null, k.does));
    group.appendChild(tr);
  }
  dialog.appendChild(table);
  // A second `?` closes the list, as Esc does; the page's own `?` binding must not reopen it.
  dialog.onkeydown = (e) => { if (e.key === '?') { e.preventDefault(); e.stopPropagation(); dialog.close(); } };
}

// The composer bar under a run's conversation. On a live run Enter sends the
// text as a steer (injected at the run's next safe boundary); on a finished
// run Enter resumes the run with the text as the follow-up instruction (empty
// = plain resume), then waits for the resumed worker to take over and
// re-renders. Shift+Enter inserts a newline. setState(s) keeps the mode in
// sync with each SSE frame. Ctrl+R (composer-focused only, so the browser
// keeps its reload elsewhere) opens the past-message search above.
function makeComposer(id) {
  const root = el('div', 'composer');
  const ta = el('textarea', 'field');
  // A typed draft survives a switch to another session and back (this browser only);
  // a sent or discarded line forgets it.
  const draftKey = 'a6-draft-' + id;
  ta.value = recall(draftKey, '');
  ta.oninput = () => { if (ta.value) remember(draftKey, ta.value); else forget(draftKey); };
  // The emptied entry reaches every input listener: the draft is forgotten and the popups close.
  const clearDraft = () => { ta.value = ''; ta.dispatchEvent(new Event('input')); };
  const hint = el('div', 'hint');
  // The preset and the model the next execution continues under (`agent6 resume
  // --preset`, `--model`): both change only between executions, so the row shows
  // only while the composer resumes, filled once from the lists the config
  // editor and the new-work composer offer. Each first option adds no flag and
  // names what the resume runs under (/resume_defaults), asked again whenever
  // the row appears (the last execution may have pinned one) and on a preset pick.
  const preset = el('select', 'field');
  const presetDefault = el('option', null, '…'); presetDefault.value = ''; preset.appendChild(presetDefault);
  const model = el('select', 'field');
  const modelDefault = el('option', null, '…'); modelDefault.value = ''; model.appendChild(modelDefault);
  const presetRow = pickerRow([['continue under preset', preset], ['model', model]]);
  // A phone's summary line names the preset and the model the resume runs under.
  const summary = pickerSummary([presetRow], () => ['preset: ' + pickedName(preset), pickedModel(model)]);
  presetRow.style.display = summary.style.display = 'none';
  let labelRequest = 0;
  const relabel = () => {
    const request = ++labelRequest;
    getJSON('/api/session/' + encodeURIComponent(id) + '/resume_defaults?preset=' + encodeURIComponent(preset.value))
      .then(d => {
        if (request !== labelRequest) return;
        presetDefault.textContent = d.preset_label;
        modelDefault.textContent = d.model_label;
        model.dispatchEvent(new Event('input', { bubbles: true })); // the summary line names them
      }).catch(() => {});
  };
  preset.onchange = relabel;
  let rowShown = false;
  let presetsFilled = false;
  const fillPresets = () => {
    if (presetsFilled) return;
    presetsFilled = true;
    getJSON('/api/config/suggest/preset').then(d => {
      for (const p of (d.values || [])) { const o = el('option', null, p); o.value = p; preset.appendChild(o); }
    }).catch(() => {});
    getJSON('/api/routes').then(d => {
      for (const r of (d.routes || [])) { const o = el('option', null, r); o.value = r; model.appendChild(o); }
    }).catch(() => {});
  };
  let finished = null; // unknown until the first SSE frame
  // A run the agent ended has nothing to continue, so resume takes an
  // instruction or is refused; every other ending resumes bare.
  let needsWork = false;
  let busy = false;
  let modelGit = false;
  const offeredCommands = STEER_COMMANDS.map(([command]) => command);
  const suggest = attachCommandSuggest(ta, root, () => finished === false, offeredCommands);
  const models = attachParallelSuggest(ta, root); // the spec before `--`
  const apply = () => {
    ta.disabled = busy;
    presetRow.style.display = summary.style.display = finished ? '' : 'none';
    if (finished) fillPresets();
    if (finished && !rowShown) relabel();
    rowShown = !!finished;
    if (busy) { hint.textContent = 'resuming…'; return; }
    if (finished && needsWork) {
      ta.placeholder = 'what should it do next…';
      hint.textContent = 'Enter resumes it with your instruction · Shift+Enter newline · Ctrl+R past messages · Ctrl+K switch session · ? keys';
    } else if (finished) {
      ta.placeholder = 'continue this session…';
      hint.textContent = 'Enter resumes this session with the instruction (empty = just resume) · Shift+Enter newline · Ctrl+R past messages · Ctrl+K switch session · ? keys';
    } else {
      ta.placeholder = 'steer this session… (type / for directives)';
      hint.textContent = 'Enter sends the instruction at the session’s next safe boundary · Ctrl+Enter sends it now · Shift+Enter newline · Ctrl+R past messages · Ctrl+K switch session · ? keys';
    }
  };
  const resume = async (text) => {
    busy = true; apply();
    try {
      await postJSON('/api/session/' + encodeURIComponent(id) + '/resume', { text, preset: preset.value, model: model.value });
      const under = [preset.value && 'preset ' + preset.value, model.value && 'model ' + model.value].filter(Boolean).join(', ');
      toast(under ? 'resuming the run under ' + under + '…' : 'resuming the run…');
      // The resume is a detached spawn: wait for it to come live, then re-open
      // the view so the SSE stream and controls come back. Waiting on
      // `finished === false` would declare takeover on the first poll, since
      // the parked and stale runs this composer offers resume for are already
      // unfinished, so a resume that died on spawn (its stderr goes to
      // DEVNULL) would report success with nothing to show.
      for (let i = 0; i < 25; i++) {
        await new Promise(r => setTimeout(r, 1000));
        if (!root.isConnected) return; // navigated away
        let s; try { s = await getJSON('/api/session/' + encodeURIComponent(id)); } catch (_) { continue; }
        if (s && s.live === true) { clearDraft(); route(); return; }
      }
      toast('the resume has not started yet; check `agent6 sessions`', true);
    } catch (e) { toast(e.message, true); }
    busy = false; apply();
  };
  if (pendingComposerFill) {
    const fill = pendingComposerFill;
    pendingComposerFill = '';
    setTimeout(() => fillAsEdit(ta, fill), 0); // after the view attaches
  }
  ta.onkeydown = (e) => {
    if (composing(e)) return;
    if (suggest.onKeyDown(e) || models.onKeyDown(e)) return;
    if (e.key === 'r' && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
      e.preventDefault(); // Ctrl+R here is history search, not a page reload
      if (busy) return;
      getJSON('/api/session/' + encodeURIComponent(id) + '/conversation').then(d => {
        const seen = new Set(), entries = [];
        for (const t of (d.operator_inputs || []).slice().reverse()) {
          const line = t.split(/\s+/).join(' ').trim();
          if (line && !seen.has(line)) { seen.add(line); entries.push(line); }
        }
        if (!entries.length) { toast('no past messages this session yet', true); return; }
        openHistorySearch(entries, (text) => fillAsEdit(ta, text));
      }).catch(err => toast(err.message, true));
      return;
    }
    if (e.key !== 'Enter' || e.shiftKey) return;
    e.preventDefault();
    if (finished === null || busy) return;
    // Ctrl+Enter is the `/now` directive: the steer aborts the call in flight.
    const typed = ta.value.trim();
    const text = e.ctrlKey && !finished && typed && !typed.startsWith('/') ? '/now ' + typed : typed;
    if (text === '/stop') {
      // Typed, so no confirm; the stop `agent6 stop` does.
      if (finished) { toast('nothing to stop: the session is not live', true); return; }
      postJSON('/api/session/' + encodeURIComponent(id) + '/stop', {})
        .then(d => { toast(d.message || 'stopped'); clearDraft(); })
        .catch(err => toast(err.message, true));
      return;
    }
    if (text === '/restate') {
      // Local and free: rendered from the journal, nothing reaches the model.
      getJSON('/api/session/' + encodeURIComponent(id) + '/restate')
        .then(d => { openRestate(d.text || ''); clearDraft(); })
        .catch(err => toast(err.message, true));
      return;
    }
    if (text === '/shells') {
      // The view is the Background shells card: bring it up (the phone menu
      // picks it; the desktop page opens the drawer through Details and scrolls
      // to it), and say so when there is nothing to show, as the CLI and TUI do.
      clearDraft();
      const card = document.querySelector('[data-w="shells"]');
      if (!card || card.style.display === 'none') { toast('no background commands this run'); return; }
      if (document.querySelector('.drawer.closed')) document.querySelector('.details-btn').click();
      const pick = document.querySelector('.wmenu button[data-w="shells"]');
      if (pick) pick.click();
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (text === '/undo') {
      if (modelGit) { toast('the model owns git in this run: no agent6 commit chain', true); return; }
      // Fork at the state before the last message; follow the fork with the
      // undone text back in the composer to edit and resend. A live run does
      // it at its next boundary (the steer channel); the SSE paint follows
      // the session.undone the loop emits. A finished run forks right here.
      if (!finished) {
        postJSON('/api/session/' + encodeURIComponent(id) + '/steer', { text })
          .then(() => { toast('undo requested; applies at the next step'); clearDraft(); })
          .catch(err => toast(err.message, true));
        return;
      }
      postJSON('/api/session/' + encodeURIComponent(id) + '/undo', {})
        .then(d => {
          toast('undone: forked to ' + d.new_session_id + (d.notice ? '\n' + d.notice : ''));
          pendingComposerFill = d.undone_text || '';
          location.hash = '#/session/' + encodeURIComponent(d.new_session_id);
        })
        .catch(err => toast(err.message, true));
      return;
    }
    if (!finished) {
      if (!text) return;
      // The server decides what the text was: `/compact [focus]` is an
      // out-of-band request, not a steer, and it says so.
      postJSON('/api/session/' + encodeURIComponent(id) + '/steer', { text })
        .then(r => { toast((r && r.message) || 'steer sent'); clearDraft(); })
        .catch(err => toast(err.message, true));
    } else {
      resume(text);
    }
  };
  root.appendChild(summary); root.appendChild(presetRow); root.appendChild(ta); root.appendChild(hint);
  // A quoted diff line lands beneath what is typed, for the words that follow it.
  root.quote = (note) => {
    const typed = ta.value.replace(/\s+$/, '');
    ta.value = (typed ? typed + '\n\n' : '') + note;
    ta.oninput();
    ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
  };
  // `live` is the dir-aware truth (a parked or stale run is not live even
  // though the fold says unfinished); fall back to the fold when a payload
  // carries no boolean `live`.
  root.setState = (s) => {
    if (busy) return;
    modelGit = s.git_control === 'model';
    offeredCommands.splice(0, offeredCommands.length, ...STEER_COMMANDS.map(([command]) => command).filter(command => !modelGit || command !== '/undo'));
    needsWork = s.needs_new_work === true; // the server's predicate, never re-derived here
    if (typeof s.live === 'boolean') { finished = !s.live; apply(); }
    else { finished = notLive(s); apply(); } // resume-style composer for any non-live run (parked/stale/ended)
  };
  apply();
  return root;
}

