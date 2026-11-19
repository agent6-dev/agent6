// --- git ---------------------------------------------------------------------
// The Git tab: the commit graph over the refs the filter keeps, the working tree's
// uncommitted files on top when there are any, a commit's files with each file's diff,
// and a compare of any two revisions. The server lists, verifies and caps (/api/git/*);
// this file lays out the lanes and paints. Filter and compare state outlive a visit to
// another tab; the graph and the working tree are fetched again on return, and on ↻.
const GIT_LANE_W = 14, GIT_ROW_H = 30, GIT_LANES_MAX = 16;
// The server's sentinel revision for the working tree, a diff or compare head.
const GIT_WORKTREE = 'WORKTREE';
// How a file's diff is shown; the choice is kept per browser.
const DIFF_MODE_KEY = 'a6-diff-mode';
const DIFF_MODES = ['inline', 'split', 'old', 'new'];
const GIT_WORKTREE_PICK = { rev: GIT_WORKTREE, label: 'working tree' };
// Lane n draws in class `l<n mod 8>`, one of the theme's eight colours (styles_git.css).
const GIT_LANE_COLORS = 8;
const GIT_KIND_TITLE = { branch: 'branch', remote: 'remote-tracking branch', tag: 'tag', agent6: 'agent6 run ref', head: 'HEAD' };
const gitState = {
  groupOn: { branch: true, remote: true, tag: true, agent6: false }, // agent6 chain refs are off until asked for
  refOff: new Set(),                 // refs unticked one by one, by full name
  compare: { base: null, head: null }, // each { rev, label } or null
};

// Lanes, as git log --graph draws them: each slot waits for one sha, and several slots may
// wait for the same one. A commit takes the lowest slot waiting for it (or a free one when
// none was) and hands it on to its first parent, so a branch keeps its lane down to its
// root; every other slot waiting for the commit curves into its dot. Each further parent
// gets the slot waiting for it, or a new one. The working tree lays out as a commit whose
// parent is HEAD; its line down to HEAD keeps a slot of its own, `wt` on each row it enters.
function gitLayout(commits) {
  const slots = [];
  const rows = [];
  let wt = -1;
  const free = () => { let i = slots.indexOf(null); if (i < 0) { slots.push(null); i = slots.length - 1; } return i; };
  for (const c of commits) {
    const before = slots.slice(), wtAbove = wt;
    let lane = slots.indexOf(c.sha);
    if (lane < 0) lane = free();
    const joins = [];
    for (let i = lane + 1; i < slots.length; i++) if (slots[i] === c.sha) { joins.push(i); slots[i] = null; }
    if (lane === wt || joins.includes(wt)) wt = -1;
    slots[lane] = c.parents[0] || null;
    if (c.sha === GIT_WORKTREE && c.parents.length) wt = lane;
    const merges = [];
    for (const p of c.parents.slice(1)) {
      let k = slots.findIndex((s, i) => s === p && i !== wt);
      if (k < 0) { k = free(); slots[k] = p; }
      merges.push(k);
    }
    while (slots.length && slots[slots.length - 1] === null) slots.pop();
    rows.push({ lane, before, after: slots.slice(), joins, merges, wt: wtAbove });
  }
  return rows;
}

// One row's lanes and its dot: `dot` is '', 'merge' (hollow) or 'wt' (the working tree,
// dashed), and HEAD's is larger. The working tree's line down to HEAD is dashed.
function gitLanesSvg(lay, cols, dot, isHead) {
  const NS = 'http://www.w3.org/2000/svg';
  const W = GIT_LANE_W, H = GIT_ROW_H, mid = H / 2;
  const x = i => W / 2 + i * W;
  const color = i => 'l' + i % GIT_LANE_COLORS;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', cols * W); svg.setAttribute('height', H);
  svg.setAttribute('class', 'gc-lanes'); svg.setAttribute('aria-hidden', 'true');
  const path = (d, lane, dashed) => { const p = document.createElementNS(NS, 'path'); p.setAttribute('d', d); p.setAttribute('class', color(lane) + (dashed ? ' wt' : '')); svg.appendChild(p); };
  // A merge curves down out of the dot into its parent's slot; a join curves down into the dot.
  const curve = (from, to) => `M${x(from)} ${mid} C${x(from)} ${H} ${x(to)} ${mid} ${x(to)} ${H}`;
  const join = (from, to) => `M${x(from)} 0 C${x(from)} ${mid} ${x(to)} 0 ${x(to)} ${mid}`;
  const n = Math.max(lay.before.length, lay.after.length);
  for (let i = 0; i < n && i < cols; i++) {
    if (i !== lay.lane && lay.before[i] && lay.after[i] === lay.before[i]) path(`M${x(i)} 0 V${H}`, i, i === lay.wt);
  }
  if (lay.lane < cols) {
    if (lay.before[lay.lane]) path(`M${x(lay.lane)} 0 V${mid}`, lay.lane, lay.lane === lay.wt);
    if (lay.after[lay.lane]) path(`M${x(lay.lane)} ${mid} V${H}`, lay.lane, dot === 'wt');
    for (const j of lay.joins) if (j < cols) path(join(j, lay.lane), j, j === lay.wt);
    for (const k of lay.merges) if (k < cols) path(curve(lay.lane, k), k);
    const circle = document.createElementNS(NS, 'circle');
    circle.setAttribute('cx', x(lay.lane)); circle.setAttribute('cy', mid); circle.setAttribute('r', isHead ? 5 : 4);
    circle.setAttribute('class', 'gc-dot ' + color(lay.lane) + (dot ? ' ' + dot : ''));
    svg.appendChild(circle);
  }
  return svg;
}

function refBadge(r) {
  const b = el('span', 'gref ' + esc(r.kind) + (r.head ? ' head' : ''), r.name);
  b.title = r.name + ' · ' + (GIT_KIND_TITLE[r.kind] || r.kind);
  return b;
}
// A commit's refs as its row and its panel list them: HEAD's first, then the branches, tags,
// remote-tracking branches and agent6 run refs, each kind in git's order.
function orderedRefs(refs) {
  const rank = (r) => r.head ? 0 : 1 + ['branch', 'tag', 'remote', 'agent6'].indexOf(r.kind);
  return refs.toSorted((a, b) => rank(a) - rank(b));
}
function statusBadge(status) {
  const s = el('span', 'gf-st st-' + esc(status), status);
  s.title = { A: 'added', M: 'modified', D: 'deleted', R: 'renamed', T: 'type changed', I: 'ignored', U: 'untracked' }[status] || status;
  return s;
}
function shortLabel(c) { return c.short + ' ' + c.subject; }
// Arrow keys move the focus between a list's rows (↑ ↓, Home, End, Page Up and Down), and Tab
// stops at one row. `stop(row)` makes `row` that one; the focused row becomes it too.
function rovingRows(list, selector) {
  const stop = (row) => { for (const r of list.querySelectorAll(selector)) r.tabIndex = r === row ? 0 : -1; };
  list.addEventListener('focusin', (e) => { if (e.target.matches(selector)) stop(e.target); });
  list.addEventListener('keydown', (e) => {
    if (!e.target.matches(selector) || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const rows = [...list.querySelectorAll(selector)].filter(r => r.checkVisibility());
    const i = rows.indexOf(e.target);
    const step = rows.length > 1 ? rows[1].getBoundingClientRect().top - rows[0].getBoundingClientRect().top : 0;
    const page = step > 0 ? Math.max(1, Math.floor(Math.min(list.clientHeight, innerHeight) / step) - 1) : 1;
    const to = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: rows.length - 1, PageDown: i + page, PageUp: i - page }[e.key];
    if (to === undefined) return;
    e.preventDefault();
    const row = rows[Math.max(0, Math.min(rows.length - 1, to))];
    row.focus({ preventScroll: true });
    // A file's button sits inside the row that draws its ring: the whole row scrolls into view.
    (row.closest('.gf') || row).scrollIntoView({ block: 'nearest' });
  });
  return stop;
}
// A commit's ISO date as git log --date=iso prints it ('2026-11-12 01:38:25 -05:00'): it may
// wrap before the zone, never inside the date or the time.
function gitDate(iso) {
  const m = /^(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)(.*)$/.exec(iso);
  if (!m) return el('span', null, iso);
  const date = el('span', 'gd-date');
  date.appendChild(el('span', null, m[1] + ' ' + m[2]));
  date.appendChild(document.createTextNode(' '));
  date.appendChild(el('span', null, m[3]));
  return date;
}
// A commit body's hard-wrapped lines rejoined into blocks, so the panel wraps each once: a line
// joins the block before it unless either is blank or code (indented four columns past the text
// above), or the line starts a list item or a trailer ('Signed-off-by: …'). Each block is
// { lead, marker, text, code }; a blank line is a block with no text.
function bodyBlocks(body) {
  const blocks = [];
  let open = null, col = 0; // the block the next line may join; the column of the last text
  for (const line of body.split('\n')) {
    const text = line.trim(), lead = line.length - line.trimStart().length;
    const marker = /^(?:[-*+]|\d+[.)])\s+/.exec(text)?.[0] || '';
    const code = !!text && lead >= col + 4;
    if (open && text && !code && !marker && !/^[\w-]+: /.test(text)) { open.text += ' ' + text; continue; }
    open = text && !code ? { lead, marker, text: text.slice(marker.length), code: false } : null;
    blocks.push(open || { lead: 0, marker: '', text: code ? line : '', code });
    if (open) col = lead + marker.length;
  }
  return blocks;
}
// The body as one element per block; a list item's later lines hang under its text.
function commitBody(body) {
  const box = el('div', 'gd-body');
  for (const b of bodyBlocks(body.trimEnd())) {
    const line = el('div', b.code ? 'gd-code' : 'gd-line');
    if (b.marker) { line.appendChild(el('span', 'gd-marker', b.marker)); line.style.setProperty('--hang', b.marker.length); }
    if (b.lead) line.style.setProperty('--lead', b.lead);
    if (b.text) line.appendChild(document.createTextNode(b.text));
    box.appendChild(line);
  }
  return box;
}
function fmtBytes(n) { return n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(1) + ' KiB' : (n / 1048576).toFixed(1) + ' MiB'; }

// An image's diff shows the file before and after, above git's "Binary files differ"
// line. The server types a blob by these extensions and sandboxes every blob response.
const IMAGE_EXTS = new Set(['avif', 'bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'svg', 'webp']);
function isImagePath(p) { const dot = p.lastIndexOf('.'); return dot >= 0 && IMAGE_EXTS.has(p.slice(dot + 1).toLowerCase()); }

// Fetch a blob with the token and show it in an <img> through a data: URL. A blob: URL
// would open under the page's origin, where the token lives, so an SVG's script opened
// from one could read it; a data: document has no origin.
async function blobImageData(path) {
  const blob = await getBlob(path);
  const url = await new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.onerror = () => rej(fr.error); fr.readAsDataURL(blob); });
  return { url, size: blob.size };
}
async function blobImage(rev, path, fig, cap, ignored = false) {
  const image = await blobImageData('/api/git/blob?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(path) + (ignored ? '&ignored=1' : ''));
  if (!fig.isConnected) return;
  const img = el('img'); img.alt = path;
  const size = cap.dataset.label + ' · ' + fmtBytes(image.size);
  img.onload = () => { cap.textContent = size + ' · ' + img.naturalWidth + '\u00d7' + img.naturalHeight; };
  img.onerror = () => { cap.textContent = size + ' · the browser cannot show this image'; };
  img.src = image.url;
  fig.appendChild(img);
}

// The pair: "before" at the base unless the file was added, "after" at the head unless it
// was deleted; a rename's "before" is the old path.
function imagePair(ctx, f) {
  const sides = [];
  if (f.status !== 'A') sides.push(['before', ctx.base, f.old_path || f.path]);
  if (f.status !== 'D') sides.push(['after', ctx.head, f.path]);
  const wrap = el('div', 'git-img' + (sides.length === 1 ? ' one' : ''));
  for (const [label, rev, path] of sides) {
    const fig = el('figure', 'git-img-side');
    const cap = el('figcaption', 'muted'); cap.dataset.label = label; cap.textContent = label + ' · loading…';
    fig.appendChild(cap);
    wrap.appendChild(fig);
    blobImage(rev, path, fig, cap).catch(e => { cap.textContent = label + ' · ' + e.message; cap.classList.add('err'); });
  }
  return wrap;
}

// A changed-file list shared by the commit, working-tree and compare views, as a flat
// list or a folder tree (the choice kept per browser): a row per file, reachable by
// `data-path`; a click opens the file's diff over the whole tab.
const FILES_VIEW_KEY = 'a6-files-view';
function fileList(files, truncated, open, actions) {
  const root = el('div', 'gfiles');
  const head = el('div', 'row gf-head'); root.appendChild(head);
  head.appendChild(el('div', 'muted gf-count', fileCount(files.length) + ' changed'));
  const view = el('span', 'gview'); head.appendChild(view);
  if (truncated) root.appendChild(el('div', 'err gf-note', 'the list is cut: more files changed than are shown'));
  const body = el('div'); root.appendChild(body);
  let mode = recall(FILES_VIEW_KEY, 'list');
  const stop = rovingRows(body, '.gf-open, .gtree-dir');
  // A working-tree file reads U while untracked, and says when its changes are staged. The path
  // is the row's one control, and a click anywhere on the row opens the diff; a working-tree
  // row's buttons sit beside it. The control covers the letter, so its tooltip names the letter.
  const fileRow = (f, short) => {
    const row = el('div', 'gf');
    row.dataset.path = f.path;
    const status = f.untracked ? 'U' : f.status;
    const index = f.staged ? (f.unstaged ? 'partly staged' : 'staged') : '';
    const badge = statusBadge(status);
    row.appendChild(badge);
    const path = filePath(f, short);
    if (index) { path.appendChild(document.createTextNode(' ')); path.appendChild(el('span', 'gf-index', index)); }
    const button = el('button', 'gf-open');
    button.setAttribute('aria-label', status + ' ' + f.path + (index ? ', ' + index : ''));
    button.title = badge.title + (index ? ', ' + index : '');
    button.onclick = () => open(files.indexOf(f));
    button.appendChild(path);
    row.appendChild(button);
    row.appendChild(fileCounts(f));
    if (actions) { const acts = actions(f); if (acts) row.appendChild(acts); }
    return row;
  };
  const folderNote = (d) => {
    const under = treeFiles(d);
    let add = 0, del = 0;
    for (const f of under) if (!f.binary) { add += f.additions; del += f.deletions; }
    return fileCount(under.length) + ' · +' + add + ' −' + del;
  };
  const paint = () => {
    body.innerHTML = '';
    for (const b of view.children) b.classList.toggle('active', b.textContent === mode);
    if (!files.length) { body.appendChild(el('div', 'muted gf-note', 'no files differ')); return; }
    if (mode === 'tree') body.appendChild(treeView(buildTree(files), (f) => fileRow(f, true), folderNote, false));
    else for (const f of files) body.appendChild(fileRow(f, false));
    stop(body.querySelector('.gf-open, .gtree-dir'));
  };
  for (const m of ['list', 'tree']) {
    const b = el('button', 'mini', m); b.title = m === 'tree' ? 'the files under their folders' : 'every file on one row';
    b.onclick = () => { mode = m; remember(FILES_VIEW_KEY, m); paint(); };
    view.appendChild(b);
  }
  paint();
  return root;
}
// The path, or in a tree the file's own name; a rename keeps its old path, shortened too
// when it sat in the same folder. A long path wraps after a slash, not inside a name.
function filePath(f, short) {
  const p = el('span', 'gf-path');
  const dir = (s) => s.slice(0, s.lastIndexOf('/') + 1), base = (s) => s.slice(s.lastIndexOf('/') + 1);
  const wrapping = (host, text) => {
    text.split('/').forEach((part, i) => {
      if (i) { host.appendChild(document.createTextNode('/')); host.appendChild(el('wbr')); }
      host.appendChild(document.createTextNode(part));
    });
    return host;
  };
  if (f.old_path) {
    const old = wrapping(el('span', 'muted'), short && dir(f.old_path) === dir(f.path) ? base(f.old_path) : f.old_path);
    old.appendChild(document.createTextNode(' → '));
    p.appendChild(old);
  }
  wrapping(p, short ? base(f.path) : f.path);
  return p;
}
function fileCounts(f) {
  const counts = el('span', 'gf-counts');
  if (f.binary) counts.appendChild(el('span', 'muted', 'binary'));
  else {
    counts.appendChild(el('span', 'add', '+' + f.additions));
    counts.appendChild(el('span', 'del', ' −' + f.deletions));
  }
  return counts;
}

// A compare picker: typed text searched as you type (/api/git/search lists matching
// branches, tags and commits), one picked by click or Enter; Enter on free text keeps it
// as the revision for the server to verify. The results render into `listHost`, a
// full-width strip under the picker row the two pickers share. `fixed` choices (the
// working tree) come first whenever the text matches their label.
function gitPicker(name, initial, onPick, listHost, fixed) {
  const root = el('div', 'gpick');
  const input = el('input', 'field');
  input.placeholder = name + ': branch, tag, sha…';
  input.setAttribute('aria-label', name);
  input.autocomplete = 'off';
  let value = initial || null;
  if (value) input.value = value.label;
  // The text is a search only once typed: a picked label opens the full list again.
  const query = () => value ? '' : input.value.trim();
  // A combobox: the list is a listbox, and the input names the option the arrow keys reached.
  const list = el('div', 'gpick-list'); list.id = 'gpick-' + name;
  list.setAttribute('role', 'listbox'); list.setAttribute('aria-label', name + ' choices');
  input.setAttribute('role', 'combobox'); input.setAttribute('aria-controls', list.id);
  input.setAttribute('aria-expanded', 'false');
  let items = [], active = -1, request = 0, timer = null;
  const show = () => { listHost.innerHTML = ''; listHost.appendChild(list); input.setAttribute('aria-expanded', 'true'); };
  // The reached option reads selected and scrolls into the list's view.
  const reach = (i) => {
    active = i;
    items.forEach((it, k) => { it.node.classList.toggle('on', k === i); it.node.setAttribute('aria-selected', String(k === i)); });
    if (i < 0) { input.removeAttribute('aria-activedescendant'); return; }
    input.setAttribute('aria-activedescendant', items[i].node.id);
    items[i].node.scrollIntoView({ block: 'nearest' });
  };
  // A note row (no match, more, an error) is an option that cannot be picked.
  const note = (cls, text) => { const n = el('div', 'ac-item ' + cls, text); n.setAttribute('role', 'option'); n.setAttribute('aria-disabled', 'true'); return n; };
  // Closing drops the search still pending or in flight, so neither can reopen the list.
  const close = () => {
    clearTimeout(timer); request++; list.remove(); list.innerHTML = ''; items = []; active = -1;
    input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant');
  };
  const choose = (v) => { value = v; input.value = v.label; close(); onPick(v); };
  const paint = (d) => {
    list.innerHTML = ''; items = [];
    const q = query().toLowerCase();
    for (const f of fixed || []) if (f.label.includes(q)) items.push({ rev: f.rev, label: f.label, node: (() => { const o = el('div', 'ac-item'); o.appendChild(el('span', 'gref wt', f.label)); return o; })() });
    for (const r of d.refs) items.push({ rev: r.full, label: r.name, node: (() => { const o = el('div', 'ac-item'); o.appendChild(refBadge(r)); o.appendChild(el('span', 'muted mono', ' ' + r.sha.slice(0, 7))); return o; })() });
    for (const c of d.commits) items.push({ rev: c.sha, label: shortLabel(c), node: (() => { const o = el('div', 'ac-item'); o.appendChild(el('span', 'mono', c.short)); o.appendChild(el('span', null, ' ' + c.subject)); o.appendChild(el('span', 'muted', ' · ' + c.when)); return o; })() });
    items.forEach((it, i) => {
      it.node.id = list.id + '-' + i; it.node.setAttribute('role', 'option');
      it.node.onmousedown = (e) => { e.preventDefault(); choose(it); }; list.appendChild(it.node);
    });
    if (!items.length) list.appendChild(note('more-note', '(no match: Enter keeps the text as the revision)'));
    else if (d.truncated) list.appendChild(note('more-note', '… more; type to narrow'));
    show(); reach(-1);
  };
  const search = async () => {
    const n = ++request;
    let d;
    try { d = await getJSON('/api/git/search?q=' + encodeURIComponent(query())); }
    catch (e) { if (n === request) { list.innerHTML = ''; items = []; list.appendChild(note('err', e.message)); show(); reach(-1); } return; }
    if (n !== request || !root.isConnected) return;
    paint(d);
  };
  input.oninput = () => { value = null; clearTimeout(timer); timer = setTimeout(search, 150); };
  input.onfocus = () => { if (!list.isConnected) search(); };
  input.onblur = () => setTimeout(close, 150);
  input.onkeydown = (e) => {
    if (composing(e)) return;
    if (e.key === 'ArrowDown' && !list.isConnected) { e.preventDefault(); search(); }
    else if (e.key === 'ArrowDown' && items.length) { e.preventDefault(); reach((active + 1) % items.length); }
    else if (e.key === 'ArrowUp' && items.length) { e.preventDefault(); reach((active - 1 + items.length) % items.length); }
    else if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      if (active >= 0) choose(items[active]);
      else { const t = input.value.trim(); if (t) choose({ rev: t, label: t }); }
    }
  };
  root.appendChild(input);
  return { root, get: () => value, set: (v) => { value = v; input.value = v ? v.label : ''; } };
}

async function renderGit(gen, link = [], query = new URLSearchParams()) {
  setCrumb('git');
  const [refsData, wtData] = await Promise.all([getJSON('/api/git/refs'), getJSON('/api/git/worktree')]);
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const app = el('div', 'git-app'); view.appendChild(app);
  let refs = refsData.refs, head = refsData.head, wt = wtData, remotes = refsData.remotes;
  const selectedRefs = () => refs.filter(r => gitState.groupOn[r.kind] && !gitState.refOff.has(r.full));

  // -- toolbar: the ref filter, the compare toggle, the refresh, the counts -----------
  const bar = el('div', 'git-bar page-pad row wrap'); app.appendChild(bar);
  const refsBtn = el('button', 'git-refs', 'Refs ▾');
  const compareBtn = el('button', null, 'Compare');
  const reloadBtn = el('button', null, '↻'); reloadBtn.title = 'refresh: re-read the refs, the graph and the working tree';
  reloadBtn.setAttribute('aria-label', 'refresh');
  const counts = el('span', 'muted git-counts');
  bar.appendChild(refsBtn); bar.appendChild(compareBtn); bar.appendChild(reloadBtn); bar.appendChild(counts);
  // A field and the buttons that act on it wrap as one group.
  const together = (...controls) => { const g = el('span', 'git-group'); for (const c of controls) g.appendChild(c); return g; };

  // -- your branch: switch to one, start one at HEAD, review one ----------------------
  // Every write goes to /api/git/<verb>, refused while a session is live in this checkout.
  const branchRow = el('span', 'row git-branch');
  const branchSel = el('select', 'field'); branchSel.title = 'the branch to switch to or review; a remote-tracking ref is for review';
  const switchBtn = el('button', null, 'Switch');
  switchBtn.title = 'check this branch out (git switch); refused while a session is live here or local changes would be lost';
  const reviewBtn = el('button', null, 'Review');
  reviewBtn.title = 'start a review session over this branch: /review <branch> in the new-work composer';
  const newName = el('input', 'field'); newName.placeholder = 'new branch…';
  const createBtn = el('button', null, 'Create'); createBtn.title = 'create the branch at HEAD and check it out';
  // A button waits for what it acts on: Create for a name, Switch for another local branch.
  // Local branches switch and review; a remote-tracking ref (a fetched pull request among
  // them) reviews only. `keep` names the ref to leave selected after a re-read.
  const gateCreate = () => { createBtn.disabled = !newName.value.trim(); };
  const isLocal = () => branchSel.selectedOptions[0]?.dataset.kind === 'branch';
  const gateSwitch = () => { switchBtn.disabled = !branchSel.value || !isLocal() || branchSel.value === head.branch; };
  const paintBranches = (keep) => {
    branchSel.textContent = '';
    for (const [kind, label] of [['branch', 'branches'], ['remote', 'remote-tracking']]) {
      const rows = refs.filter(r => r.kind === kind && !r.name.endsWith('/HEAD'));
      if (!rows.length) continue;
      const group = el('optgroup'); group.label = label;
      for (const r of rows) {
        const o = el('option', null, r.name); o.value = r.name; o.dataset.kind = kind;
        if (keep ? r.name === keep : kind === 'branch' && r.name === head.branch) o.selected = true;
        group.appendChild(o);
      }
      branchSel.appendChild(group);
    }
    gateSwitch();
    reviewBtn.disabled = !branchSel.value;
  };
  branchSel.onchange = gateSwitch;
  reviewBtn.onclick = () => { handToNewWork({ task: '/review ' + branchSel.value }); location.hash = '#/'; };
  branchRow.appendChild(together(branchSel, switchBtn, reviewBtn));
  branchRow.appendChild(together(newName, createBtn));
  bar.appendChild(branchRow);

  // -- your remotes: fetch one's branches, or one pull request, into its remote-tracking refs.
  // Nothing is merged and no branch of yours moves; the URL shown is where the fetch goes.
  const fetchRow = el('span', 'row git-branch git-fetch');
  const remoteSel = el('select', 'field'); remoteSel.title = 'the remote to fetch from';
  const remoteUrl = el('span', 'muted git-counts');
  const fetchBtn = el('button', null, 'Fetch');
  fetchBtn.title = "fetch this remote's branches into <remote>/<branch>; nothing is merged, no branch of yours moves, tags and submodules are left alone";
  const pullNum = el('input', 'field'); pullNum.placeholder = 'PR #'; pullNum.inputMode = 'numeric';
  pullNum.title = 'a pull request number on this remote';
  const pullBtn = el('button', null, 'Fetch PR');
  pullBtn.title = "fetch that pull request (the remote's refs/pull/<number>/head) into <remote>/pr/<number>, then Review or Compare it";
  // The number typed, '#' allowed before it; 0, and Fetch PR waits, until it is a whole number above zero.
  const pullNumber = () => { const t = pullNum.value.trim().replace(/^#/, ''); return /^\d+$/.test(t) ? Number(t) : 0; };
  const gatePull = () => { pullBtn.disabled = !pullNumber(); };
  const paintRemotes = () => {
    const kept = remoteSel.value;
    remoteSel.textContent = '';
    for (const r of remotes) { const o = el('option', null, r.name); o.value = r.name; if (r.name === kept) o.selected = true; remoteSel.appendChild(o); }
    fetchRow.hidden = !remotes.length;
    remoteSel.onchange();
  };
  remoteSel.onchange = () => { const url = (remotes.find(r => r.name === remoteSel.value) || { url: '' }).url; remoteUrl.textContent = url && 'from ' + url; };
  for (const x of [together(remoteSel, fetchBtn), together(pullNum, pullBtn), remoteUrl]) fetchRow.appendChild(x);
  bar.appendChild(fetchRow);

  const body = el('div', 'git-body'); app.appendChild(body);
  const graph = el('div', 'git-graph'); body.appendChild(graph);
  body.appendChild(resizeHandle(graph, {
    label: 'Graph width', key: 'a6-git-graph-w', property: '--git-graph-w',
    min: 320, max: () => body.getBoundingClientRect().width - 260,
  }));
  const side = el('div', 'git-side'); body.appendChild(side);

  const {closeDiff, showDiff} = gitDiffViewer(body);

  // -- the graph -------------------------------------------------------------------
  let commits = [], tips = null, hasMore = false, selected = null, loading = 0;
  const settle = rovingRows(graph, '.gc-row');
  // The selected row reads selected, and Tab enters the graph there.
  const markRow = (sha) => {
    let hit = null;
    for (const r of graph.querySelectorAll('.gc-row')) { const on = r.dataset.sha === sha; r.classList.toggle('sel', on); if (on) hit = r; }
    if (hit) settle(hit);
    return hit;
  };
  const graphUrl = (skip) => {
    const revs = tips === null ? selectedRefs().map(r => r.full).concat(head.sha ? ['HEAD'] : []) : tips;
    const params = (revs.length ? revs : ['']).map(r => 'refs=' + encodeURIComponent(r));
    params.push('skip=' + skip);
    return '/api/git/graph?' + params.join('&');
  };
  const paintGraph = () => {
    graph.innerHTML = '';
    // The working tree tops the graph as a commit whose parent is HEAD.
    const dirty = wt.files.length > 0;
    const lays = gitLayout(dirty ? [{ sha: GIT_WORKTREE, parents: head.sha ? [head.sha] : [] }, ...commits] : commits);
    let cols = 1;
    for (const l of lays) cols = Math.max(cols, l.before.length, l.after.length, l.lane + 1);
    cols = Math.min(cols, GIT_LANES_MAX);
    if (dirty) {
      const row = el('div', 'gc-row gc-wt');
      row.dataset.sha = GIT_WORKTREE;
      row.appendChild(gitLanesSvg(lays.shift(), cols, 'wt', false));
      const n = wt.files.length + (wt.truncated ? '+' : '');
      const subj = el('div', 'gc-subj');
      subj.appendChild(el('span', 'gc-text', 'Working tree · ' + n + (wt.files.length === 1 && !wt.truncated ? ' uncommitted file' : ' uncommitted files')));
      row.appendChild(subj);
      actionable(row, showWorktree, 'working tree, ' + n + ' uncommitted files');
      graph.appendChild(row);
    }
    commits.forEach((c, i) => {
      const row = el('div', 'gc-row');
      row.dataset.sha = c.sha;
      const isHead = c.sha === head.sha;
      row.appendChild(gitLanesSvg(lays[i], cols, c.is_merge ? 'merge' : '', isHead));
      const subj = el('div', 'gc-subj');
      // The row shows its first ref and counts the rest; its name and the commit panel list them all.
      const named = orderedRefs((isHead && !c.refs.some(r => r.head) ? [{ name: 'HEAD', kind: 'head', head: true }] : []).concat(c.refs));
      if (named.length) subj.appendChild(refBadge(named[0]));
      if (named.length > 1) {
        const more = el('span', 'gref more', '+' + (named.length - 1)); more.title = named.slice(1).map(r => r.name).join(', ');
        subj.appendChild(more);
      }
      subj.appendChild(el('span', 'gc-text', c.subject));
      row.appendChild(subj);
      row.appendChild(el('span', 'gc-author muted', c.author));
      row.appendChild(el('span', 'gc-when muted mono', c.when));
      actionable(row, () => showCommit(c.sha), named.map(r => r.name).concat(c.subject, c.author, c.when).join(', '));
      graph.appendChild(row);
    });
    if (!commits.length) graph.appendChild(el('div', 'empty', selectedRefs().length || head.sha ? 'no commits yet' : 'no refs selected'));
    if (hasMore) {
      // The next page waits on a disabled button; the focus it held goes to the first row it adds.
      const more = el('button', 'git-more', 'Load more');
      more.onclick = async () => {
        const had = document.activeElement === more, from = commits.length;
        more.disabled = true; more.textContent = 'Loading…';
        await loadGraph(false);
        if (had) (commits[from] ? graph.querySelector('[data-sha="' + commits[from].sha + '"]') : graph.querySelector('.git-more'))?.focus();
      };
      graph.appendChild(more);
    }
    if (!markRow(selected)) settle(graph.querySelector('.gc-row'));
    const n = selectedRefs().length;
    counts.textContent = commits.length + (hasMore ? '+' : '') + ' commits · ' + n + ' of ' + refs.length + ' refs';
  };
  // While a page loads, the graph reads busy and, past a moment, dims.
  const loadGraph = async (reset) => {
    const n = ++loading;
    if (reset) { commits = []; tips = null; hasMore = false; }
    graph.setAttribute('aria-busy', 'true');
    try {
      const d = await getJSON(graphUrl(commits.length));
      if (n !== loading || !app.isConnected) return;
      commits = commits.concat(d.commits); tips = d.tips; hasMore = d.has_more;
    } catch (e) { if (n === loading) toast(e.message, true); }
    if (n === loading) graph.setAttribute('aria-busy', 'false');
    paintGraph();
  };
  // ↻: the refs, the working tree and the graph again; an open commit panel stays, an
  // open working-tree panel is painted afresh. ↻ waits until they are in, keeping its focus.
  const reload = async (keep) => {
    const had = document.activeElement === reloadBtn;
    reloadBtn.disabled = true;
    try {
      const [r, w] = await Promise.all([getJSON('/api/git/refs'), getJSON('/api/git/worktree')]);
      if (!app.isConnected) return;
      refs = r.refs; head = r.head; wt = w; remotes = r.remotes;
      paintBranches(typeof keep === 'string' ? keep : '');
      paintRemotes();
      if (selected === GIT_WORKTREE) showWorktree();
      await loadGraph(true);
    } catch (e) { toast(e.message, true); }
    finally {
      reloadBtn.disabled = false;
      if (had && document.activeElement === document.body) reloadBtn.focus();
    }
  };
  reloadBtn.onclick = reload;
  // One write, then the page re-reads what git now holds.
  const GIT_WRITES = {
    stage: '/api/git/stage', unstage: '/api/git/unstage', discard: '/api/git/discard',
    commit: '/api/git/commit', branch: '/api/git/branch', switch: '/api/git/switch',
  };
  const gitWrite = async (verb, body) => {
    try { const d = await postJSON(GIT_WRITES[verb], body); toast(d.message || verb); await reload(); }
    catch (e) { toast(e.message, true); }
  };
  // A fetch can take a network's time: its two buttons wait, the pressed one saying so. A
  // fetched pull request is left selected, so Review is the next click.
  const gitFetch = async (pull) => {
    const remote = remoteSel.value, button = pull ? pullBtn : fetchBtn, label = button.textContent;
    const had = document.activeElement === button;
    fetchBtn.disabled = pullBtn.disabled = true; button.textContent = 'Fetching…';
    try {
      const d = await postJSON('/api/git/fetch', pull ? { remote, pull } : { remote });
      toast(d.message || 'fetched');
      await reload(pull ? remote + '/pr/' + pull : '');
    } catch (e) { toast(e.message, true); }
    button.textContent = label; fetchBtn.disabled = false; gatePull();
    if (had && document.activeElement === document.body) (button.disabled ? pullNum : button).focus();
  };
  fetchBtn.onclick = () => gitFetch(0);
  pullBtn.onclick = () => { const n = pullNumber(); pullNum.value = ''; gitFetch(n); };
  pullNum.oninput = gatePull;
  pullNum.onkeydown = (e) => { if (e.key === 'Enter' && !composing(e)) { e.preventDefault(); pullBtn.click(); } };
  switchBtn.onclick = () => gitWrite('switch', { name: branchSel.value });
  // The focus moves to the name field before Create disables itself to wait for the next name.
  createBtn.onclick = () => { const name = newName.value.trim(); newName.value = ''; newName.focus(); gateCreate(); gitWrite('branch', { name }); };
  newName.oninput = gateCreate;
  newName.onkeydown = (e) => { if (e.key === 'Enter' && !composing(e)) { e.preventDefault(); createBtn.click(); } };
  paintBranches();
  paintRemotes();
  gateCreate(); gatePull();

  // -- the side panel: a commit, or the compare --------------------------------------
  // A phone's panel covers the graph; ‹ graph closes it, and the focus it held goes to the graph.
  const sideHead = (title) => {
    const h = el('div', 'git-side-head row');
    const back = el('button', 'mini git-back', '‹ graph');
    back.onclick = () => {
      const had = document.activeElement === back;
      side.classList.remove('open');
      if (had) graph.querySelector('.gc-row[tabindex="0"]')?.focus();
    };
    h.appendChild(back); h.appendChild(el('h3', null, title));
    return h;
  };
  const openSide = () => side.classList.add('open');
  const setCompare = (which, v) => { query = new URLSearchParams(); gitState.compare[which] = v; showCompare(); };
  // The working tree's panel: no message; HEAD is the base every file diffs against.
  const showWorktree = () => {
    closeDiff();
    selected = GIT_WORKTREE;
    setViewHash('#/git/worktree');
    markRow(GIT_WORKTREE);
    side.innerHTML = '';
    side.appendChild(sideHead('Working tree'));
    const pad = el('div', 'gd page-pad'); side.appendChild(pad);
    pad.appendChild(el('div', 'gd-subject', 'Uncommitted changes'));
    const kv = el('div', 'kv gd-meta');
    kv.appendChild(el('span', 'k', 'base'));
    const base = el('span', 'v');
    if (head.sha) {
      const b = el('button', 'mini mono', wt.base.slice(0, 7)); b.title = wt.base;
      b.onclick = () => showCommit(wt.base);
      base.appendChild(b); base.appendChild(el('span', 'muted', 'HEAD' + (head.branch ? ' (' + head.branch + ')' : '')));
    } else base.appendChild(el('span', 'muted', 'the empty tree: no commits yet'));
    kv.appendChild(base);
    pad.appendChild(kv);
    const acts = el('div', 'row wrap gd-actions');
    const asHead = el('button', 'mini', 'Set as head'); asHead.onclick = () => setCompare('head', GIT_WORKTREE_PICK);
    acts.appendChild(asHead);
    pad.appendChild(acts);
    // Your own hands on the index: stage and unstage by file, discard one file's unstaged
    // edits after a confirm, commit what is staged. Nothing here touches an untracked file.
    const staged = wt.files.filter(f => f.staged);
    const commitBox = el('div', 'row git-commit');
    const msg = el('textarea', 'field');
    msg.placeholder = staged.length ? 'commit message for ' + fileCount(staged.length) + ' staged…' : 'stage files to commit';
    msg.disabled = !staged.length;
    // Commit staged waits for staged files and a message.
    const commitBtn = el('button', 'primary mini', 'Commit staged');
    const gateCommit = () => { commitBtn.disabled = !staged.length || !msg.value.trim(); };
    msg.oninput = gateCommit; gateCommit();
    commitBtn.onclick = () => gitWrite('commit', { message: msg.value });
    commitBox.appendChild(msg); commitBox.appendChild(commitBtn);
    pad.appendChild(commitBox);
    const fileActions = (f) => {
      const row = el('span', 'gf-acts');
      const add = (label, title, run) => {
        const b = el('button', 'mini', label); b.title = title; b.dataset.act = label.toLowerCase();
        b.onclick = run;
        row.appendChild(b);
      };
      if (f.unstaged || f.untracked) add('Stage', 'add this file to the index (git add)', () => gitWrite('stage', { paths: [f.path] }));
      if (f.staged) add('Unstage', 'take this file out of the index; its content stays (git restore --staged)', () => gitWrite('unstage', { paths: [f.path] }));
      if (f.unstaged && !f.untracked) add('Discard', 'restore this file from the index, dropping its unstaged edits (git restore)', () => {
        if (confirm('Discard the unstaged edits to ' + f.path + '? The file is restored from the index; the edits are not kept anywhere.')) gitWrite('discard', { path: f.path });
      });
      return row.children.length ? row : null;
    };
    pad.appendChild(fileList(wt.files, wt.truncated, (i) => showDiff({ base: wt.base, head: GIT_WORKTREE, files: wt.files }, i), fileActions));
    openSide();
  };
  const showCommit = async (sha, selection = new URLSearchParams()) => {
    closeDiff();
    selected = sha;
    const gen = routeGen; // a response landing after a route change paints and publishes nothing
    const suffix = selection.size ? '?' + selection : '';
    setViewHash('#/git/commit/' + encodeURIComponent(sha) + suffix);
    markRow(sha);
    side.innerHTML = ''; side.appendChild(sideHead('Commit')); side.appendChild(el('div', 'muted page-pad', 'loading…'));
    openSide();
    let c;
    try { c = await getJSON('/api/git/commit/' + encodeURIComponent(sha)); }
    catch (e) { if (selected === sha && app.isConnected && gen === routeGen) { side.innerHTML = ''; side.appendChild(sideHead('Commit')); side.appendChild(el('div', 'err page-pad', e.message)); } return; }
    if (selected !== sha || !app.isConnected || gen !== routeGen) return;
    selected = c.sha;
    setViewHash('#/git/commit/' + encodeURIComponent(c.sha) + suffix, true);
    markRow(c.sha)?.scrollIntoView({ block: 'nearest' });
    side.innerHTML = '';
    side.appendChild(sideHead('Commit'));
    const pad = el('div', 'gd page-pad'); side.appendChild(pad);
    const title = el('div', 'gd-subject');
    for (const r of orderedRefs(c.refs)) title.appendChild(refBadge(r));
    title.appendChild(el('span', null, c.subject));
    pad.appendChild(title);
    const kv = el('div', 'kv gd-meta');
    const add = (k, v) => { kv.appendChild(el('span', 'k', k)); const node = typeof v === 'string' ? el('span', 'v', v) : v; node.classList.add('v'); kv.appendChild(node); };
    const full = el('span', 'mono gd-sha', c.sha); full.title = c.sha;
    add('commit', full);
    const who = (p) => {
      const v = el('span'); v.appendChild(document.createTextNode(p.name + ' <' + p.email + '> · '));
      v.appendChild(gitDate(p.date)); return v;
    };
    add('author', who(c.author));
    if (c.committer.name !== c.author.name || c.committer.email !== c.author.email || c.committer.date !== c.author.date) {
      add('committer', who(c.committer));
    }
    const parents = el('span');
    if (!c.parents.length) parents.appendChild(el('span', 'muted', 'none (root commit)'));
    for (const p of c.parents) {
      const b = el('button', 'mini mono', p.slice(0, 7)); b.title = p;
      b.onclick = () => showCommit(p);
      parents.appendChild(b);
    }
    add(c.is_merge ? 'parents' : 'parent', parents);
    pad.appendChild(kv);
    const acts = el('div', 'row wrap gd-actions');
    const me = { rev: c.sha, label: shortLabel(c) };
    const asBase = el('button', 'mini', 'Set as base'); asBase.onclick = () => setCompare('base', me);
    const asHead = el('button', 'mini', 'Set as head'); asHead.onclick = () => setCompare('head', me);
    acts.appendChild(asBase); acts.appendChild(asHead);
    pad.appendChild(acts);
    if (c.body) pad.appendChild(commitBody(c.body));
    const ctx = {base: c.base, head: c.sha, files: c.files, commit: true, query: selection};
    pad.appendChild(fileList(c.files, c.truncated, (i) => showDiff(ctx, i)));
    const path = selection.get('path');
    const selectedFile = c.files.findIndex(f => f.path === path);
    if (selectedFile >= 0) await showDiff(ctx, selectedFile);
    else if (path) pad.appendChild(el('div', 'err', 'This commit has no diff for ' + path));
  };

  let compareRun = 0;
  const showCompare = () => {
    closeDiff();
    selected = null;
    markRow(null);
    side.innerHTML = '';
    side.appendChild(sideHead('Compare'));
    const pad = el('div', 'gcmp page-pad'); side.appendChild(pad);
    const result = el('div', 'gcmp-result');
    const run = async () => {
      const { base, head: h } = gitState.compare;
      result.innerHTML = '';
      if (!base || !h) { setViewHash('#/git'); result.appendChild(el('div', 'muted', 'pick a base and a head: a branch, a tag, a commit, or text from a commit message')); return; }
      const n = ++compareRun;
      setViewHash('#/git/compare/' + encodeURIComponent(base.rev) + '..' + encodeURIComponent(h.rev) + (query.size ? '?' + query : ''));
      result.appendChild(el('div', 'muted', 'comparing…'));
      try {
        const panel = query.has('panel') ? await getJSON('/api/reviews/' + encodeURIComponent(query.get('panel'))) : null;
        const target = panel?.panel.target;
        const reviewed = target && [target.base_tree, target.base_commit].includes(base.rev) && [target.head_tree, target.head_commit].includes(h.rev);
        const d = reviewed ? {base: {sha: target.base_tree, short: target.base_tree.slice(0, 7), subject: 'reviewed base'},
          head: {sha: target.head_tree, short: target.head_tree.slice(0, 7), subject: 'reviewed head'}, files: panel.files, truncated: panel.truncated}
          : await getJSON('/api/git/compare?base=' + encodeURIComponent(base.rev) + '&head=' + encodeURIComponent(h.rev));
        if (n !== compareRun || !result.isConnected) return;
        result.innerHTML = '';
        const sum = el('div', 'gcmp-sum muted');
        sum.appendChild(el('span', 'mono', d.base.short)); sum.appendChild(el('span', null, ' ' + d.base.subject + '  →  '));
        sum.appendChild(el('span', 'mono', d.head.short)); sum.appendChild(el('span', null, ' ' + d.head.subject));
        result.appendChild(sum);
        if (reviewed) result.appendChild(reviewRules(panel.panel));
        const path = query.get('path');
        let selectedFile = d.files.findIndex(f => path === f.path || path === f.old_path);
        if (reviewed && path && selectedFile < 0) {
          selectedFile = d.files.length;
          d.files.push({path, status: 'M', old_path: '', additions: 0, deletions: 0});
        }
        const ctx = {base: d.base.sha, head: d.head.sha, files: d.files, panel: reviewed ? query.get('panel') : '', query};
        result.appendChild(fileList(d.files, d.truncated, (i) => showDiff(ctx, i)));
        if (path && selectedFile >= 0) await showDiff(ctx, selectedFile);
      } catch (e) { if (n === compareRun) { result.innerHTML = ''; result.appendChild(el('div', 'err', e.message)); } }
    };
    const listHost = el('div', 'gpick-host');
    const pickBase = gitPicker('base', gitState.compare.base, (v) => { query = new URLSearchParams(); gitState.compare.base = v; run(); }, listHost);
    const pickHead = gitPicker('head', gitState.compare.head, (v) => { query = new URLSearchParams(); gitState.compare.head = v; run(); }, listHost, [GIT_WORKTREE_PICK]);
    const swap = el('button', 'mini', '⇄'); swap.title = 'swap base and head'; swap.setAttribute('aria-label', 'swap base and head');
    swap.onclick = () => { query = new URLSearchParams(); const { base, head: h } = gitState.compare; gitState.compare = { base: h, head: base }; pickBase.set(h); pickHead.set(base); run(); };
    const row = el('div', 'gcmp-pick');
    row.appendChild(pickBase.root); row.appendChild(swap); row.appendChild(pickHead.root);
    pad.appendChild(row);
    pad.appendChild(listHost);
    pad.appendChild(el('div', 'hint', 'the files that differ between the two revisions, as git diff base head lists them'));
    pad.appendChild(result);
    openSide();
    return run();
  };
  compareBtn.onclick = showCompare;

  // -- the ref filter: group toggles and one checkbox per ref, a popover under Refs ---
  // The browser closes it on Escape (focus back on Refs) or a click outside.
  const filter = el('div', 'gfilter'); filter.popover = 'auto';
  refsBtn.popoverTargetElement = filter;
  refsBtn.setAttribute('aria-expanded', 'false');
  bar.appendChild(filter);
  filter.addEventListener('beforetoggle', (e) => { if (e.newState === 'open') paintFilter(); });
  filter.addEventListener('toggle', (e) => {
    const open = e.newState === 'open';
    refsBtn.classList.toggle('active', open); refsBtn.setAttribute('aria-expanded', String(open));
  });
  const paintFilter = () => {
    filter.innerHTML = '';
    const groups = el('div', 'row wrap gfilter-groups');
    const kinds = [['branch', 'Branches'], ['remote', 'Remotes'], ['tag', 'Tags'], ['agent6', 'agent6 runs']];
    for (const [kind, label] of kinds) {
      const n = refs.filter(r => r.kind === kind).length;
      const l = el('label', 'gfilter-group');
      const cb = el('input'); cb.type = 'checkbox'; cb.checked = gitState.groupOn[kind];
      cb.onchange = () => { gitState.groupOn[kind] = cb.checked; for (const r of refs) if (r.kind === kind) gitState.refOff.delete(r.full); paintList(); loadGraph(true); };
      l.appendChild(cb); l.appendChild(el('span', null, label + ' (' + n + ')'));
      groups.appendChild(l);
    }
    filter.appendChild(groups);
    const search = el('input', 'filter'); search.type = 'search'; search.placeholder = 'filter refs…';
    search.autofocus = true;
    filter.appendChild(search);
    const list = el('div', 'gfilter-list'); filter.appendChild(list);
    const paintList = () => {
      list.innerHTML = '';
      const q = search.value.toLowerCase();
      for (const r of refs) {
        if (q && !r.name.toLowerCase().includes(q)) continue;
        const l = el('label', 'gfilter-item' + (gitState.groupOn[r.kind] ? '' : ' off'));
        const cb = el('input'); cb.type = 'checkbox';
        cb.checked = gitState.groupOn[r.kind] && !gitState.refOff.has(r.full);
        cb.disabled = !gitState.groupOn[r.kind];
        cb.onchange = () => { if (cb.checked) gitState.refOff.delete(r.full); else gitState.refOff.add(r.full); loadGraph(true); };
        l.appendChild(cb); l.appendChild(refBadge(r)); l.appendChild(el('span', 'muted mono', r.sha.slice(0, 7)));
        list.appendChild(l);
      }
      if (!list.children.length) list.appendChild(el('div', 'more-note', refs.length ? '(no ref matches)' : '(no refs: an empty repository)'));
    };
    search.oninput = paintList;
    paintList();
  };

  await loadGraph(true);
  if (!app.isConnected) return;
  if (link[0] === 'commit' && link[1]) await showCommit(link[1], query);
  else if (link[0] === 'compare' && link[1]) {
    const [base, head] = link[1].split('..');
    gitState.compare = { base: { rev: base || '', label: base || '' }, head: { rev: head || '', label: head || '' } };
    await showCompare();
  }
  else if (link[0] === 'worktree' && wt.files.length) showWorktree();
  else side.appendChild(el('div', 'empty', 'click a commit for its files and diff, or Compare for any two revisions'));
}
