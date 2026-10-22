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
const GIT_LANE_COLORS = ['#6ea8fe', '#4ec9a5', '#e2c08d', '#f07178', '#b48ead', '#56b6c2', '#d19a66', '#98c379'];
const GIT_KIND_TITLE = { branch: 'branch', remote: 'remote-tracking branch', tag: 'tag', agent6: 'agent6 run ref', head: 'HEAD' };
const gitState = {
  groupOn: { branch: true, remote: true, tag: true, agent6: false }, // agent6 chain refs are off until asked for
  refOff: new Set(),                 // refs unticked one by one, by full name
  compare: { base: null, head: null }, // each { rev, label } or null
  side: 'commit',                    // what the side panel shows: 'commit', 'worktree' or 'compare'
};

// Lanes: each slot waits for one sha. A commit takes the slot waiting for it (or a free
// one when nothing was), hands the slot on to its first parent, and curves to the slot
// of every other parent (allocating one when that parent is new). A first parent some
// other slot already waits for ends this lane with a curve into that slot.
function gitLayout(commits) {
  const slots = [];
  const rows = [];
  const free = () => { let i = slots.indexOf(null); if (i < 0) { slots.push(null); i = slots.length - 1; } return i; };
  for (const c of commits) {
    const before = slots.slice();
    let lane = slots.indexOf(c.sha);
    if (lane < 0) lane = free();
    const p0 = c.parents[0] || null;
    let joinTo = null;
    const k0 = p0 ? slots.indexOf(p0) : -1;
    if (k0 >= 0 && k0 !== lane) { joinTo = k0; slots[lane] = null; }
    else slots[lane] = p0;
    const merges = [];
    for (const p of c.parents.slice(1)) {
      let k = slots.indexOf(p);
      if (k < 0) { k = free(); slots[k] = p; }
      merges.push(k);
    }
    while (slots.length && slots[slots.length - 1] === null) slots.pop();
    rows.push({ lane, before, after: slots.slice(), joinTo, merges });
  }
  return rows;
}

function gitLanesSvg(lay, cols, isMerge, isHead) {
  const NS = 'http://www.w3.org/2000/svg';
  const W = GIT_LANE_W, H = GIT_ROW_H, mid = H / 2;
  const x = i => W / 2 + i * W;
  const color = i => GIT_LANE_COLORS[i % GIT_LANE_COLORS.length];
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', cols * W); svg.setAttribute('height', H);
  svg.setAttribute('class', 'gc-lanes');
  const path = (d, stroke) => { const p = document.createElementNS(NS, 'path'); p.setAttribute('d', d); p.setAttribute('stroke', stroke); svg.appendChild(p); };
  const curve = (from, to) => `M${x(from)} ${mid} C${x(from)} ${H} ${x(to)} ${mid} ${x(to)} ${H}`;
  const n = Math.max(lay.before.length, lay.after.length);
  for (let i = 0; i < n && i < cols; i++) {
    if (i !== lay.lane && lay.before[i] && lay.after[i] === lay.before[i]) path(`M${x(i)} 0 V${H}`, color(i));
  }
  if (lay.lane < cols) {
    if (lay.before[lay.lane]) path(`M${x(lay.lane)} 0 V${mid}`, color(lay.lane));
    if (lay.after[lay.lane]) path(`M${x(lay.lane)} ${mid} V${H}`, color(lay.lane));
    if (lay.joinTo !== null && lay.joinTo < cols) path(curve(lay.lane, lay.joinTo), color(lay.joinTo));
    for (const k of lay.merges) if (k < cols) path(curve(lay.lane, k), color(k));
    const dot = document.createElementNS(NS, 'circle');
    dot.setAttribute('cx', x(lay.lane)); dot.setAttribute('cy', mid); dot.setAttribute('r', isHead ? 5 : 4);
    dot.setAttribute('stroke', color(lay.lane));
    dot.setAttribute('class', 'gc-dot' + (isMerge ? ' merge' : ''));
    svg.appendChild(dot);
  }
  return svg;
}

function gitWorktreeSvg(cols) {
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', cols * GIT_LANE_W); svg.setAttribute('height', GIT_ROW_H);
  svg.setAttribute('class', 'gc-lanes');
  const dot = document.createElementNS(NS, 'circle');
  dot.setAttribute('cx', GIT_LANE_W / 2); dot.setAttribute('cy', GIT_ROW_H / 2); dot.setAttribute('r', 4);
  dot.setAttribute('stroke', GIT_LANE_COLORS[0]);
  dot.setAttribute('class', 'gc-dot wt');
  svg.appendChild(dot);
  return svg;
}

function refBadge(r) {
  const b = el('span', 'gref ' + esc(r.kind) + (r.head ? ' head' : ''), r.name);
  b.title = GIT_KIND_TITLE[r.kind] || r.kind;
  return b;
}
function statusBadge(status) {
  const s = el('span', 'gf-st st-' + esc(status), status);
  s.title = { A: 'added', M: 'modified', D: 'deleted', R: 'renamed', T: 'type changed' }[status] || status;
  return s;
}
function shortLabel(c) { return c.short + ' ' + c.subject; }
function fmtBytes(n) { return n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(1) + ' KiB' : (n / 1048576).toFixed(1) + ' MiB'; }

// An image's diff shows the file before and after, above git's "Binary files differ"
// line. The server types a blob by these extensions and sandboxes every blob response.
const IMAGE_EXTS = new Set(['avif', 'bmp', 'gif', 'ico', 'jpeg', 'jpg', 'png', 'svg', 'webp']);
function isImagePath(p) { const dot = p.lastIndexOf('.'); return dot >= 0 && IMAGE_EXTS.has(p.slice(dot + 1).toLowerCase()); }

// Fetch a blob with the token and show it in an <img> through a data: URL. A blob: URL
// would open under the page's origin, where the token lives, so an SVG's script opened
// from one could read it; a data: document has no origin.
async function blobImage(rev, path, fig, cap) {
  const blob = await getBlob('/api/git/blob?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(path));
  const url = await new Promise((res, rej) => { const fr = new FileReader(); fr.onload = () => res(fr.result); fr.onerror = () => rej(fr.error); fr.readAsDataURL(blob); });
  if (!fig.isConnected) return;
  const img = el('img'); img.alt = path;
  const size = cap.dataset.label + ' · ' + fmtBytes(blob.size);
  img.onload = () => { cap.textContent = size + ' · ' + img.naturalWidth + '\u00d7' + img.naturalHeight; };
  img.onerror = () => { cap.textContent = size + ' · the browser cannot show this image'; };
  img.src = url;
  fig.appendChild(img);
}

// The pair: "before" at the base unless the file was added, "after" at the head unless it
// was deleted; a rename's "before" is the old path.
function imagePair(ctx, f) {
  const wrap = el('div', 'git-img');
  const sides = [];
  if (f.status !== 'A') sides.push(['before', ctx.base, f.old_path || f.path]);
  if (f.status !== 'D') sides.push(['after', ctx.head, f.path]);
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
function fileList(files, truncated, open) {
  const root = el('div', 'gfiles');
  const head = el('div', 'row gf-head'); root.appendChild(head);
  head.appendChild(el('div', 'muted gf-count', fileCount(files.length) + ' changed'));
  const view = el('span', 'gview'); head.appendChild(view);
  if (truncated) root.appendChild(el('div', 'err gf-note', 'the list is cut: more files changed than are shown'));
  const body = el('div'); root.appendChild(body);
  let mode = recall(FILES_VIEW_KEY, 'list');
  const fileRow = (f, short) => {
    const row = el('div', 'gf');
    row.dataset.path = f.path;
    row.appendChild(statusBadge(f.status));
    row.appendChild(filePath(f, short));
    row.appendChild(fileCounts(f));
    actionable(row, () => open(files.indexOf(f)), f.status + ' ' + f.path);
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
// when it sat in the same folder.
function filePath(f, short) {
  const p = el('span', 'gf-path');
  const dir = (s) => s.slice(0, s.lastIndexOf('/') + 1), base = (s) => s.slice(s.lastIndexOf('/') + 1);
  if (f.old_path) p.appendChild(el('span', 'muted', (short && dir(f.old_path) === dir(f.path) ? base(f.old_path) : f.old_path) + ' → '));
  p.appendChild(document.createTextNode(short ? base(f.path) : f.path));
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
  input.placeholder = name + ': branch, tag, sha, message…';
  input.setAttribute('aria-label', name);
  input.autocomplete = 'off';
  let value = initial || null;
  if (value) input.value = value.label;
  // The text is a search only once typed: a picked label opens the full list again.
  const query = () => value ? '' : input.value.trim();
  const list = el('div', 'gpick-list');
  let items = [], active = -1, request = 0, timer = null;
  // Closing drops the search still pending or in flight, so neither can reopen the list.
  const close = () => { clearTimeout(timer); request++; list.remove(); list.innerHTML = ''; items = []; active = -1; };
  const choose = (v) => { value = v; input.value = v.label; close(); onPick(v); };
  const paint = (d) => {
    list.innerHTML = ''; items = [];
    const q = query().toLowerCase();
    for (const f of fixed || []) if (f.label.includes(q)) items.push({ rev: f.rev, label: f.label, node: (() => { const o = el('div', 'ac-item'); o.appendChild(el('span', 'gref wt', f.label)); return o; })() });
    for (const r of d.refs) items.push({ rev: r.full, label: r.name, node: (() => { const o = el('div', 'ac-item'); o.appendChild(refBadge(r)); o.appendChild(el('span', 'muted mono', ' ' + r.sha.slice(0, 7))); return o; })() });
    for (const c of d.commits) items.push({ rev: c.sha, label: shortLabel(c), node: (() => { const o = el('div', 'ac-item'); o.appendChild(el('span', 'mono', c.short)); o.appendChild(el('span', null, ' ' + c.subject)); o.appendChild(el('span', 'muted', ' · ' + c.when)); return o; })() });
    items.forEach((it, i) => { it.node.classList.toggle('on', i === active); it.node.onmousedown = (e) => { e.preventDefault(); choose(it); }; list.appendChild(it.node); });
    if (!items.length) list.appendChild(el('div', 'more-note ac-item', '(no match: Enter keeps the text as the revision)'));
    else if (d.truncated) list.appendChild(el('div', 'more-note ac-item', '… more; type to narrow'));
    listHost.innerHTML = ''; listHost.appendChild(list);
  };
  const search = async () => {
    const n = ++request;
    let d;
    try { d = await getJSON('/api/git/search?q=' + encodeURIComponent(query())); }
    catch (e) { if (n === request) { list.innerHTML = ''; list.appendChild(el('div', 'ac-item err', e.message)); listHost.innerHTML = ''; listHost.appendChild(list); } return; }
    if (n !== request || !root.isConnected) return;
    active = -1; paint(d);
  };
  input.oninput = () => { value = null; clearTimeout(timer); timer = setTimeout(search, 150); };
  input.onfocus = () => { if (!list.isConnected) search(); };
  input.onblur = () => setTimeout(close, 150);
  input.onkeydown = (e) => {
    if (e.key === 'ArrowDown' && items.length) { e.preventDefault(); active = (active + 1) % items.length; items.forEach((it, i) => it.node.classList.toggle('on', i === active)); }
    else if (e.key === 'ArrowUp' && items.length) { e.preventDefault(); active = (active - 1 + items.length) % items.length; items.forEach((it, i) => it.node.classList.toggle('on', i === active)); }
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

async function renderGit(gen) {
  setCrumb('git');
  const [refsData, wtData] = await Promise.all([getJSON('/api/git/refs'), getJSON('/api/git/worktree')]);
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const app = el('div', 'git-app'); view.appendChild(app);
  let refs = refsData.refs, head = refsData.head, wt = wtData;
  const selectedRefs = () => refs.filter(r => gitState.groupOn[r.kind] && !gitState.refOff.has(r.full));

  // -- toolbar: the ref filter, the compare toggle, the refresh, the counts -----------
  const bar = el('div', 'git-bar page-pad row wrap'); app.appendChild(bar);
  const refsBtn = el('button', null, 'Refs ▾');
  const compareBtn = el('button', null, 'Compare');
  const reloadBtn = el('button', null, '↻'); reloadBtn.title = 'refresh: re-read the refs, the graph and the working tree';
  reloadBtn.setAttribute('aria-label', 'refresh');
  const counts = el('span', 'muted git-counts');
  bar.appendChild(refsBtn); bar.appendChild(compareBtn); bar.appendChild(reloadBtn); bar.appendChild(counts);

  const body = el('div', 'git-body'); app.appendChild(body);
  const graph = el('div', 'git-graph'); body.appendChild(graph);
  const side = el('div', 'git-side'); body.appendChild(side);

  // -- a file's diff: it takes the whole body; the list is one step back --------------
  const diffView = el('div', 'git-diff'); body.appendChild(diffView);
  const diffs = new Map();
  const closeDiff = () => { body.classList.remove('diffing'); diffView.innerHTML = ''; delete diffView.dataset.key; };
  const showDiff = async (ctx, i, whole) => {
    const f = ctx.files[i];
    const key = ctx.base + '|' + ctx.head + '|' + f.path + (whole ? '|whole' : '');
    diffView.dataset.key = key;
    diffView.innerHTML = '';
    body.classList.add('diffing');
    const head = el('div', 'git-diff-head row');
    const back = el('button', 'mini', '‹ files'); back.onclick = closeDiff;
    head.appendChild(back);
    head.appendChild(statusBadge(f.status));
    head.appendChild(filePath(f));
    head.appendChild(fileCounts(f));
    head.appendChild(el('span', 'muted git-diff-pos', (i + 1) + ' / ' + ctx.files.length));
    const prev = el('button', 'mini', '‹'); prev.title = 'previous file'; prev.disabled = i === 0;
    prev.onclick = () => showDiff(ctx, i - 1);
    const next = el('button', 'mini', '›'); next.title = 'next file'; next.disabled = i === ctx.files.length - 1;
    next.onclick = () => showDiff(ctx, i + 1);
    head.appendChild(prev); head.appendChild(next);
    let mode = recall(DIFF_MODE_KEY, 'inline');
    if (!DIFF_MODES.includes(mode)) mode = 'inline';
    const modes = el('span', 'gview');
    for (const m of DIFF_MODES) {
      const b = el('button', 'mini' + (m === mode ? ' active' : ''), m);
      b.title = { inline: 'the unified patch', split: 'old and new side by side', old: 'the file at the base, deletions marked', new: 'the file at the head, additions marked' }[m];
      b.onclick = () => { remember(DIFF_MODE_KEY, m); showDiff(ctx, i, whole); };
      modes.appendChild(b);
    }
    head.appendChild(modes);
    diffView.appendChild(head);
    const box = el('div', 'git-diff-body'); diffView.appendChild(box);
    let d = diffs.get(key);
    if (!d) {
      box.appendChild(el('div', 'muted', 'loading…'));
      try {
        const q = '?base=' + encodeURIComponent(ctx.base) + '&head=' + encodeURIComponent(ctx.head)
          + '&path=' + encodeURIComponent(f.path) + '&old_path=' + encodeURIComponent(f.old_path || '')
          + (whole ? '&whole=1' : '');
        d = await getJSON('/api/git/diff' + q);
      } catch (e) {
        if (diffView.dataset.key === key) { box.innerHTML = ''; box.appendChild(el('div', 'err', e.message)); }
        return;
      }
      if (ctx.head !== GIT_WORKTREE) diffs.set(key, d); // the working tree changes under the tab
      if (diffView.dataset.key !== key || !diffView.isConnected) return; // superseded
      box.innerHTML = '';
    }
    // The states come first, above the patch, so a cut patch reads as cut.
    if (isImagePath(f.path)) box.appendChild(imagePair(ctx, f));
    if (d.binary) box.appendChild(el('div', 'muted gf-note', 'binary file: no text diff'));
    if (d.truncated) {
      const note = el('div', 'err gf-note', 'the patch is cut at ' + Math.round(d.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole patch'); all.onclick = () => showDiff(ctx, i, true);
      note.appendChild(all); box.appendChild(note);
    }
    const patch = el('div', 'gf-patch'); box.appendChild(patch);
    const parsed = parseUnifiedDiff(d.patch || '');
    const file = parsed[0] || { preamble: [], hunks: [], binary: d.binary };
    if (mode === 'split' && window.matchMedia('(max-width: 780px)').matches) {
      patch.appendChild(el('div', 'muted gf-note', 'split needs a wider screen: showing inline'));
      mode = 'inline';
    }
    if (mode === 'inline') patch.appendChild(renderUnified(parsed, true));
    else if (mode === 'split') patch.appendChild(renderSplit(file));
    else await showSide(patch, ctx, f, file, mode, key, whole, () => showDiff(ctx, i, true));
  };
  // The file at one side of a diff, its changed lines marked: the deletions on the old
  // side, the additions on the new. The text comes highlighted from /api/git/text.
  const showSide = async (host, ctx, f, file, side, key, whole, openWhole) => {
    if ((side === 'old' && f.status === 'A') || (side === 'new' && f.status === 'D')) {
      host.appendChild(el('div', 'muted gf-note', 'no ' + side + ' side: the file ' + (side === 'old' ? 'was added' : 'was deleted')));
      return;
    }
    const rev = side === 'old' ? ctx.base : ctx.head, path = side === 'old' ? (f.old_path || f.path) : f.path;
    host.appendChild(el('div', 'muted', 'loading…'));
    let t;
    try { t = await getJSON('/api/git/text?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(path) + (whole ? '&whole=1' : '')); }
    catch (e) { if (diffView.dataset.key === key) { host.innerHTML = ''; host.appendChild(el('div', 'err', e.message)); } return; }
    if (diffView.dataset.key !== key || !host.isConnected) return;
    host.innerHTML = '';
    if (t.binary) { host.appendChild(el('div', 'muted gf-note', 'binary file: no text')); return; }
    if (t.truncated) {
      const note = el('div', 'err gf-note', 'the file is cut at ' + Math.round(t.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole file'); all.onclick = openWhole;
      note.appendChild(all); host.appendChild(note);
    }
    const marks = new Set();
    for (const h of file.hunks) for (const l of h.lines) {
      if (side === 'old' && l.kind === 'del') marks.add(l.old);
      if (side === 'new' && l.kind === 'add') marks.add(l.new);
    }
    host.appendChild(renderLines(splitLines(t.text), t.spans, marks));
  };

  // -- the graph -------------------------------------------------------------------
  let commits = [], hasMore = false, selected = null, loading = 0;
  const graphUrl = (skip) => {
    const params = selectedRefs().map(r => 'refs=' + encodeURIComponent(r.full));
    if (head.sha) params.push('refs=HEAD');
    params.push('skip=' + skip);
    return '/api/git/graph?' + params.join('&');
  };
  const paintGraph = () => {
    graph.innerHTML = '';
    const lays = gitLayout(commits);
    let cols = 1;
    for (const l of lays) cols = Math.max(cols, l.before.length, l.after.length, l.lane + 1);
    cols = Math.min(cols, GIT_LANES_MAX);
    if (wt.files.length) {
      const row = el('div', 'gc-row gc-wt' + (selected === GIT_WORKTREE ? ' sel' : ''));
      row.dataset.sha = GIT_WORKTREE;
      row.appendChild(gitWorktreeSvg(cols));
      const n = wt.files.length + (wt.truncated ? '+' : '');
      const subj = el('div', 'gc-subj');
      subj.appendChild(el('span', 'gc-text', 'Working tree · ' + n + (wt.files.length === 1 && !wt.truncated ? ' uncommitted file' : ' uncommitted files')));
      row.appendChild(subj);
      actionable(row, showWorktree, 'working tree, ' + n + ' uncommitted files');
      graph.appendChild(row);
    }
    commits.forEach((c, i) => {
      const row = el('div', 'gc-row' + (c.sha === selected ? ' sel' : ''));
      row.dataset.sha = c.sha;
      const isHead = c.sha === head.sha;
      row.appendChild(gitLanesSvg(lays[i], cols, c.is_merge, isHead));
      const subj = el('div', 'gc-subj');
      if (isHead && !c.refs.some(r => r.head)) subj.appendChild(refBadge({ name: 'HEAD', kind: 'head', head: true }));
      for (const r of c.refs) subj.appendChild(refBadge(r));
      subj.appendChild(el('span', 'gc-text', c.subject));
      row.appendChild(subj);
      row.appendChild(el('span', 'gc-author muted', c.author));
      row.appendChild(el('span', 'gc-when muted mono', c.when));
      actionable(row, () => showCommit(c.sha), shortLabel(c));
      graph.appendChild(row);
    });
    if (!commits.length) graph.appendChild(el('div', 'empty', selectedRefs().length || head.sha ? 'no commits yet' : 'no refs selected'));
    if (hasMore) {
      const more = el('button', 'git-more', 'Load more');
      more.onclick = () => loadGraph(false);
      graph.appendChild(more);
    }
    const n = selectedRefs().length;
    counts.textContent = commits.length + (hasMore ? '+' : '') + ' commits · ' + n + ' of ' + refs.length + ' refs';
  };
  const loadGraph = async (reset) => {
    const n = ++loading;
    if (reset) { commits = []; hasMore = false; }
    try {
      const d = await getJSON(graphUrl(commits.length));
      if (n !== loading || !app.isConnected) return;
      commits = commits.concat(d.commits); hasMore = d.has_more;
    } catch (e) { if (n === loading) toast(e.message, true); }
    paintGraph();
  };
  // ↻: the refs, the working tree and the graph again; an open commit panel stays, an
  // open working-tree panel is painted afresh.
  const reload = async () => {
    let r, w;
    try { [r, w] = await Promise.all([getJSON('/api/git/refs'), getJSON('/api/git/worktree')]); }
    catch (e) { toast(e.message, true); return; }
    if (!app.isConnected) return;
    refs = r.refs; head = r.head; wt = w;
    if (gitState.side === 'worktree') showWorktree();
    await loadGraph(true);
  };
  reloadBtn.onclick = reload;

  // -- the side panel: a commit, or the compare --------------------------------------
  const sideHead = (title) => {
    const h = el('div', 'git-side-head row');
    const back = el('button', 'mini git-back', '‹ graph');
    back.onclick = () => side.classList.remove('open');
    h.appendChild(back); h.appendChild(el('h3', null, title));
    return h;
  };
  const openSide = () => side.classList.add('open');
  const setCompare = (which, v) => { gitState.compare[which] = v; showCompare(); };
  // The working tree's panel: no message; HEAD is the base every file diffs against.
  const showWorktree = () => {
    closeDiff();
    selected = GIT_WORKTREE; gitState.side = 'worktree';
    graph.querySelectorAll('.gc-row').forEach(r => r.classList.toggle('sel', r.dataset.sha === GIT_WORKTREE));
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
    pad.appendChild(fileList(wt.files, wt.truncated, (i) => showDiff({ base: wt.base, head: GIT_WORKTREE, files: wt.files }, i)));
    openSide();
  };
  const showCommit = async (sha) => {
    closeDiff();
    selected = sha; gitState.side = 'commit';
    graph.querySelectorAll('.gc-row').forEach(r => r.classList.toggle('sel', r.dataset.sha === sha));
    side.innerHTML = ''; side.appendChild(sideHead('Commit')); side.appendChild(el('div', 'muted page-pad', 'loading…'));
    openSide();
    let c;
    try { c = await getJSON('/api/git/commit/' + encodeURIComponent(sha)); }
    catch (e) { side.innerHTML = ''; side.appendChild(sideHead('Commit')); side.appendChild(el('div', 'err page-pad', e.message)); return; }
    if (selected !== sha || !app.isConnected) return;
    side.innerHTML = '';
    side.appendChild(sideHead('Commit'));
    const pad = el('div', 'gd page-pad'); side.appendChild(pad);
    const title = el('div', 'gd-subject');
    for (const r of c.refs) title.appendChild(refBadge(r));
    title.appendChild(el('span', null, c.subject));
    pad.appendChild(title);
    const kv = el('div', 'kv gd-meta');
    const add = (k, v) => { kv.appendChild(el('span', 'k', k)); const node = typeof v === 'string' ? el('span', 'v', v) : v; node.classList.add('v'); kv.appendChild(node); };
    add('commit', el('span', 'mono', c.sha));
    add('author', c.author.name + ' <' + c.author.email + '> · ' + c.author.date);
    if (c.committer.name !== c.author.name || c.committer.email !== c.author.email || c.committer.date !== c.author.date) {
      add('committer', c.committer.name + ' <' + c.committer.email + '> · ' + c.committer.date);
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
    if (c.body) pad.appendChild(el('pre', 'gd-body', c.body));
    pad.appendChild(fileList(c.files, c.truncated, (i) => showDiff({ base: c.base, head: c.sha, files: c.files }, i)));
  };

  let compareRun = 0;
  const showCompare = () => {
    closeDiff();
    gitState.side = 'compare';
    selected = null;
    graph.querySelectorAll('.gc-row.sel').forEach(r => r.classList.remove('sel'));
    side.innerHTML = '';
    side.appendChild(sideHead('Compare'));
    const pad = el('div', 'gcmp page-pad'); side.appendChild(pad);
    const result = el('div', 'gcmp-result');
    const run = async () => {
      const { base, head: h } = gitState.compare;
      result.innerHTML = '';
      if (!base || !h) { result.appendChild(el('div', 'muted', 'pick a base and a head: a branch, a tag, a commit, or text from a commit message')); return; }
      const n = ++compareRun;
      result.appendChild(el('div', 'muted', 'comparing…'));
      try {
        const d = await getJSON('/api/git/compare?base=' + encodeURIComponent(base.rev) + '&head=' + encodeURIComponent(h.rev));
        if (n !== compareRun || !result.isConnected) return;
        result.innerHTML = '';
        const sum = el('div', 'gcmp-sum muted');
        sum.appendChild(el('span', 'mono', d.base.short)); sum.appendChild(el('span', null, ' ' + d.base.subject + '  →  '));
        sum.appendChild(el('span', 'mono', d.head.short)); sum.appendChild(el('span', null, ' ' + d.head.subject));
        result.appendChild(sum);
        result.appendChild(fileList(d.files, d.truncated, (i) => showDiff({ base: d.base.sha, head: d.head.sha, files: d.files }, i)));
      } catch (e) { if (n === compareRun) { result.innerHTML = ''; result.appendChild(el('div', 'err', e.message)); } }
    };
    const listHost = el('div', 'gpick-host');
    const pickBase = gitPicker('base', gitState.compare.base, (v) => { gitState.compare.base = v; run(); }, listHost);
    const pickHead = gitPicker('head', gitState.compare.head, (v) => { gitState.compare.head = v; run(); }, listHost, [GIT_WORKTREE_PICK]);
    const swap = el('button', 'mini', '⇄'); swap.title = 'swap base and head';
    swap.onclick = () => { const { base, head: h } = gitState.compare; gitState.compare = { base: h, head: base }; pickBase.set(h); pickHead.set(base); run(); };
    const row = el('div', 'gcmp-pick');
    row.appendChild(pickBase.root); row.appendChild(swap); row.appendChild(pickHead.root);
    pad.appendChild(row);
    pad.appendChild(listHost);
    pad.appendChild(el('div', 'hint', 'the files that differ between the two revisions, as git diff base head lists them'));
    pad.appendChild(result);
    openSide();
    run();
  };
  compareBtn.onclick = showCompare;

  // -- the ref filter: group toggles and one checkbox per ref ------------------------
  let filter = null;
  const closeFilter = () => { if (filter) { filter.remove(); filter = null; document.removeEventListener('pointerdown', outside, true); document.removeEventListener('keydown', onKey, true); } };
  const outside = (e) => { if (filter && !filter.contains(e.target) && e.target !== refsBtn) closeFilter(); };
  const onKey = (e) => { if (e.key === 'Escape') closeFilter(); };
  const openFilter = () => {
    if (filter) { closeFilter(); return; }
    filter = el('div', 'gfilter');
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
    bar.appendChild(filter);
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', onKey, true);
    search.focus();
  };
  refsBtn.onclick = openFilter;

  if (gitState.side === 'compare' && (gitState.compare.base || gitState.compare.head)) showCompare();
  else if (gitState.side === 'worktree' && wt.files.length) showWorktree();
  else side.appendChild(el('div', 'empty', 'click a commit for its files and diff, or Compare for any two revisions'));
  await loadGraph(true);
}
