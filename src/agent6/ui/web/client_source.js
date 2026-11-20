// --- source views shared by the Git and Files tabs ---------------------------
// The helpers the two tabs render files with: a folder tree from paths, the unified-diff
// parser with its inline and split renderers, and the line renderer with highlight spans.
// Each takes plain data (entries with a `path`, a patch, text with spans) and returns
// elements; the tabs own the state. Every rendered line carries its numbers as data
// attributes (`data-old` and `data-new` on a diff line, `data-line` on a file line).

// A folder tree from entries that carry a `path`. A node is { name, path, dirs: Map(name →
// node), files: [entry] }; folders and files keep the order the entries came in.
function buildTree(entries) {
  const root = { name: '', path: '', dirs: new Map(), files: [] };
  for (const e of entries) {
    const parts = e.path.split('/');
    let node = root;
    for (let i = 0; i < parts.length - 1; i++) {
      let d = node.dirs.get(parts[i]);
      if (!d) { d = { name: parts[i], path: node.path ? node.path + '/' + parts[i] : parts[i], dirs: new Map(), files: [] }; node.dirs.set(parts[i], d); }
      node = d;
    }
    node.files.push(e);
  }
  return root;
}
// Every entry under a node, folders first, as `buildTree` ordered them.
function treeFiles(node) {
  const out = [];
  for (const d of node.dirs.values()) out.push(...treeFiles(d));
  out.push(...node.files);
  return out;
}
const fileCount = (n) => n === 1 ? '1 file' : n + ' files';

// Render a tree: a folder row (a caret, the name, what `folderNote(node)` says of its
// contents) that folds and unfolds its children on a click, and `fileRow(entry)` for each
// file; both carry `data-path`, the path as their title, and a `--depth` the stylesheet
// indents by. `folded` starts every folder closed; `opened`, a Map of folder path to open,
// overrides that per folder and records each toggle, so a choice outlives a repaint. The
// tree is one Tab stop: ↑ and ↓ move between the rows shown, → opens a folder or enters it,
// ← closes it or goes to its parent, and Home and End go to the first and last row.
function treeView(root, fileRow, folderNote, folded, folderMark, opened) {
  const box = el('div', 'gtree');
  box.setAttribute('role', 'tree'); box.setAttribute('aria-label', 'Files');
  const item = (row, depth) => {
    row.setAttribute('role', 'treeitem'); row.setAttribute('aria-level', depth + 1); row.tabIndex = -1;
    row.title ||= row.dataset.path;
    row.style.setProperty('--depth', depth);
  };
  const paint = (node, host, depth) => {
    for (const d of node.dirs.values()) {
      const row = el('div', 'gtree-dir'); row.dataset.path = d.path;
      const caret = el('span', 'gtree-caret');
      row.appendChild(caret);
      if (folderMark) { const mark = folderMark(d); if (mark) row.appendChild(mark); }
      row.appendChild(el('span', 'gtree-name', d.name)); row.appendChild(el('span', 'muted gtree-note', folderNote(d)));
      const kids = el('div', 'gtree-kids'); kids.setAttribute('role', 'group');
      let open = opened?.has(d.path) ? opened.get(d.path) : !folded;
      caret.textContent = '▾'; // one glyph; the stylesheet turns it for a folded folder
      const apply = () => { kids.hidden = !open; caret.classList.toggle('folded', !open); row.setAttribute('aria-expanded', String(open)); };
      actionable(row, () => { open = !open; opened?.set(d.path, open); apply(); }, 'folder ' + d.path);
      item(row, depth);
      apply();
      host.appendChild(row); host.appendChild(kids);
      paint(d, kids, depth + 1);
    }
    for (const f of node.files) { const r = fileRow(f); item(r, depth); host.appendChild(r); }
  };
  paint(root, box, 0);
  const shown = () => [...box.querySelectorAll('[role=treeitem]')].filter(r => !r.closest('.gtree-kids[hidden]'));
  // The Tab stop is the selected row, or the nearest folder showing where it is folded away.
  let stop = box.querySelector('[aria-selected=true]') || box.querySelector('[role=treeitem]');
  for (let kids; (kids = stop?.closest('.gtree-kids[hidden]'));) stop = kids.previousElementSibling;
  if (stop) stop.tabIndex = 0;
  box.addEventListener('focusin', (e) => {
    if (e.target.getAttribute('role') !== 'treeitem') return;
    for (const r of box.querySelectorAll('[role=treeitem][tabindex="0"]')) r.tabIndex = -1;
    e.target.tabIndex = 0;
  });
  box.addEventListener('keydown', (e) => {
    const row = e.target;
    if (row.getAttribute('role') !== 'treeitem') return;
    const rows = shown(), i = rows.indexOf(row), state = row.getAttribute('aria-expanded');
    let to = null;
    if (e.key === 'ArrowDown') to = rows[i + 1];
    else if (e.key === 'ArrowUp') to = rows[i - 1];
    else if (e.key === 'Home') to = rows[0];
    else if (e.key === 'End') to = rows[rows.length - 1];
    else if (e.key === 'ArrowRight') { if (state === 'false') row.click(); else if (state === 'true') to = rows[i + 1]; }
    else if (e.key === 'ArrowLeft') {
      if (state === 'true') row.click();
      else to = rows.slice(0, i).findLast(r => +r.getAttribute('aria-level') < +row.getAttribute('aria-level'));
    } else return;
    e.preventDefault();
    to?.focus();
  });
  return box;
}

// --- unified diffs ---------------------------------------------------------------
// Git quotes control characters and UTF-8 bytes in paths, using C escapes and octal bytes.
function diffPath(raw) {
  let path = raw.split('\t')[0];
  if (path.startsWith('"')) {
    const bytes = [], encoder = new TextEncoder();
    const escapes = {a: 7, b: 8, t: 9, n: 10, v: 11, f: 12, r: 13, '"': 34, '\\': 92};
    for (const m of path.slice(1, -1).matchAll(/\\([0-7]{1,3}|[abfnrtv"\\])|([^\\]+)/g)) {
      if (m[2]) bytes.push(...encoder.encode(m[2]));
      else bytes.push(/^[0-7]/.test(m[1]) ? parseInt(m[1], 8) : escapes[m[1]]);
    }
    path = new TextDecoder().decode(new Uint8Array(bytes));
  }
  return path === '/dev/null' ? '' : path.replace(/^[ab]\//, '');
}
// Parse a unified diff into files, each { preamble: [line], hunks: [{ header, oldStart,
// newStart, lines }], binary }. A hunk line is { kind: 'ctx' | 'add' | 'del' | 'meta',
// text (without its leading space, + or -), old, new }: its line number on each side, null
// where it has none; 'meta' is git's "\ No newline at end of file" note. The preamble holds
// the lines before a file's first hunk (diff --git, index, ---, +++, rename, Binary files).
function parseUnifiedDiff(text) {
  const lines = splitLines(text);
  const files = [];
  let file = null, hunk = null, o = 0, n = 0, oLeft = 0, nLeft = 0;
  const newFile = () => { file = { preamble: [], hunks: [], binary: false, oldPath: '', newPath: '' }; files.push(file); hunk = null; };
  for (const line of lines) {
    const m = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/.exec(line);
    if (m) {
      if (!file) newFile();
      o = +m[1]; n = +m[3]; oLeft = m[2] === undefined ? 1 : +m[2]; nLeft = m[4] === undefined ? 1 : +m[4];
      hunk = { header: line, oldStart: o, newStart: n, lines: [] };
      file.hunks.push(hunk);
      continue;
    }
    if (hunk && line[0] === '\\') { hunk.lines.push({ kind: 'meta', text: line, old: null, new: null }); continue; }
    if (hunk && (oLeft > 0 || nLeft > 0)) {
      const c = line[0], t = line.slice(1);
      if (c === '+') { hunk.lines.push({ kind: 'add', text: t, old: null, new: n++ }); nLeft--; }
      else if (c === '-') { hunk.lines.push({ kind: 'del', text: t, old: o++, new: null }); oLeft--; }
      else { hunk.lines.push({ kind: 'ctx', text: t, old: o++, new: n++ }); oLeft--; nLeft--; }
      continue;
    }
    if (!file || line.startsWith('diff ')) newFile();
    hunk = null;
    file.preamble.push(line);
    if (line.startsWith('--- ')) file.oldPath = diffPath(line.slice(4));
    if (line.startsWith('+++ ')) file.newPath = diffPath(line.slice(4));
    if (/^Binary files .* differ$/.test(line)) file.binary = true;
  }
  return files;
}
// A file's lines as the server counts them: a final newline starts no empty line.
function splitLines(text) { const lines = text.split('\n'); if (lines.length && lines[lines.length - 1] === '') lines.pop(); return lines; }
// A line-number gutter: an empty element whose number is drawn by the stylesheet from
// `data-n`, so copied text carries no numbers.
function gutter(n) { const g = el('i', 'g'); g.dataset.n = n == null ? '' : String(n); return g; }
const gutterWidth = (max) => String(Math.max(1, max)).length + 'ch';
// A line's code: its text with highlight spans, then its newline. The newline sits inside the
// code because a line is a flex row, which renders no bare whitespace, and a copy then loses
// blank lines.
function codeSpan(text, spans, marker = '') {
  const code = el('span', 'source-code');
  if (marker) code.appendChild(el('span', 'diff-marker', marker));
  appendSpans(code, text, spans);
  code.appendChild(document.createTextNode('\n'));
  return code;
}

// Render parsed files as one unified patch: a <pre class="diff"> with one span per line
// carrying `data-old` and `data-new`; `numbered` draws the two gutters.
function renderUnified(files, numbered, oldSpans = {}, newSpans = {}) {
  const box = el('pre', 'diff' + (numbered ? ' numbered' : ''));
  let max = 0;
  for (const f of files) for (const h of f.hunks) for (const l of h.lines) max = Math.max(max, l.old || 0, l.new || 0);
  box.style.setProperty('--gw', gutterWidth(max));
  const line = (cls, text, old, nw, marker = '', spans, f = null) => {
    const s = el('span', 'dl' + (cls ? ' ' + cls : ''));
    if (old != null) s.dataset.old = old;
    if (nw != null) s.dataset.new = nw;
    if (f) { if (f.oldPath) s.dataset.oldPath = f.oldPath; if (f.newPath) s.dataset.newPath = f.newPath; }
    if (numbered) { s.appendChild(gutter(old)); s.appendChild(gutter(nw)); }
    s.appendChild(codeSpan(text, spans, marker));
    return s;
  };
  const mark = { add: '+', del: '-', ctx: ' ' };
  for (const f of files) {
    const old = oldSpans[f.oldPath] || [], news = newSpans[f.newPath] || [];
    for (const p of f.preamble) box.appendChild(line(null, p, null, null));
    for (const h of f.hunks) {
      box.appendChild(line('hunk', h.header, null, null));
      for (const l of h.lines) {
        const spans = l.kind === 'meta' ? null : l.kind === 'del' ? old[l.old - 1] : news[l.new - 1] || old[l.old - 1];
        box.appendChild(line(l.kind === 'ctx' ? null : l.kind, l.text, l.old, l.new, mark[l.kind], spans, f));
      }
    }
  }
  return box;
}

// Render one parsed file side by side: a row per pair of lines, deletions beside the
// additions that replace them, context on both sides; a row carries `data-old` and
// `data-new`, each cell its own gutter. A drag selects within the side it starts in.
function renderSplit(file, oldSpans = {}, newSpans = {}) {
  const old = oldSpans[file.oldPath] || [], news = newSpans[file.newPath] || [];
  const box = el('div', 'dsplit numbered');
  box.addEventListener('pointerdown', (e) => {
    const c = e.target.closest('.ds-cell');
    if (c) box.dataset.select = c === c.parentElement.children[0] ? 'old' : 'new';
    else delete box.dataset.select;
  });
  if (file.oldPath) box.dataset.oldPath = file.oldPath;
  if (file.newPath) box.dataset.newPath = file.newPath;
  let max = 0;
  for (const h of file.hunks) for (const l of h.lines) max = Math.max(max, l.old || 0, l.new || 0);
  box.style.setProperty('--gw', gutterWidth(max));
  const cell = (l, side) => {
    const kind = l ? (side === 'old' ? (l.kind === 'del' ? ' del' : '') : (l.kind === 'add' ? ' add' : '')) : ' ds-empty';
    const c = el('span', 'ds-cell' + kind);
    c.appendChild(gutter(l ? (side === 'old' ? l.old : l.new) : null));
    if (l) c.appendChild(codeSpan(l.text, side === 'old' ? old[l.old - 1] : news[l.new - 1]));
    return c;
  };
  let segment = 0;
  const row = (left, right) => {
    const r = el('div', 'ds-row');
    r.dataset.segment = segment;
    if (left) r.dataset.old = left.old;
    if (right) r.dataset.new = right.new;
    r.appendChild(cell(left, 'old')); r.appendChild(cell(right, 'new'));
    box.appendChild(r);
  };
  for (const p of file.preamble) box.appendChild(el('div', 'ds-head', p));
  for (const h of file.hunks) {
    segment++;
    box.appendChild(el('div', 'ds-head hunk', h.header));
    let dels = [], adds = [];
    const flush = () => { for (let i = 0; i < Math.max(dels.length, adds.length); i++) row(dels[i] || null, adds[i] || null); dels = []; adds = []; };
    for (const l of h.lines) {
      if (l.kind === 'del') dels.push(l);
      else if (l.kind === 'add') adds.push(l);
      else { flush(); if (l.kind === 'ctx') row(l, l); else box.appendChild(el('div', 'ds-head meta', l.text)); }
    }
    flush();
  }
  return box;
}

// A click on a line number quotes that line for a composer: the path, the side, the number
// and the text, with the operator's words to follow beneath. `where(side)` names a side's
// revision; `deliver(note)` hands the text to a composer. The rows carry their paths
// (`data-old-path`, `data-new-path`), or the host does; a one-sided view names its side.
// The keyboard reaches the numbers through one Tab stop: ↑ and ↓ move between them, and
// Enter or Space quotes. The host may be filled after this call, and again later.
function quoteDiffLines(host, where, deliver) {
  host.classList.add('quotable');
  // A number's parent is its line: a diff line, a split cell or a file line.
  const sideOf = (g) => {
    const row = g.parentElement;
    if (row.classList.contains('dl')) return g === row.querySelector('.g') ? 'base' : 'head';
    if (row.classList.contains('ds-cell')) return row === row.parentElement.children[0] ? 'base' : 'head';
    return host.dataset.side || 'head';
  };
  const quote = (g) => {
    const row = g.parentElement, side = sideOf(g);
    const code = row.querySelector('.source-code');
    let text = code ? code.textContent.replace(/\n$/, '') : '';
    if (code && code.querySelector('.diff-marker')) text = text.slice(1);
    const scope = row.closest('[data-old-path], [data-new-path]') || host;
    const path = (side === 'base' ? scope.dataset.oldPath : scope.dataset.newPath) || scope.dataset.newPath || scope.dataset.oldPath || '';
    deliver(path + ':' + g.dataset.n + ' (' + side + ' side, ' + where(side) + '):\n> ' + text + '\n');
  };
  const numbers = () => [...host.querySelectorAll('.g')].filter(g => g.dataset.n);
  const seat = () => numbers().forEach((g, i) => {
    g.tabIndex = i ? -1 : 0;
    g.setAttribute('role', 'button');
    g.setAttribute('aria-label', 'Quote line ' + g.dataset.n + ', ' + sideOf(g) + ' side');
  });
  new MutationObserver(seat).observe(host, {childList: true});
  seat();
  host.addEventListener('click', (e) => {
    const g = e.target.closest('.g');
    if (g && g.dataset.n && host.contains(g)) quote(g);
  });
  host.addEventListener('keydown', (e) => {
    const g = e.target;
    if (!g.matches('.g[role=button]') || !host.contains(g)) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); quote(g); return; }
    const step = {ArrowDown: 1, ArrowUp: -1}[e.key];
    if (!step) return;
    e.preventDefault();
    const all = numbers(), next = all[all.indexOf(g) + step];
    if (next) { g.tabIndex = -1; next.tabIndex = 0; next.focus(); }
  });
}

// --- a file's lines ----------------------------------------------------------------
// Render a file's lines with their numbers and highlight spans (the server's, per line as
// [start, end, class]); `marks` is a Set of line numbers to mark as changed. Each line is
// one element carrying `data-line`.
function renderLines(lines, spans, marks) {
  const box = el('pre', 'src numbered');
  box.style.setProperty('--gw', gutterWidth(lines.length));
  lines.forEach((text, i) => {
    const n = i + 1;
    const s = el('span', 'sl' + (marks && marks.has(n) ? ' changed' : ''));
    s.dataset.line = n;
    s.appendChild(gutter(n));
    s.appendChild(codeSpan(text, spans && spans[i]));
    box.appendChild(s);
  });
  return box;
}
// Rows name their side, segment (hunk), displayed path, node and BlameLine. A blame cell
// leads its line. A commit group's card opens over the page by the cell the pointer or the
// focus is on: the card moves into that cell, which the stylesheet makes its anchor.
const blameCommits = new Map();
function clearBlame(host) {
  host.querySelectorAll('.blame-cell').forEach(c => c.remove());
}
function renderBlame(rows) {
  let group = null, previous = null;
  for (const row of rows) {
    const line = row.line;
    if (!row.node || !line) { group = previous = null; continue; }
    const same = previous && previous.side === row.side && previous.segment === row.segment
      && previous.line.sha === line.sha && previous.line.filename === line.filename
      && previous.line.line + 1 === line.line;
    if (!same) group = {cells: [], card: null, timer: null, dismissed: false};
    const g = group;
    const cell = el('span', 'blame-cell');
    cell.dataset.sha = line.sha;
    cell.dataset.side = row.side;
    cell.dataset.filename = line.filename;
    if (!same) {
      cell.classList.add('blame-start');
      const label = el('span', 'blame-label');
      if (line.sha) {
        const link = el('a', 'blame-commit', line.sha.slice(0, 7));
        link.href = '#/git/commit/' + encodeURIComponent(line.sha) + '?' + new URLSearchParams({path: line.filename, line: line.original_line});
        link.title = line.sha;
        label.appendChild(link);
        label.appendChild(el('span', 'blame-author', line.author));
        label.appendChild(el('span', 'blame-date', line.date.slice(0, 10)));
      } else {
        // The narrow column shows a dash in place of the words (styles_source.css).
        const none = el('span', 'blame-none'); none.title = 'not committed';
        none.appendChild(el('span', null, 'not committed'));
        label.appendChild(none);
      }
      cell.appendChild(label);
    }
    g.cells.push(cell);
    const hide = () => {
      g.dismissed = false;
      clearTimeout(g.timer); // leaving the card also leaves its cell: one hide stays pending
      g.timer = setTimeout(() => {
        g.cells.forEach(c => c.classList.remove('blame-hover'));
        if (g.card?.matches(':popover-open')) g.card.hidePopover();
      }, 150);
    };
    const show = async () => {
      clearTimeout(g.timer);
      g.cells.forEach(c => c.classList.add('blame-hover'));
      if (!line.sha || g.dismissed) return;
      if (!g.card) {
        g.card = el('span', 'blame-card');
        g.card.popover = 'manual';
        // Escape closes the open card, which stays closed until the pointer leaves or the focus moves.
        const escape = (e) => {
          if (!g.card.isConnected) document.removeEventListener('keydown', escape); // removed while open: no toggle
          if (e.key !== 'Escape' || !g.card.matches(':popover-open')) return;
          g.card.hidePopover();
          g.dismissed = g.cells.some(c => c.matches(':hover, :focus-within'));
        };
        g.card.addEventListener('toggle', (e) => document[e.newState === 'open' ? 'addEventListener' : 'removeEventListener']('keydown', escape));
        g.card.appendChild(el('span', 'mono', line.sha));
        const who = el('span', null, line.author + ' · '); who.appendChild(gitDate(line.date)); g.card.appendChild(who);
        if (line.filename !== row.path) g.card.appendChild(el('span', null, line.filename));
        const title = el('a', 'blame-title', 'loading commit…');
        title.href = '#/git/commit/' + encodeURIComponent(line.sha);
        g.card.appendChild(title);
        g.card.onmouseenter = () => clearTimeout(g.timer);
        g.card.onmouseleave = hide;
        if (!blameCommits.has(line.sha)) blameCommits.set(line.sha, getJSON('/api/git/commit/' + encodeURIComponent(line.sha)));
        blameCommits.get(line.sha).then(c => { if (g.card.isConnected) title.textContent = c.subject; })
          .catch(e => { if (g.card.isConnected) title.textContent = 'Commit unavailable: ' + e.message; blameCommits.delete(line.sha); });
      }
      if (g.card.parentNode !== cell) cell.appendChild(g.card); // a moved popover closes
      if (!g.card.matches(':popover-open')) g.card.showPopover();
    };
    cell.onmouseenter = show; cell.onmouseleave = hide;
    // focusin bubbles from the link; Chromium has no onfocusin property handler.
    cell.addEventListener('focusin', show); cell.addEventListener('focusout', hide);
    row.node.prepend(cell);
    previous = row;
  }
}

// Append a line's text, its spans wrapped as <span class="hl-<class>">.
function appendSpans(node, text, row) {
  let at = 0;
  for (const [a, b, cls] of row || []) {
    if (a > at) node.appendChild(document.createTextNode(text.slice(at, a)));
    node.appendChild(el('span', 'hl-' + cls, text.slice(a, b)));
    at = b;
  }
  if (at < text.length) node.appendChild(document.createTextNode(text.slice(at)));
}
