// --- files -------------------------------------------------------------------
// The Files tab: the files at one revision (HEAD to start, the working tree, or any
// revision the picker finds) as a folder tree with a filter, and one file read-only: text
// with line numbers and highlighting, an image, or a binary file's size and kind. The
// picked revision and the folders opened outlive a visit to another tab; the tree is fetched
// again on return.
// The picker's { rev, label }; each folder path → open; the Markdown file shown, if any.
const filesState = { rev: null, folders: new Map(), markdown: null };
const MARKDOWN_MODE_KEY = 'a6-markdown-mode';
async function renderFiles(gen, revision = '', path = '', query = new URLSearchParams()) {
  if (query.has('panel')) return renderReviewedFile(gen, path, query);
  // A hash naming the Markdown file shown moves within it: a link to a heading, and Back and
  // Forward between them, scroll in place.
  const markdown = filesState.markdown;
  if (markdown?.view.isConnected && markdown.rev === revision && markdown.path === path) return markdown.scroll(query.get('anchor'));
  const refsData = await getJSON('/api/git/refs');
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const app = el('div', 'files-app'); view.appendChild(app);
  if (revision && revision !== filesState.rev?.rev) filesState.rev = { rev: revision, label: revision === GIT_WORKTREE ? 'working tree' : revision };
  if (!filesState.rev) filesState.rev = refsData.head.sha ? { rev: 'HEAD', label: 'HEAD' } : GIT_WORKTREE_PICK;
  // The hash names the revision picked, so a reload keeps HEAD or a branch; the reads use the
  // commit its listing resolved to.
  const filesHash = (file = '') => '#/files/' + encodeURIComponent(filesState.rev.rev) + (file && '/' + file.split('/').map(encodeURIComponent).join('/'));

  // -- toolbar: the revision picker, the filter, the count ---------------------------
  const bar = el('div', 'files-bar page-pad row wrap'); app.appendChild(bar);
  const listHost = el('div', 'gpick-host');
  const pick = gitPicker('revision', filesState.rev, (v) => { filesState.rev = v; setViewHash(filesHash()); relist(); }, listHost, [GIT_WORKTREE_PICK]);
  pick.root.appendChild(listHost); // the list opens under the field it searches
  const filter = el('input', 'filter'); filter.type = 'search'; filter.placeholder = 'filter files…'; filter.setAttribute('aria-label', 'filter files');
  const counts = el('span', 'muted files-counts');
  let showIgnored = recall('a6-files-ignored', 'off') === 'on';
  const ignoredToggle = el('button', 'files-ignored', 'Show ignored');
  ignoredToggle.title = 'Include files excluded by any Git ignore source';
  ignoredToggle.setAttribute('aria-pressed', String(showIgnored));
  ignoredToggle.onclick = () => {
    showIgnored = !showIgnored; remember('a6-files-ignored', showIgnored ? 'on' : 'off');
    ignoredToggle.setAttribute('aria-pressed', String(showIgnored));
    return relist();
  };
  bar.appendChild(pick.root); bar.appendChild(filter); bar.appendChild(ignoredToggle); bar.appendChild(counts);
  const body = el('div', 'files-body'); app.appendChild(body);
  const tree = el('div', 'files-tree'); tree.id = 'files-tree'; body.appendChild(tree);
  body.appendChild(resizeHandle(tree, {
    label: 'File tree width', key: 'a6-files-tree-w', property: '--files-tree-w',
    min: 180, max: () => body.getBoundingClientRect().width - 246,
  }));
  const pane = el('div', 'files-view'); body.appendChild(pane);
  // With no file open the pane says how to open one: from the tree, or by showing the tree first.
  const unopened = el('div', 'empty'); pane.appendChild(unopened);
  const toggle = el('button', 'files-tree-toggle');
  toggle.setAttribute('aria-controls', tree.id);
  const setTreeOpen = open => {
    body.classList.toggle('tree-closed', !open);
    toggle.textContent = open ? 'Hide tree' : 'Show tree';
    toggle.setAttribute('aria-expanded', String(open));
    unopened.textContent = open ? 'click a file to read it' : 'show the tree to pick a file';
  };
  setTreeOpen(recall('a6-files-tree', 'open') !== 'closed');
  toggle.onclick = () => {
    const open = body.classList.contains('tree-closed');
    setTreeOpen(open);
    remember('a6-files-tree', open ? 'open' : 'closed');
  };
  bar.insertBefore(toggle, pick.root);

  // -- the tree ----------------------------------------------------------------------
  let entries = [], truncated = false, rev = '', ignored = false, loading = 0, shown = null;
  // A working-tree row's mark says how it differs from HEAD: M modified, A untracked, I ignored. A
  // clean row keeps an empty slot, so the names stay in line; a commit's rows have no marks.
  const stateMark = (files) => {
    const counts = { tracked: 0, untracked: 0, modified: 0, ignored: 0 };
    for (const f of files) if (f.state) counts[f.state]++;
    const state = counts.modified ? 'modified' : counts.untracked ? 'untracked' : counts.tracked ? 'tracked' : counts.ignored ? 'ignored' : '';
    if (!state) return null;
    if (state === 'tracked') return el('span', 'gf-st');
    const mark = statusBadge({ untracked: 'U', modified: 'M', ignored: 'I' }[state]);
    mark.title = Object.entries(counts).filter(([, n]) => n).map(([s, n]) => n + ' ' + s).join(', ');
    return mark;
  };
  const fileRow = (f) => {
    const row = el('div', 'gf');
    row.dataset.path = f.path;
    row.setAttribute('aria-selected', String(shown === f.path));
    const mark = stateMark([f]); if (mark) row.appendChild(mark);
    row.appendChild(el('span', 'gf-path', f.path.slice(f.path.lastIndexOf('/') + 1)));
    row.appendChild(el('span', 'gf-counts muted', f.kind === 'file' ? fmtBytes(f.size) : f.kind));
    actionable(row, () => showFile(f), f.path);
    return row;
  };
  const folderNote = (d) => { const under = treeFiles(d); let size = 0; for (const f of under) size += f.size; return fileCount(under.length) + ' · ' + fmtBytes(size); };
  const paintTree = () => {
    tree.innerHTML = '';
    const q = filter.value.trim().toLowerCase();
    const kept = q ? entries.filter(f => f.path.toLowerCase().includes(q)) : entries;
    if (truncated) tree.appendChild(el('div', 'err gf-note', 'the list is cut: more files than are shown'));
    // A large tree starts folded and keeps the folders opened by hand; a filter opens every folder holding a match.
    if (!kept.length) tree.appendChild(el('div', 'empty', entries.length ? 'no file matches' : 'no files'));
    else tree.appendChild(treeView(buildTree(kept), fileRow, folderNote, !q && kept.length > 200, d => stateMark(treeFiles(d)), q ? null : filesState.folders));
    counts.textContent = (q ? kept.length + ' of ' : '') + fileCount(entries.length) + (truncated ? '+' : '');
    counts.dataset.room = entries.length + ' of ' + fileCount(entries.length) + (truncated ? '+' : '');
  };
  const closeFile = () => { pane.innerHTML = ''; pane.appendChild(unopened); body.classList.remove('viewing'); };
  // A new listing (another revision, ignored files shown or hidden) keeps the open file open
  // when it still lists it; the file stays on screen until the listing answers, and the hash
  // names it again only then.
  const relist = () => {
    query = new URLSearchParams(); setViewHash(filesHash(), true);
    return loadTree(body.classList.contains('viewing') ? shown : '', true);
  };
  const loadTree = async (openPath = '', relisting = false) => {
    const n = ++loading;
    ignoredToggle.hidden = filesState.rev.rev !== GIT_WORKTREE;
    pane.dataset.key = ''; shown = null; rev = ''; ignored = false; entries = [];
    if (!relisting) closeFile();
    tree.innerHTML = ''; tree.appendChild(el('div', 'empty', 'loading…'));
    try {
      const d = await getJSON('/api/git/tree?rev=' + encodeURIComponent(filesState.rev.rev) + (showIgnored && !ignoredToggle.hidden ? '&ignored=1' : ''));
      if (n !== loading || !app.isConnected) return;
      entries = d.files; truncated = d.truncated; rev = d.rev; ignored = !!d.ignored;
    } catch (e) { if (n === loading && app.isConnected) { closeFile(); tree.innerHTML = ''; tree.appendChild(el('div', 'empty err', e.message)); } return; }
    const file = entries.find(f => f.path === openPath) || (openPath && !relisting ? { path: openPath, kind: 'file', size: null } : null);
    if (file) { reveal(file.path); shown = file.path; }
    else if (relisting) closeFile();
    paintTree();
    if (file) await showFile(file, false, true);
  };
  filter.oninput = paintTree;
  // The listed path a symlink names, or null: a target outside the repository or not listed.
  const targetPath = (f) => {
    if (f.target.startsWith('/')) return null;
    const parts = [];
    for (const part of (f.path.slice(0, f.path.lastIndexOf('/') + 1) + f.target).split('/')) {
      if (part === '..') { if (!parts.pop()) return null; } else if (part && part !== '.') parts.push(part);
    }
    const path = parts.join('/');
    return entries.some(e => e.path === path) ? path : null;
  };
  // Open a file's folders, so its row shows in every later paint of the tree.
  const reveal = (path) => {
    const parts = path.split('/');
    for (let i = 1; i < parts.length; i++) filesState.folders.set(parts.slice(0, i).join('/'), true);
  };

  // -- one file: it takes the right pane, or the whole screen on a phone -------------
  let viewing = 0;
  const showFile = async (f, whole, linked = false) => {
    const key = JSON.stringify([++viewing, rev, f.path, !!whole]);
    pane.dataset.key = key;
    const current = () => pane.dataset.key === key && pane.isConnected;
    shown = f.path;
    const anchor = linked ? query.get('anchor') : null;
    setViewHash(filesHash(f.path) + (anchor ? '?' + new URLSearchParams({ anchor }) : ''), linked);
    reveal(f.path);
    for (const r of tree.querySelectorAll('.gf')) {
      r.setAttribute('aria-selected', String(r.dataset.path === f.path));
      if (r.dataset.path === f.path) r.scrollIntoView({ block: 'nearest' });
    }
    body.classList.add('viewing');
    pane.innerHTML = '';
    const head = el('div', 'files-head row');
    const back = el('button', 'mini files-back', '‹ Files');
    back.onclick = () => { body.classList.remove('viewing'); tree.querySelector('[aria-selected=true]')?.focus(); };
    head.appendChild(back);
    // A long path wraps after a slash, not inside a name.
    const pathText = el('span', 'gf-path'); head.appendChild(pathText);
    pathText.append(...f.path.split('/').flatMap((part, i) => i ? ['/', el('wbr'), part] : [part]));
    const facts = f.size === null ? [] : [f.kind === 'file' ? fmtBytes(f.size) : f.kind];
    const note = el('span', 'muted files-note', facts.join(' · '));
    head.appendChild(note);
    pane.appendChild(head);
    // On a narrow screen the file covers the tree: a file opened from it takes the focus to Back.
    if (!linked && back.checkVisibility()) back.focus();
    const box = el('div', 'files-text'); pane.appendChild(box);
    if (f.kind === 'symlink') {
      const symlink = el('div', 'muted gf-note', 'symlink to '), to = targetPath(f);
      if (to === null) symlink.append(f.target);
      else { const a = el('a', null, f.target); a.href = filesHash(to); symlink.appendChild(a); }
      box.appendChild(symlink);
      return;
    }
    if (f.kind !== 'file') { box.appendChild(el('div', 'muted gf-note', 'a submodule: its files are in another repository')); return; }
    if (isImagePath(f.path)) {
      // The head's note takes the image's size and dimensions, as it takes a text file's facts.
      // The image takes the place of "loading…" once it has loaded or failed.
      const loading = el('div', 'muted', 'loading…'); box.appendChild(loading);
      blobImageData('/api/git/blob?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(f.path) + (ignored ? '&ignored=1' : '')).then(image => {
        if (!current()) return;
        if (!facts.length) facts.push(fmtBytes(image.size));
        const wrap = el('div', 'git-img one'), fig = el('figure', 'git-img-side'), img = el('img');
        img.alt = f.path; fig.appendChild(img); wrap.appendChild(fig);
        img.onload = () => { facts.push(img.naturalWidth + '\u00d7' + img.naturalHeight); note.textContent = facts.join(' · '); loading.replaceWith(wrap); };
        img.onerror = () => { note.textContent = facts.concat('the browser cannot show this image').join(' · '); loading.replaceWith(wrap); };
        img.src = image.url;
      }).catch(e => { if (current()) loading.replaceWith(el('div', 'err gf-note', e.message)); });
      return;
    }
    box.appendChild(el('div', 'muted', 'loading…'));
    // The buttons hold their places in the head while the text loads: Blame, and Rendered and
    // Source for a name the server renders as Markdown (.md or .markdown, as text_payload in
    // git_model.py decides).
    const blame = el('button', 'mini files-blame', 'Blame');
    blame.setAttribute('aria-pressed', 'false'); blame.disabled = true; head.insertBefore(blame, note);
    const rendersMarkdown = /\.(md|markdown)$/i.test(f.path);
    const modes = el('span', 'gview files-md-mode'), sourceBox = el('div', 'files-source');
    let md = null;
    const selectMode = mode => {
      const rendered = mode !== 'source';
      blame.hidden = rendered;
      for (const button of modes.children) {
        const active = button.dataset.mode === (rendered ? 'rendered' : 'source');
        button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
      }
      if (!md) return;
      md.hidden = !rendered; sourceBox.hidden = rendered;
      if (rendered && anchor) [...md.querySelectorAll('[id]')].find(h => h.id === anchor)?.scrollIntoView({ block: 'start' });
    };
    if (rendersMarkdown) {
      modes.setAttribute('role', 'group'); modes.setAttribute('aria-label', 'Markdown view');
      for (const [mode, label] of [['rendered', 'Rendered'], ['source', 'Source']]) {
        const button = el('button', 'mini', label); button.dataset.mode = mode; button.disabled = true;
        button.onclick = () => { remember(MARKDOWN_MODE_KEY, mode); selectMode(mode); };
        modes.appendChild(button);
      }
      head.insertBefore(modes, blame);
      selectMode(recall(MARKDOWN_MODE_KEY, 'rendered'));
    }
    let t;
    const readQuery = 'rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(f.path) + (whole ? '&whole=1' : '') + (ignored ? '&ignored=1' : '');
    try { t = await getJSON('/api/git/text?' + readQuery); }
    catch (e) { if (current()) { blame.remove(); modes.remove(); box.innerHTML = ''; box.appendChild(el('div', 'err', e.message)); } return; }
    if (!current()) return;
    box.innerHTML = '';
    if (t.language) facts.push(t.language);
    if (t.binary) {
      // Blame stays, disabled, so the head keeps the shape it had while the file loaded.
      blame.title = 'a binary file has no lines to blame'; modes.remove(); blame.hidden = false;
      note.textContent = facts.join(' · '); box.appendChild(el('div', 'muted gf-note', 'binary file: no text')); return;
    }
    facts.push(t.lines === 1 ? '1 line' : t.lines + ' lines');
    note.textContent = facts.join(' · ');
    if (t.truncated) {
      const cut = el('div', 'err gf-note', 'the file is cut at ' + Math.round(t.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole file'); all.onclick = () => showFile(f, true);
      cut.appendChild(all); box.appendChild(cut);
    }
    box.appendChild(sourceBox);
    const source = renderLines(splitLines(t.text), t.spans, null);
    sourceBox.appendChild(source);
    blame.disabled = false;
    let blaming = false, blameRequest = 0;
    blame.onclick = async () => {
      const request = ++blameRequest;
      blaming = !blaming;
      blame.setAttribute('aria-pressed', String(blaming)); blame.classList.toggle('active', blaming);
      clearBlame(source);
      sourceBox.querySelectorAll('.blame-note').forEach(c => c.remove());
      if (!blaming) return;
      const notice = el('div', 'muted blame-note', 'loading blame…'); sourceBox.insertBefore(notice, source);
      try {
        const d = await getJSON('/api/git/blame?' + readQuery);
        if (!current() || request !== blameRequest) return;
        if (d.digest !== t.digest) { notice.classList.add('err'); notice.textContent = 'the file changed since it was opened: open it again to blame'; return; }
        notice.remove();
        const lines = new Map(d.lines.map(l => [l.line, l]));
        renderBlame([...source.querySelectorAll('[data-line]')].map(node => ({
          node, line: lines.get(Number(node.dataset.line)), side: 'head', path: f.path,
        })));
      } catch (e) { if (current() && request === blameRequest) { notice.classList.add('err'); notice.textContent = e.message; } }
    };
    if (!rendersMarkdown) return;
    // A Markdown file cut at the size cap has no rendering: its text alone shows.
    if (typeof t.rendered_html !== 'string') { modes.remove(); blame.hidden = false; return; }
    md = el('div', 'md-view'); md.innerHTML = t.rendered_html; box.appendChild(md);
    // The server's links to files name the commit listed; the page's name the revision picked.
    const listed = '#/files/' + encodeURIComponent(rev) + '/';
    for (const a of md.querySelectorAll('a[href^="#/files/"]')) {
      const href = a.getAttribute('href');
      if (href.startsWith(listed)) a.setAttribute('href', filesHash() + '/' + href.slice(listed.length));
    }
    filesState.markdown = { rev: filesState.rev.rev, path: f.path, view: md, scroll: (id) => {
      const heading = id && [...md.querySelectorAll('[id]')].find(h => h.id === id);
      if (heading) heading.scrollIntoView({ block: 'start' }); else box.scrollTo(0, 0);
    } };
    // A link to the hash already shown changes nothing a hashchange sees: route it again.
    md.addEventListener('click', (e) => { if (e.target.closest('a')?.href === location.href) route(); });
    for (const image of md.querySelectorAll('img[data-blob-src]')) {
      blobImageData(image.dataset.blobSrc).then(data => {
        if (current()) image.src = data.url;
      }).catch(e => {
        if (!current()) return;
        const missing = el('span', 'md-missing');
        missing.appendChild(el('span', null, image.alt || 'image'));
        missing.appendChild(el('span', 'muted', e.message));
        image.replaceWith(missing);
      });
    }
    for (const button of modes.children) button.disabled = false;
    selectMode(recall(MARKDOWN_MODE_KEY, 'rendered'));
  };
  await loadTree(path);
}
boot();
