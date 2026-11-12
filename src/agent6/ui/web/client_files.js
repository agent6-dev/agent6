// --- files -------------------------------------------------------------------
// The Files tab: the files at one revision (HEAD to start, the working tree, or any
// revision the picker finds) as a folder tree with a filter, and one file read-only: text
// with line numbers and highlighting, an image, or a binary file's size and kind. The
// picked revision outlives a visit to another tab; the tree is fetched again on return.
const filesState = { rev: null }; // the picker's { rev, label }; null until the first visit
const MARKDOWN_MODE_KEY = 'a6-markdown-mode';
async function renderFiles(gen, revision = '', path = '', query = new URLSearchParams()) {
  if (query.has('panel')) return renderReviewedFile(gen, path, query);
  setCrumb('files');
  const refsData = await getJSON('/api/git/refs');
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const app = el('div', 'files-app'); view.appendChild(app);
  if (revision) filesState.rev = { rev: revision, label: revision === GIT_WORKTREE ? 'working tree' : revision };
  if (!filesState.rev) filesState.rev = refsData.head.sha ? { rev: 'HEAD', label: 'HEAD' } : GIT_WORKTREE_PICK;

  // -- toolbar: the revision picker, the filter, the count ---------------------------
  const bar = el('div', 'files-bar page-pad row wrap'); app.appendChild(bar);
  const listHost = el('div', 'gpick-host page-pad');
  const pick = gitPicker('revision', filesState.rev, (v) => { filesState.rev = v; setViewHash('#/files/' + encodeURIComponent(v.rev)); loadTree(); }, listHost, [GIT_WORKTREE_PICK]);
  const filter = el('input', 'filter'); filter.type = 'search'; filter.placeholder = 'filter files…'; filter.setAttribute('aria-label', 'filter files');
  const counts = el('span', 'muted files-counts');
  let showIgnored = recall('a6-files-ignored', 'off') === 'on';
  const ignoredToggle = el('button', 'files-ignored', 'show ignored');
  ignoredToggle.title = 'Include files excluded by any Git ignore source';
  ignoredToggle.setAttribute('aria-pressed', String(showIgnored));
  ignoredToggle.onclick = () => {
    showIgnored = !showIgnored; remember('a6-files-ignored', showIgnored ? 'on' : 'off');
    ignoredToggle.setAttribute('aria-pressed', String(showIgnored));
    setViewHash('#/files/' + encodeURIComponent(filesState.rev.rev));
    return loadTree();
  };
  bar.appendChild(pick.root); bar.appendChild(filter); bar.appendChild(ignoredToggle); bar.appendChild(counts);
  app.appendChild(listHost);
  const body = el('div', 'files-body'); app.appendChild(body);
  const tree = el('div', 'files-tree'); tree.id = 'files-tree'; body.appendChild(tree);
  body.appendChild(resizeHandle(tree, {
    label: 'File tree width', key: 'a6-files-tree-w', property: '--files-tree-w',
    min: 180, max: () => body.getBoundingClientRect().width - 246,
  }));
  const pane = el('div', 'files-view'); body.appendChild(pane);
  pane.appendChild(el('div', 'empty', 'click a file to read it'));
  const toggle = el('button', 'files-tree-toggle');
  toggle.setAttribute('aria-controls', tree.id);
  const setTreeOpen = open => {
    body.classList.toggle('tree-closed', !open);
    toggle.textContent = open ? 'Hide tree' : 'Show tree';
    toggle.setAttribute('aria-expanded', String(open));
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
  const stateMark = (files) => {
    const counts = { tracked: 0, untracked: 0, modified: 0, ignored: 0 };
    for (const f of files) if (f.state) counts[f.state]++;
    const state = counts.modified ? 'modified' : counts.untracked ? 'untracked' : counts.tracked ? 'tracked' : counts.ignored ? 'ignored' : '';
    if (!state) return null;
    const mark = statusBadge({ tracked: 'H', untracked: 'A', modified: 'M', ignored: 'I' }[state]);
    mark.title = Object.entries(counts).filter(([, n]) => n).map(([s, n]) => n + ' ' + s).join(', ');
    return mark;
  };
  const fileRow = (f) => {
    const row = el('div', 'gf' + (shown === f.path ? ' sel' : ''));
    row.dataset.path = f.path;
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
    if (truncated) tree.appendChild(el('div', 'err gf-note page-pad', 'the list is cut: more files than are shown'));
    if (!kept.length) tree.appendChild(el('div', 'empty', entries.length ? 'no file matches' : 'no files'));
    else tree.appendChild(treeView(buildTree(kept), fileRow, folderNote, !q && kept.length > 200, d => stateMark(treeFiles(d)))); // a large tree starts folded
    counts.textContent = (q ? kept.length + ' of ' : '') + fileCount(entries.length) + (truncated ? '+' : '');
  };
  const loadTree = async (openPath = '') => {
    const n = ++loading;
    ignoredToggle.hidden = filesState.rev.rev !== GIT_WORKTREE;
    pane.dataset.key = ''; shown = null; rev = ''; ignored = false; entries = [];
    pane.innerHTML = ''; pane.appendChild(el('div', 'empty', 'click a file to read it'));
    body.classList.remove('viewing');
    tree.innerHTML = ''; tree.appendChild(el('div', 'muted page-pad', 'loading…'));
    try {
      const d = await getJSON('/api/git/tree?rev=' + encodeURIComponent(filesState.rev.rev) + (showIgnored && !ignoredToggle.hidden ? '&ignored=1' : ''));
      if (n !== loading || !app.isConnected) return;
      entries = d.files; truncated = d.truncated; rev = d.rev; ignored = !!d.ignored;
    } catch (e) { if (n === loading && app.isConnected) { tree.innerHTML = ''; tree.appendChild(el('div', 'err page-pad', e.message)); } return; }
    paintTree();
    if (openPath) await showFile(entries.find(f => f.path === openPath) || { path: openPath, kind: 'file', size: null }, false, true);
  };
  filter.oninput = paintTree;

  // -- one file: it takes the right pane, or the whole screen on a phone -------------
  let viewing = 0;
  const showFile = async (f, whole, linked = false) => {
    const key = JSON.stringify([++viewing, rev, f.path, !!whole]);
    pane.dataset.key = key;
    const current = () => pane.dataset.key === key && pane.isConnected;
    shown = f.path;
    const anchor = linked ? query.get('anchor') : null;
    const fileHash = '#/files/' + encodeURIComponent(rev) + '/' + f.path.split('/').map(encodeURIComponent).join('/');
    setViewHash(fileHash + (anchor ? '?' + new URLSearchParams({ anchor }) : ''), linked);
    tree.querySelectorAll('.gf').forEach(r => r.classList.toggle('sel', r.dataset.path === f.path));
    body.classList.add('viewing');
    pane.innerHTML = '';
    const head = el('div', 'files-head row');
    const back = el('button', 'mini files-back', '‹ files'); back.onclick = () => body.classList.remove('viewing');
    head.appendChild(back);
    head.appendChild(el('span', 'gf-path', f.path));
    const facts = f.size === null ? [] : [f.kind === 'file' ? fmtBytes(f.size) : f.kind];
    const note = el('span', 'muted files-note', facts.join(' · '));
    head.appendChild(note);
    pane.appendChild(head);
    const box = el('div', 'files-text'); pane.appendChild(box);
    if (f.kind !== 'file') {
      box.appendChild(el('div', 'muted gf-note', f.kind === 'symlink' ? 'symlink: ' + f.target : 'a submodule: its files are in another repository'));
      return;
    }
    if (isImagePath(f.path)) {
      const wrap = el('div', 'git-img one'), fig = el('figure', 'git-img-side'), cap = el('figcaption', 'muted');
      cap.dataset.label = 'image'; cap.textContent = 'image · loading…';
      fig.appendChild(cap); wrap.appendChild(fig); box.appendChild(wrap);
      blobImage(rev, f.path, fig, cap, ignored).catch(e => { if (current()) { cap.textContent = 'image · ' + e.message; cap.classList.add('err'); } });
      return;
    }
    box.appendChild(el('div', 'muted', 'loading…'));
    let t;
    const readQuery = 'rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(f.path) + (whole ? '&whole=1' : '') + (ignored ? '&ignored=1' : '');
    try { t = await getJSON('/api/git/text?' + readQuery); }
    catch (e) { if (current()) { box.innerHTML = ''; box.appendChild(el('div', 'err', e.message)); } return; }
    if (!current()) return;
    box.innerHTML = '';
    if (t.language) facts.push(t.language);
    if (t.binary) { note.textContent = facts.join(' · '); box.appendChild(el('div', 'muted gf-note', 'binary file: no text')); return; }
    facts.push(t.lines === 1 ? '1 line' : t.lines + ' lines');
    note.textContent = facts.join(' · ');
    if (t.truncated) {
      const cut = el('div', 'err gf-note', 'the file is cut at ' + Math.round(t.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole file'); all.onclick = () => showFile(f, true);
      cut.appendChild(all); box.appendChild(cut);
    }
    const sourceBox = el('div', 'files-source'); box.appendChild(sourceBox);
    const source = renderLines(splitLines(t.text), t.spans, null);
    sourceBox.appendChild(source);
    const blame = el('button', 'mini files-blame', 'blame');
    blame.setAttribute('aria-pressed', 'false'); head.insertBefore(blame, note);
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
    if (typeof t.rendered_html === 'string') {
      const md = el('div', 'md-view'); md.innerHTML = t.rendered_html; box.appendChild(md);
      for (const image of md.querySelectorAll('img[data-blob-src]')) {
        blobImageData(image.dataset.blobSrc).then(data => {
          if (current()) image.src = data.url;
        }).catch(e => {
          if (current()) image.replaceWith(el('span', 'err', (image.alt || 'image') + ': ' + e.message));
        });
      }
      const modes = el('span', 'gview files-md-mode');
      modes.setAttribute('role', 'group'); modes.setAttribute('aria-label', 'Markdown view');
      head.insertBefore(modes, blame);
      const selectMode = mode => {
        const rendered = mode !== 'source';
        md.hidden = !rendered; sourceBox.hidden = rendered; blame.hidden = rendered;
        for (const button of modes.children) {
          const active = button.textContent === (rendered ? 'rendered' : 'source');
          button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
        }
        if (rendered && anchor) [...md.querySelectorAll('[id]')].find(h => h.id === anchor)?.scrollIntoView({ block: 'start' });
      };
      for (const mode of ['rendered', 'source']) {
        const button = el('button', 'mini', mode);
        button.onclick = () => { remember(MARKDOWN_MODE_KEY, mode); selectMode(mode); };
        modes.appendChild(button);
      }
      selectMode(recall(MARKDOWN_MODE_KEY, 'rendered'));
    }
  };
  await loadTree(path);
}
boot();
