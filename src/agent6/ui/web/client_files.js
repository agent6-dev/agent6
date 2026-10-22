// --- files -------------------------------------------------------------------
// The Files tab: the files at one revision (HEAD to start, the working tree, or any
// revision the picker finds) as a folder tree with a filter, and one file read-only: text
// with line numbers and highlighting, an image, or a binary file's size and kind. The
// picked revision outlives a visit to another tab; the tree is fetched again on return.
const filesState = { rev: null }; // the picker's { rev, label }; null until the first visit
async function renderFiles(gen) {
  setCrumb('files');
  const refsData = await getJSON('/api/git/refs');
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const app = el('div', 'files-app'); view.appendChild(app);
  if (!filesState.rev) filesState.rev = refsData.head.sha ? { rev: 'HEAD', label: 'HEAD' } : GIT_WORKTREE_PICK;

  // -- toolbar: the revision picker, the filter, the count ---------------------------
  const bar = el('div', 'files-bar page-pad row wrap'); app.appendChild(bar);
  const listHost = el('div', 'gpick-host page-pad');
  const pick = gitPicker('revision', filesState.rev, (v) => { filesState.rev = v; loadTree(); }, listHost, [GIT_WORKTREE_PICK]);
  const filter = el('input', 'filter'); filter.type = 'search'; filter.placeholder = 'filter files…'; filter.setAttribute('aria-label', 'filter files');
  const counts = el('span', 'muted files-counts');
  bar.appendChild(pick.root); bar.appendChild(filter); bar.appendChild(counts);
  app.appendChild(listHost);
  const body = el('div', 'files-body'); app.appendChild(body);
  const tree = el('div', 'files-tree'); body.appendChild(tree);
  const pane = el('div', 'files-view'); body.appendChild(pane);
  pane.appendChild(el('div', 'empty', 'click a file to read it'));

  // -- the tree ----------------------------------------------------------------------
  let entries = [], truncated = false, rev = '', loading = 0, shown = null;
  const fileRow = (f) => {
    const row = el('div', 'gf' + (shown === f.path ? ' sel' : ''));
    row.dataset.path = f.path;
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
    else tree.appendChild(treeView(buildTree(kept), fileRow, folderNote, !q && kept.length > 200)); // a large tree starts folded
    counts.textContent = (q ? kept.length + ' of ' : '') + fileCount(entries.length) + (truncated ? '+' : '');
  };
  const loadTree = async () => {
    const n = ++loading;
    tree.innerHTML = ''; tree.appendChild(el('div', 'muted page-pad', 'loading…'));
    try {
      const d = await getJSON('/api/git/tree?rev=' + encodeURIComponent(filesState.rev.rev));
      if (n !== loading || !app.isConnected) return;
      entries = d.files; truncated = d.truncated; rev = d.rev;
    } catch (e) { if (n === loading && app.isConnected) { tree.innerHTML = ''; tree.appendChild(el('div', 'err page-pad', e.message)); } return; }
    paintTree();
  };
  filter.oninput = paintTree;

  // -- one file: it takes the right pane, or the whole screen on a phone -------------
  let viewing = 0;
  const showFile = async (f, whole) => {
    const n = ++viewing;
    shown = f.path;
    tree.querySelectorAll('.gf').forEach(r => r.classList.toggle('sel', r.dataset.path === f.path));
    body.classList.add('viewing');
    pane.innerHTML = '';
    const head = el('div', 'files-head row');
    const back = el('button', 'mini files-back', '‹ files'); back.onclick = () => body.classList.remove('viewing');
    head.appendChild(back);
    head.appendChild(el('span', 'gf-path', f.path));
    const facts = [f.kind === 'file' ? fmtBytes(f.size) : f.kind];
    const note = el('span', 'muted files-note', facts.join(' · '));
    head.appendChild(note);
    pane.appendChild(head);
    const box = el('div', 'files-text'); pane.appendChild(box);
    if (f.kind !== 'file') {
      box.appendChild(el('div', 'muted gf-note', f.kind === 'symlink' ? 'a symlink: git stores its target, not a file' : 'a submodule: its files are in another repository'));
      return;
    }
    if (isImagePath(f.path)) {
      const wrap = el('div', 'git-img one'), fig = el('figure', 'git-img-side'), cap = el('figcaption', 'muted');
      cap.dataset.label = 'image'; cap.textContent = 'image · loading…';
      fig.appendChild(cap); wrap.appendChild(fig); box.appendChild(wrap);
      blobImage(rev, f.path, fig, cap).catch(e => { cap.textContent = 'image · ' + e.message; cap.classList.add('err'); });
      return;
    }
    box.appendChild(el('div', 'muted', 'loading…'));
    let t;
    try { t = await getJSON('/api/git/text?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(f.path) + (whole ? '&whole=1' : '')); }
    catch (e) { if (n === viewing && pane.isConnected) { box.innerHTML = ''; box.appendChild(el('div', 'err', e.message)); } return; }
    if (n !== viewing || !pane.isConnected) return;
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
    box.appendChild(renderLines(splitLines(t.text), t.spans, null));
  };
  await loadTree();
}
boot();
