// --- files -------------------------------------------------------------------
// Files opens on the working tree unless the link names a revision.
// Folder choices outlive a visit to another page.
// The picker's { rev, label }; each folder path → open; the page's in-place opener, if any.
const filesState = { rev: null, folders: new Map(), open: null, workingTree: null, closePreview: null, previewScroll: null };
const MARKDOWN_MODE_KEY = 'a6-markdown-mode';

function readFileTree(rev, showIgnored = recall('a6-files-ignored', 'off') === 'on') {
  return getJSON('/api/git/tree?rev=' + encodeURIComponent(rev) + (showIgnored && rev === GIT_WORKTREE ? '&ignored=1' : ''));
}
function fileHash(rev, path = '') {
  return '#/files/' + encodeURIComponent(rev) + (path && '/' + path.split('/').map(encodeURIComponent).join('/'));
}
function plainFileClick(e) { return e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey && !e.altKey; }

function closeFilePreview() { filesState.closePreview?.(); }

async function openFilePreview(host, rev, path, ignored, {digest = '', anchor = '', filesPrefix = '', afterSave} = {}) {
  closeFilePreview();
  if (!window.isSecureContext) {
    host.replaceChildren(el('p', 'err gf-note', 'Previews need HTTPS or localhost.'));
    return;
  }
  const note = el('p', 'files-preview-note', 'Opening preview…');
  const frame = el('iframe'); frame.title = 'Preview of ' + path;
  frame.setAttribute('sandbox', ''); frame.referrerPolicy = 'no-referrer';
  host.replaceChildren(note);
  let key = null, closed = false, timer, opened, previewDialog;
  const viewState = (scroll = false) => frame.contentWindow?.postMessage({type: 'preview-state', ...(scroll ? {anchor: anchor || ''} : {}),
    theme: ['light', 'dark'].find(name => document.documentElement.classList.contains(name)) || ''}, '*');
  const theme = new MutationObserver(() => viewState());
  const message = event => {
    if (closed || opened?.kind !== 'markdown' || event.source !== frame.contentWindow) return;
    if (event.data?.type === 'preview-link' && typeof event.data.href === 'string') {
      const href = event.data.href, listed = '#/files/' + encodeURIComponent(opened.rev) + '/';
      if (href.startsWith(listed)) {
        const target = filesPrefix + href.slice(listed.length);
        if (event.data.newTab) window.open(target, '_blank', 'noopener,noreferrer');
        else if (location.hash === target) route(); else location.hash = target;
      } else if (/^https?:\/\//i.test(href)) window.open(href, '_blank', 'noopener,noreferrer');
      return;
    }
    if (event.data?.type === 'preview-save') {
      if (opened.rev !== GIT_WORKTREE || event.data.name !== path.split('/').at(-1) || typeof event.data.text !== 'string') return;
      review('Save checkbox change?', {path, text: event.data.text, digest});
    } else if (event.data?.type === 'preview-remote' && (opened.images.length || opened.media.length)) {
      review('Load remote files?', null);
    }
  };
  const review = (title, proposal) => {
    if (closed || previewDialog?.open) return;
    const dialog = openDialog(title, 'confirm');
    if (!dialog) return;
    previewDialog = dialog;
    const form = opened.kind === 'html';
    if (form) dialog.addEventListener('close', () => { if (!closed) frame.src = opened.url; }, {once: true});
    if (proposal) {
      const verb = proposal.digest === null ? 'Create' : 'Replace';
      dialog.appendChild(el('p', null, `${verb} ${proposal.path} (${new Blob([proposal.text]).size} bytes) in the working tree.`));
    } else {
      dialog.appendChild(el('p', null, 'This page can send anything it can read to these servers. Loading a file sends its full address and your IP address.'));
      const addresses = el('ul', 'files-remote-addresses');
      for (const address of [...opened.images, ...opened.media].sort()) addresses.appendChild(el('li', null, address));
      dialog.appendChild(addresses);
    }
    const row = el('div', 'form-row'), accept = el('button', 'primary', proposal ? 'Save' : 'Load'), cancel = el('button', null, 'Cancel');
    const error = el('p', 'err'); error.setAttribute('role', 'status');
    row.append(accept, cancel); dialog.append(row, error);
    cancel.onclick = () => dialog.close();
    accept.onclick = async () => {
      accept.disabled = true; error.textContent = '';
      try {
        if (proposal) await postJSON('/api/files/text', proposal);
        else await postJSON('/api/files/preview/remote', {key});
        if (closed) return;
        dialog.close();
        if (form) toast('Saved ' + proposal.path);
        else if (proposal) await afterSave(); else frame.src = opened.url;
      } catch (e) { error.textContent = e.message; accept.disabled = false; }
    };
  };
  const close = () => {
    closed = true; clearTimeout(timer); frame.remove(); theme.disconnect(); previewDialog?.close();
    window.removeEventListener('message', message);
    if (filesState.closePreview === close) { filesState.closePreview = null; filesState.previewScroll = null; }
    if (key) {
      const expired = key; key = null;
      postJSON('/api/files/preview/close', {key: expired}, {keepalive: true}).catch(e => toast(e.message, true));
    }
  };
  filesState.closePreview = close;
  const renew = async () => {
    if (closed) return;
    try {
      await postJSON('/api/files/preview/renew', {key});
      if (!closed) timer = setTimeout(renew, 10000);
    } catch (e) {
      if (closed) return;
      close(); host.replaceChildren(el('p', 'err gf-note', e.message + ' Choose Source, then Preview to reopen.'));
    }
  };
  try {
    opened = await postJSON('/api/files/preview', {rev, path, ignored, digest});
    key = opened.key;
    if (closed) { close(); return; }
    if (opened.kind === 'markdown') {
      note.remove(); frame.setAttribute('sandbox', 'allow-scripts');
      window.addEventListener('message', message);
      frame.onload = () => viewState(true);
      theme.observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
      filesState.previewScroll = id => { anchor = id || ''; viewState(true); };
    } else {
      if (opened.rev === GIT_WORKTREE) {
        frame.sandbox.add('allow-forms');
        frame.onload = async () => {
          if (closed) return;
          try {
            const {pending} = await postJSON('/api/files/preview/answers', {key});
            if (pending) review('Save form answers?', pending);
          } catch (e) { if (!closed) toast(e.message, true); }
        };
      }
      const run = el('button', null, 'Run scripts');
      note.replaceChildren(el('span', null, 'Scripts are off for this page.'), ' ', run);
      run.onclick = async () => {
        // The grant covers the page's folder and everything under it: the whole listing for a top-level page.
        const folder = path.slice(0, path.lastIndexOf('/') + 1);
        const reach = folder ? 'every file under ' + folder : 'every file Files shows in this repository';
        const warning = "This page's scripts can read " + reach + ' and send them to any server. The browser cannot stop that.';
        if (!await confirmDialog('Run scripts?', warning, 'Run scripts', true) || closed) return;
        run.disabled = true;
        try {
          await postJSON('/api/files/preview/scripts', {key});
          if (closed) return;
          frame.sandbox.add('allow-scripts'); frame.src = opened.url;
          note.textContent = 'Scripts are on for this page.';
        } catch (e) { if (!closed) { toast(e.message, true); run.disabled = false; } }
      };
    }
    frame.src = opened.url; host.appendChild(frame);
    timer = setTimeout(renew, 10000);
  } catch (e) {
    if (!closed) { close(); host.replaceChildren(el('p', 'err gf-note', e.message)); }
  }
}

async function editFileText(path, afterSave) {
  const dialog = openDialog('Edit ' + path, 'wide files-editor');
  if (!dialog) return;
  dialog.setAttribute('closedby', 'closerequest');
  const field = (name, control, parent) => {
    const label = el('label', null, name); label.appendChild(control); parent.appendChild(label); return control;
  };
  const fields = el('div', 'files-fields'); dialog.appendChild(fields);
  const text = field('File text', el('textarea', 'field file-text'), fields); text.spellcheck = false; text.disabled = true;
  const conflict = el('section', 'files-conflict'); conflict.hidden = true; fields.appendChild(conflict);
  conflict.appendChild(el('p', null, 'The file changed on disk.'));
  const disk = field('Disk version', el('textarea', 'field file-disk'), conflict); disk.readOnly = true; disk.rows = 3;
  const load = el('button', null, 'Load new file'), overwrite = el('button', null, 'Overwrite with mine');
  const copyAs = el('div', 'row wrap files-copy');
  const name = field('New file name', el('input', 'field'), copyAs); name.placeholder = 'copy.txt';
  const copy = el('button', null, 'Save as new file'); copyAs.appendChild(copy);
  const row = el('footer', 'row wrap'), save = el('button', 'primary', 'Save'), cancel = el('button', null, 'Cancel');
  row.append(save, cancel); dialog.appendChild(row);
  const error = el('p', 'err'); error.setAttribute('role', 'status'); dialog.appendChild(error);
  let loaded = null, latest = null, busy = false, timer;
  const read = () => getJSON('/api/files/text?path=' + encodeURIComponent(path));
  const buttons = () => {
    save.disabled = busy || !loaded;
    load.disabled = busy || latest?.text == null;
    overwrite.disabled = copy.disabled = busy;
    text.readOnly = busy;
  };
  const changed = snapshot => {
    latest = snapshot;
    disk.value = snapshot.text ?? 'The file was removed.';
    conflict.hidden = false; row.replaceChildren(load, overwrite, copyAs, cancel); buttons();
  };
  const poll = async () => {
    try {
      if (!busy && loaded && conflict.hidden) {
        const snapshot = await read();
        if (dialog.open && !busy && snapshot.digest !== loaded.digest) changed(snapshot);
      }
    } catch (e) { if (dialog.open) error.textContent = e.message; }
    if (dialog.open) timer = setTimeout(poll, 2000);
  };
  const write = async (destination, digest) => {
    busy = true; buttons(); error.textContent = '';
    try {
      const response = await serverFetch('/api/files/text', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({path: destination, text: text.value, digest})});
      const result = await response.json();
      if (!dialog.open) return;
      if (!response.ok) {
        if (response.status === 409 && destination === path) changed(await read());
        throw new Error(result.error || 'Save failed');
      }
      dialog.close();
      await afterSave(destination);
    } catch (e) { if (dialog.open) error.textContent = e.message; else toast(e.message, true); }
    finally { busy = false; buttons(); }
  };
  load.onclick = () => {
    loaded = latest; text.value = loaded.text; conflict.hidden = true; error.textContent = '';
    row.replaceChildren(save, cancel); buttons(); text.focus();
  };
  overwrite.onclick = () => write(path, latest.digest);
  copy.onclick = () => {
    const value = name.value;
    if (!value || value.startsWith('.') || /[/\\\x00-\x1f\x7f]/.test(value)) {
      error.textContent = 'Use a new file name in this folder, without slashes or a leading dot.'; name.focus(); return;
    }
    return write(path.slice(0, path.lastIndexOf('/') + 1) + value, null);
  };
  save.onclick = () => write(path, loaded.digest);
  cancel.onclick = () => { if (!busy) dialog.close(); };
  dialog.addEventListener('cancel', e => {
    if (busy || (loaded && text.value !== loaded.text)) {
      e.preventDefault(); error.textContent = busy ? 'Saving…' : 'Unsaved edits. Cancel discards them.';
    }
  });
  dialog.querySelector('.dialog-head button').onclick = cancel.onclick;
  dialog.addEventListener('close', () => clearTimeout(timer));
  buttons();
  try {
    loaded = await read();
    if (!dialog.open) return;
    if (loaded.text === null) throw new Error('The file was removed. Close the editor and refresh the tree.');
    text.value = loaded.text; text.disabled = false; buttons(); text.focus();
    timer = setTimeout(poll, 2000);
  } catch (e) { if (dialog.open) { loaded = null; buttons(); error.textContent = e.message; } }
}

async function renderFiles(gen, revision, path = '', query = new URLSearchParams()) {
  if (query.has('panel')) return renderReviewedFile(gen, path, query);
  revision ||= GIT_WORKTREE;
  const inPlace = revision === filesState.rev?.rev && filesState.open?.(path, query);
  if (inPlace) return inPlace;
  closeFilePreview();
  const refsData = await getJSON('/api/git/refs');
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const app = el('div', 'files-app'); view.appendChild(app);
  // A revision from the hash reads as the picker names it: a ref by its name, a full sha short.
  if (revision === GIT_WORKTREE) filesState.rev = GIT_WORKTREE_PICK;
  else if (revision !== filesState.rev?.rev) {
    const ref = refsData.refs.find(r => r.full === revision), sha = /^([0-9a-f]{40}|[0-9a-f]{64})$/.test(revision);
    filesState.rev = { rev: revision, label: ref ? ref.name : sha ? revision.slice(0, 7) : revision };
  }
  // The hash names the revision picked, so a reload keeps HEAD or a branch; the reads use the
  // commit its listing resolved to.
  const filesHash = file => fileHash(filesState.rev.rev, file);

  // -- toolbar: the revision picker, the filter, the count ---------------------------
  const bar = el('div', 'files-bar page-pad row wrap'); app.appendChild(bar);
  const listHost = el('div', 'gpick-host');
  const pick = gitPicker('revision', filesState.rev, (v) => { filesState.rev = v; setViewHash(filesHash()); relist(); }, listHost, [GIT_WORKTREE_PICK]);
  pick.root.appendChild(listHost); // the list opens under the field it searches
  const filter = el('input', 'filter'); filter.type = 'search'; filter.placeholder = 'filter files…'; filter.setAttribute('aria-label', 'filter files');
  let showIgnored = recall('a6-files-ignored', 'off') === 'on';
  const ignoredToggle = el('button', 'files-ignored', 'Show ignored');
  ignoredToggle.title = 'Include files excluded by any Git ignore source';
  ignoredToggle.setAttribute('aria-pressed', String(showIgnored));
  ignoredToggle.onclick = () => {
    showIgnored = !showIgnored; remember('a6-files-ignored', showIgnored ? 'on' : 'off');
    ignoredToggle.setAttribute('aria-pressed', String(showIgnored));
    return relist();
  };
  const refresh = el('button', 'files-refresh', 'Refresh');
  refresh.onclick = () => relist();
  bar.append(pick.root, refresh, fileContentSearch(() => rev || filesState.rev.rev), filter, ignoredToggle);
  const body = el('div', 'files-body'); app.appendChild(body);
  const tree = el('div', 'files-tree'); tree.id = 'files-tree'; body.appendChild(tree);
  body.appendChild(resizeHandle(tree, {
    label: 'File tree width', key: 'a6-files-tree-w', property: '--files-tree-w',
    min: rem => 11.25 * rem, max: rem => body.getBoundingClientRect().width - 15.375 * rem,
  }));
  const pane = el('div', 'files-view'); body.appendChild(pane);
  // With no file open and rows to click, the pane says how to open one: from the tree, or by showing it.
  const unopened = el('div', 'empty');
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
  filesState.workingTree = () => app.isConnected && rev === GIT_WORKTREE && {files: entries, truncated};
  // The count heads the tree it counts, so its arrival neither moves nor narrows a field on the bar.
  const counts = el('div', 'muted files-counts');
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
    const linked = f.path.isWellFormed(), note = linked ? '' : ' (path is not UTF-8)';
    const row = el(linked ? 'a' : 'div', 'gf'); row.dataset.path = f.path;
    row.setAttribute('aria-label', f.path + note);
    row.setAttribute('aria-selected', String(shown === f.path));
    const mark = stateMark([f]); if (mark) row.appendChild(mark);
    row.appendChild(el('span', 'gf-path', f.path.slice(f.path.lastIndexOf('/') + 1) + note));
    row.appendChild(el('span', 'gf-counts muted', f.kind === 'file' ? fmtBytes(f.size) : f.kind));
    if (!linked) { row.setAttribute('aria-disabled', 'true'); row.title = f.path + note; return row; }
    row.href = filesHash(f.path);
    row.onclick = e => {
      if (!plainFileClick(e)) return;
      e.preventDefault(); return showFile(f);
    };
    row.onkeydown = e => { if (e.key === ' ') { e.preventDefault(); row.click(); } };
    return row;
  };
  const folderNote = (d) => { const under = treeFiles(d); let size = 0; for (const f of under) size += f.size; return fileCount(under.length) + ' · ' + fmtBytes(size); };
  const paintTree = () => {
    tree.innerHTML = '';
    const q = filter.value.trim().toLowerCase();
    const kept = q ? entries.filter(f => f.path.toLowerCase().includes(q)) : entries;
    unopened.hidden = !kept.length;
    counts.textContent = (q ? kept.length + ' of ' : '') + fileCount(entries.length) + (truncated ? '+' : '');
    if (entries.length) tree.appendChild(counts);
    if (truncated) tree.appendChild(el('div', 'err gf-note', 'the list is cut: more files than are shown'));
    // A large tree starts folded and keeps the folders opened by hand; a filter opens every folder holding a match.
    if (!kept.length) tree.appendChild(el('div', 'empty', entries.length ? 'no file matches' : 'no files'));
    else tree.appendChild(treeView(buildTree(kept), fileRow, folderNote, !q && kept.length > 200, d => stateMark(treeFiles(d)), q ? null : filesState.folders));
  };
  const closeFile = () => {
    closeFilePreview();
    pane.innerHTML = ''; if (entries.length) pane.appendChild(unopened); body.classList.remove('viewing');
    shown = null; for (const r of tree.querySelectorAll('.gf[aria-selected=true]')) r.setAttribute('aria-selected', 'false');
  };
  // A failed read says why, and offers to read again unless the server refused it with a 4xx.
  const failure = (cls, e, retry) => {
    const note = el('div', cls, e.message);
    if (!(e.status < 500)) { const again = el('button', 'mini', 'Try again'); again.onclick = retry; note.append(' ', again); }
    return note;
  };
  // A new listing (another revision, ignored files shown or hidden) keeps the open file open
  // when it still lists it; the file stays on screen until the listing answers, and the hash
  // names it again only then.
  const relist = () => {
    query = new URLSearchParams(); setViewHash(filesHash(), true);
    return loadTree(body.classList.contains('viewing') ? shown : '', true);
  };
  const loadTree = async (openPath = '', relisting = false) => {
    closeFilePreview();
    const n = ++loading;
    ignoredToggle.hidden = filesState.rev.rev !== GIT_WORKTREE;
    pane.dataset.key = ''; shown = null; rev = ''; ignored = false; entries = [];
    if (!relisting) closeFile();
    tree.innerHTML = ''; tree.appendChild(el('div', 'empty', 'loading…'));
    try {
      const d = await readFileTree(filesState.rev.rev, showIgnored);
      if (n !== loading || !app.isConnected) return;
      entries = d.files; truncated = d.truncated; rev = d.rev; ignored = !!d.ignored;
    } catch (e) { if (n === loading && app.isConnected) { closeFile(); tree.innerHTML = ''; tree.appendChild(failure('empty err', e, () => loadTree(openPath, relisting))); } return; }
    const file = entries.find(f => f.path === openPath) || (openPath && !relisting ? { path: openPath, kind: 'file', size: null } : null);
    if (file) { reveal(file.path); shown = file.path; }
    paintTree();
    if (file) await showFile(file, false, true); else closeFile();
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
    return path.isWellFormed() && entries.some(e => e.path === path) ? path : null;
  };
  // Open a file's folders, so its row shows in every later paint of the tree.
  const reveal = (path) => {
    const parts = path.split('/');
    for (let i = 1; i < parts.length; i++) filesState.folders.set(parts.slice(0, i).join('/'), true);
  };

  // -- one file: it takes the right pane, or the whole screen on a phone -------------
  let viewing = 0;
  const showFile = async (f, whole, linked = false) => {
    closeFilePreview();
    const key = JSON.stringify([++viewing, rev, f.path, !!whole]);
    pane.dataset.key = key;
    const current = () => pane.dataset.key === key && pane.isConnected;
    shown = f.path;
    const anchor = linked ? query.get('anchor') : null;
    const number = linked ? Number(query.get('line')) : 0;
    const line = Number.isSafeInteger(number) && number > 0 ? number : 0;
    const at = line ? '?line=' + line : anchor ? '?' + new URLSearchParams({anchor}) : '';
    setViewHash(filesHash(f.path) + at, linked);
    reveal(f.path);
    for (const r of tree.querySelectorAll('.gf')) {
      r.setAttribute('aria-selected', String(r.dataset.path === f.path));
      if (r.dataset.path === f.path) r.scrollIntoView({ block: 'nearest' });
    }
    body.classList.add('viewing');
    pane.innerHTML = '';
    const head = el('div', 'files-head row');
    const back = el('button', 'mini files-back'); back.append(icon('back'), ' Files');
    back.onclick = () => { closeFilePreview(); body.classList.remove('viewing'); tree.querySelector('[aria-selected=true]')?.focus(); };
    head.appendChild(back);
    const refreshFile = el('button', 'mini files-refresh', 'Refresh');
    refreshFile.onclick = () => relist();
    head.appendChild(refreshFile);
    // A long path wraps after a slash, not inside a name.
    const pathText = el('span', 'gf-path'); head.appendChild(pathText);
    breakable(pathText, f.path, '/');
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
    if (!line && isImagePath(f.path)) {
      // The image replaces "loading…" once loaded; one the browser cannot show becomes a note.
      const loading = el('div', 'muted gf-note', 'loading…'); box.appendChild(loading);
      blobImageData('/api/git/blob?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(f.path) + (ignored ? '&ignored=1' : '')).then(image => {
        if (!current()) return;
        if (!facts.length) facts.push(fmtBytes(image.size));
        const wrap = el('div', 'git-img one'), fig = el('figure', 'git-img-side'), img = el('img');
        img.alt = f.path; fig.appendChild(img); wrap.appendChild(fig);
        img.onload = () => {
          // An SVG states no pixel size: a browser guesses one (150 or 300 square) where it has none.
          if (!/\.svg$/i.test(f.path)) facts.push(img.naturalWidth + '\u00d7' + img.naturalHeight);
          note.textContent = facts.join(' · '); loading.replaceWith(wrap);
        };
        img.onerror = () => { note.textContent = facts.join(' · '); loading.replaceWith(el('div', 'muted gf-note', 'the browser cannot show this image')); };
        img.src = image.url;
      }).catch(e => { if (current()) loading.replaceWith(failure('err gf-note', e, () => showFile(f, whole, linked))); });
      return;
    }
    box.appendChild(el('div', 'muted gf-note', 'loading…'));
    // The buttons hold their places in the head while the text loads: Blame, and Rendered and
    // Source for a name the server renders as Markdown (.md or .markdown, as text_payload in
    // git_model.py decides).
    const blame = el('button', 'mini files-blame', 'Blame');
    blame.setAttribute('aria-pressed', 'false'); blame.disabled = true; head.insertBefore(blame, note);
    const rendersMarkdown = /\.(md|markdown)$/i.test(f.path);
    const modes = el('span', 'gview files-md-mode'), sourceBox = el('div', 'files-source');
    const markdownMode = line ? 'source' : recall(MARKDOWN_MODE_KEY, 'rendered');
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
      if (rendered) openFilePreview(md, rev, f.path, ignored, {digest: t.digest, anchor, filesPrefix: filesHash() + '/', afterSave: () => current() && loadTree(f.path)});
      else closeFilePreview();
    };
    if (rendersMarkdown) {
      modes.setAttribute('role', 'group'); modes.setAttribute('aria-label', 'Markdown view');
      for (const [mode, label] of [['rendered', 'Rendered'], ['source', 'Source']]) {
        const button = el('button', 'mini', label); button.dataset.mode = mode; button.disabled = true;
        button.onclick = () => { remember(MARKDOWN_MODE_KEY, mode); selectMode(mode); };
        modes.appendChild(button);
      }
      head.insertBefore(modes, blame);
      selectMode(markdownMode);
    }
    let t;
    const readQuery = 'rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(f.path) + (whole ? '&whole=1' : '') + (ignored ? '&ignored=1' : '');
    try { t = await getJSON('/api/git/text?' + readQuery); }
    catch (e) { if (current()) { blame.remove(); modes.remove(); box.innerHTML = ''; box.appendChild(failure('err gf-note', e, () => showFile(f, whole, linked))); } return; }
    if (!current()) return;
    if (!whole && t.truncated && line > t.lines) return showFile(f, true, linked);
    box.innerHTML = '';
    if (t.language) facts.push(t.language);
    if (t.binary) {
      // Blame stays, disabled, so the head keeps the shape it had while the file loaded.
      blame.title = 'a binary file has no lines to blame'; modes.remove(); blame.hidden = false;
      note.textContent = facts.join(' · '); box.appendChild(el('div', 'muted gf-note', 'binary file: no text')); return;
    }
    if (rev === GIT_WORKTREE && !f.path.split('/').some(part => part.startsWith('.'))) {
      const edit = el('button', 'mini', 'Edit'); edit.setAttribute('aria-haspopup', 'dialog');
      edit.onclick = () => editFileText(f.path, savedPath => current() && loadTree(savedPath));
      head.insertBefore(edit, note);
    }
    facts.push(t.lines === 1 ? '1 line' : t.lines + ' lines');
    note.textContent = facts.join(' · ');
    if (t.truncated) {
      const cut = el('div', 'err gf-note', 'the file is cut at ' + Math.round(t.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole file'); all.onclick = () => showFile(f, true, linked);
      cut.appendChild(all); box.appendChild(cut);
    }
    box.appendChild(sourceBox);
    const source = renderLines(splitLines(t.text), t.spans, null);
    sourceBox.appendChild(source);
    if (line) {
      const row = source.querySelector(`[data-line="${line}"]`);
      if (row) {
        row.setAttribute('aria-current', 'location'); row.tabIndex = -1;
        row.focus({preventScroll: true}); row.scrollIntoView({block: 'center'});
      } else sourceBox.prepend(el('div', 'err gf-note', `Line ${line} is not in this file.`));
    }
    blame.disabled = false;
    let blaming = false, blameRequest = 0;
    // The button reads busy while blame loads, so the code holds still; a note says why it failed.
    const failed = (reason) => sourceBox.insertBefore(el('div', 'err gf-note blame-note', reason), source);
    blame.onclick = async () => {
      const request = ++blameRequest;
      blaming = !blaming;
      blame.setAttribute('aria-pressed', String(blaming)); blame.classList.toggle('active', blaming);
      blame.setAttribute('aria-busy', String(blaming));
      clearBlame(source);
      sourceBox.querySelectorAll('.blame-note').forEach(c => c.remove());
      if (!blaming) return;
      try {
        const d = await getJSON('/api/git/blame?' + readQuery);
        if (!current() || request !== blameRequest) return;
        blame.setAttribute('aria-busy', 'false');
        if (d.digest !== t.digest) { failed('the file changed since it was opened: open it again to blame'); return; }
        const lines = new Map(d.lines.map(l => [l.line, l]));
        renderBlame([...source.querySelectorAll('[data-line]')].map(node => ({
          node, line: lines.get(Number(node.dataset.line)), side: 'head', path: f.path,
        })));
      } catch (e) { if (current() && request === blameRequest) { blame.setAttribute('aria-busy', 'false'); failed(e.message); } }
    };
    if (/\.html?$/i.test(f.path)) {
      const preview = el('div', 'files-preview'), choices = el('span', 'gview');
      choices.setAttribute('role', 'group'); choices.setAttribute('aria-label', 'HTML view');
      box.appendChild(preview); head.insertBefore(choices, blame);
      const select = mode => {
        const active = mode === 'preview';
        preview.hidden = !active; sourceBox.hidden = active; blame.hidden = active;
        for (const button of choices.children) {
          const selected = button.dataset.mode === mode;
          button.classList.toggle('active', selected); button.setAttribute('aria-pressed', String(selected));
        }
        if (active) openFilePreview(preview, rev, f.path, ignored); else closeFilePreview();
      };
      for (const [mode, label] of [['source', 'Source'], ['preview', 'Preview']]) {
        const button = el('button', 'mini', label); button.dataset.mode = mode;
        button.onclick = () => select(mode); choices.appendChild(button);
      }
      select('source');
    }
    if (!rendersMarkdown) return;
    // A Markdown file cut at the size cap has no rendering: its text alone shows.
    if (t.truncated) { modes.remove(); blame.hidden = false; return; }
    md = el('div', 'files-preview'); box.appendChild(md);
    for (const button of modes.children) button.disabled = false;
    selectMode(markdownMode);
  };
  // Back, Forward or a link opens another listed file, scrolls the Markdown shown, or closes the file.
  filesState.open = (file, hashQuery) => {
    if (!app.isConnected || !rev) return null; // no page, or no listing yet: render anew
    app.removeAttribute('aria-busy'); // route() marked the page it expected to replace
    query = hashQuery;
    if (!file) { closeFile(); return true; }
    if (file === shown && filesState.previewScroll && !query.has('line')) {
      body.classList.add('viewing'); filesState.previewScroll(query.get('anchor')); return true;
    }
    shown = file; reveal(file);
    if (![...tree.querySelectorAll('.gf')].find(r => r.dataset.path === file)?.checkVisibility()) paintTree();
    return showFile(entries.find(f => f.path === file) || { path: file, kind: 'file', size: null }, false, true);
  };
  await loadTree(path);
}
window.addEventListener('pagehide', closeFilePreview);
window.addEventListener('pageshow', e => { if (e.persisted) route(); });
