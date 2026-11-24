// A listbox option cannot contain buttons; a picker with row actions uses a grid.
function openPicker({ title, placeholder, rows, haystack, paint, pick, empty = '', initial = '', prefixes, listStatus, limit = 12, typed = false, actions = null }) {
  const dialog = openDialog(title, 'wide picker');
  if (!dialog) return;
  const name = dialog.getAttribute('aria-labelledby');
  const aria = (e, attrs) => { for (const k in attrs) e.setAttribute(k, attrs[k]); };
  const popup = actions ? 'grid' : 'listbox';
  const field = el('input', 'field'); field.placeholder = placeholder; field.value = initial;
  const list = el('div', 'hs-list');
  list.id = name + '-list';
  list.tabIndex = -1;
  aria(list, { role: popup, 'aria-labelledby': name });
  aria(field, { role: 'combobox', 'aria-labelledby': name, 'aria-controls': list.id, 'aria-expanded': 'true', 'aria-autocomplete': 'list', 'aria-haspopup': popup });
  const note = el('div', 'more-note');
  note.setAttribute('role', 'status');
  dialog.append(field, list, note);
  const close = () => dialog.close();
  const choose = (r) => { close(); pick(r); };
  let loaded = null, items = [], active = 0;
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
    const all = (loaded || []).filter(r => !q || haystack(r).includes(q));
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
    const status = listStatus ? listStatus(all.length) : items.length ? '' : loaded?.length ? '(no match)' : empty;
    note.textContent = !loaded ? 'loading…' : [status, more && '… ' + more + ' more (type to narrow)'].filter(Boolean).join(' · ');
  };
  const move = (step) => { if (items.length) { active = (active + step + items.length) % items.length; mark(true); } };
  // Leave Enter and Space to a focused row button.
  dialog.onkeydown = (e) => {
    if (composing(e) || e.target.closest('button')) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
    else if (e.key === 'Enter') { e.preventDefault(); if (loaded && (items.length || typed)) choose(items.length ? items[active] : field.value); }
  };
  field.oninput = () => {
    const open = prefixes?.[field.value[0]];
    if (open) { const rest = field.value.slice(1); close(); open(rest); return; }
    active = 0; render();
  };
  field.focus();
  render();
  Promise.resolve().then(rows).then(
    (r) => { if (dialog.open) { loaded = r; render(); } },
    (err) => { if (dialog.open) { close(); toast(err.message, true); } });
}

function openSwitcher() {
  const flatten = (r) => [r, ...(r.children || []).flatMap(flatten)];
  openPicker({
    title: 'Switch session',
    placeholder: '/ for files, > for commands',
    rows: () => getJSON('/api/hub').then(data => (data.sessions || []).flatMap(flatten)),
    empty: '(no sessions yet)',
    prefixes: { '>': openPalette, '/': openFilePicker },
    haystack: (r) => [r.session_id, r.mode, r.status_label, r.task_line || r.task || '', r.branch || '', r.worktree || ''].join(' ').toLowerCase(),
    pick: (r) => { location.hash = '#/session/' + encodeURIComponent(r.session_id); },
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
      const review = el('button', 'mini', 'Review');
      review.onclick = () => { close(); toNewWork({ task: '/review ' + r.branch }); };
      acts.appendChild(review);
      if (r.unmerged) {
        const merge = el('button', 'mini', 'Merge');
        merge.onclick = async () => {
          close();
          if (!await confirmMerge(r)) return;
          await postMerge(r);
        };
        acts.appendChild(merge);
      }
      return acts;
    },
  });
}

function openFilePicker(initial) {
  let cut = false;
  openPicker({
    title: 'Go to file: working tree', placeholder: 'part of a file path', initial,
    rows: async () => {
      const d = filesState.workingTree?.() || await readFileTree(GIT_WORKTREE);
      cut = d.truncated;
      return d.files;
    },
    listStatus: count => [cut && 'file list is cut', !count && 'no match in the listed files'].filter(Boolean).join(' · '),
    haystack: f => f.path.toLowerCase(),
    paint: (f, o) => { o.textContent = f.path + (f.path.isWellFormed() ? '' : ' (path is not UTF-8)'); },
    pick: f => {
      if (!f.path.isWellFormed()) { toast('This path is not UTF-8 and has no Files link.', true); return; }
      const hash = fileHash(GIT_WORKTREE, f.path);
      if (location.hash === hash) route(); else location.hash = hash;
    },
  });
}

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
    { label: 'New session', hint: 'the Sessions composer', run: () => toNewWork({}) },
  ];
  for (const b of view.querySelectorAll('.toolbar button, .actions button')) {
    if (b.matches(':disabled, [aria-disabled=true]') || b.offsetParent === null) continue;
    items.push({ label: b.textContent.replace(/^[■▶]\s*/, ''), hint: 'this page', run: () => b.click() });
  }
  return items;
}
function openPalette(initial = '') {
  openPicker({
    title: 'Commands',
    placeholder: 'type a command…',
    rows: paletteItems,
    limit: Infinity,
    empty: '(no commands)',
    initial,
    haystack: (c) => (c.label + ' ' + c.hint).toLowerCase(),
    pick: (c) => c.run(),
    paint: (c, o) => { o.appendChild(el('span', null, c.label)); o.appendChild(el('span', 'muted', ' · ' + c.hint)); },
  });
}

// IME confirmation is typing; some engines mark it only with keyCode 229.
function composing(e) { return e.isComposing || e.keyCode === 229; }
function inField(e) { const t = e.target; return !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)); }
// A touch keyboard may have no Shift+Enter for a newline.
function enterSends(e) { return e.key === 'Enter' && !e.shiftKey && !composing(e) && (e.ctrlKey || !matchMedia('(pointer: coarse)').matches); }
function enterHint(keys, touch) { return [el('span', 'on-keys', keys), el('span', 'on-touch', touch)]; }
const KEYS = [
  { label: 'Ctrl+K', where: 'Anywhere', does: 'switch session (/ for working-tree files, > for commands)', match: (e) => e.key.toLowerCase() === 'k' && (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey, run: () => openSwitcher() },
  { label: '?', where: 'Outside a field', does: 'this list', match: (e) => e.key === '?' && !inField(e) && !e.ctrlKey && !e.metaKey && !e.altKey, run: () => openKeys() },
  { label: 'Esc', where: 'In a dialog or popup', does: 'close it' },
  // Let a nonempty search field consume Escape to clear itself.
  { label: 'Esc', where: 'Outside a dialog or popup', does: 'dismiss the newest message', match: (e) => e.key === 'Escape' && !e.defaultPrevented && !composing(e) && !(e.target.matches?.('input[type=search]') && e.target.value) && !document.querySelector('dialog[open], :popover-open') && !!newestHeld(), run: () => newestHeld().remove() },
  { label: 'Enter', where: 'In the composer', does: 'send the instruction; resume an ended session' },
  { label: 'Shift+Enter', where: 'In the composer', does: 'newline' },
  { label: '/', where: 'In the composer', does: 'directive completion; Tab or Enter accepts' },
  { label: 'Ctrl+Enter', where: "In a session's composer", does: 'send now, interrupting the call in flight (/now)' },
  { label: 'Ctrl+R', where: "In a session's composer", does: "search this session's past messages" },
  { label: '↑ ↓', where: 'In a list', does: 'move' },
  { label: 'Home End', where: 'In a list on the page', does: 'first, last' },
  { label: 'Enter or Space', where: 'On a focused row', does: 'open it' },
];
document.addEventListener('keydown', (e) => {
  for (const k of KEYS) if (k.run && k.match(e)) { e.preventDefault(); k.run(); return; }
});
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
  // Closing with ? must not trigger the page binding that reopens the dialog.
  dialog.onkeydown = (e) => { if (e.key === '?') { e.preventDefault(); e.stopPropagation(); dialog.close(); } };
}
