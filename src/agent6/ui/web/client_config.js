// --- config ------------------------------------------------------------------
async function renderConfig(gen) {
  setCrumb('config');
  const [data, registry] = await Promise.all([getJSON('/api/config'), getJSON('/api/config/entry_tables')]);
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Config'));
  // The filter and entry actions share a row, in the TUI config page's words.
  const bar = el('div', 'row cfg-bar');
  const filter = el('input', 'filter'); filter.placeholder = 'filter settings…'; filter.type = 'search';
  bar.appendChild(filter);
  for (const table of registry.tables) {
    const addBtn = el('button', null, 'Add ' + table.label + '…');
    addBtn.onclick = () => addEntry(table);
    bar.appendChild(addBtn);
  }
  const reviewerBtn = el('button', null, 'Add reviewer…');
  reviewerBtn.onclick = () => addReviewer();
  bar.appendChild(reviewerBtn);
  card.appendChild(bar);
  const tbl = el('table', 'cfg');
  const head = el('tr'); ['setting','value','source'].forEach(h => head.appendChild(el('th', null, h))); tbl.appendChild(head);
  const keys = Object.keys(data).sort();
  const rows = [];
  for (const k of keys) {
    const s = data[k];
    const tr = el('tr', s.modified ? 'mod' : '');
    tr.appendChild(el('td', 'key', k));
    tr.appendChild(el('td', 'val', s.display)); // the one shared value column (viewmodel.display_value)
    tr.appendChild(el('td', 'src', esc(s.source)));
    // Hover text: the leaf's meaning (the docs table cell), from the same
    // per-key payload the editor overlay shows it in.
    tr.title = s.description ? plainDescription(s.description) : 'click to edit';
    tr.style.cursor = 'pointer';
    actionable(tr, () => editConfig(k, s, registry.tables), 'edit ' + k);
    tbl.appendChild(tr); rows.push([k.toLowerCase(), tr]);
  }
  card.appendChild(tbl);
  filter.oninput = () => { const q = filter.value.toLowerCase(); for (const [key, tr] of rows) tr.style.display = key.includes(q) ? '' : 'none'; };
  view.appendChild(card);
}
// A leaf's description is the docs table cell: markdown-lite. Backtick spans
// render as code; `**bold**` is dropped (the terminal renderers do the same).
function plainDescription(text) { return String(text).replace(/\*\*/g, ''); }
function descriptionNode(text) {
  const frag = document.createDocumentFragment();
  const parts = plainDescription(text).split('`');
  parts.forEach((part, i) => {
    if (!part) return;
    frag.appendChild(i % 2 ? el('code', null, part) : document.createTextNode(part));
  });
  return frag;
}

// A blank field writes nothing, preserving an existing entry's value.
function addEntry(table) {
  const back = el('div', 'overlay');
  const opener = document.activeElement;
  const box = el('div', 'card'); box.style.width = 'min(560px, 92vw)';
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', 'add ' + table.label);
  const title = el('h2', null, 'Add ' + table.label); title.style.textTransform = 'none';
  box.appendChild(title);
  box.appendChild(el('div', 'sub muted', table.path + '.<name>. A blank field keeps an existing value or uses the default.'));
  const labelled = (text, field) => { const l = el('label', 'sub muted', text); l.style.display = 'block'; l.style.marginTop = '8px'; field.name = text; l.appendChild(field); box.appendChild(l); return field; };
  const name = labelled('name', el('input', 'field'));
  const fields = {};
  for (const spec of table.fields) {
    const field = el(spec.choices.length ? 'select' : 'input', 'field');
    if (spec.choices.length) {
      const values = ['api_format', 'state'].includes(spec.name) ? spec.choices : ['', ...spec.choices];
      for (const value of values) {
        const option = el('option', null, value || 'blank = unchanged, or default');
        option.value = value; field.appendChild(option);
      }
    } else {
      field.placeholder = plainDescription(spec.description);
    }
    field.title = plainDescription(spec.description);
    fields[spec.name] = labelled(spec.name, field);
  }
  if (table.path === 'providers') {
    const format = fields.api_format, baseUrl = fields.base_url;
    let autofilled = '';
    name.oninput = () => {
      const preset = Object.hasOwn(table.defaults, name.value.trim()) ? table.defaults[name.value.trim()] : null;
      if (!preset) {
        if (autofilled && baseUrl.value === autofilled) baseUrl.value = '';
        autofilled = '';
        return;
      }
      format.value = preset.api_format;
      if (baseUrl.value === '' || baseUrl.value === autofilled) { autofilled = preset.base_url || ''; baseUrl.value = autofilled; }
    };
  }
  const repoRow = el('label', 'row'); repoRow.style.marginTop = '8px'; repoRow.style.cursor = 'pointer';
  const repoCb = el('input'); repoCb.type = 'checkbox';
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  box.appendChild(repoRow);
  const row = el('div', 'form-row');
  const add = el('button', 'primary', 'Add'), cancel = el('button', null, 'Cancel');
  row.appendChild(add); row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back);
  const close = () => {
    activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey);
    if (opener && opener.isConnected) opener.focus();
  };
  activeOverlayClose = close;
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  cancel.onclick = close;
  back.onclick = (e) => { if (e.target === back) close(); };
  name.focus();
  add.onclick = async () => {
    add.disabled = true;
    try {
      const d = await postJSON('/api/config/entry', {
        table: table.path, name: name.value,
        fields: Object.fromEntries(Object.entries(fields).map(([key, field]) => [key, field.value])),
        repo: repoCb.checked,
      });
      toast(d.message || 'added'); close(); await renderConfig();
    } catch (e) { toast(e.message, true); add.disabled = false; }
  };
}

async function addReviewer() {
  let choices;
  try { choices = await getJSON('/api/config/reviewer_choices'); } catch (e) { toast(e.message, true); return; }
  const opener = document.activeElement;
  const back = el('div', 'overlay'), box = el('div', 'card');
  box.style.width = 'min(560px, 92vw)';
  box.setAttribute('role', 'dialog'); box.setAttribute('aria-label', 'add reviewer');
  box.appendChild(el('h2', null, 'Add reviewer'));
  box.appendChild(el('div', 'sub muted', 'Focus is required. Set provider and model together, or leave both blank. Blank optional fields inherit the route\'s settings.'));
  const fields = {};
  for (const name of ['focus', 'provider', 'model', 'effort', 'temperature']) {
    const label = el('label', 'sub muted', name); label.style.display = 'block'; label.style.marginTop = '8px';
    const field = el(choices[name] ? 'select' : 'input', 'field');
    field.id = 'reviewer-' + name; label.htmlFor = field.id;
    if (choices[name]) for (const value of ['', ...choices[name]]) {
      const option = el('option', null, value || 'blank = inherit'); option.value = value; field.appendChild(option);
    }
    fields[name] = field; box.appendChild(label); box.appendChild(field);
  }
  const repoRow = el('label', 'row'); repoRow.style.marginTop = '8px';
  const repo = el('input'); repo.type = 'checkbox'; repo.id = 'reviewer-repo';
  repoRow.appendChild(repo); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  box.appendChild(repoRow);
  const row = el('div', 'form-row'), add = el('button', 'primary', 'Add'), cancel = el('button', null, 'Cancel');
  row.appendChild(add); row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back);
  const close = () => {
    activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey);
    if (opener && opener.isConnected) opener.focus();
  };
  activeOverlayClose = close;
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  cancel.onclick = close;
  back.onclick = e => { if (e.target === back) close(); };
  fields.focus.focus();
  add.onclick = async () => {
    add.disabled = true;
    try {
      const body = Object.fromEntries(Object.entries(fields).map(([name, field]) => [name, field.value]));
      const result = await postJSON('/api/config/reviewer', {...body, repo: repo.checked});
      toast(result.message); close(); renderConfig();
    } catch (e) { toast(e.message, true); add.disabled = false; }
  };
}

// A proper editor overlay (not a browser prompt): choices and booleans get a
// select, everything else a text field, with the default, source, and type
// shown; "set for this repo" writes the per-repo config instead of the global.
function editConfig(key, s, tables) {
  const cur = s.input;
  const back = el('div', 'overlay');
  const opener = document.activeElement;
  const box = el('div', 'card'); box.style.width = 'min(560px, 92vw)';
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', 'edit ' + key);
  const title = el('h2', null, key);
  title.style.textTransform = 'none'; // a config key is a case-sensitive identifier
  box.appendChild(title);
  const meta = el('div', 'sub muted');
  meta.textContent = `${esc(s.type)} · default: ${s.default_display} · set from: ${esc(s.source)}` + (s.adaptive ? ' · adaptive' : '');
  meta.style.marginBottom = '10px';
  box.appendChild(meta);
  if (s.description) {
    const desc = el('div', 'cfg-desc');
    desc.appendChild(descriptionNode(s.description));
    box.appendChild(desc);
  }
  let field;
  const choices = s.choices || (s.type === 'bool' ? ['true', 'false'] : null);
  if (choices) {
    field = el('select', 'field');
    for (const c of choices) { const o = el('option', null, c); o.value = c; field.appendChild(o); }
    field.value = cur || String(s.default ?? '');
  } else {
    field = el('input', 'field');
    field.value = cur;
    if (s.type === 'list') field.placeholder = 'TOML list';
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
  // Default the target layer to the value's origin: editing a repo-sourced value
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
    unsetBtn.title = 'remove from the ' + s.source + ' config; the next layer below takes over';
    row.appendChild(unsetBtn);
  }
  const entry = s.entry;
  const table = entry && tables.find(t => t.path === entry.table);
  let removeBtn = null;
  if (table) {
    removeBtn = el('button', null, 'Remove ' + table.label);
    removeBtn.disabled = !['global', 'repo'].includes(entry.source);
    removeBtn.title = removeBtn.disabled
      ? 'This entry comes from ' + entry.source + '; Remove edits only global and repo config.'
      : 'Remove ' + table.label + " '" + entry.name + "' and all its values from " + entry.source + ' config';
    row.appendChild(removeBtn);
  }
  row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back);
  const close = () => {
    activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey);
    if (opener && opener.isConnected) opener.focus();
  };
  activeOverlayClose = close; // navigating away dismisses it
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  cancel.onclick = close;
  back.onclick = (e) => { if (e.target === back) close(); };
  if (key === 'review.reviewers') {
    const roster = el('div');
    const editable = s.source === 'repo' || s.source === 'global';
    roster.appendChild(el('div', 'sub muted', editable
      ? `Remove saves the remaining roster to the ${s.source} config, including an empty list after the last reviewer.`
      : `Remove edits only the global and repo config, not the ${s.source} layer.`));
    (s.value || []).forEach((reviewer, index) => {
      const item = el('div', 'row');
      const label = `${index + 1}: ${reviewer.focus}`;
      item.appendChild(el('span', null, label));
      const remove = el('button', null, 'Remove'); remove.disabled = !editable;
      remove.setAttribute('aria-label', 'Remove reviewer ' + label);
      remove.onclick = async () => {
        if (!confirm(`Remove reviewer ${label} from ${s.source} config?`)) return;
        remove.disabled = true;
        try {
          const result = await postJSON('/api/config/reviewer/remove', {index, repo: s.source === 'repo'});
          toast(result.message); close(); renderConfig();
        } catch (e) { toast(e.message, true); remove.disabled = false; }
      };
      item.appendChild(remove); roster.appendChild(item);
    });
    if (!s.value || !s.value.length) roster.appendChild(el('div', 'sub muted', 'The reviewer roster is empty.'));
    box.insertBefore(roster, field);
  }
  field.focus();
  const submit = async () => {
    save.disabled = true;
    try {
      // A string-valued leaf travels as a TOML string, as the TUI sends it, so
      // `true`, `42` or `[a]` typed into a str field stays a string.
      const value = s.type === 'str' ? JSON.stringify(field.value) : field.value;
      const d = await postJSON('/api/config', { key, value, repo: repoCb.checked });
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
  if (removeBtn) removeBtn.onclick = async () => {
    if (!confirm('Remove ' + table.label + " '" + entry.name + "' and all its values from " + entry.source + ' config? Values from lower layers may become visible.')) return;
    removeBtn.disabled = true;
    try {
      const d = await postJSON('/api/config/entry/remove', {table: entry.table, name: entry.name, repo: entry.source === 'repo'});
      toast(d.message || 'removed'); close(); await renderConfig();
    } catch (e) { toast(e.message, true); removeBtn.disabled = false; }
  };
  field.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } };
}
