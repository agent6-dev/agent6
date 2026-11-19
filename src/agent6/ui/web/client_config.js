// --- config ------------------------------------------------------------------
async function renderConfig(gen = routeGen, readState = () => ({})) {
  if (gen !== routeGen) return;
  const state = readState();
  const [data, registry] = await Promise.all([
    getJSON('/api/config'),
    getJSON('/api/config/entry_tables'),
  ]);
  if (gen !== routeGen) return;
  // The page edits the selected preset: a save, an unset or an entry change lands in its
  // [presets.<name>] table (the server's rule); none edits the global and repo config.
  const selected = selectedPreset(data);
  view.innerHTML = '';
  const card = el('div', 'card cfg-card');
  // The filter and the page's actions share a toolbar, in the TUI config page's words; the command
  // palette offers its buttons.
  const bar = el('div', 'toolbar cfg-bar');
  const filter = el('input', 'filter'); filter.placeholder = 'filter settings…'; filter.type = 'search';
  filter.value = state.filter || '';
  bar.appendChild(filter);
  let modified = state.modified || false;
  const modifiedBtn = el('button', null, 'Modified only');
  modifiedBtn.setAttribute('aria-pressed', String(modified));
  modifiedBtn.onclick = () => {
    modified = !modified;
    modifiedBtn.setAttribute('aria-pressed', String(modified));
    filter.oninput();
  };
  bar.appendChild(modifiedBtn);
  const refresh = () => {
    if (gen !== routeGen || !card.isConnected) return;
    return renderConfig(gen, () => ({
      filter: filter.value, modified, x: window.scrollX, y: window.scrollY,
      folded: folds(),
      key: rows.find(row => row.edit === document.activeElement)?.key,
      filtering: document.activeElement === filter,
    }));
  };
  // Preset… opens the `agent6.preset` row's editor; the rows are the config under the selected preset.
  const presetBtn = el('button', null, 'Preset…');
  presetBtn.onclick = () => editConfig('agent6.preset', data['agent6.preset'], registry.tables, refresh, data);
  bar.appendChild(presetBtn);
  const addPresetBtn = el('button', null, 'Add preset…');
  addPresetBtn.onclick = () => addPreset(refresh);
  bar.appendChild(addPresetBtn);
  const removePresetBtn = el('button', 'danger', 'Remove preset');
  removePresetBtn.disabled = !selected;
  removePresetBtn.title = selected
    ? 'Remove [presets.' + selected + '] from the global and repo config and reset preset to none'
    : 'No preset is selected';
  removePresetBtn.onclick = async () => {
    if (!confirm('Remove [presets.' + selected + '] and every value under it from the global and repo config, and reset preset to none?')) return;
    removePresetBtn.disabled = true;
    try {
      const d = await postJSON('/api/config/preset/remove', {name: selected});
      toast(d.message || 'removed'); await refresh();
    } catch (e) { toast(e.message, true); removePresetBtn.disabled = false; }
  };
  bar.appendChild(removePresetBtn);
  for (const table of registry.tables) {
    const addBtn = el('button', null, 'Add ' + table.label + '…');
    addBtn.onclick = () => addEntry(table, refresh);
    bar.appendChild(addBtn);
  }
  const reviewerBtn = el('button', null, 'Add reviewer…');
  reviewerBtn.onclick = () => addReviewer(refresh);
  bar.appendChild(reviewerBtn);
  card.appendChild(bar);
  // While a preset is selected, a line under the bar says where saves go, in the TUI's words.
  if (selected) {
    card.appendChild(el('div', 'notif-banner'))
      .appendChild(el('div', 'grow nb-msg', `preset ${selected}: saves write [presets.${selected}]`));
  }
  // The payload comes in the order every surface shares (`config_view`); one collapsible
  // section per table, as the TUI's page shows them; a render keeps the sections folded before it.
  const rows = [], sections = new Map();
  let section = null, tbl = null;
  for (const k of Object.keys(data)) {
    const s = data[k];
    const name = k.includes('.') ? k.slice(0, k.indexOf('.')) : k;
    if (name !== section) {
      section = name;
      const box = el('details', 'cfg-section'); box.open = !state.folded?.includes(section);
      box.appendChild(el('summary', null, section));
      tbl = el('table', 'cfg');
      const head = el('tr'); ['setting', 'value', 'source'].forEach(h => head.appendChild(el('th', null, h)));
      tbl.appendChild(el('thead')).appendChild(head);
      box.appendChild(tbl); card.appendChild(box); sections.set(section, box);
    }
    const tr = el('tr', s.modified ? 'mod' : '');
    // The key is the row's edit button, so the row keeps its cells for a screen reader. A click
    // anywhere on the row presses it, unless the pointer moved between press and release: that
    // drag selects text to copy.
    const edit = breakable(el('button', 'cfg-edit'), k, '.');
    edit.type = 'button'; edit.setAttribute('aria-label', 'edit ' + k); edit.setAttribute('aria-haspopup', 'dialog');
    tr.appendChild(el('td', 'key')).appendChild(edit);
    tr.appendChild(breakable(el('td', 'val'), s.display, '/')); // the one shared value column (viewmodel.display_value)
    tr.appendChild(el('td', 'source', esc(s.source)));
    // Hover text: the leaf's meaning (the docs table cell), from the same
    // per-key payload the editor dialog shows it in. The edit button, which takes the focus,
    // carries it as its description.
    tr.title = s.description ? (s.description_points || [s.description]).join('\n') : 'click to edit';
    if (s.description) edit.setAttribute('aria-description', tr.title);
    tr.setAttribute('data-key', k);
    let press = null;
    tr.onpointerdown = e => { press = [e.clientX, e.clientY]; };
    tr.onclick = e => {
      const dragged = press && e.detail && Math.hypot(e.clientX - press[0], e.clientY - press[1]) > 4;
      press = null;
      if (dragged) return;
      edit.focus({preventScroll: true});
      editConfig(k, s, registry.tables, refresh, data);
    };
    tbl.appendChild(tr); rows.push({key: k, tr, edit, box: sections.get(section), mod: s.modified});
  }
  // While the filter or Modified only narrows the rows, every section with a match shows open; the
  // folds come back when both clear. A line says when nothing matches.
  const none = el('div', 'empty cfg-none'); none.setAttribute('role', 'status');
  card.appendChild(none);
  let narrowed = false, folded = [];
  const folds = () => narrowed ? folded : [...sections].filter(([, box]) => !box.open).map(([name]) => name);
  filter.oninput = () => {
    const q = filter.value.toLowerCase();
    const narrowing = q !== '' || modified;
    if (narrowing && !narrowed) folded = folds();
    const shown = new Set();
    for (const {key, tr, box, mod} of rows) {
      const show = key.toLowerCase().includes(q) && (!modified || mod);
      tr.style.display = show ? '' : 'none';
      if (show) shown.add(box);
    }
    for (const [name, box] of sections) {
      box.style.display = shown.has(box) ? '' : 'none';
      if (narrowing) box.open = true;
      else if (narrowed) box.open = !folded.includes(name);
    }
    narrowed = narrowing;
    none.textContent = shown.size ? '' : q ? `no ${modified ? 'modified ' : ''}setting matches` : 'no modified settings';
  };
  view.appendChild(card);
  filter.oninput();
  if (state.y !== undefined) window.scrollTo(state.x, state.y);
  // The focus goes back to the edited row, scrolled into view if the page moved it.
  const focused = rows.find(row => row.key === state.key && row.tr.style.display !== 'none');
  if (state.filtering || (state.key && !focused)) filter.focus({preventScroll: true});
  else if (focused) { focused.edit.focus({preventScroll: true}); focused.tr.scrollIntoView({block: 'nearest'}); }
}
// Fill `node` with text that may wrap after each `mark` (a key's dots, a value's or a path's
// slashes): a <wbr> marks the place, so a copy of the text holds no extra character. The <wbr> is
// hidden from screen readers, as Chrome reads it as a space.
function breakable(node, text, mark) {
  const parts = String(text).split(mark);
  parts.forEach((part, i) => {
    if (i) node.appendChild(el('wbr')).setAttribute('aria-hidden', 'true');
    node.appendChild(document.createTextNode(i < parts.length - 1 ? part + mark : part));
  });
  return node;
}
// The preset the page edits, read off the payload's `agent6.preset` row; '' for none.
function selectedPreset(data) {
  const value = data['agent6.preset'] && data['agent6.preset'].value;
  return value && value !== 'none' ? String(value) : '';
}
// A leaf's description is the docs table cell: markdown-lite. Backtick spans
// render as code; `**bold**` is dropped (the terminal renderers do the same).
function plainDescription(text) { return String(text).replace(/\*\*/g, ''); }
// One line per point: the server splits the field's semicolon chain (`description_points`).
function descriptionNode(points) {
  const frag = document.createDocumentFragment();
  for (const point of Array.isArray(points) ? points : [points]) {
    const line = el('div', 'cfg-point');
    plainDescription(point).split('`').forEach((part, i) => {
      if (!part) return;
      line.appendChild(i % 2 ? el('code', null, part) : document.createTextNode(part));
    });
    frag.appendChild(line);
  }
  return frag;
}

// A dialog's button row, then the line its own actions report a failure on. Enter in any field
// presses `primary` (Add or Save); Shift+Enter breaks a long value's line. A held Enter's repeats
// press nothing: the Enter that opened the dialog would save it unchanged.
// `act(button, run)` holds the button while `run` waits: a failure shows on that line and frees the
// button while the dialog is open, and toasts once the dialog has closed (the page's refresh
// after a write).
function actionRow(dialog, row, primary) {
  const err = el('p', 'err'); err.setAttribute('role', 'alert');
  dialog.appendChild(row); dialog.appendChild(err);
  dialog.onkeydown = e => {
    if (e.key !== 'Enter' || e.repeat || e.shiftKey || e.defaultPrevented || composing(e)) return;
    if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    e.preventDefault(); primary.click();
  };
  return async (button, run) => {
    if (button.disabled) return;
    button.disabled = true; err.textContent = '';
    try { await run(); } catch (e) {
      if (!dialog.open) { toast(e.message, true); return; }
      err.textContent = e.message; button.disabled = false;
      if (document.activeElement === document.body) button.focus(); // disabling it dropped the focus
    }
  };
}

// An Add dialog's Add and Cancel row; Cancel closes the dialog.
function addButtons(dialog) {
  const row = el('div', 'form-row');
  const add = el('button', 'primary', 'Add'), cancel = el('button', null, 'Cancel');
  cancel.onclick = () => dialog.close();
  row.appendChild(add); row.appendChild(cancel);
  return { add, act: actionRow(dialog, row, add) };
}

// A blank field writes nothing, preserving an existing entry's value.
function addEntry(table, refresh) {
  const dialog = openDialog('Add ' + table.label);
  if (!dialog) return;
  dialog.appendChild(el('div', 'sub muted', table.path + '.<name>. A blank field keeps an existing value or uses the default.'));
  const labelled = (text, field) => { const l = el('label', 'sub muted cfg-label', text); field.name = text; l.appendChild(field); dialog.appendChild(l); return field; };
  const name = labelled('name', el('input', 'field'));
  const fields = {};
  for (const spec of table.fields) {
    const field = el(spec.choices.length ? 'select' : 'input', 'field');
    if (spec.choices.length) {
      const values = ['backend', 'state'].includes(spec.name) ? spec.choices : ['', ...spec.choices];
      for (const value of values) {
        const option = el('option', null, value || 'blank = unchanged, or default');
        option.value = value; field.appendChild(option);
      }
    }
    fields[spec.name] = labelled(spec.name, field);
    // The field's description, whole, under the field: it stays in view while the field is typed in.
    const help = el('div', 'cfg-desc cfg-help'); help.id = 'cfg-help-' + spec.name;
    help.appendChild(descriptionNode(spec.description));
    field.setAttribute('aria-describedby', help.id);
    dialog.appendChild(help);
  }
  if (table.path === 'providers') {
    const format = fields.backend, baseUrl = fields.base_url;
    let autofilled = '';
    name.oninput = () => {
      const preset = Object.hasOwn(table.defaults, name.value.trim()) ? table.defaults[name.value.trim()] : null;
      if (!preset) {
        if (autofilled && baseUrl.value === autofilled) baseUrl.value = '';
        autofilled = '';
        return;
      }
      format.value = preset.backend;
      if (baseUrl.value === '' || baseUrl.value === autofilled) { autofilled = preset.base_url || ''; baseUrl.value = autofilled; }
    };
  }
  const repoRow = el('label', 'row cfg-check');
  const repoCb = el('input'); repoCb.type = 'checkbox';
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  dialog.appendChild(repoRow);
  const { add, act } = addButtons(dialog);
  name.focus();
  add.onclick = () => act(add, async () => {
    const d = await postJSON('/api/config/entry', {
      table: table.path, name: name.value,
      fields: Object.fromEntries(Object.entries(fields).map(([key, field]) => [key, field.value])),
      repo: repoCb.checked,
    });
    toast(d.message || 'added'); dialog.close(); await refresh();
  });
}

// Name and layer for a new preset table; the server selects it, so the page then shows and
// edits it.
function addPreset(onAdded) {
  const dialog = openDialog('Add preset');
  if (!dialog) return;
  dialog.appendChild(el('div', 'sub muted', 'An empty [presets.<name>] table, selected as preset in the chosen layer; edit the rows to fill it.'));
  const label = el('label', 'sub muted cfg-label', 'name');
  const name = el('input', 'field'); name.name = 'name'; label.appendChild(name); dialog.appendChild(label);
  const repoRow = el('label', 'row cfg-check');
  const repoCb = el('input'); repoCb.type = 'checkbox';
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'create in this repo\'s config (not the global config)'));
  dialog.appendChild(repoRow);
  const { add, act } = addButtons(dialog);
  add.onclick = () => act(add, async () => {
    const d = await postJSON('/api/config/preset', {name: name.value.trim(), repo: repoCb.checked});
    toast(d.message || 'added'); dialog.close(); await onAdded();
  });
  name.focus();
}

async function addReviewer(refresh) {
  const gen = routeGen;
  let choices;
  try { choices = await getJSON('/api/config/reviewer_choices'); } catch (e) { toast(e.message, true); return; }
  if (gen !== routeGen) return;
  const dialog = openDialog('Add reviewer');
  if (!dialog) return;
  dialog.appendChild(el('div', 'sub muted', 'Focus is required. Set model to provider/model, or leave it blank. Blank optional fields inherit the route\'s settings.'));
  const fields = {};
  for (const name of ['focus', 'model', 'effort', 'temperature']) {
    const label = el('label', 'sub muted cfg-label', name);
    // The model suggests from the catalog, as the model editor does.
    const combo = name === 'model' ? configCombobox('', 'models.reviewer.model', 'field') : null;
    const field = combo ? combo.field : el(choices[name] ? 'select' : 'input', 'field');
    field.id = 'reviewer-' + name; label.htmlFor = field.id;
    if (choices[name]) for (const value of ['', ...choices[name]]) {
      const option = el('option', null, value || 'blank = inherit'); option.value = value; field.appendChild(option);
    } else if (name !== 'focus') field.placeholder = 'blank = inherit';
    fields[name] = field; dialog.appendChild(label); dialog.appendChild(combo ? combo.element : field);
  }
  const repoRow = el('label', 'row cfg-check');
  const repo = el('input'); repo.type = 'checkbox'; repo.id = 'reviewer-repo';
  repoRow.appendChild(repo); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  dialog.appendChild(repoRow);
  const { add, act } = addButtons(dialog);
  fields.focus.focus();
  add.onclick = () => act(add, async () => {
    const body = Object.fromEntries(Object.entries(fields).map(([name, field]) => [name, field.value]));
    const result = await postJSON('/api/config/reviewer', {...body, repo: repo.checked});
    toast(result.message); dialog.close(); await refresh();
  });
}

// A model field over the catalog's suggestions for the leaf `key`. Suggestions guide text entry;
// catalog membership is not validation. The list stays open under the field, so the dialog's
// buttons stay in view. Enter picks the active suggestion; with none active it reaches the dialog,
// which presses its Add or Save.
function configCombobox(current, key, cls) {
  const element = el('div', 'cfg-combo'), field = el('input', cls);
  field.value = current; field.autocomplete = 'off';
  field.setAttribute('role', 'combobox');
  field.setAttribute('aria-autocomplete', 'list');
  field.setAttribute('aria-expanded', 'true');
  const list = el('div', 'cfg-options'); list.id = 'cfg-model-options';
  list.tabIndex = -1; // a scroller is a Tab stop; the field's arrow keys move through the list
  list.setAttribute('role', 'listbox'); list.setAttribute('aria-label', 'Model suggestions');
  field.setAttribute('aria-controls', list.id);
  element.appendChild(field); element.appendChild(list);
  let values = [], matches = [], query = '', index = -1, message = 'Loading models…';
  const choose = value => { field.value = value; query = value; index = -1; paint(); };
  const paint = () => {
    list.innerHTML = '';
    matches = values.filter(value => value.toLowerCase().includes(query.toLowerCase()));
    field.removeAttribute('aria-activedescendant');
    if (!matches.length) {
      const status = el('div', 'cfg-catalog sub muted', message || 'No matches. You can save the typed model id.');
      status.setAttribute('role', 'status'); list.appendChild(status);
    }
    matches.forEach((value, i) => {
      const option = el('div', 'cfg-option', value); option.id = list.id + '-' + i;
      option.setAttribute('role', 'option'); option.setAttribute('aria-selected', String(i === index));
      option.onmousedown = e => e.preventDefault();
      option.onclick = () => choose(value);
      list.appendChild(option);
      if (i === index) {
        field.setAttribute('aria-activedescendant', option.id);
        option.scrollIntoView({block: 'nearest'});
      }
    });
  };
  field.oninput = () => { query = field.value; index = -1; paint(); };
  field.onkeydown = e => {
    if (composing(e)) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const step = e.key === 'ArrowDown' ? 1 : -1;
      index = matches.length ? (index < 0 ? (step > 0 ? 0 : matches.length - 1) : (index + step + matches.length) % matches.length) : -1;
      paint();
    } else if (e.key === 'Enter' && index >= 0) {
      e.preventDefault(); choose(matches[index]);
    }
  };
  // The list says it is loading until the catalog answers; a dialog closed meanwhile stays closed.
  const load = async () => {
    try {
      values = (await getJSON('/api/config/suggest/' + key)).values || [];
      message = values.length ? '' : 'Catalog empty or unavailable. You can type a model id.';
    } catch (e) {
      message = 'Catalog unavailable: ' + e.message.replace(/\.$/, '') + '. You can type a model id.';
    }
    if (element.isConnected) paint();
  };
  paint(); load();
  return {element, field};
}

function editConfig(key, s, tables, refresh = () => {}, data = {}) {
  const cur = s.input;
  // The [agent6] table, which holds the selector, is never nested under a preset.
  const preset = key.startsWith('agent6.') ? '' : selectedPreset(data);
  const role = s.model_role || null; // the view's one rule for the model picker
  const section = role ? 'models.' + role : null;
  const dialog = openDialog((role ? 'Model: ' + role : key) + (preset ? ' under preset ' + preset : ''));
  if (!dialog) return;
  const close = () => dialog.close();
  const meta = el('div', 'sub muted cfg-meta');
  meta.textContent = `${esc(s.type)} · default: ${s.default_display} · set from: ${esc(s.source)}` + (s.adaptive ? ' · adaptive' : '');
  dialog.appendChild(meta);
  if (s.description) {
    const desc = el('div', 'cfg-desc');
    desc.appendChild(descriptionNode(s.description_points || [s.description]));
    dialog.appendChild(desc);
  }
  let field, combo, meaning;
  const choices = s.choices || (s.type === 'bool' ? ['true', 'false'] : null);
  if (role) {
    combo = configCombobox(data[section + '.model']?.input || '', section + '.model', 'field cfg-value');
    field = combo.field; field.id = 'cfg-model';
    const modelLabel = el('label', 'sub muted', 'Model'); modelLabel.htmlFor = field.id;
    dialog.appendChild(modelLabel); dialog.appendChild(combo.element);
  } else if (choices) {
    field = el('select', 'field cfg-value');
    const current = cur || String(s.default ?? '');
    if (!choices.includes(current)) {
      // A leaf no layer sets and with no default (an adaptive effort) matches no choice: the first
      // option reads what its row reads, so the select never opens blank.
      const shown = el('option', null, s.display.replace(/\s+/g, ' '));
      shown.value = ''; shown.disabled = true; shown.selected = true; field.appendChild(shown);
    }
    for (const c of choices) {
      const option = el('option', null, c); option.value = c; field.appendChild(option);
    }
    if (choices.includes(current)) field.value = current;
    // An option holds only its value; the chosen one's meaning reads whole under the select,
    // where a long meaning wraps instead of being cut at the select's edge.
    if (Object.keys(s.choice_meanings || {}).length) {
      meaning = el('div', 'cfg-desc cfg-help'); meaning.id = 'cfg-meaning';
      field.setAttribute('aria-describedby', meaning.id);
      field.oninput = () => { meaning.textContent = s.choice_meanings[field.value || s.effective] || ''; };
      field.oninput();
    }
  } else if (['list', 'table', 'str'].includes(s.type)) {
    // A list, table or string shows whole: the field grows with the value.
    field = el('textarea', 'field cfg-value');
    field.value = cur;
    if (s.type === 'list') field.placeholder = 'TOML list';
  } else {
    field = el('input', 'field cfg-value');
    field.value = cur;
  }
  if (!combo) {
    field.setAttribute('aria-labelledby', dialog.getAttribute('aria-labelledby')); // named by the key
    dialog.appendChild(field);
    if (meaning) dialog.appendChild(meaning);
  }
  const repoRow = el('label', 'row cfg-check');
  const repoCb = el('input'); repoCb.type = 'checkbox';
  // Default the target layer to the value's origin: editing a repo-sourced value
  // must write the repo config, or the repo overlay keeps masking the edit and it
  // looks like the save vanished.
  repoCb.checked = (s.source === 'repo');
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  dialog.appendChild(repoRow);
  const row = el('div', 'form-row cfg-actions');
  const save = el('button', 'primary', 'Save'), cancel = el('button', null, 'Cancel');
  row.appendChild(save);
  // A value set from an editable layer gets an Unset: remove it from that layer
  // so it reverts to the next-lower one / the built-in default. (default-sourced
  // values have nothing to unset; flag/machine layers are not editable here.)
  let unsetBtn = null;
  if (s.source === 'repo' || s.source === 'global' || (preset && s.source === 'preset')) {
    unsetBtn = el('button', null, role ? 'Unset ' + key : 'Unset');
    unsetBtn.title = preset
      ? 'remove from [presets.' + preset + '] in the chosen layer; the layer below takes over'
      : 'remove from the ' + s.source + ' config; the next layer below takes over';
    row.appendChild(unsetBtn);
  }
  const entry = s.entry;
  const table = entry && tables.find(t => t.path === entry.table);
  // Under a preset the entry's source reads `preset`; the server removes it from the layer
  // holding [presets.<name>].
  const entryWhere = preset ? '[presets.' + preset + ']' : entry && entry.source + ' config';
  let removeBtn = null;
  if (table) {
    const addBtn = el('button', null, 'Add ' + table.label + '…');
    addBtn.onclick = () => { close(); addEntry(table, refresh); };
    row.appendChild(addBtn);
    removeBtn = el('button', 'danger', 'Remove ' + table.label);
    removeBtn.disabled = !(['global', 'repo'].includes(entry.source) || (preset && entry.source === 'preset'));
    removeBtn.title = removeBtn.disabled
      ? 'This entry comes from ' + entry.source + '; Remove edits only global and repo config.'
      : 'Remove ' + table.label + " '" + entry.name + "' and all its values from " + entryWhere;
  }
  if (key === 'review.reviewers') {
    const addBtn = el('button', null, 'Add reviewer…');
    addBtn.onclick = () => { close(); addReviewer(refresh); };
    row.appendChild(addBtn);
  }
  row.appendChild(cancel);
  if (removeBtn) row.appendChild(removeBtn); // last, at the row's end: it removes the whole entry
  const act = actionRow(dialog, row, save);
  cancel.onclick = close;
  if (key === 'review.reviewers') {
    const roster = el('div', 'cfg-roster');
    const editable = s.source === 'repo' || s.source === 'global' || (preset && s.source === 'preset');
    const rosterWhere = preset ? '[presets.' + preset + ']' : 'the ' + s.source + ' config';
    const reviewers = s.value || [];
    // The roster's first line explains its Remove buttons, or says it has none.
    const note = !reviewers.length ? 'The reviewer roster is empty.'
      : editable ? `Remove saves the remaining roster to ${rosterWhere}, including an empty list after the last reviewer.`
      : `Remove edits only the global and repo config, not the ${s.source} layer.`;
    roster.appendChild(el('div', 'sub muted', note));
    reviewers.forEach((reviewer, index) => {
      const item = el('div', 'row');
      const label = `${index + 1}: ${reviewer.focus}`;
      item.appendChild(el('span', null, label));
      const remove = el('button', 'danger', 'Remove'); remove.disabled = !editable;
      remove.setAttribute('aria-label', 'Remove reviewer ' + label);
      remove.onclick = async () => {
        if (!confirm(`Remove reviewer ${label} from ${rosterWhere}?`)) return;
        await act(remove, async () => {
          const result = await postJSON('/api/config/reviewer/remove', {index, repo: s.source === 'repo'});
          toast(result.message); close(); await refresh();
        });
      };
      item.appendChild(remove); roster.appendChild(item);
    });
    dialog.insertBefore(roster, field);
  }
  field.focus();
  save.onclick = () => act(save, async () => {
    // A string-valued leaf travels as a TOML string, as the TUI sends it, so
    // `true`, `42` or `[a]` typed into a str field stays a string.
    const value = role || s.type === 'str' ? JSON.stringify(field.value) : field.value;
    const d = await postJSON('/api/config', { key: role ? section + '.model' : key, value, repo: repoCb.checked });
    toast(d.message || 'set ' + key); close(); await refresh();
  });
  if (unsetBtn) unsetBtn.onclick = () => act(unsetBtn, async () => {
    // Under a preset the layer is the checkbox's: the row's source reads `preset`, not a file.
    const d = await postJSON('/api/config', { key, unset: true, repo: preset ? repoCb.checked : s.source === 'repo' });
    toast(d.message || 'unset ' + key); close(); await refresh();
  });
  if (removeBtn) removeBtn.onclick = async () => {
    if (!confirm('Remove ' + table.label + " '" + entry.name + "' and all its values from " + entryWhere + '? Values from lower layers may become visible.')) return;
    await act(removeBtn, async () => {
      const d = await postJSON('/api/config/entry/remove', {table: entry.table, name: entry.name, repo: entry.source === 'repo'});
      toast(d.message || 'removed'); close(); await refresh();
    });
  };
}
