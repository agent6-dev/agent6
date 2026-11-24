// execCommand preserves native undo history for a programmatic fill.
function fillAsEdit(ta, text) {
  ta.focus();
  ta.setSelectionRange(0, ta.value.length);
  let ok = false;
  try { ok = document.execCommand('insertText', false, text); } catch (_) { ok = false; }
  if (!ok) {
    ta.value = text;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

// POST keeps the operator's task out of URLs.
function attachParallelPreview(field, root, options) {
  const box = el('div', 'parallel-preview');
  box.setAttribute('role', 'status');
  root.appendChild(box);
  let timer, generation = 0;
  const refresh = () => {
    clearTimeout(timer);
    const current = ++generation;
    const text = field.value;
    const extra = options();
    box.hidden = true;
    if (!extra || !/^\s*\/parallel(?:\s|$)/i.test(text)) return;
    timer = setTimeout(async () => {
      try {
        const reply = await postJSON('/api/parallel-preview', {text, ...extra});
        if (current !== generation) return;
        box.textContent = reply.preview;
      } catch (error) {
        if (current !== generation) return;
        box.textContent = 'Preview unavailable: ' + error.message;
      }
      box.hidden = !box.textContent;
    }, 100);
  };
  field.addEventListener('input', refresh);
  refresh();
  return refresh;
}

// The command and model popups share one entry and must not clear each other's selection.
let suggestLists = 0;
function unpointPopup(entry, box) {
  if (entry.getAttribute('aria-controls') !== box.id) return;
  entry.removeAttribute('aria-controls'); entry.removeAttribute('aria-activedescendant');
}
// Only the comma fragment under the caret, before the task separator, names a model.
function attachParallelSuggest(task, root) {
  let models = null;
  let box = null, items = [], active = -1;
  task.setAttribute('aria-autocomplete', 'list');
  const ensureModels = () => {
    if (models !== null) return;
    models = [];
    getJSON('/api/config/suggest/parallel.models').then(d => { models = d.values || []; render(); }).catch(() => {});
  };
  const frag = () => {
    const v = task.value, caret = task.selectionStart;
    const markers = [...v.matchAll(/(^|\s)\/parallel(?=\s|$)/gi)].map(m => ({
      start: m.index + m[1].length,
      end: m.index + m[0].length,
    }));
    if (!markers.length || v.slice(0, markers[0].start).trim()) return null;
    const marker = markers.filter(m => m.end < caret).pop();
    if (!marker || !/\s/.test(v[marker.end])) return null;
    const segmentEnd = markers.find(m => m.start > marker.start)?.start ?? v.length;
    const separator = /(^|\s)--(?=\s|$)/.exec(v.slice(marker.end, segmentEnd));
    const end = separator ? marker.end + separator.index + separator[1].length : segmentEnd;
    let start = marker.end;
    while (start < end && /\s/.test(v[start])) start++;
    if (caret < start || caret > end) return null;
    if (/^\p{Decimal_Number}+$/u.test(v.slice(start, end).trim())) return null;
    let fragStart = start + v.slice(start, caret).lastIndexOf(',') + 1;
    let fragEnd = caret;
    while (fragEnd < end && v[fragEnd] !== ',') fragEnd++;
    while (fragStart < fragEnd && /\s/.test(v[fragStart])) fragStart++;
    while (fragEnd > fragStart && /\s/.test(v[fragEnd - 1])) fragEnd--;
    return { fragStart, fragEnd, text: v.slice(fragStart, Math.min(caret, fragEnd)) };
  };
  const close = () => {
    if (box) { unpointPopup(task, box); box.remove(); box = null; }
    items = []; active = -1;
  };
  const insert = (model) => {
    const f = frag(); if (!f) { close(); return; }
    const v = task.value;
    task.value = v.slice(0, f.fragStart) + model + v.slice(f.fragEnd);
    const pos = f.fragStart + model.length;
    task.setSelectionRange(pos, pos); close(); task.focus();
    task.dispatchEvent(new Event('input', {bubbles: true}));
  };
  const render = () => {
    const f = frag();
    if (!f) { close(); return; }
    ensureModels();
    const q = f.text.toLowerCase();
    const hit = m => m.toLowerCase();
    items = models.filter(m => hit(m).startsWith(q)).concat(models.filter(m => hit(m).includes(q) && !hit(m).startsWith(q))).slice(0, 8);
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) {
      box = el('div', 'ac-pop'); box.id = 'suggest-' + ++suggestLists; box.setAttribute('role', 'listbox');
      box.setAttribute('aria-label', 'Model ids');
      root.appendChild(box); task.setAttribute('aria-controls', box.id);
    }
    box.textContent = '';
    items.forEach((m, i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''), m);
      o.id = box.id + '-' + i; o.setAttribute('role', 'option'); o.setAttribute('aria-selected', String(i === active));
      o.onmousedown = (e) => { e.preventDefault(); insert(m); };
      box.appendChild(o);
    });
    if (active >= 0) task.setAttribute('aria-activedescendant', box.id + '-' + active);
    else task.removeAttribute('aria-activedescendant');
  };
  task.addEventListener('input', () => { active = -1; render(); });
  task.addEventListener('click', render);
  task.addEventListener('blur', () => setTimeout(close, 120));
  return { onKeyDown(e) {
    if (!box || !items.length || composing(e)) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

// A message's images, pasted, dropped or picked. The limits mirror agent6.media; the server checks each image again.
const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
const IMAGE_MAX_BYTES = 2 * 1024 * 1024, IMAGE_MAX_COUNT = 4;
function attachImages(ta, root, onChange = () => {}) {
  const tray = el('div', 'attachments'); tray.hidden = true;
  const input = el('input'); input.type = 'file'; input.accept = IMAGE_TYPES.join(','); input.multiple = true; input.hidden = true;
  const button = el('button', 'quiet attach', 'Image'); button.type = 'button';
  button.title = 'Attach an image; a paste or a drop attaches one too';
  let items = [];
  const paint = () => {
    tray.replaceChildren(...items.map((item, i) => {
      const chip = el('span', 'attachment');
      const thumb = el('img'); thumb.src = 'data:' + item.media_type + ';base64,' + item.data; thumb.alt = '';
      const remove = el('button', 'quiet x-btn', '×'); remove.type = 'button';
      remove.setAttribute('aria-label', 'Remove ' + item.name);
      remove.onclick = () => { items.splice(i, 1); paint(); };
      chip.append(thumb, el('span', null, item.name + ' · ' + Math.ceil(item.size / 1024) + ' KiB'), remove);
      return chip;
    }));
    tray.hidden = !items.length;
    onChange();
  };
  const base64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.slice(reader.result.indexOf(',') + 1));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  const add = async (files) => {
    for (const file of files) {
      const name = file.name || 'image';
      if (!IMAGE_TYPES.includes(file.type)) { toast(name + ' is not a PNG, JPEG, GIF or WebP image', true); continue; }
      if (file.size > IMAGE_MAX_BYTES) { toast(name + ' is over the 2 MiB limit', true); continue; }
      if (items.length >= IMAGE_MAX_COUNT) { toast('a message carries at most four images', true); break; }
      try { items.push({ media_type: file.type, data: await base64(file), name, size: file.size }); }
      catch (e) { toast(name + ' could not be read: ' + e.message, true); }
    }
    paint();
  };
  button.onclick = () => input.click();
  input.onchange = () => { add([...input.files]); input.value = ''; };
  ta.addEventListener('paste', (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) { e.preventDefault(); add(files); }
  });
  root.addEventListener('dragover', (e) => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); });
  root.addEventListener('drop', (e) => {
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) { e.preventDefault(); add(files); }
  });
  return {
    tray,
    // The entry's label line, with the button at its end: the entry's row keeps its width.
    head: (label) => { const line = el('div', 'entry-head'); line.append(label, button, input); return line; },
    count: () => items.length,
    // A body without images stays the body a text message always sent.
    into: (body) => items.length ? { ...body, images: items.map(({ media_type, data }) => ({ media_type, data })) } : body,
    clear: () => { items = []; paint(); },
  };
}

// The fragment matches agent6.ui.path_completion.ref_fragment.
function attachRefSuggest(ta, root, session = () => '') {
  let box = null, items = [], more = 0, active = -1, timer, generation = 0;
  ta.setAttribute('aria-autocomplete', 'list');
  const frag = () => {
    const caret = ta.selectionStart;
    const m = /(?<![\p{L}\p{N}_@/])@([A-Za-z0-9_./-]*)$/u.exec(ta.value.slice(0, caret));
    return m ? { start: m.index, end: caret, text: m[1] } : null;
  };
  const close = () => {
    clearTimeout(timer); generation++;
    if (box) { unpointPopup(ta, box); box.remove(); box = null; }
    items = []; more = 0; active = -1;
  };
  const insert = (path) => {
    const f = frag(); if (!f) { close(); return; }
    ta.focus(); ta.setSelectionRange(f.start, f.end);
    let ok = false;
    try { ok = document.execCommand('insertText', false, path + ' '); } catch (_) { ok = false; }
    if (!ok) {
      ta.setRangeText(path + ' ', f.start, f.end, 'end');
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
    close();
  };
  const paint = () => {
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) {
      box = el('div', 'ac-pop'); box.id = 'suggest-' + ++suggestLists; box.setAttribute('role', 'listbox');
      box.setAttribute('aria-label', 'Workspace files');
      root.appendChild(box); ta.setAttribute('aria-controls', box.id);
    }
    box.textContent = '';
    items.forEach((path, i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''), path);
      o.id = box.id + '-' + i; o.setAttribute('role', 'option'); o.setAttribute('aria-selected', String(i === active));
      o.onmousedown = (e) => { e.preventDefault(); insert(path); };
      box.appendChild(o);
    });
    if (more) box.appendChild(el('div', 'ac-more muted', '+' + more + ' more, keep typing'));
    if (active >= 0) ta.setAttribute('aria-activedescendant', box.id + '-' + active);
    else ta.removeAttribute('aria-activedescendant');
  };
  const render = () => {
    clearTimeout(timer);
    const current = ++generation;
    const f = frag();
    if (!f) { close(); return; }
    timer = setTimeout(async () => {
      let d;
      try { d = await getJSON('/api/refs?q=' + encodeURIComponent(f.text) + '&session=' + encodeURIComponent(session())); }
      catch (_) { if (current === generation) close(); return; }
      if (current !== generation) return;
      items = d.paths || []; more = d.more || 0; active = -1;
      paint();
    }, 100);
  };
  ta.addEventListener('input', render);
  ta.addEventListener('click', render);
  ta.addEventListener('blur', () => setTimeout(close, 120));
  return { onKeyDown(e) {
    if (!box || !items.length || composing(e)) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; paint(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; paint(); return true; }
    if (e.key === 'Tab') { e.preventDefault(); insert(items[active >= 0 ? active : 0]); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

function attachCommandSuggest(ta, root, offered) {
  let box = null, items = [], active = -1;
  ta.setAttribute('aria-autocomplete', 'list');
  const word = () => {
    const v = ta.value;
    return v.startsWith('/') && !/\s/.test(v) ? v : null;
  };
  const close = () => {
    if (box) { unpointPopup(ta, box); box.remove(); box = null; }
    items = []; active = -1;
  };
  const insert = (cmd) => {
    fillAsEdit(ta, cmd + ' ');
    close();
  };
  const render = () => {
    const w = word();
    if (w === null) { close(); return; }
    items = offered().filter(([command]) => command.startsWith(w));
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) {
      box = el('div', 'ac-pop'); box.id = 'suggest-' + ++suggestLists; box.setAttribute('role', 'listbox');
      box.setAttribute('aria-label', 'Directives');
      root.appendChild(box); ta.setAttribute('aria-controls', box.id);
    }
    box.textContent = '';
    items.forEach(([c, help], i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''));
      o.id = box.id + '-' + i; o.setAttribute('role', 'option'); o.setAttribute('aria-selected', String(i === active));
      o.appendChild(el('span', null, c));
      o.appendChild(el('span', 'muted', ' · ' + help));
      o.onmousedown = (e) => { e.preventDefault(); insert(c); };
      box.appendChild(o);
    });
    if (active >= 0) {
      const option = box.children[active];
      ta.setAttribute('aria-activedescendant', option.id);
      option.scrollIntoView({block: 'nearest', container: 'nearest'});
    } else ta.removeAttribute('aria-activedescendant');
  };
  ta.addEventListener('input', () => { active = -1; render(); });
  ta.addEventListener('click', render);
  ta.addEventListener('blur', () => setTimeout(close, 120));
  return { onKeyDown(e) {
    if (!box || !items.length || composing(e)) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Tab') { e.preventDefault(); insert(items[active >= 0 ? active : 0][0]); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active][0]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}
