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

// POST keeps the operator’s task out of URLs.
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

// The command and model popups share one entry and must not clear each other’s selection.
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

// Completion mirrors agent6.directive, including which commands need a live run.
const STEER_COMMANDS = [
  ['/pin', 'pin an instruction that survives compaction: /pin <text>'],
  ['/compact', 'compact the context now; /compact <focus> steers the summary'],
  ['/parallel', 'dispatch children: /parallel [SPEC --] TASK (SPEC: count or models; repeat to queue more)'],
  ['/restate', 'restate the conversation since your last message (local, no model call)'],
  ['/review', 'review changes in this session: /review [TARGET] (a ref or A..B or A...B)'],
  ['/undo', 'fork back to before your last message (the text returns to edit and resend)'],
  ['/btw', 'ask a question beside the run: /btw <question> (answers inline, later)'],
  ['/task', 'queue work into the task graph: /task <text> (worked when the queue drains)'],
  ['/metric', 'Show the metric, set maximize|minimize ["capture pattern"] -- command, or turn it off.'],
  ['/standing', 'set the goal the run returns to when the queue drains: /standing <text>'],
  ['/retire', 'drop a task from the graph: /retire <task id>, the number the task tree shows'],
  ['/now', 'steer at once, aborting the call in flight: /now <text> (Ctrl+Enter on the web)'],
  ['/stop', 'stop the run now, as `agent6 stop` does (resumable)'],
  ['/shells', 'background commands this run started, and how they ended'],
];
function attachCommandSuggest(ta, root, liveNow, offered = null) {
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
    items = STEER_COMMANDS.filter(([c]) => (liveNow() || (c !== '/compact' && c !== '/btw' && c !== '/now' && c !== '/stop' && c !== '/task' && c !== '/metric' && c !== '/standing' && c !== '/retire')) && (!offered || offered.includes(c)) && c.startsWith(w));
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
