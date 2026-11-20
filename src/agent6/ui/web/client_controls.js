function icon(name) {
  const paths = {
    back: 'M15 5 8 12l7 7', forward: 'm9 5 7 7-7 7',
    stop: 'M6 6h12v12H6z', help: 'M9 8a3 3 0 0 1 6 0c0 3-3 2-3 5m0 4v.1',
    refresh: 'M20 5v6h-6M20 11a8 8 0 1 0-1 7',
    swap: 'm7 3-4 4 4 4M3 7h18M17 13l4 4-4 4M21 17H3',
  };
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  for (const [key, value] of Object.entries({class: 'ic', viewBox: '0 0 24 24',
    fill: 'none', stroke: 'currentColor', 'stroke-width': '1.75',
    'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true'})) svg.setAttribute(key, value);
  const path = document.createElementNS(svg.namespaceURI, 'path');
  path.setAttribute('d', paths[name]); svg.appendChild(path);
  return svg;
}
// Keep both the row and the tooltip in a repeated button’s accessible description.
let titleIds = 0;
function describedBy(title, control) {
  title.id ||= 'title-' + ++titleIds;
  const ids = [title.id];
  if (control.title) {
    const own = el('span', null, control.title);
    own.id = 'title-' + ++titleIds; own.hidden = true;
    control.after(own); ids.push(own.id);
  }
  control.setAttribute('aria-describedby', ids.join(' '));
}
const FOCUSABLE = 'a[href], button, input, select, textarea, summary, [tabindex]';
function rowOf(e) { while (e && e.rowKey === undefined) e = e.parentElement; return e; }
function controlKey(e) {
  const name = e.getAttribute('aria-label') || e.getAttribute('href') || e.labels?.[0]?.textContent
    || (e.matches('button, summary') ? e.textContent.trim() : '');
  return JSON.stringify([rowOf(e)?.rowKey ?? null, e.tagName, name]);
}
// Restore focus only if the repaint dropped it and the user has not moved it since.
function refocus(held, scope, fallback) {
  const now = document.activeElement;
  if (!held || held === document.body || (now && now !== document.body)) return;
  if (held.isConnected && held.checkVisibility()) { held.focus({ preventScroll: true }); return; }
  const key = controlKey(held), row = rowOf(held);
  const controls = scope ? [...scope.querySelectorAll(FOCUSABLE)].filter(c => c.checkVisibility()) : [];
  const same = controls.find(c => controlKey(c) === key);
  const sameRow = row ? controls.find(c => rowOf(c)?.rowKey === row.rowKey) : null;
  for (const c of [same, sameRow]) { c?.focus(); if (c && document.activeElement === c) return; }
  fallback?.focus();
}

// A busy button stays focusable while its action is in flight.
function runOnce(button, action, ready = () => true, home = null) {
  let busy = false;
  button.sync = () => {
    button.disabled = !ready();
    if (busy) button.setAttribute('aria-disabled', 'true'); else button.removeAttribute('aria-disabled');
  };
  const run = async (...args) => {
    if (busy || !ready()) return;
    const focused = document.activeElement === button;
    busy = true; button.sync();
    try { await action(...args); } finally {
      busy = false; button.sync();
      if (focused && button.disabled && home && [button, document.body].includes(document.activeElement)) home.focus();
    }
  };
  button.onclick = () => run();
  button.sync();
  return run;
}

function pickerRow(pairs) {
  const row = el('div', 'row pickers');
  for (const [text, control] of pairs) {
    const pick = el('label', 'pick');
    const label = el('span', 'muted', text);
    if (control.type === 'checkbox') pick.append(control, label); else pick.append(label, control);
    row.appendChild(pick);
  }
  return row;
}
// A programmatic pick must dispatch input to refresh the phone summary.
function pickerSummary(rows, words) {
  const fold = el('button', 'fold picker-summary');
  const line = fold.appendChild(el('span'));
  const paint = () => { line.textContent = words().filter(Boolean).join(' · '); };
  fold.setAttribute('aria-expanded', 'false');
  fold.onclick = () => fold.setAttribute('aria-expanded', String(fold.getAttribute('aria-expanded') !== 'true'));
  for (const row of rows) row.addEventListener('input', paint);
  paint();
  return fold;
}
const pickedName = (select) => (select.selectedOptions[0]?.text || '').replace(/ \((config default|as recorded)\)$/, '');
const pickedModel = (select) => { const m = pickedName(select).split('/').pop(); return m === 'none' ? 'no model' : m; };
