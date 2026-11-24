function handToNewWork(fields) {
  window.pendingNewWork = Object.assign({}, window.pendingNewWork || {}, fields);
}
function toNewWork(fields) {
  const shown = !location.hash.replace(/^#\/?/, '') && view.querySelector('.dock textarea');
  if (shown) { if (fields.task) fillAsEdit(shown, fields.task); else shown.focus(); return; }
  handToNewWork(fields);
  if (location.hash.replace(/^#\/?/, '')) location.hash = '#/'; else route(); // no hashchange on #/
}

function newWorkDock(presets, presetDefault, sessions) {
  const root = el('div', 'composer dock');
  const row = el('div', 'row');
  const pending = window.pendingNewWork || {}; delete window.pendingNewWork;
  let draft = {};
  try { draft = pending.task || pending.seed_from ? {} : JSON.parse(recall('a6-draft-new', '{}')) || {}; } catch (_) {}
  const task = el('textarea', 'field'); task.placeholder = 'task / question…';
  task.value = pending.task || draft.task || '';
  const mode = el('select', 'field');
  for (const m of ['run', 'plan', 'ask']) { const o = el('option', null, m); o.value = m; mode.appendChild(o); }
  const worktree = el('input'); worktree.type = 'checkbox';
  worktree.title = 'start the run in a new git worktree of this repository (run --worktree)';
  const seed = el('select', 'field');
  seed.title = 'seed the run with a plan: its text becomes the task when you type none (run --from)';
  const noSeed = el('option', null, 'none'); noSeed.value = ''; seed.appendChild(noSeed);
  const live = new Set(LIVE_WORDS);
  for (const r of (sessions || []).filter(r => r.mode === 'plan' && !live.has(r.status))) {
    const o = el('option', null, r.session_id + ' · ' + firstLine(r.task_line, 41));
    o.value = r.session_id; seed.appendChild(o);
  }
  const cap = el('input', 'field'); cap.placeholder = 'config';
  cap.title = 'the metered spend ceiling in USD for this session (--max-usd); -1 is unlimited';
  // Chromium can read 5,50 in a number input as 550.
  const capOk = () => /^(-?(\d+\.?\d*|\.\d+))?$/.test(cap.value.trim());
  const more = pickerRow([['from plan', seed], ['max USD', cap]]);
  const advanced = el('button', 'fold', 'advanced');
  const fold = (open) => { advanced.setAttribute('aria-expanded', String(open)); more.hidden = !open; };
  advanced.onclick = () => fold(more.hidden || !capOk()); // an unreadable cap stays in view: it holds Start
  const followMode = () => { worktree.disabled = seed.disabled = mode.value !== 'run'; };
  const preset = el('select', 'field');
  preset.title = 'config preset for this run (a preset cannot change mid-run)';
  const dflt = el('option', null, presetDefault); dflt.value = ''; preset.appendChild(dflt);
  for (const p of (presets || [])) { const o = el('option', null, p); o.value = p; preset.appendChild(o); }
  const model = el('select', 'field');
  model.title = 'the model for this run, over every config layer; re-resolved whenever the mode or preset changes';
  // The first option follows the resolved route and sets no override.
  const resolved = el('option', null, '…'); resolved.value = ''; model.appendChild(resolved);
  const restore = (select, value) => { if (value && [...select.options].some(o => o.value === value)) select.value = value; };
  let routeRequest = 0;
  const fillRoutes = async () => {
    const request = ++routeRequest;
    model.value = '';
    model.disabled = true;
    let d;
    try {
      d = await getJSON('/api/routes?mode=' + encodeURIComponent(mode.value) + '&preset=' + encodeURIComponent(preset.value));
    } catch (e) {
      if (request === routeRequest) {
        resolved.textContent = 'config default'; model.disabled = false;
        model.dispatchEvent(new Event('input', { bubbles: true }));
        toast('could not list the models: ' + e.message, true);
      }
      return;
    }
    if (request !== routeRequest) return;
    resolved.textContent = d.default_label;
    model.replaceChildren(resolved, ...(d.routes || []).map(r => new Option(r, r)));
    model.value = '';
    restore(model, draft.model); draft.model = '';
    model.disabled = false;
    model.dispatchEvent(new Event('input', { bubbles: true }));
  };
  restore(mode, draft.mode); restore(preset, draft.preset); restore(seed, pending.seed_from || draft.seed);
  worktree.checked = !!draft.worktree; cap.value = draft.cap || '';
  fold(!!(seed.value || cap.value));
  mode.onchange = () => { followMode(); fillRoutes(); }; preset.onchange = fillRoutes; fillRoutes(); followMode();
  const go = el('button', 'primary', 'Start');
  // Running a picked plan as written needs no task text.
  runOnce(go, async () => {
    task.readOnly = true;
    try {
      const d = await postJSON('/api/new', {
        mode: mode.value, task: task.value, preset: preset.value, model: model.value,
        worktree: mode.value === 'run' && worktree.checked,
        seed_from: mode.value === 'run' ? seed.value : '',
        max_usd: cap.value.trim() ? Number(cap.value) : null,
      });
      if (d.session_id) { forget('a6-draft-new'); location.hash = '#/session/' + encodeURIComponent(d.session_id); }
    } catch (e) { toast(e.message, true); }
    task.readOnly = false;
  }, () => capOk() && (!!task.value.trim() || (!!seed.value && mode.value === 'run')));
  task.addEventListener('input', go.sync); mode.addEventListener('change', go.sync); seed.onchange = go.sync;
  const checkCap = () => { cap.setAttribute('aria-invalid', String(!capOk())); go.sync(); };
  cap.oninput = checkCap; checkCap();
  const save = () => {
    const d = { task: task.value, mode: mode.value, preset: preset.value, model: model.value || draft.model,
      worktree: worktree.checked, seed: seed.value, cap: cap.value };
    if (d.task || d.mode !== 'run' || d.preset || d.model || d.worktree || d.seed || d.cap) remember('a6-draft-new', JSON.stringify(d));
    else forget('a6-draft-new');
  };
  root.addEventListener('input', save); root.addEventListener('change', save);
  const ac = attachParallelSuggest(task, root);
  const commands = attachCommandSuggest(task, root, () => false, ['/parallel', '/review']);
  const refs = attachRefSuggest(task, root);
  task.onkeydown = (e) => {
    if (composing(e)) return;
    if (commands.onKeyDown(e)) return;
    if (ac.onKeyDown(e)) return;
    if (refs.onKeyDown(e)) return;
    if (enterSends(e)) { e.preventDefault(); go.click(); }
  };
  row.appendChild(task); row.appendChild(go);
  const picks = pickerRow([['mode', mode], ['worktree', worktree], ['preset', preset], ['model', model]]);
  picks.appendChild(advanced);
  root.appendChild(pickerSummary([picks, more], () => {
    const run = mode.value === 'run';
    return [mode.value, 'preset: ' + pickedName(preset), pickedModel(model),
      run && worktree.checked && 'worktree', run && seed.value && 'from plan: ' + seed.value,
      cap.value.trim() && 'max USD: ' + cap.value.trim() + (capOk() ? '' : ' (not a number)')];
  }));
  root.appendChild(picks);
  root.appendChild(more);
  root.appendChild(fieldLabel(task, 'Task or question'));
  root.appendChild(row);
  const preview = attachParallelPreview(task, root, () => mode.value === 'run' ? {preset: preset.value} : null);
  mode.addEventListener('change', preview); preset.addEventListener('change', preview);
  const hint = root.appendChild(el('div', 'hint'));
  hint.append(...enterHint('Enter starts the run / plan / ask (a run from a plan needs no task) · Shift+Enter newline',
    'Start begins the run / plan / ask (a run from a plan needs no task) · Enter newline'),
    ' · /parallel [SPEC --] TASK previews child attempts (repeat /parallel for another task) · Ctrl+K switches sessions');
  return root;
}

function createMachineDock() {
  const root = el('div', 'composer dock');
  const row = el('div', 'row');
  const ct = el('textarea', 'field'); ct.placeholder = 'describe a machine to create…';
  const key = 'a6-machine-description';
  ct.value = recall(key, '');
  const cbtn = el('button', 'primary', 'Create machine');
  runOnce(cbtn, async () => {
    ct.readOnly = true;
    try {
      const d = await postJSON('/api/machine/create', { task: ct.value });
      if (recall(key, '') === ct.value) forget(key);
      ct.value = '';
      if (root.isConnected && d.draft) location.hash = '#/draft/' + encodeURIComponent(d.draft);
    } catch (e) { toast(e.message, true); }
    finally { ct.readOnly = false; }
  }, () => !!ct.value.trim());
  ct.oninput = () => { if (ct.value) remember(key, ct.value); else forget(key); cbtn.sync(); };
  ct.onkeydown = (e) => { if (enterSends(e)) { e.preventDefault(); cbtn.click(); } };
  row.appendChild(ct); row.appendChild(cbtn);
  root.appendChild(fieldLabel(ct, 'Machine description'));
  root.appendChild(row);
  root.appendChild(el('div', 'hint')).append(...enterHint('Enter creates a machine draft from the description · Shift+Enter newline',
    'Create machine drafts a machine from the description · Enter newline'));
  return root;
}

const dockHeight = new ResizeObserver((entries) => {
  for (const e of entries) {
    const h = e.borderBoxSize[0].blockSize;
    const covers = getComputedStyle(e.target).position === 'sticky' ? h : 0;
    document.documentElement.style.setProperty('--dock-h', covers + 'px');
    if (!h) dockHeight.unobserve(e.target); // a dock that left the page reports 0
  }
});
new MutationObserver(() => { const dock = view.querySelector('.composer.dock'); if (dock) dockHeight.observe(dock); }).observe(view, { childList: true });
// At the short-screen breakpoint a dock can stop sticking without changing size.
matchMedia('(height < 36em)').addEventListener('change', () => {
  const dock = view.querySelector(':scope > .composer.dock');
  if (dock) { dockHeight.unobserve(dock); dockHeight.observe(dock); }
});
