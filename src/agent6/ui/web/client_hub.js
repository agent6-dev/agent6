async function confirmMerge(r) {
  let same = '';
  try { same = (await getJSON('/api/session/' + encodeURIComponent(r.session_id) + '/merge')).same_paths; }
  catch (err) { toast(err.message, true); return false; }
  return confirmDialog('Merge run ' + r.session_id + '?', 'Lands its branch on its base with your git.merge_method, as `agent6 sessions merge` does; the checkout never moves.' + (same ? '\n\n' + same : ''), 'Merge');
}
async function postMerge(r) {
  try { const d = await postJSON('/api/session/' + encodeURIComponent(r.session_id) + '/merge', {}); toast(d.message || 'merged'); route(); return true; }
  catch (err) { toast(err.message, true); return false; }
}

// Do not replace a row under a press or selection during a poll.
const focusables = (row) => [row, ...row.querySelectorAll(FOCUSABLE)];
let refocusing = false;
function patchRows(list, rows, hold = false) {
  const old = new Map([...list.children].map(n => [n.rowKey, n]));
  const kept = new Set();
  let prev = null;
  for (const { key, data, paint } of rows) {
    const sig = JSON.stringify(data);
    let node = old.get(key);
    if (!node || (node.rowSig !== sig && !(hold && (node.matches(':active') || selecting(node))))) {
      const fresh = paint();
      fresh.rowKey = key; fresh.rowSig = sig;
      if (node) {
        const focused = focusables(node).indexOf(document.activeElement);
        node.replaceWith(fresh);
        if (focused >= 0) { refocusing = true; focusables(fresh)[focused]?.focus({ preventScroll: true }); refocusing = false; }
      }
      node = fresh;
    }
    kept.add(node);
    const want = prev ? prev.nextSibling : list.firstChild;
    if (node !== want) {
      if (node.parentNode === list) list.moveBefore(node, want); else list.insertBefore(node, want);
    }
    prev = node;
  }
  for (const n of [...list.children]) if (!kept.has(n)) n.remove();
}
function selecting(node) {
  const s = getSelection();
  return !!s && !s.isCollapsed && s.containsNode(node, true);
}

function listCard(title, empty, rows) {
  const card = el('div', 'card');
  if (title) card.appendChild(el('h2', null, title));
  const list = el('div', 'list');
  list.setAttribute('role', 'list');
  card.appendChild(list);
  // A click after dragging a selection is for copying, not navigation.
  let down = [0, 0];
  list.onpointerdown = (e) => { down = [e.clientX, e.clientY]; };
  list.onclick = list.onauxclick = (e) => {
    const it = e.target.closest('.item');
    if (!it || e.button > 1) return;
    if (e.detail && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) { e.preventDefault(); return; }
    if (e.target.closest('a, button, input, select, textarea, summary, label')) return;
    const link = it.querySelector('a.title');
    if (!link) return;
    // Firefox follows only primary clicks dispatched to a link.
    const middle = e.button === 1;
    if (middle || e.ctrlKey || e.metaKey || e.shiftKey) {
      link.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, ctrlKey: e.ctrlKey || middle, metaKey: e.metaKey || middle, shiftKey: e.shiftKey }));
    } else location.hash = link.hash;
  };
  let stopKey = null;
  const settle = () => {
    const items = [...list.children].filter(n => n.matches('.item'));
    const stop = items.find(n => n.rowKey === stopKey) || items[0];
    for (const n of items) for (const c of n.querySelectorAll(FOCUSABLE)) c.tabIndex = n === stop ? 0 : -1;
  };
  list.onkeydown = (e) => {
    if (!e.target.matches('a.title')) return;
    if (e.key === ' ') { e.preventDefault(); e.target.click(); return; }
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    const titles = [...list.querySelectorAll('.item a.title')], at = titles.indexOf(e.target);
    const to = { ArrowDown: titles[at + 1], ArrowUp: titles[at - 1], Home: titles[0], End: titles.at(-1) }[e.key];
    if (to) { e.preventDefault(); to.focus(); }
  };
  // The browser scrolls the focused link, not the whole row above the dock.
  list.addEventListener('focusin', (e) => {
    const it = e.target.closest('.item');
    if (it && it.rowKey !== stopKey) { stopKey = it.rowKey; settle(); }
    if (refocusing || !e.target.matches(':focus-visible')) return;
    it?.scrollIntoView({ block: 'nearest' });
    e.target.scrollIntoView({ block: 'nearest' });
  });
  card.show = (entries, hold) => {
    patchRows(list, entries.length ? entries.flatMap(rows)
      : [{ key: '', data: empty, paint: () => { const row = el('div', 'empty', empty); row.setAttribute('role', 'listitem'); return row; } }], hold);
    settle();
  };
  return card;
}
function itemRow(paint) {
  const it = el('div', 'item');
  it.setAttribute('role', 'listitem');
  const g = el('div', 'grow');
  it.appendChild(g);
  paint(it, g);
  return it;
}

const expandedGroups = new Set();

function titleLink(cls, text, href) {
  const title = el('a', cls, text);
  title.href = href; title.draggable = false;
  return title;
}
function taskTitle(parent, taskLine, href) {
  const [task, continued] = taskLine.split('\n');
  parent.appendChild(titleLink('title', task, href));
  if (continued) parent.appendChild(el('div', 'continued', continued));
}
// Keep each separator with the part before it when the row wraps.
function subLine(parts) {
  const sub = el('div', 'sub');
  parts = parts.filter(Boolean);
  parts.forEach((part, i) => {
    const f = typeof part === 'string' ? el('span', 'f', part) : el('span');
    if (typeof part !== 'string') f.append(part);
    if (i < parts.length - 1) f.append('\u00a0·');
    sub.append(...(i ? [' ', f] : [f]));
  });
  return sub;
}

// The hub has its own mode column; the wire label includes the mode for listings without one.
function statusOnly(r) {
  return r.status_label.startsWith(r.mode + ' · ') ? r.status_label.slice(r.mode.length + 3) : r.status_label;
}
function paintSession(r, depth, regroup) {
  return itemRow((it, g) => {
    if (depth) { it.classList.add('child'); it.style.setProperty('--depth', depth); }
    taskTitle(g, r.task_line || firstLine(r.task, 200) || '(no task)', '#/session/' + encodeURIComponent(r.session_id));
    const status = [statusOnly(r), r.merge_note].filter(Boolean).join(' · ');
    const cell = esc(r.id_cell || r.session_id), cut = cell.indexOf(' (', r.session_id.length);
    let id = cell;
    if (cut >= 0) { id = el('span'); id.append(el('span', 'f', cell.slice(0, cut)), cell.slice(cut)); }
    g.appendChild(subLine([pill(r.status_level, status), esc(r.mode), id, esc(r.when), esc(r.usage)]));
    // A missing worktree already has its path in the id cell.
    const where = r.worktree && !cell.includes(r.worktree);
    if (where || r.branch) g.appendChild(subLine([where && el('span', null, 'worktree ' + r.worktree), r.branch && el('span', null, 'branch ' + r.branch)]));
    const live = LIVE_WORDS.includes(r.status);
    const title = g.querySelector('a.title');
    if (r.branch && !live) {
      const actions = el('div', 'item-actions');
      const review = el('button', 'mini', 'Review');
      review.onclick = () => fillAsEdit(view.querySelector('.dock textarea'), '/review ' + r.branch);
      actions.appendChild(review);
      describedBy(title, review);
      if (r.unmerged) {
        const merge = el('button', 'mini', 'Merge');
        runOnce(merge, async () => {
          if (!await confirmMerge(r)) return;
          await postMerge(r);
        });
        actions.appendChild(merge);
        describedBy(title, merge);
      }
      it.appendChild(actions);
    }
    const children = r.children || [];
    if (!children.length) return;
    const open = expandedGroups.has(r.session_id);
    const toggle = el('button', 'fold', `children: ${children.length} · ${open ? 'hide' : 'show'}`);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.onclick = regroup;
    describedBy(title, toggle);
    g.appendChild(toggle);
  });
}

function sessionsCard() {
  let shown = [];
  const rows = (r, depth) => {
    const open = expandedGroups.has(r.session_id);
    const regroup = () => {
      if (open) expandedGroups.delete(r.session_id); else expandedGroups.add(r.session_id);
      card.show(shown);
    };
    return [{ key: r.session_id, data: [r, open], paint: () => paintSession(r, depth, regroup) },
      ...(open ? (r.children || []).flatMap(c => rows(c, depth + 1)) : [])];
  };
  const card = listCard('', 'no sessions yet (describe a task below to start one)', r => rows(r, 0));
  const show = card.show;
  card.show = (sessions, hold) => { shown = sessions; show(sessions, hold); };
  const maintenance = el('details', 'maintenance');
  maintenance.appendChild(el('summary', 'chevron', 'Maintenance'));
  const toolbar = el('div', 'toolbar');
  toolbar.appendChild(notificationsButton('mini quiet'));
  const action = (label, run) => { const b = el('button', 'mini quiet danger', label); runOnce(b, run); toolbar.appendChild(b); };
  // Squash merges leave the source commits unreachable; plain prune keeps them.
  const prune = async (deleteSquashed) => {
    if (deleteSquashed && !await confirmDialog('Force-delete branches and chain refs recorded as squash-merged?', 'Their commits stay in the base; each deletion prints an undelete command.', 'Force-delete', true)) return;
    try { const d = await postJSON('/api/sessions/prune', { delete_squashed: deleteSquashed }); toast(d.message || 'pruned', 'hold'); route(); }
    catch (e) { toast(e.message, true); }
  };
  action('Prune merged runs', () => prune(false));
  action('Prune merged runs, squash-merged too', () => prune(true));
  action('Clear saved asks', async () => {
    if (!await confirmDialog('Delete every saved ask?', 'Runs `agent6 sessions remove --asks`: removes every saved `agent6 ask` transcript for this directory. Asks run elsewhere are untouched.', 'Delete', true)) return;
    try { const d = await postJSON('/api/sessions/rm_asks', {}); toast(d.message || 'cleared'); route(); }
    catch (e) { toast(e.message, true); }
  });
  maintenance.appendChild(toolbar);
  card.insertBefore(maintenance, card.firstChild);
  return card;
}

function machinesCard() {
  return listCard('Instances', 'no machines yet (describe one below to create it)', m => [{ key: m.name, data: m, paint: () => itemRow((it, g) => {
    g.appendChild(titleLink('title name', m.name, '#/machine/' + encodeURIComponent(m.name)));
    const machine = m.machine && m.machine !== m.name && el('span', null, 'machine ' + m.machine);
    const at = m.current && el('span', null, 'at ' + m.current);
    g.appendChild(subLine([pill(m.status_level, m.status_label), machine, at, esc(m.when)]));
  }) }]);
}

function draftsCard() {
  return listCard('Drafts', '', d => [{ key: d.session_id, data: d, paint: () => itemRow((it, g) => {
    taskTitle(g, d.task_line || firstLine(d.task, 200) || d.session_id, '#/draft/' + encodeURIComponent(d.session_id));
    g.appendChild(subLine([pill(d.status_level, statusOnly(d)), d.session_id, esc(d.when)]));
  }) }]);
}

function machineFilesCard() {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Run a machine'));
  const frow = el('div', 'list machine-files'); frow.setAttribute('role', 'list');
  const run = (mf) => {
    const b = el('button', null, 'Run');
    b.setAttribute('aria-label', 'Run ' + mf.name);
    runOnce(b, async () => {
      // Starting an instance spends against the configured provider.
      if (!await confirmDialog('Run ' + mf.name + '?', 'It starts a paid machine run now and spends against your provider.', 'Run')) return;
      try { await postJSON('/api/machine/run', { file: mf.path }); toast('started ' + mf.name); setTimeout(route, 800); } catch (e) { toast(e.message, true); }
    });
    return itemRow((row, text) => {
      text.append(el('div', 'title name', mf.name), el('div', 'sub', mf.path));
      row.appendChild(b);
    });
  };
  card.appendChild(frow);
  card.show = (files, hold) => patchRows(frow, files.map(mf => ({ key: mf.path, data: mf, paint: () => run(mf) })), hold);
  return card;
}

async function renderHub(focus, gen) {
  const machinesTab = focus === 'machines';
  let data;
  try { data = await getJSON('/api/hub'); }
  catch (e) {
    if (gen !== undefined && gen !== routeGen) return;
    showProblem('Could not load this page', e.message, true);
    view.querySelector('.problem p').after(el('p', 'muted', `retrying every ${HUB_POLL_MS / 1000} s`));
    hubTimer = setInterval(() => getJSON('/api/hub').then(route, () => {}), HUB_POLL_MS);
    return;
  }
  if (gen !== undefined && gen !== routeGen) return;
  view.innerHTML = '';
  const lists = el('div', 'grid');
  let show;
  if (machinesTab) {
    const cards = { files: machineFilesCard(), machines: machinesCard(), drafts: draftsCard() };
    show = (d, hold) => {
      const entries = { files: d.machine_files || [], machines: d.machines || [], drafts: d.drafts || [] };
      for (const k in cards) cards[k].show(entries[k], hold);
      patchRows(lists, Object.keys(cards).filter(k => k === 'machines' || entries[k].length)
        .map(k => ({ key: k, data: k, paint: () => cards[k] })));
    };
  } else {
    const sessions = sessionsCard();
    lists.appendChild(sessions);
    show = (d, hold) => sessions.show(d.sessions, hold);
  }
  show(data);
  let statuses = seedStatuses(data.sessions);
  view.appendChild(lists);
  const handed = !machinesTab && !!window.pendingNewWork;
  const dock = machinesTab ? createMachineDock() : newWorkDock(data.presets, data.preset_default_label, data.sessions);
  view.appendChild(dock);
  if (handed) dock.querySelector('textarea').focus();
  watchLive();
  // Poll the lists in place so a refresh cannot discard a draft.
  const refreshLists = async () => {
    if (document.hidden && !osNotifyOn()) return;
    if (gen !== undefined && gen !== routeGen) return;
    let next;
    try { next = await getJSON('/api/hub'); }
    catch (e) { if (gen === undefined || gen === routeGen) unanswered(e.message, HUB_POLL_MS); return; }
    if (gen !== undefined && gen !== routeGen) return;
    answered();
    announceSessions(statuses, next.sessions);
    statuses = seedStatuses(next.sessions);
    show(next, true);
  };
  hubTimer = setInterval(refreshLists, HUB_POLL_MS);
  // A foreground tab should not have to wait out the poll interval.
  hubVisWake = () => { refreshLists(); };
  document.addEventListener('visibilitychange', hubVisWake);
}
