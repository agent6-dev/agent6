// Findings keep their recorded comparison and side when the surrounding view changes.
function reviewQuery(entry) {
  const loc = entry.observation.finding.location;
  return new URLSearchParams({panel: entry.panel.panel_id, finding: entry.finding.finding_id,
    path: loc.path, side: loc.side}).toString();
}
function reviewLink(entry) {
  const t = entry.panel.target;
  return '#/git/compare/' + (t.base_commit || t.base_tree) + '..' + (t.head_commit || t.head_tree) + '?' + reviewQuery(entry);
}
function reviewFileLink(entry) {
  const loc = entry.observation.finding.location, t = entry.panel.target;
  return '#/files/' + (loc.side === 'base' ? t.base_tree : t.head_tree) + '/'
    + loc.path.split('/').map(encodeURIComponent).join('/') + '?' + reviewQuery(entry);
}
function reviewRules(panel) {
  const box = el('div', 'review-rules');
  const status = panel.rules_complete == null ? 'Rule sources not recorded'
    : panel.rules_complete ? 'Rules complete' : 'Rules incomplete';
  const sources = panel.rule_sources || [];
  const none = panel.rules_complete && !sources.length ? ' · No base rules exist.' : '';
  box.appendChild(el('div', panel.rules_complete === false ? 'err' : 'muted', status + none));
  if (panel.rules_reason) box.appendChild(el('div', 'err', panel.rules_reason));
  for (const source of sources) {
    const row = el('div', 'row wrap'); box.appendChild(row);
    const link = el('a', null, source.path);
    link.href = '#/files/' + panel.target.base_tree + '/'
      + source.path.split('/').map(encodeURIComponent).join('/') + '?'
      + new URLSearchParams({panel: panel.panel_id, side: 'base', path: source.path});
    row.appendChild(link);
    row.appendChild(el('span', 'muted', 'blob ' + source.blob_id + ' · ' + source.supplied_bytes + ' bytes'));
  }
  return box;
}
function findingCard(entry, card = el('span', 'finding')) {
  const f = entry.observation.finding, loc = f.location;
  card.dataset.finding = entry.finding.finding_id;
  card.rowKey = entry.finding.finding_id; // a repaint finds the focused control in it again (refocus)
  card.dataset.side = loc.side;
  const record = card.firstChild || card.appendChild(el('span', 'finding-record'));
  const titleId = record.firstChild?.id; // the form's buttons describe themselves with this title
  record.replaceChildren();
  const title = record.appendChild(el('span', 'finding-title', f.severity + ' · ' + entry.finding.status + ' · ' + f.title));
  if (titleId) title.id = titleId;
  const range = loc.start_line == null ? ' (file)' : ':' + loc.start_line
    + (loc.end_line !== loc.start_line ? '-' + loc.end_line : '');
  record.appendChild(el('span', 'muted', entry.finding.finding_id + ' · ' + f.category + ' · ' + loc.side + ' · ' + loc.path + range));
  record.appendChild(el('span', 'finding-detail', f.detail));
  if (f.suggestion) record.appendChild(el('span', 'finding-detail', 'Suggestion: ' + f.suggestion));
  const links = el('span', 'row wrap'); record.appendChild(links);
  const origin = entry.observation.line_origin;
  if (origin?.kind === 'pre_existing') {
    const base = origin.base_location, badge = el('a', 'finding-origin', 'Pre-existing line');
    badge.title = origin.reason;
    badge.href = '#/files/' + entry.panel.target.base_tree + '/' + base.path.split('/').map(encodeURIComponent).join('/')
      + '?' + new URLSearchParams({panel: entry.panel.panel_id, side: 'base', line: base.start_line, end_line: base.end_line});
    links.appendChild(badge);
    for (const sha of origin.commits) {
      const blame = el('a', null, 'Blame ' + sha.slice(0, 7));
      blame.href = '#/git/commit/' + encodeURIComponent(sha); blame.title = sha;
      links.appendChild(blame);
    }
  }
  const diff = el('a', null, 'Reviewed comparison'); diff.href = reviewLink(entry); links.appendChild(diff);
  const file = el('a', null, 'Reviewed file'); file.href = reviewFileLink(entry); links.appendChild(file);
  if (entry.session_id) {
    const run = el('a', null, 'Session findings'); run.href = '#/session/' + encodeURIComponent(entry.session_id);
    links.appendChild(run);
  }
  for (const link of [...links.children]) describedBy(title, link); // a copy: describedBy() puts a sibling after a titled link
  return card;
}
function drawDiffFindings(host, entries, file, mode) {
  for (const entry of entries) {
    const loc = entry.observation.finding.location;
    if (loc.path !== (loc.side === 'base' ? file.old_path || file.path : file.path)) continue;
    const card = findingCard(entry);
    let row = null;
    if (entry.matches && loc.start_line != null) {
      const side = loc.side === 'base' ? 'old' : 'new';
      const attr = mode === 'old' || mode === 'new' ? (mode === side ? 'line' : null) : side;
      if (attr) row = host.querySelector('[data-' + attr + '="' + loc.start_line + '"]');
    }
    if (row) {
      (mode === 'split' ? row.children[loc.side === 'base' ? 0 : 1] : row).appendChild(card);
    } else {
      if (!entry.matches) card.appendChild(el('span', 'muted', 'Recorded on a different comparison.'));
      else if (loc.start_line != null) card.appendChild(el('span', 'muted', 'Location is outside the displayed lines or side.'));
      host.appendChild(card);
    }
  }
}
function reviewMergeReason(review) {
  const guard = review?.merge_guard;
  if (guard?.read_failure) return 'Cannot read review evidence: ' + guard.read_failure;
  return guard?.open_block_ids?.length ? 'Open block findings: ' + guard.open_block_ids.join(', ') + '. Resolve with a reason before merging.' : '';
}
function reviewEntries(review, session_id, panel_id = '') {
  const entries = [];
  for (const finding of review.findings || []) {
    const observation = panel_id ? finding.observations.find(o => o.panel_id === panel_id) : finding.observations.at(-1);
    if (!observation) continue;
    const panel = review.panels.find(p => p.panel_id === observation.panel_id);
    if (panel?.target) entries.push({finding, observation, panel, session_id, matches: true});
  }
  return entries;
}
function findingRequestId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2, '0')).join('');
}
function findingActions(card, entry, cards) {
  const base = cards._base + '/findings/' + encodeURIComponent(entry.finding.finding_id);
  // A button busy with its request reads disabled (aria-disabled) but keeps the keyboard focus.
  const send = async (button, verb, fields) => {
    if (button.hasAttribute('aria-disabled')) return;
    button.setAttribute('aria-disabled', 'true');
    const intent = JSON.stringify(fields);
    // Until the worker answers, a retry sends the same envelope, including its version.
    if (button.attempt?.intent !== intent) button.attempt = {intent, body: {
      ...fields, request_id: findingRequestId(), expected_version: cards._review.version,
    }};
    try {
      const result = await postJSON(base + '/' + verb, button.attempt.body);
      if (!['queued', 'delivery_unknown'].includes(result.status)) button.attempt = null;
      toast(result.message || result.status);
      const snap = await getJSON(cards._base);
      if (cards.findings.isConnected) paintRun(cards, snap);
    } catch (e) {
      if (e.data?.error && [404, 409, 422].includes(e.status)) button.attempt = null;
      toast(e.message, true);
    }
    finally { button.removeAttribute('aria-disabled'); }
  };
  const row = el('span', 'row wrap'); card.appendChild(row);
  if (entry.finding.status === 'open') {
    const fix = el('button', 'mini'); card.fix = fix;
    fix.onclick = async () => {
      const dead = cards._reviewDead;
      if (fix.hasAttribute('aria-disabled') || (dead && !await confirmDialog('Resume this session to reproduce and fix ' + entry.finding.finding_id + '?', 'The finding stays open.', 'Resume to fix'))) return;
      return send(fix, 'fix', {resume: dead});
    };
    row.appendChild(fix);
  }
  const verb = entry.finding.status === 'open' ? 'resolve' : 'reopen';
  const button = el('button', 'mini', verb === 'resolve' ? 'Resolve' : 'Reopen'); row.appendChild(button);
  const form = el('form', 'finding-form'); card.appendChild(form);
  const disposition = el('select', 'field'); disposition.setAttribute('aria-label', 'Disposition');
  for (const [value, label] of Object.entries(dispositionLabels)) {
    const option = el('option', null, label); option.value = value; disposition.appendChild(option);
  }
  const duplicate = el('input', 'field'); duplicate.setAttribute('aria-label', 'Duplicate finding ID');
  duplicate.placeholder = 'Another finding ID in this session'; duplicate.hidden = true;
  disposition.onchange = () => { duplicate.hidden = disposition.value !== 'duplicate'; duplicate.required = !duplicate.hidden; };
  if (verb === 'resolve') { form.appendChild(disposition); form.appendChild(duplicate); }
  const reason = el('textarea', 'field'); reason.maxLength = 2000; reason.required = true;
  form.append(fieldLabel(reason, reasonLabel), reason);
  const submit = el('button', 'mini', verb === 'resolve' ? 'Submit resolution' : 'Submit reopen'); submit.type = 'submit'; form.appendChild(submit);
  for (const control of card.querySelectorAll('button')) describedBy(card.querySelector('.finding-title'), control);
  form.onsubmit = event => {
    event.preventDefault();
    if (!reason.value.trim()) { reason.focus(); return; }
    const body = verb === 'reopen' ? {reason: reason.value} : {resolution: {
      disposition: disposition.value, reason: reason.value,
      ...(disposition.value === 'duplicate' ? {duplicate_of: duplicate.value.trim()} : {}),
    }};
    return send(submit, verb, body);
  };
  // The button shows and hides its form and says which.
  const show = open => { form.hidden = !open; button.setAttribute('aria-expanded', String(open)); button.classList.toggle('active', open); };
  show(false);
  button.onclick = () => { show(form.hidden); if (!form.hidden) reason.focus(); };
}
function reviewPanel(panel) {
  const result = panel.result, block = el('div', 'review-panel');
  const line = el('div', 'muted', 'Panel ' + panel.panel_id.slice(0, 12) + ': ' + panel.completion
    + (panel.reason ? ' (' + panel.reason + ')' : '')
    + (result ? ' · worker finish ' + (result.blocked ? 'rejected' : 'not rejected') + ' (review.block_on=' + result.block_on + (result.disarmed ? ', rejection cap reached' : '') + ')' : ''));
  line.title = panel.panel_id;
  block.appendChild(line);
  block.appendChild(reviewRules(panel));
  // One line per reviewer: its session holds the reviewer's own conversation.
  for (const reviewer of panel.reviewers || []) if (reviewer.session_id) {
    const who = el('div', 'sub', `Reviewer ${reviewer.reviewer_index + 1} (${reviewer.focus}): `);
    const link = el('a', null, reviewer.session_id);
    link.href = '#/session/' + encodeURIComponent(reviewer.session_id);
    who.appendChild(link);
    block.appendChild(who);
  }
  return block;
}
function paintSessionFinding(card, entry, cards) {
  const sig = JSON.stringify([entry.finding, entry.panel.target]);
  if (sig !== card.recordSig) {
    const fresh = !card.firstChild;
    card.recordSig = sig;
    findingCard(entry, card);
    const record = card.firstChild;
    if (entry.finding.fix_state !== 'none') record.appendChild(el('span', 'muted', 'Fix: ' + entry.finding.fix_state.replaceAll('_', ' ')));
    for (const action of entry.finding.actions) record.appendChild(el('span', 'finding-history', action.type.split('.').at(-1)
      + (action.resolution ? ' · ' + dispositionLabels[action.resolution.disposition] : '') + ': '
      + (action.resolution ? action.resolution.reason : action.reason || action.error || action.action || '')));
    if (fresh && !cards._readOnly) findingActions(card, entry, cards);
  }
  if (card.fix) {
    card.fix.textContent = cards._reviewDead ? 'Resume to fix' : 'Fix';
    card.fix.disabled = entry.finding.fix_state === 'delivery_unknown';
  }
}
function makeFindings(cards) {
  const host = cards.findings, head = el('div', 'row finding-head');
  const counts = head.appendChild(el('div', 'finding-counts'));
  host.appendChild(head);
  if (!cards._readOnly) {
    const inspect = head.appendChild(el('button', 'mini', 'Receipt'));
    inspect.setAttribute('aria-haspopup', 'dialog');
    inspect.onclick = async () => {
      const dialog = openDialog('Receipt', 'wide');
      if (!dialog) return;
      const receipt = dialog.appendChild(el('div', 'receipt'));
      receipt.appendChild(el('div', 'muted', 'loading the receipt…'));
      try {
        const data = await getJSON(cards._base + '/receipt');
        if (dialog.open) drawReceipt(receipt, data);
      } catch (e) { if (dialog.open) receipt.replaceChildren(el('div', 'err', e.message)); }
    };
  }
  const facts = el('div', 'kv'), summary = el('div', 'v'), merge = el('div', 'v');
  facts.append(el('div', 'k', 'review'), summary, el('div', 'k', 'merge'), merge);
  const legacy = el('div', 'muted'), panels = el('div');
  host.append(facts, legacy);
  const all = el('button', 'mini');
  if (!cards._readOnly) {
    host.appendChild(all);
    all.onclick = async () => {
      const dead = cards._reviewDead;
      if (all.hasAttribute('aria-disabled') || (dead && !await confirmDialog('Resume this session to reproduce and fix every open finding?', 'The findings stay open.', 'Resume to fix'))) return;
      all.setAttribute('aria-disabled', 'true');
      try {
        const result = await postJSON(cards._base + '/findings/fix', {all_open: true, request_id: findingRequestId(), resume: dead});
        toast(result.results.map(r => r.finding_id + ': ' + r.message).join('; '));
        const snap = await getJSON(cards._base);
        if (host.isConnected) paintRun(cards, snap);
      } catch (e) { toast(e.message, true); }
      finally { all.removeAttribute('aria-disabled'); }
    };
  }
  const status = el('select', 'field'); status.setAttribute('aria-label', 'Finding status');
  const severity = el('select', 'field'); severity.setAttribute('aria-label', 'Finding severity');
  for (const [select, values] of [[status, ['open', 'resolved', 'all']], [severity, ['all', 'block', 'warn', 'nit']]]) {
    for (const item of values) { const option = el('option', null, item); option.value = item; select.appendChild(option); }
    select.value = values[0];
  }
  const list = el('div'), empty = el('div', 'muted', 'no findings in this filter');
  host.append(panels, pickerRow([['status', status], ['severity', severity]]), list, empty);
  const paintList = () => {
    const entries = reviewEntries(cards._review, '').filter(e =>
      (status.value === 'all' || e.finding.status === status.value) &&
      (severity.value === 'all' || e.observation.finding.severity === severity.value));
    // Resolve and Reopen have different forms. Other changes update the record beside the form.
    patchRows(list, entries.map(e => ({key: e.finding.finding_id, data: e.finding.status, paint: () => el('span', 'finding')})));
    entries.forEach((e, i) => paintSessionFinding(list.children[i], e, cards));
    list.hidden = !entries.length; empty.hidden = !!entries.length;
  };
  status.onchange = severity.onchange = paintList;
  return () => {
    const review = cards._review, held = document.activeElement;
    counts.textContent = ['block', 'warn', 'nit'].map(s => (review.open_counts?.[s] || 0) + ' ' + s).join(' · ');
    summary.textContent = review.summary || 'unknown';
    const guard = reviewMergeReason(review);
    merge.textContent = guard || 'no open block findings'; merge.classList.toggle('err', !!guard);
    legacy.textContent = review.legacy_notice || ''; legacy.hidden = !review.legacy_notice;
    all.textContent = cards._reviewDead ? 'Resume to fix all open' : 'Fix all open';
    all.hidden = !(review.findings || []).some(f => f.status === 'open');
    patchRows(panels, (review.panels || []).map(panel => ({key: panel.panel_id, data: panel, paint: () => reviewPanel(panel)})));
    panels.hidden = !panels.children.length;
    paintList();
    refocus(held, host, host.querySelector(FOCUSABLE));
  };
}
function paintFindings(cards, state) {
  if (!cards.findings) return;
  const review = state.review || {version: 0, panels: [], findings: []};
  const dead = notLive(state), key = JSON.stringify([review, dead]);
  if (cards._findings_key === key) return;
  cards._findings_key = key; cards._review = review; cards._reviewDead = dead;
  // A draft has no Receipt, so its card waits for the first review event.
  if (cards._readOnly) cards.findings.parentElement.style.display = review.version ? '' : 'none';
  cards._paintFindings ||= makeFindings(cards);
  cards._paintFindings();
}
async function renderReviewedFile(gen, path, query) {
  setCrumb('Files · reviewed version');
  const panel_id = query.get('panel'), side = query.get('side');
  const root = '/api/reviews/' + encodeURIComponent(panel_id);
  const panel = await getJSON(root);
  if (gen !== undefined && gen !== routeGen) return;
  view.innerHTML = '';
  const app = el('div', 'files-app page-pad'); view.appendChild(app);
  app.appendChild(el('h3', null, side + ' · ' + path));
  app.appendChild(reviewRules(panel.panel));
  const box = el('div', 'files-text'); app.appendChild(box);
  const load = async whole => {
    const t = await getJSON(root + '/file?' + new URLSearchParams({path, side, ...(whole ? {whole: '1'} : {})}));
    if (!app.isConnected) return;
    box.innerHTML = '';
    box.appendChild(el('div', 'muted', 'Reviewed tree: ' + t.rev));
    if (t.truncated) {
      box.appendChild(el('div', 'err', 'The file is truncated.'));
      const more = el('button', 'mini', 'Show the whole file'); more.onclick = () => load(true).catch(e => toast(e.message, true)); box.appendChild(more);
    }
    if (t.binary) box.appendChild(el('div', 'muted', 'binary file: no text'));
    else box.appendChild(renderLines(splitLines(t.text), t.spans));
    const entries = reviewEntries(panel.review, panel.session_id, panel_id).filter(e => e.observation.finding.location.side === side);
    drawDiffFindings(box, entries, {path}, side === 'base' ? 'old' : 'new');
    focusFinding(box, query.get('finding'));
    const start = Number(query.get('line')), end = Number(query.get('end_line')) || start;
    if (start > 0 && end >= start) {
      for (const row of box.querySelectorAll('[data-line]')) {
        if (Number(row.dataset.line) >= start && Number(row.dataset.line) <= end) row.classList.add('finding-selected');
      }
      box.querySelector('[data-line="' + start + '"]')?.scrollIntoView({block: 'center'});
    }
  };
  await load(false);
}
function focusFinding(host, finding_id) {
  for (const card of host.querySelectorAll('.finding')) if (card.dataset.finding === finding_id) {
    card.classList.add('finding-selected'); card.scrollIntoView({block: 'center'});
  }
}


// The receipt text is the shared rendering (the CLI, TUI and ACP print the same); the page adds
// the retained output bodies as collapsibles and the new-work button.
function drawReceipt(host, receipt) {
  host.innerHTML = '';
  // One block per line, so a wrapped line (a long hash) hangs under its own start.
  const text = el('pre', 'receipt-text');
  for (const line of receipt.text.split('\n')) text.appendChild(el('div', null, line || '\u00a0'));
  host.appendChild(text);
  // Each child the receipt names opens through the session navigation.
  for (const child of receipt.children || []) {
    const line = el('div', 'sub');
    const link = el('a', null, child.session_id);
    link.href = '#/session/' + encodeURIComponent(child.session_id);
    const reviewer = (receipt.review?.panels || []).flatMap(p => p.reviewers || [])
      .find(s => s.session_id === child.session_id);
    const label = child.source === 'panel' ? 'reviewer' : 'attempt';
    line.appendChild(el('span', null, `${child.group} ${label} ${child.number}${reviewer ? ` (${reviewer.focus})` : ''} `));
    line.appendChild(link);
    host.appendChild(line);
  }
  for (const v of receipt.verifications) {
    for (const stream of ['stdout', 'stderr']) {
      if (!v[stream + '_tail']) continue; // a stream that wrote nothing has no fold to open
      const output = el('details'); host.appendChild(output);
      // The check's id at the twelve characters the panel lines use, then its stream.
      output.appendChild(el('summary', 'chevron', (v.verify_id ? v.verify_id.slice(0, 12) : 'unrecorded id') + ' ' + stream));
      output.appendChild(el('pre', null, v[stream + '_tail']));
    }
  }
  const ask = el('button', 'mini', 'Ask a run to revert'); host.appendChild(ask);
  ask.onclick = () => {
    handToNewWork({ task: receipt.revert_request });
    location.hash = '#/';
  };
}
