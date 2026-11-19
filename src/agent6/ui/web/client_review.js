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
function findingCard(entry) {
  const f = entry.observation.finding, loc = f.location;
  const card = el('span', 'finding'); card.dataset.finding = entry.finding.finding_id;
  card.dataset.side = loc.side;
  card.appendChild(el('span', 'finding-title', f.severity + ' · ' + entry.finding.status + ' · ' + f.title));
  const range = loc.start_line == null ? ' (file)' : ':' + loc.start_line
    + (loc.end_line !== loc.start_line ? '-' + loc.end_line : '');
  card.appendChild(el('span', 'muted', entry.finding.finding_id + ' · ' + loc.side + ' · ' + loc.path + range));
  card.appendChild(el('span', 'finding-detail', f.detail));
  const links = el('span', 'row wrap'); card.appendChild(links);
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
    if (panel?.target) entries.push({finding, observation, panel, session_id, version: review.version, matches: true});
  }
  return entries;
}
function findingRequestId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2, '0')).join('');
}
function findingActions(card, entry, cards, state) {
  const base = cards._base + '/findings/' + encodeURIComponent(entry.finding.finding_id);
  // A button busy with its request reads disabled (aria-disabled) but keeps the keyboard focus.
  const send = async (button, verb, body) => {
    if (button.hasAttribute('aria-disabled')) return;
    button.setAttribute('aria-disabled', 'true');
    try {
      const result = await postJSON(base + '/' + verb, body);
      toast(result.message || result.status);
      const snap = await getJSON(cards._base);
      if (cards.findings.isConnected) paintRun(cards, snap);
    } catch (e) { toast(e.message, true); }
    finally { button.removeAttribute('aria-disabled'); }
  };
  const row = el('span', 'row wrap'); card.appendChild(row);
  if (entry.finding.status === 'open') {
    const dead = notLive(state), fix = el('button', 'mini', dead ? 'Resume to fix' : 'Fix');
    const request_id = findingRequestId();
    fix.disabled = entry.finding.fix_state === 'delivery_unknown';
    fix.onclick = async () => {
      if (fix.hasAttribute('aria-disabled') || (dead && !await confirmDialog('Resume this session to reproduce and fix ' + entry.finding.finding_id + '?', 'The finding stays open.', 'Resume to fix'))) return;
      return send(fix, 'fix', {request_id, expected_version: entry.version, resume: dead});
    };
    row.appendChild(fix);
  }
  const verb = entry.finding.status === 'open' ? 'resolve' : 'reopen';
  const button = el('button', 'mini', verb === 'resolve' ? 'Resolve' : 'Reopen'); row.appendChild(button);
  const form = el('form', 'finding-form'); card.appendChild(form);
  const disposition = el('select', 'field'); disposition.setAttribute('aria-label', 'Disposition');
  for (const value of ['fixed', 'false_positive', 'intentional', 'wont_fix', 'duplicate', 'pre_existing']) {
    const option = el('option', null, value); option.value = value; disposition.appendChild(option);
  }
  const duplicate = el('input', 'field'); duplicate.setAttribute('aria-label', 'Duplicate finding ID');
  duplicate.placeholder = 'Another finding ID in this session'; duplicate.hidden = true;
  disposition.onchange = () => { duplicate.hidden = disposition.value !== 'duplicate'; duplicate.required = !duplicate.hidden; };
  if (verb === 'resolve') { form.appendChild(disposition); form.appendChild(duplicate); }
  const reason = el('textarea', 'field'); reason.setAttribute('aria-label', 'Reason'); reason.maxLength = 2000; reason.required = true;
  reason.placeholder = 'Reason (required, at most 2,000 characters)'; form.appendChild(reason);
  const submit = el('button', 'mini', verb === 'resolve' ? 'Submit resolution' : 'Submit reopen'); submit.type = 'submit'; form.appendChild(submit);
  const request_id = findingRequestId();
  form.onsubmit = event => {
    event.preventDefault();
    if (!reason.value.trim()) { reason.focus(); return; }
    const body = {request_id, expected_version: entry.version};
    if (verb === 'resolve') body.resolution = {disposition: disposition.value, reason: reason.value,
      ...(disposition.value === 'duplicate' ? {duplicate_of: duplicate.value.trim()} : {})};
    else body.reason = reason.value;
    return send(submit, verb, body);
  };
  // The button shows and hides its form and says which, as Receipt does.
  const show = open => { form.hidden = !open; button.setAttribute('aria-expanded', String(open)); button.classList.toggle('active', open); };
  show(false);
  button.onclick = () => { show(form.hidden); if (!form.hidden) reason.focus(); };
}
function paintFindings(cards, state) {
  if (!cards.findings) return;
  const review = state.review || {version: 0, panels: [], findings: []};
  const key = JSON.stringify([review, notLive(state)]);
  if (cards._findings_key === key) return;
  cards._findings_key = key;
  const host = cards.findings; host.innerHTML = '';
  // The counts lead the card with the Receipt button at their end; the receipt opens under them.
  const counts = review.open_counts || {};
  const head = el('div', 'row finding-head');
  head.appendChild(el('div', 'finding-counts', ['block', 'warn', 'nit'].map(s => (counts[s] || 0) + ' ' + s).join(' · ')));
  host.appendChild(head);
  if (!cards._readOnly) {
    // Keep the receipt's fetch target and output across streamed findings refreshes.
    if (!cards._receipt) {
      // A press during the fetch does nothing: disabling the button would drop the keyboard focus.
      const inspect = el('button', 'mini', 'Receipt'), receipt = el('div', 'receipt');
      let open = false, loading = false;
      const mark = shown => {
        open = shown;
        inspect.setAttribute('aria-expanded', String(shown)); inspect.classList.toggle('active', shown);
      };
      mark(false);
      inspect.onclick = async () => {
        if (loading) return;
        receipt.innerHTML = '';
        if (open) { mark(false); return; }
        mark(true);
        loading = true;
        receipt.appendChild(el('div', 'muted', 'loading the receipt…'));
        try { drawReceipt(receipt, await getJSON(cards._base + '/receipt')); }
        catch (e) { receipt.innerHTML = ''; mark(false); toast(e.message, true); }
        finally { loading = false; }
      };
      cards._receipt = {inspect, receipt};
    }
    head.appendChild(cards._receipt.inspect);
    host.appendChild(cards._receipt.receipt);
  }
  const guard = reviewMergeReason(review);
  const facts = el('div', 'kv');
  facts.appendChild(el('div', 'k', 'review')); facts.appendChild(el('div', 'v', review.summary || 'unknown'));
  facts.appendChild(el('div', 'k', 'merge'));
  facts.appendChild(el('div', guard ? 'v err' : 'v', guard || 'no open block findings'));
  host.appendChild(facts);
  if (review.legacy_notice) host.appendChild(el('div', 'muted', review.legacy_notice));
  if (!cards._readOnly && (review.findings || []).some(f => f.status === 'open')) {
    // One steer for every open finding without a queued Fix; the per-finding Fix stays below.
    const dead = notLive(state), all = el('button', 'mini', dead ? 'Resume to fix all open' : 'Fix all open');
    all.onclick = async () => {
      if (all.hasAttribute('aria-disabled') || (dead && !await confirmDialog('Resume this session to reproduce and fix every open finding?', 'The findings stay open.', 'Resume to fix'))) return;
      all.setAttribute('aria-disabled', 'true'); // reads disabled, and keeps the keyboard focus
      try {
        const result = await postJSON(cards._base + '/findings/fix', {all_open: true, request_id: findingRequestId(), resume: dead});
        toast(result.results.map(r => r.finding_id + ': ' + r.message).join('; '));
        const snap = await getJSON(cards._base);
        if (cards.findings.isConnected) paintRun(cards, snap);
      } catch (e) { toast(e.message, true); }
      finally { all.removeAttribute('aria-disabled'); }
    };
    host.appendChild(all);
  }
  // One block per panel: its outcome (the twelve-character id the conversation prints) over its rules.
  for (const panel of review.panels || []) {
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
    host.appendChild(block);
  }
  const status = el('select', 'field'); status.setAttribute('aria-label', 'Finding status');
  const severity = el('select', 'field'); severity.setAttribute('aria-label', 'Finding severity');
  for (const [select, values, value] of [[status, ['open', 'resolved', 'all'], cards._finding_status || 'open'], [severity, ['all', 'block', 'warn', 'nit'], cards._finding_severity || 'all']]) {
    for (const item of values) { const option = el('option', null, item); option.value = item; select.appendChild(option); }
    select.value = value;
  }
  host.appendChild(pickerRow([['status', status], ['severity', severity]]));
  const list = el('div'); host.appendChild(list);
  const paint = () => {
    cards._finding_status = status.value; cards._finding_severity = severity.value; list.innerHTML = '';
    // No session id: a card on its own session's page links to no Session findings.
    for (const entry of reviewEntries(review, '')) {
      if (status.value !== 'all' && entry.finding.status !== status.value) continue;
      if (severity.value !== 'all' && entry.observation.finding.severity !== severity.value) continue;
      const card = findingCard(entry); list.appendChild(card);
      if (entry.finding.fix_state !== 'none') card.appendChild(el('span', 'muted', 'Fix: ' + entry.finding.fix_state.replaceAll('_', ' ')));
      for (const action of entry.finding.actions) card.appendChild(el('span', 'finding-history', action.type.split('.').at(-1)
        + (action.resolution ? ' · ' + action.resolution.disposition : '') + ': '
        + (action.resolution ? action.resolution.reason : action.reason || action.error || action.action || '')));
      if (!cards._readOnly) findingActions(card, entry, cards, state);
    }
    if (!list.children.length) list.appendChild(el('div', 'muted', 'No findings in this filter.'));
  };
  status.onchange = severity.onchange = paint; paint();
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
    if (t.binary) box.appendChild(el('div', 'muted', 'Binary file: no text.'));
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
    line.appendChild(el('span', null, reviewer ? `reviewer ${reviewer.reviewer_index + 1} (${reviewer.focus}) ` : `child ${child.number} `));
    line.appendChild(link);
    host.appendChild(line);
  }
  for (const v of receipt.verifications) {
    for (const stream of ['stdout', 'stderr']) {
      if (!v[stream + '_tail']) continue; // a stream that wrote nothing has no fold to open
      const output = el('details'); host.appendChild(output);
      // The check's id at the twelve characters the panel lines use, then its stream.
      output.appendChild(el('summary', null, (v.verify_id ? v.verify_id.slice(0, 12) : 'unrecorded id') + ' ' + stream));
      output.appendChild(el('pre', null, v[stream + '_tail']));
    }
  }
  const ask = el('button', 'mini', 'Ask a run to revert'); host.appendChild(ask);
  ask.onclick = () => {
    handToNewWork({ task: receipt.revert_request });
    location.hash = '#/';
  };
}
