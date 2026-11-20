// Children in the session's Details column. Only a disclosed row reads its conversation.
const CHILD_REFRESH_MS = 1000;

function childFacts(c) {
  const model = [c.model, c.route !== c.model && c.route].filter(Boolean).join(' · ');
  const usage = c.launched || c.settled
    ? (c.usd_partial && !c.spend_usd ? '$?' : `${c.usd_partial ? '~' : ''}$${c.spend_usd}`) : '';
  const outcome = c.outcome === 'joined'
    ? `joined ${c.joined_to}'s chain` + (c.join_sha ? ` at ${c.join_sha.slice(0, 12)}` : '')
    : c.outcome.replaceAll('_', ' ');
  return [
    [model, c.current_tool, c.turn != null && `turn ${c.turn}`, c.tokens != null && `${c.tokens.toLocaleString()} tokens`,
     usage, c.elapsed_s != null && `${c.elapsed_s}s elapsed`],
    [!c.launched && c.status === 'failed' && 'never started', outcome && outcome !== c.status && outcome,
     c.commits > 0 && `${c.commits} ${c.outcome === 'stopped' ? 'unmerged ' : ''}commit${c.commits === 1 ? '' : 's'}`, c.imported_ref, c.error],
    [c.verdict_order.length && `verdict: ${c.verdict_order.join(', ')}`, c.verdict_summary],
  ].map(line => line.filter(Boolean).join(' · ')).filter(Boolean).join('\n');
}

function childRow(sid, card, readOnly) {
  const base = '/api/session/' + encodeURIComponent(sid);
  const row = el('details', 'child-row');
  const summary = row.appendChild(el('summary', 'chevron'));
  const headline = summary.appendChild(el('strong'));
  const facts = summary.appendChild(el('span', 'child-facts muted sub'));
  const body = row.appendChild(el('div', 'child-body'));
  const link = el('a', null, sid);
  link.href = '#/session/' + encodeURIComponent(sid);
  const recent = el('pre', 'plan');
  const conversation = el('div');
  conversation.append(el('strong', 'sub', 'Recent conversation'), recent);
  const identity = el('span', 'sub', sid);
  const task = el('div');
  body.append(link, identity, task, conversation);

  const controls = el('div');
  const entry = el('textarea', 'field');
  entry.rows = 2;
  const send = el('button', 'primary', 'Send');
  const stop = el('button', 'danger', 'Stop');
  stop.onclick = () => stopRun(base, sid);
  const buttons = el('div', 'row wrap');
  buttons.append(send, stop);
  controls.append(fieldLabel(entry, 'Message'), entry, buttons);
  send.onclick = async () => {
    const draft = entry.value;
    if (!draft.trim() || send.disabled) return;
    send.disabled = true;
    try {
      const d = await postJSON(base + '/steer', { text: draft });
      if (entry.value === draft) entry.value = '';
      toast(d.message || 'message sent');
    } catch (err) { toast(err.message, true); }
    finally { send.disabled = false; }
  };
  entry.onkeydown = e => { if (enterSends(e)) { e.preventDefault(); send.click(); } };

  let child, busy = false, live = false, seenStatus = null;
  row.paint = c => {
    if (child?.task !== c.task) {
      const preview = c.task.slice(0, 120).split('\n', 3).join('\n');
      task.replaceChildren(el('strong', 'sub', 'Prompt'));
      if (preview === c.task) task.append(el('pre', 'child-prompt', c.task));
      else {
        const full = el('details');
        full.append(el('summary', 'chevron', 'Full prompt'), el('pre', 'child-prompt', c.task));
        task.append(el('pre', 'child-prompt', preview.trimEnd() + '…'), full);
      }
    }
    child = c;
    const role = c.source === 'panel' ? 'reviewer' : 'child';
    const title = `${role} ${c.number} · ${c.status} · ${firstLine(c.task, 60) || sid}`;
    if (headline.textContent !== title) headline.textContent = title;
    const meta = childFacts(c);
    if (facts.textContent !== meta) facts.textContent = meta;
    if (!c.launched) { live = false; seenStatus = null; controls.remove(); }
    link.hidden = conversation.hidden = !c.launched;
    identity.hidden = c.launched;
  };
  row.refresh = async () => {
    if (busy || !child?.launched || !row.open || !card.open || !row.checkVisibility()) return;
    if (!live && seenStatus === child.status) return;
    busy = true;
    const status = child.status;
    try {
      const d = await getJSON(base + '/conversation?recent=1');
      if (!row.isConnected || !child.launched) return;
      seenStatus = status;
      live = d.live;
      const text = d.text || (live ? 'No conversation yet.' : 'No conversation recorded.');
      if (recent.textContent !== text) recent.textContent = text;
      if (live && !readOnly) {
        if (!controls.isConnected) body.appendChild(controls);
      } else controls.remove();
    } catch (err) {
      recent.textContent = err.message;
      seenStatus = null;
      controls.remove();
    } finally { busy = false; }
  };
  row.ontoggle = () => { if (row.open) { seenStatus = null; row.refresh(); } };
  return row;
}

function childrenCard(drawer, readOnly) {
  const card = drawer.appendChild(el('details', 'card card-children'));
  card.dataset.w = 'head'; // On a phone the card belongs to Overview.
  const summary = card.appendChild(el('summary', 'chevron'));
  const countsText = summary.appendChild(el('span'));
  const rows = card.appendChild(el('div'));
  let timer;
  const poll = () => {
    clearTimeout(timer);
    if (!card.open || !card.isConnected) return;
    for (const row of rows.children) row.refresh();
    timer = setTimeout(poll, CHILD_REFRESH_MS);
  };
  card.ontoggle = poll;
  return children => {
    card.style.display = children.length ? '' : 'none';
    const counts = new Map();
    for (const c of children) counts.set(c.status, (counts.get(c.status) || 0) + 1);
    const label = 'children: ' + [...counts].map(([word, count]) => `${count} ${word}`).join(', ');
    if (countsText.textContent !== label) countsText.textContent = summary.title = label;
    // Keep each disclosure and its input. patchRows moves existing rows without taking focus.
    patchRows(rows, children.map(c => ({ key: c.session_id, data: c.session_id,
      paint: () => childRow(c.session_id, card, readOnly) })));
    children.forEach((c, i) => rows.children[i].paint(c));
  };
}
