// --- machine watch -----------------------------------------------------------
async function renderMachine(name, gen) {
  const base = '/api/machine/' + encodeURIComponent(name);
  // Existence + readability probe: a bad name or a corrupt machine throws here
  // and route() shows the error (the SSE error frame alone leaves a hollow view).
  const initial = await getJSON(base);
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint or open a stream
  setCrumb(name);
  view.innerHTML = '';
  // Ephemeral notification banners live here; the prompts host holds pending
  // approval/question boxes; both are appended to, never wiped, so a repaint can
  // never clear a half-typed answer.
  const notifs = el('div', 'page-pad'); view.appendChild(notifs);
  const prompts = el('div', 'page-pad'); view.appendChild(prompts);
  const cards = { _prompts: prompts, _base: base };

  const controls = el('div', 'row wrap page-pad'); controls.style.marginBottom = '10px';
  const bell = el('button', null, '🔔 Notifications');
  bell.onclick = enableNotifications;
  controls.appendChild(bell);
  view.appendChild(controls);

  const grid = el('div', 'grid cols2');
  const structCard = el('div', 'card scroll'); structCard.appendChild(el('h2', null, 'States')); const structBody = el('div'); structCard.appendChild(structBody);
  const pathCard = el('div', 'card scroll'); pathCard.appendChild(el('h2', null, 'Path')); const pathBody = el('div'); pathCard.appendChild(pathBody);
  // The current agent state's conversation: the same folded view a run shows,
  // full-width under the states/path pair.
  const cc = convCard(base + '/conversation', 'Current state', 'span2');
  cards._conv = cc.conv;
  grid.appendChild(structCard); grid.appendChild(pathCard); grid.appendChild(cc.card);
  view.appendChild(grid);
  cc.conv.refresh();

  // The machine composer, docked at the bottom: one text entry with the two
  // machine verbs, matching the TUI machine watch (s = Steer, m = Message).
  // Steer injects into the current agent state at its next safe boundary
  // (blank = continue); Message is a poke payload a waiting machine's next
  // tool reads (blank = a bare wake). Created once, so the input survives
  // repaints; paintMachine gates the buttons.
  const dock = el('div', 'composer dock dock-fixed');
  const drow = el('div', 'row');
  const din = el('textarea', 'field');
  din.placeholder = 'steer the agent state, or message the machine…';
  const steerBtn = el('button', 'primary', 'Steer');
  steerBtn.onclick = async () => {
    // cards._state is set each frame to the agent state currently rendered, so
    // the steer routes to that state, not whichever is newest at click time.
    const body = cards._state ? { text: din.value, state: cards._state } : { text: din.value };
    try { await postJSON(base + '/steer', body); toast('steer sent'); din.value = ''; } catch (e) { toast(e.message, true); }
  };
  const msgBtn = el('button', null, 'Message');
  msgBtn.onclick = async () => {
    try { await postJSON(base + '/poke', { message: din.value }); toast('message sent'); din.value = ''; } catch (e) { toast(e.message, true); }
  };
  din.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!steerBtn.disabled) steerBtn.click();
      else if (!msgBtn.disabled) msgBtn.click();
    }
  };
  // Stop, as the CLI (`agent6 machine stop`) and the TUI (x) have: the machine
  // parks at its next transition and `machine run` resumes it.
  const stopBtn = el('button', 'danger', 'Stop');
  stopBtn.onclick = async () => {
    if (!confirm('Stop this machine at its next transition? `agent6 machine run` resumes it.')) return;
    try { const d = await postJSON(base + '/stop', {}); toast(d.message || 'stop requested'); }
    catch (e) { toast(e.message, true); }
  };
  drow.appendChild(din); drow.appendChild(steerBtn); drow.appendChild(msgBtn); drow.appendChild(stopBtn);
  dock.appendChild(growGrip(din));
  dock.appendChild(drow);
  dock.appendChild(el('div', 'hint', 'Steer injects into the current agent state · Message wakes a waiting machine (its next tool reads it) · Enter sends, Shift+Enter newline'));
  view.appendChild(dock);
  cards._input = din; cards._steer_btn = steerBtn; cards._msg_btn = msgBtn; cards._stop_btn = stopBtn;

  // Notification de-dup across repaints: seed with history on the first frame so
  // opening a machine does not replay every past notification; banner + OS-notify
  // only genuinely new ones.
  const ctx = { notifsHost: notifs, seen: null, lostNotified: false, endedNotified: false };
  paintMachine(structBody, pathBody, cards, ctx, { machine: initial, reasoning: {} });

  live = new EventSource(base + '/events');
  live.onmessage = ev => {
    let data; try { data = JSON.parse(ev.data); } catch (_) { return; }
    if (data.type === 'error') { closeLive(); toast('stream error: ' + data.error, true); return; }
    paintMachine(structBody, pathBody, cards, ctx, data);
    hbState.spin++;
    // A stopped machine is resumable and its stream stays open; a journaled end closes it.
    if (data.machine && data.machine.ended) closeLive();
  };
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
}

function machineNotify(ctx, m) {
  const notes = m.notifications || [];
  const keyOf = n => (n.ts || '') + '|' + (n.state || '') + '|' + (n.message || '');
  if (ctx.seen === null) {
    // First frame: seed history (notifications and an already-ended machine)
    // silently, so opening a finished machine does not replay past notifications
    // or fire a spurious "ended" banner/OS-notify. Only events that happen while
    // watching fire.
    ctx.seen = new Set(notes.map(keyOf));
    if (m.ended || m.worker_lost) ctx.endedNotified = true;
    return;
  }
  for (const n of notes) {
    const k = keyOf(n);
    if (ctx.seen.has(k)) continue;
    ctx.seen.add(k);
    const banner = el('div', 'notif-banner ' + esc(n.level || 'info'));
    const g = el('div', 'grow');
    g.appendChild(el('div', 'nb-msg', n.message || ''));
    g.appendChild(el('div', 'nb-sub', `${esc(m.machine || '')} · ${esc(n.state || '')}`));
    const x = el('button', 'nb-x', '×'); x.onclick = () => banner.remove();
    banner.appendChild(g); banner.appendChild(x);
    ctx.notifsHost.appendChild(banner);
    osNotify('agent6: ' + (m.machine || 'machine'), n.message || '');
  }
}

function paintMachine(structBody, pathBody, cards, ctx, data) {
  if (data.error) { structBody.innerHTML=''; structBody.appendChild(el('div', 'err', data.error)); return; }
  const m = data.machine || {};
  // Pending approval/question/steer come from the current agent state's SessionState.
  // Track which per-state dir this frame rendered so prompt answers + steer route
  // to that exact state (ids reset per state; the machine may advance meanwhile).
  cards._state = (data.reasoning || {}).state_dir || '';
  // What each verb can reach, decided once server-side (`machine_verb_refusals`,
  // the same reading the CLI and the TUI gate on). A machine whose answer verb
  // is refused takes no prompt: painting {} reconciles the boxes away. The
  // status word cannot stand in for this: a live machine blocked on an approval
  // reads "waiting", and gating on the word would hide the box it is blocked on
  // from the page that claimed the instance.
  const refusals = m.refusals || {};
  const canAnswer = !refusals.answer;
  const agentLive = !refusals.steer;
  paintPrompts(cards, canAnswer ? (data.reasoning || {}) : {});
  machineNotify(ctx, m);
  if (!m.worker_lost) ctx.lostNotified = false; // a resume re-arms the stop banner
  if (m.worker_lost && !ctx.lostNotified) {
    // Supervisor loss, not a journaled end: the instance is resumable.
    ctx.lostNotified = true;
    const banner = el('div', 'notif-banner error');
    banner.appendChild(el('div', 'grow',
      `${esc(m.machine || '')} stopped: ${esc(m.worker_lost.reason)}; resumable with agent6 machine run`));
    const x = el('button', 'nb-x', '×'); x.onclick = () => banner.remove();
    banner.appendChild(x); ctx.notifsHost.appendChild(banner);
    osNotify('agent6: ' + (m.machine || 'machine') + ' stopped', m.worker_lost.reason || '');
  }
  if (m.ended && !ctx.endedNotified) {
    ctx.endedNotified = true;
    const banner = el('div', 'notif-banner ' + (m.ended.status === 'ok' ? 'info' : 'error'));
    banner.appendChild(el('div', 'grow', `${esc(m.machine || '')} ended: ${esc(m.ended.status)} (${esc(m.ended.reason)})`));
    const x = el('button', 'nb-x', '×'); x.onclick = () => banner.remove();
    banner.appendChild(x); ctx.notifsHost.appendChild(banner);
    osNotify('agent6: ' + (m.machine || 'machine') + ' ' + m.ended.status, m.ended.reason || '');
  }
  structBody.innerHTML = '';
  // `spend` is the instance total, as `machine status` reports it, rendered server-side.
  const sp = m.spend || {};
  const cost = sp.text ? ' · ' + sp.text : '';
  const summary = el('div', 'row wrap');
  summary.appendChild(el('div', 'sub muted',
    `${esc(m.machine)} v${esc(m.version)} · current: ${esc(m.current)}${cost}`));
  summary.appendChild(pill(m.level, m.status));
  structBody.appendChild(summary);
  const tree = el('div', 'tree');
  for (const st of m.states || []) {
    const line = el('div', 'node' + (st.is_current ? ' cursor' : ''));
    line.textContent = `${st.mark} ${st.name}  (${st.kind})`; // machine_state_mark, server-side
    tree.appendChild(line);
  }
  structBody.appendChild(tree);

  pathBody.innerHTML = '';
  const path = el('div', 'tree');
  for (const t of m.transitions || []) path.appendChild(el('div', 'node', t.line)); // format_transition, server-side
  if (!(m.transitions||[]).length) path.appendChild(el('div', 'muted', 'no transitions yet'));
  pathBody.appendChild(path);
  if (m.ended) pathBody.appendChild(el('div', 'sub muted', `ended: ${m.ended.status} (${m.ended.reason}) at ${m.ended.state}`));
  if (m.worker_lost) pathBody.appendChild(el('div', 'sub muted', `stopped: ${esc(m.worker_lost.reason)} at ${esc(m.worker_lost.state)}; resumable`));

  // Every control is gated and labelled from the wire refusals, the one
  // decision the CLI and the TUI run: a steer needs an open agent state, a
  // poke an open wait, a stop a live worker; the input follows steer and poke.
  if (cards._steer_btn) {
    cards._steer_btn.disabled = !!refusals.steer;
    cards._steer_btn.title = refusals.steer
      || 'inject into the current agent state at its next safe boundary';
  }
  if (cards._msg_btn) {
    cards._msg_btn.disabled = !!refusals.poke;
    cards._msg_btn.title = refusals.poke || 'wake a waiting machine; its next tool reads the message';
  }
  if (cards._stop_btn) {
    cards._stop_btn.disabled = !!refusals.stop;
    cards._stop_btn.title = refusals.stop || 'park the machine at its next transition';
  }
  if (cards._input) {
    cards._input.disabled = !!refusals.steer && !!refusals.poke;
    cards._input.title = cards._input.disabled
      ? [refusals.steer, refusals.poke].filter(Boolean).join(' · ') : '';
  }

  // The current state's conversation: live turn from this frame, completed
  // turns re-folded on a debounce. A live-but-silent state ticks the heartbeat.
  const r = data.reasoning || {};
  // Gate on the machine's own liveness (agentLive), not the reasoning fold's
  // `finished`: an agent-state per-state log carries no session.end, so the dir-less
  // fold's `finished` is structurally always false, and a parked/ended/stopped
  // machine would otherwise show its last (finished) turn as live and tick
  // "agent working…" forever. Matches the TUI, which gates on worker liveness.
  cards._conv.setLive(agentLive ? r : { finished: true });
  cards._conv.poke();
  const streaming = r.last_role && (r.last_role.streamed_thinking || r.last_role.streamed_text);
  hbState = {
    // agentLive quiets the beat for a machine that is not working;
    // operator_blocked still quiets a running state blocked on an
    // approval/question (the run pane's rule).
    active: agentLive && !!r.last_role && !streaming && !r.operator_blocked,
    role: (r.last_role && r.last_role.role) || 'agent',
    // Server-computed age, as the run pane uses: anchoring to this frame's
    // arrival would show a state wedged for forty minutes as "working… 3s".
    last: Date.now() - 1000 * (r.last_event_age_s || 0),
    spin: hbState.spin,
  };
  hbTick();
}

