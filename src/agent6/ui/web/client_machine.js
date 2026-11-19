// --- machine watch -----------------------------------------------------------
async function renderMachine(name, gen) {
  const base = '/api/machine/' + encodeURIComponent(name);
  // The top bar names the instance first, so an error view stands under the name it is about.
  setCrumb(name);
  // Existence + readability probe: a bad name or a corrupt machine throws here
  // and route() shows the error (the SSE error frame alone leaves a hollow view).
  const initial = await getJSON(base);
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint or open a stream
  view.innerHTML = '';
  const app = el('div', 'machine-app');
  // The prompts and the actions flow in one block, so their margins collapse;
  // the view is a flex column, where margins add up.
  const top = el('div', 'machine-top'); app.appendChild(top);
  // The prompts host holds pending approval/question boxes; it is appended to, never wiped,
  // so a repaint can never clear a half-typed answer.
  const prompts = el('div', 'page-pad'); top.appendChild(prompts);
  const cards = { _prompts: prompts, _base: base };

  // The page's actions sit on top, as the run view's do, and the command palette offers them.
  // Stop, as `agent6 machine stop` and the TUI machine watch do: the machine parks at its next
  // transition and `machine run` resumes it.
  const actions = el('div', 'row wrap page-pad actions');
  const stopBtn = el('button', 'danger', '■ Stop');
  stopBtn.onclick = async () => {
    if (!confirm(`Stop ${name}? It parks at its next transition, and agent6 machine run ${name} resumes it.`)) return;
    try { const d = await postJSON(base + '/stop', {}); toast(d.message || 'stop requested'); }
    catch (e) { toast(e.message, true); }
  };
  actions.append(stopBtn, notificationsButton());
  top.appendChild(actions);

  // The States and Path cards, each a heading over a body that scrolls on a wide screen.
  const grid = el('div', 'grid machine-grid');
  const addCard = (title) => {
    const card = el('div', 'card'); card.appendChild(el('h2', null, title));
    const body = el('div', 'card-body'); card.appendChild(body);
    grid.appendChild(card);
    return body;
  };
  const structBody = addCard('States');
  const pathBody = addCard('Path');
  // The machine's summary and status stay over the States list as it scrolls.
  cards._summary = el('div', 'row wrap'); structBody.before(cards._summary);
  // The newest agent state's conversation, the same folded view a run shows, under the pair;
  // paintMachine titles it by that state.
  const cc = convCard(base + '/conversation', 'Conversation');
  cards._conv = cc.conv;
  cards._conv_title = cc.card.querySelector('h2');
  grid.appendChild(cc.card);
  app.appendChild(grid);
  cc.conv.refresh();

  // The machine composer, docked at the bottom: one text entry with the two
  // machine verbs the TUI machine watch offers (ui/keymap.py names its keys).
  // Steer injects into the current agent state at its next safe boundary
  // (blank = continue); Message is a poke payload a waiting machine's next
  // tool reads (blank = a bare wake). Created once, so the input survives
  // repaints; paintMachine gates the buttons and fills the one Enter sends.
  const dock = el('div', 'composer dock');
  const drow = el('div', 'row');
  const din = el('textarea', 'field');
  const steerBtn = el('button', null, 'Steer');
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
    if (e.key === 'Enter' && !e.shiftKey && !composing(e)) {
      e.preventDefault();
      if (!steerBtn.disabled) steerBtn.click();
      else if (!msgBtn.disabled) msgBtn.click();
    }
  };
  drow.append(din, steerBtn, msgBtn);
  // Why a dimmed verb is refused, on every screen: a hover title never reaches a phone.
  const refusal = el('div', 'refusal');
  const hint = el('div', 'hint', 'Steer injects into the current agent state · Message wakes a waiting machine (its next tool reads it) · Enter sends, Shift+Enter newline');
  dock.append(drow, refusal, hint);
  app.appendChild(dock);
  view.appendChild(app);
  cards._input = din; cards._steer_btn = steerBtn; cards._msg_btn = msgBtn; cards._stop_btn = stopBtn;
  cards._refusal = refusal; cards._hint = hint;

  // Notification de-dup across repaints: seed with history on the first frame so
  // opening a machine does not replay every past notification; toast + OS-notify
  // only genuinely new ones, named by the instance.
  const ctx = {
    name, seen: null, lostNotified: false, endedNotified: false, shownCurrent: null, pathAtEnd: true,
  };
  // Path follows its newest transition while the reader stays at its end: each scroll records
  // whether they do, and a resize (a prompt above it, the entry growing below) scrolls back to it.
  pathBody.addEventListener('scroll', () => {
    ctx.pathAtEnd = pathBody.scrollTop + pathBody.clientHeight >= pathBody.scrollHeight - 8;
  });
  new ResizeObserver(() => { if (ctx.pathAtEnd) pathBody.scrollTop = pathBody.scrollHeight; }).observe(pathBody);
  paintMachine(structBody, pathBody, cards, ctx, { machine: initial, reasoning: {} });

  live = openStream(base + '/events', raw => {
    let data; try { data = JSON.parse(raw); } catch (_) { return; }
    if (data.type === 'error') { closeLive(); toast('stream error: ' + data.error, true); return; }
    paintMachine(structBody, pathBody, cards, ctx, data);
    hbState.spin++;
    // A stopped machine is resumable and its stream stays open; a journaled end closes it.
    if (data.machine && data.machine.ended) closeLive();
  });
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
}

// A notification, a stop or an end that lands while the page watches shows once as a toast, as
// the hub announces a session and the TUI a machine; a warning, an error or a failed end holds
// until dismissed.
function machineNotify(ctx, m) {
  const notes = m.notifications || [];
  const keyOf = n => (n.ts || '') + '|' + (n.state || '') + '|' + (n.message || '');
  if (ctx.seen === null) {
    // First frame: seed history (notifications and an already-ended machine)
    // silently, so opening a finished machine does not replay past notifications
    // or fire a spurious "ended" toast/OS-notify. Only events that happen while
    // watching fire.
    ctx.seen = new Set(notes.map(keyOf));
    ctx.endedNotified = !!m.ended;
    ctx.lostNotified = !!m.worker_lost; // a stop the page opened on is history, not news
    return;
  }
  for (const n of notes) {
    const k = keyOf(n);
    if (ctx.seen.has(k)) continue;
    ctx.seen.add(k);
    toast(`${ctx.name} · ${esc(n.state)}: ${esc(n.message)}`, n.level);
    osNotify('agent6: ' + ctx.name, n.message || '');
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
  if (!m.worker_lost) ctx.lostNotified = false; // a resume re-arms the stop notice
  if (m.worker_lost && !ctx.lostNotified) {
    // Supervisor loss, not a journaled end: the instance is resumable.
    ctx.lostNotified = true;
    toast(`${ctx.name} stopped: ${esc(m.worker_lost.reason)}; resume it with agent6 machine run ${ctx.name}`);
    osNotify('agent6: ' + ctx.name + ' stopped', m.worker_lost.reason || '');
  }
  if (m.ended && !ctx.endedNotified) {
    ctx.endedNotified = true;
    toast(`${ctx.name} ended: ${esc(m.ended.status)} (${esc(m.ended.reason)})`, m.ended.status !== 'ok');
    osNotify('agent6: ' + ctx.name + ' ' + m.ended.status, m.ended.reason || '');
  }
  // `spend` is the instance total, as `machine status` reports it, rendered server-side.
  const sp = m.spend || {};
  const cost = sp.text ? ' · ' + sp.text : '';
  cards._summary.replaceChildren(
    el('div', 'sub muted', `${esc(m.machine)} v${esc(m.version)} · current: ${esc(m.current)}${cost}`),
    pill(m.status_level, m.status));
  // Each line leads with its mark, or its `[seq] `; a wrapped line continues under the text
  // after the lead (--hang: the lead's width).
  structBody.innerHTML = '';
  const tree = el('div', 'tree');
  let cursor = null;
  for (const st of m.states || []) {
    const line = el('div', 'node' + (st.is_current ? ' cursor' : ''));
    line.append(el('span', 'lead', st.mark + ' '), `${st.name}  (${st.kind})`); // machine_state_mark, server-side
    if (st.is_current) cursor = line;
    tree.appendChild(line);
  }
  structBody.appendChild(tree);

  pathBody.innerHTML = '';
  const path = el('div', 'tree');
  const transitions = m.transitions || [];
  // Each lead is padded to the widest, so every transition starts in one column.
  const width = transitions.reduce((w, t) => Math.max(w, `[${t.seq}] `.length), 0);
  path.style.setProperty('--hang', width + 'ch');
  for (const t of transitions) {
    const lead = `[${t.seq}] `; // format_transition's prefix; the line is rendered server-side
    const line = el('div', 'node');
    line.append(el('span', 'lead', lead.padStart(width)), t.line.slice(lead.length));
    path.appendChild(line);
  }
  if (!transitions.length) path.appendChild(el('div', 'muted', 'no transitions yet'));
  // How the path ends, in the list's own type: a journaled end, or a stop it resumes from.
  if (m.ended) path.appendChild(el('div', 'sub muted', `ended: ${m.ended.status} (${m.ended.reason}) at ${m.ended.state}`));
  if (m.worker_lost) path.appendChild(el('div', 'sub muted', `stopped: ${esc(m.worker_lost.reason)} at ${esc(m.worker_lost.state)}; resumable`));
  pathBody.appendChild(path);
  if (ctx.pathAtEnd) pathBody.scrollTop = pathBody.scrollHeight;
  // States brings the current state into view when the machine moves; a phone's cards flow, so
  // nothing scrolls there.
  if (cursor && ctx.shownCurrent !== m.current) {
    ctx.shownCurrent = m.current;
    const box = structBody.getBoundingClientRect(), at = cursor.getBoundingClientRect();
    if (at.top < box.top || at.bottom > box.bottom) structBody.scrollTop += (at.top + at.bottom - box.top - box.bottom) / 2;
  }
  // The conversation is the newest agent state's. Its dir is `<seq>-<state>`, seq being the
  // transition that execution makes, so the current state's own execution has the path's length.
  const [, seq, ran] = /^(\d+)-(.+)$/.exec(cards._state) || [];
  const current = ran === m.current && Number(seq) === (m.transitions || []).length;
  if (!ran) cards._conv_title.textContent = 'Conversation';
  else cards._conv_title.replaceChildren(current ? 'Current state · ' : 'Last agent state · ', el('span', 'name', ran));

  // Every control is gated and labelled from the wire refusals, the one
  // decision the CLI and the TUI run: a steer needs an open agent state, a
  // poke an open wait, a stop a live worker; the input follows steer and poke.
  // The filled button is the verb Enter sends: Steer, else Message.
  if (cards._steer_btn) {
    cards._steer_btn.disabled = !!refusals.steer;
    cards._steer_btn.classList.toggle('primary', !refusals.steer);
    cards._steer_btn.title = refusals.steer
      || 'inject into the current agent state at its next safe boundary';
  }
  if (cards._msg_btn) {
    cards._msg_btn.disabled = !!refusals.poke;
    cards._msg_btn.classList.toggle('primary', !!refusals.steer && !refusals.poke);
    cards._msg_btn.title = refusals.poke || 'wake a waiting machine; its next tool reads the message';
  }
  if (cards._stop_btn) {
    cards._stop_btn.disabled = !!refusals.stop;
    cards._stop_btn.title = refusals.stop || 'park the machine at its next transition';
  }
  if (cards._input) {
    cards._input.disabled = !!refusals.steer && !!refusals.poke;
    // The line under the dock says why each dimmed verb is refused: a placeholder is one line and
    // cuts a long reason. An end refuses both verbs for one cause, so it says that once. A closed
    // box invites no typing and gives no keyboard tips.
    const why = m.ended ? refusals.steer : [refusals.steer, refusals.poke].filter(Boolean).join(' · ');
    cards._refusal.textContent = why;
    cards._refusal.classList.toggle('ended', !!m.ended);
    cards._input.title = cards._input.disabled ? why : '';
    cards._input.placeholder = cards._input.disabled ? '' : 'steer or message the machine…';
    cards._hint.hidden = cards._input.disabled;
  }

  // The current state's conversation: live turn from this frame, completed
  // turns re-folded on a debounce. A live-but-silent state ticks the heartbeat.
  const r = data.reasoning || {};
  // Gate on the machine's own liveness (agentLive), not the reasoning fold's
  // `finished`: an agent-state per-state log carries no session.stop, so the dir-less
  // fold's `finished` is structurally always false, and a parked/ended/stopped
  // machine would otherwise show its last (finished) turn as live and tick
  // "agent working…" forever. Matches the TUI, which gates on worker liveness.
  cards._conv.setLive(agentLive ? r : { finished: true });
  cards._conv.poke();
  const streaming = r.last_model_call && (r.last_model_call.streamed_thinking || r.last_model_call.streamed_text);
  hbState = {
    // agentLive quiets the beat for a machine that is not working;
    // operator_blocked still quiets a running state blocked on an
    // approval/question (the run pane's rule).
    active: agentLive && !!r.last_model_call && !streaming && !r.operator_blocked,
    role: (r.last_model_call && r.last_model_call.role) || 'agent',
    // Server-computed age, as the run pane uses: anchoring to this frame's
    // arrival would show a state wedged for forty minutes as "working… 3s".
    last: Date.now() - 1000 * (r.last_event_age_s || 0),
    spin: hbState.spin,
  };
  hbTick();
}

