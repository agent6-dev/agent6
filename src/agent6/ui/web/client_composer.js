// A fork created by /undo restores the text in its own composer.
let pendingComposerFill = '';

function openRestate(text) {
  const dialog = openDialog('Since your last message', 'wide');
  if (!dialog) return;
  dialog.appendChild(el('pre', null, text));
}

// Journal history includes instructions sent before resume and from other surfaces.
function openHistorySearch(id, onPick) {
  const entries = () => getJSON('/api/session/' + encodeURIComponent(id) + '/conversation').then(d => {
    const seen = new Set(), lines = [];
    for (const t of (d.operator_inputs || []).slice().reverse()) {
      const line = t.split(/\s+/).join(' ').trim();
      if (line && !seen.has(line)) { seen.add(line); lines.push(line); }
    }
    return lines;
  });
  openPicker({
    title: 'Search past messages',
    placeholder: 'type to narrow…',
    rows: entries,
    empty: '(no past messages yet)',
    limit: 8,
    typed: true,
    haystack: (t) => t.toLowerCase(),
    paint: (t, o) => { o.textContent = t; },
    pick: onPick,
  });
}

const SEND_ICON = '<svg class="ic" viewBox="0 0 24 24" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" fill="none" stroke="currentColor" aria-hidden="true"><path d="M4 4 21.5 12 4 20l3-8z"/><path d="M7 12h7"/></svg>';
function makeComposer(id) {
  const root = el('div', 'composer');
  const row = el('div', 'row');
  const ta = el('textarea', 'field');
  const sendBtn = el('button', 'primary send');
  sendBtn.innerHTML = SEND_ICON;
  const verb = sendBtn.appendChild(el('span'));
  sendBtn.onmousedown = (e) => e.preventDefault(); // a click or a tap leaves the focus, and a phone's keyboard, in the entry
  const draftKey = 'a6-draft-' + id;
  ta.value = recall(draftKey, '');
  ta.oninput = () => { if (ta.value) remember(draftKey, ta.value); else forget(draftKey); sendBtn.sync(); };
  const clearDraft = () => { ta.value = ''; ta.dispatchEvent(new Event('input')); };
  const hint = el('div', 'hint');
  const refusal = el('div', 'refusal');
  // A resume default follows the last execution and the selected preset.
  const preset = el('select', 'field');
  const presetDefault = el('option', null, '…'); presetDefault.value = ''; preset.appendChild(presetDefault);
  const model = el('select', 'field');
  const modelDefault = el('option', null, '…'); modelDefault.value = ''; model.appendChild(modelDefault);
  const presetRow = pickerRow([['continue under preset', preset], ['model', model]]);
  const summary = pickerSummary([presetRow], () => ['preset: ' + pickedName(preset), pickedModel(model)]);
  presetRow.style.display = summary.style.display = 'none';
  let labelRequest = 0;
  const relabel = () => {
    const request = ++labelRequest;
    getJSON('/api/session/' + encodeURIComponent(id) + '/resume_defaults?preset=' + encodeURIComponent(preset.value))
      .then(d => {
        if (request !== labelRequest) return;
        presetDefault.textContent = d.preset_label;
        modelDefault.textContent = d.model_label;
        model.dispatchEvent(new Event('input', { bubbles: true }));
      }).catch(() => {});
  };
  preset.onchange = relabel;
  let rowShown = false;
  let presetsFilled = false;
  const fillPresets = () => {
    if (presetsFilled) return;
    presetsFilled = true;
    getJSON('/api/config/suggest/preset').then(d => {
      for (const p of (d.values || [])) { const o = el('option', null, p); o.value = p; preset.appendChild(o); }
    }).catch(() => {});
    getJSON('/api/routes').then(d => {
      for (const r of (d.routes || [])) { const o = el('option', null, r); o.value = r; model.appendChild(o); }
    }).catch(() => {});
  };
  let finished = null; // unknown until the first SSE frame
  let needsWork = false;
  let refused = '';
  let busy = false;
  let modelGit = false;
  const suggest = attachCommandSuggest(ta, root, () => {
    const rows = finished === false ? composerCommands.steer : composerCommands.resume;
    return modelGit ? rows.filter(([command]) => command !== '/undo') : rows;
  });
  const models = attachParallelSuggest(ta, root);
  const refs = attachRefSuggest(ta, root, () => id);
  const attached = attachImages(ta, root, () => sendBtn.sync());
  const apply = () => {
    const closed = !!finished && !!refused;
    ta.readOnly = busy; // not disabled: the entry keeps the keyboard focus
    ta.disabled = closed;
    refusal.textContent = closed ? 'cannot resume: ' + refused : '';
    ta.title = refusal.textContent;
    hint.hidden = closed;
    const word = finished ? 'Resume' : 'Send';
    verb.textContent = sendBtn.title = word;
    sendBtn.setAttribute('aria-label', word);
    sendBtn.sync();
    if (closed) {
      presetRow.style.display = summary.style.display = 'none';
      ta.placeholder = '';
      rowShown = false;
      return;
    }
    presetRow.style.display = summary.style.display = finished ? '' : 'none';
    if (finished) fillPresets();
    if (finished && !rowShown) relabel();
    rowShown = !!finished;
    if (busy) { hint.textContent = 'resuming…'; return; }
    const keys = ' · Ctrl+R past messages · Ctrl+K switch session · ? keys';
    if (finished && needsWork) {
      ta.placeholder = 'what should it do next…';
      hint.replaceChildren(...enterHint('Enter resumes it with your instruction · Shift+Enter newline',
        'Resume continues it with your instruction · Enter newline'), keys);
    } else if (finished) {
      ta.placeholder = 'continue this session…';
      hint.replaceChildren(...enterHint('Enter resumes this session with the instruction (empty = just resume) · Shift+Enter newline',
        'Resume continues this session with the instruction (empty = just resume) · Enter newline'), keys);
    } else {
      ta.placeholder = 'steer… (/ for directives)';
      hint.replaceChildren(...enterHint('Enter sends the instruction at the session\'s next safe boundary · Ctrl+Enter sends it now · Shift+Enter newline',
        'Send injects the instruction at the session\'s next safe boundary · Enter newline'), keys);
    }
  };
  const resume = async (text) => {
    busy = true; apply();
    try {
      await postJSON('/api/session/' + encodeURIComponent(id) + '/resume', attached.into({ text, preset: preset.value, model: model.value }));
      const under = [preset.value && 'preset ' + preset.value, model.value && 'model ' + model.value].filter(Boolean).join(', ');
      toast(under ? 'resuming the run under ' + under + '…' : 'resuming the run…');
      // A parked run is already unfinished; only worker liveness confirms a resume.
      for (let i = 0; i < 25; i++) {
        await new Promise(r => setTimeout(r, 1000));
        if (!root.isConnected) return;
        let s; try { s = await getJSON('/api/session/' + encodeURIComponent(id)); } catch (_) { continue; }
        if (s && s.live === true) { clearDraft(); attached.clear(); route(); return; }
      }
      toast('the resume has not started yet; check `agent6 sessions`', true);
    } catch (e) { toast(e.message, true); }
    busy = false; apply();
  };
  if (pendingComposerFill) {
    const fill = pendingComposerFill;
    pendingComposerFill = '';
    setTimeout(() => fillAsEdit(ta, fill), 0); // after the view attaches
  }
  ta.onkeydown = (e) => {
    if (composing(e)) return;
    if (suggest.onKeyDown(e) || models.onKeyDown(e) || refs.onKeyDown(e)) return;
    if (e.key === 'r' && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
      e.preventDefault();
      if (!busy) openHistorySearch(id, (text) => fillAsEdit(ta, text));
      return;
    }
    if (!enterSends(e)) return;
    e.preventDefault();
    send(e.ctrlKey);
  };
  const send = runOnce(sendBtn, (now) => {
    const typed = ta.value.trim();
    const text = now && !finished && typed && !typed.startsWith('/') ? '/now ' + typed : typed;
    if (text === '/stop') {
      // The typed stop is already deliberate; the Stop button asks for confirmation.
      if (finished) { toast('nothing to stop: the session is not live', true); return; }
      return postJSON('/api/session/' + encodeURIComponent(id) + '/stop', {})
        .then(d => { toast(d.message || 'stopped'); clearDraft(); })
        .catch(err => toast(err.message, true));
    }
    if (text === '/restate') {
      return getJSON('/api/session/' + encodeURIComponent(id) + '/restate')
        .then(d => { openRestate(d.text || ''); clearDraft(); })
        .catch(err => toast(err.message, true));
    }
    if (text === '/shells') {
      clearDraft();
      const card = document.querySelector('[data-w="shells"]');
      if (!card || card.style.display === 'none') { toast('no background commands this run'); return; }
      if (document.querySelector('.drawer.closed')) document.querySelector('.details-btn').click();
      const pick = document.querySelector('.wmenu button[data-w="shells"]');
      if (pick) pick.click();
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (text === '/undo') {
      if (modelGit) { toast('the model owns git in this run: no agent6 commit chain', true); return; }
      // A live undo forks at the next boundary; its stream identifies the new session.
      if (!finished) {
        return postJSON('/api/session/' + encodeURIComponent(id) + '/steer', { text })
          .then(() => { toast('undo requested; applies at the next step'); clearDraft(); })
          .catch(err => toast(err.message, true));
      }
      return postJSON('/api/session/' + encodeURIComponent(id) + '/undo', {})
        .then(d => {
          toast('undone: forked to ' + d.new_session_id + (d.notice ? '\n' + d.notice : ''));
          pendingComposerFill = d.undone_text || '';
          location.hash = '#/session/' + encodeURIComponent(d.new_session_id);
        })
        .catch(err => toast(err.message, true));
    }
    if (!finished) {
      return postJSON('/api/session/' + encodeURIComponent(id) + '/steer', attached.into({ text }))
        .then(r => { toast((r && r.message) || 'steer sent'); clearDraft(); attached.clear(); })
        .catch(err => toast(err.message, true));
    }
    return resume(text);
  }, () => finished !== null && !busy && !(finished && refused) && (!!ta.value.trim() || attached.count() > 0 || (finished && !needsWork)), ta);
  row.appendChild(ta); row.appendChild(sendBtn);
  root.appendChild(summary); root.appendChild(presetRow); root.appendChild(attached.head(fieldLabel(ta, 'Message to the run'))); root.appendChild(attached.tray); root.appendChild(row); root.appendChild(refusal); root.appendChild(hint);
  const preview = attachParallelPreview(ta, root, () => ta.disabled ? null : {preset: finished ? preset.value : '', session_id: id});
  preset.addEventListener('change', preview);
  root.quote = (note) => {
    const typed = ta.value.replace(/\s+$/, '');
    ta.value = (typed ? typed + '\n\n' : '') + note;
    ta.dispatchEvent(new Event('input'));
    ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
  };
  root.setState = (s) => {
    if (busy) return;
    modelGit = s.git_control === 'model';
    needsWork = s.needs_new_work === true;
    refused = s.unreadable || s.manifest_resume_blocker || '';
    if (typeof s.live === 'boolean') { finished = !s.live; apply(); }
    else { finished = notLive(s); apply(); }
  };
  apply();
  return root;
}
