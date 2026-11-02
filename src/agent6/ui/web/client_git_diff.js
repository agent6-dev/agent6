// Each reference names the displayed side and line before attribution is attached.
function diffBlameRows(host, mode, file) {
  const row = (node, side, number, segment) => ({node, side, number: Number(number) || 0, segment,
    path: side === 'base' ? file.old_path || file.path : file.path});
  if (mode === 'split') {
    const rows = [...host.querySelectorAll('.ds-row')];
    return rows.map(r => row(r.children[0], 'base', r.dataset.old, r.dataset.segment))
      .concat([row(null, '', 0)], rows.map(r => row(r.children[1], 'head', r.dataset.new, r.dataset.segment)));
  }
  if (mode === 'old' || mode === 'new') return [...host.querySelectorAll('.sl')]
    .map(r => row(r, mode === 'old' ? 'base' : 'head', r.dataset.line));
  return [...host.querySelectorAll('.dl')].map(r => r.dataset.new
    ? row(r, 'head', r.dataset.new) : row(r, 'base', r.dataset.old));
}

// A file diff owns its request lifetime and its exact-comparison finding anchors.
function gitDiffViewer(body) {
  // -- a file's diff: it takes the whole body; the list is one step back --------------
  const diffView = el('div', 'git-diff'); body.appendChild(diffView);
  const diffs = new Map();
  let showing = 0, blameOn = false;
  const closeDiff = () => { showing++; body.classList.remove('diffing'); diffView.innerHTML = ''; delete diffView.dataset.key; };
  const showDiff = async (ctx, i, whole, fullContext = false) => {
    const request = ++showing;
    const f = ctx.files[i];
    if (ctx.commit) {
      const query = new URLSearchParams({path: f.path});
      if (ctx.query?.get('path') === f.path && ctx.query.has('line')) query.set('line', ctx.query.get('line'));
      setViewHash('#/git/commit/' + encodeURIComponent(ctx.head) + '?' + query);
    }
    const generation = routeGen;
    const current = () => request === showing && generation === routeGen && diffView.isConnected;
    const key = JSON.stringify([ctx.panel || '', ctx.base, ctx.head, f.path, f.old_path, !!whole, fullContext]);
    diffView.dataset.key = key;
    diffView.innerHTML = '';
    body.classList.add('diffing');
    const head = el('div', 'git-diff-head row');
    const back = el('button', 'mini', '‹ files');
    back.onclick = () => { closeDiff(); if (ctx.commit) setViewHash('#/git/commit/' + encodeURIComponent(ctx.head)); };
    head.appendChild(back);
    head.appendChild(statusBadge(f.status));
    head.appendChild(filePath(f));
    head.appendChild(fileCounts(f));
    head.appendChild(el('span', 'muted git-diff-pos', (i + 1) + ' / ' + ctx.files.length));
    const prev = el('button', 'mini', '‹'); prev.title = 'previous file'; prev.disabled = i === 0;
    prev.onclick = () => showDiff(ctx, i - 1);
    const next = el('button', 'mini', '›'); next.title = 'next file'; next.disabled = i === ctx.files.length - 1;
    next.onclick = () => showDiff(ctx, i + 1);
    head.appendChild(prev); head.appendChild(next);
    const fileLink = (side, label, cls) => {
      const rev = side === 'base' ? ctx.base : ctx.head;
      const path = side === 'base' ? f.old_path || f.path : f.path;
      const link = el('a', 'mini ' + cls, label);
      link.href = '#/files/' + encodeURIComponent(rev) + '/' + path.split('/').map(encodeURIComponent).join('/');
      if (ctx.panel) {
        const query = new URLSearchParams(ctx.query);
        query.delete('path'); query.delete('line'); query.delete('end_line');
        query.set('panel', ctx.panel); query.set('side', side);
        link.href += '?' + query;
      }
      return link;
    };
    if (f.status === 'D') {
      const absent = el('button', 'mini diff-open-file', 'Open in Files');
      absent.disabled = true; absent.title = 'Deleted: no file at the head revision';
      head.appendChild(absent);
      head.appendChild(fileLink('base', 'Open base file in Files', 'diff-open-base'));
    } else head.appendChild(fileLink('head', 'Open in Files', 'diff-open-file'));
    let mode = recall(DIFF_MODE_KEY, 'inline');
    if (!DIFF_MODES.includes(mode)) mode = 'inline';
    const modes = el('span', 'gview');
    for (const m of DIFF_MODES) {
      const b = el('button', 'mini' + (m === mode ? ' active' : ''), m);
      b.title = { inline: 'the unified patch', split: 'old and new side by side', old: 'the file at the base, deletions marked', new: 'the file at the head, additions marked' }[m];
      b.onclick = () => { remember(DIFF_MODE_KEY, m); showDiff(ctx, i, whole, fullContext); };
      modes.appendChild(b);
    }
    head.appendChild(modes);
    if (mode === 'inline' || mode === 'split') {
      const context = el('button', 'mini diff-context', fullContext ? 'Show hunks only' : 'Show full context');
      context.setAttribute('aria-pressed', String(fullContext));
      context.disabled = !!ctx.panel;
      if (ctx.panel) context.title = 'Full context is unavailable for recorded review comparisons';
      context.onclick = () => showDiff(ctx, i, whole, !fullContext);
      head.appendChild(context);
    }
    diffView.appendChild(head);
    const box = el('div', 'git-diff-body'); diffView.appendChild(box);
    let d = diffs.get(key);
    if (!d) {
      box.appendChild(el('div', 'muted', 'loading…'));
      try {
        const q = '?base=' + encodeURIComponent(ctx.base) + '&head=' + encodeURIComponent(ctx.head)
          + '&path=' + encodeURIComponent(f.path) + '&old_path=' + encodeURIComponent(f.old_path || '')
          + (whole ? '&whole=1' : '') + (fullContext ? '&full_context=1' : '');
        d = await getJSON(ctx.panel ? '/api/reviews/' + encodeURIComponent(ctx.panel) + '/diff?path=' + encodeURIComponent(f.path) + (whole ? '&whole=1' : '') : '/api/git/diff' + q);
      } catch (e) {
        if (current()) { box.innerHTML = ''; box.appendChild(el('div', 'err', e.message)); }
        return;
      }
      if (ctx.head !== GIT_WORKTREE) diffs.set(key, d); // the working tree changes under the tab
      if (!current()) return; // superseded
      box.innerHTML = '';
    }
    // The states come first, above the patch, so a cut patch reads as cut.
    if (isImagePath(f.path)) box.appendChild(imagePair(ctx, f));
    if (d.binary) box.appendChild(el('div', 'muted gf-note', 'binary file: no text diff'));
    if (d.truncated) {
      const note = el('div', 'err gf-note', 'the patch is cut at ' + Math.round(d.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole patch'); all.onclick = () => showDiff(ctx, i, true, fullContext);
      note.appendChild(all); box.appendChild(note);
    }
    const patch = el('div', 'gf-patch'); box.appendChild(patch);
    const parsed = parseUnifiedDiff(d.patch || '');
    const file = parsed[0] || { preamble: [], hunks: [], binary: d.binary };
    if (fullContext && d.context != null && (mode === 'inline' || mode === 'split')) {
      patch.appendChild(el('div', 'muted gf-note', 'This change has no text hunks; showing the file.'));
      file.hunks = [{header: 'Full file', lines: splitLines(d.context).map((text, n) => ({kind: 'ctx', text, old: n + 1, new: n + 1}))}];
      if (!parsed.length) parsed.push(file);
    }
    if (mode === 'split' && window.matchMedia('(max-width: 780px)').matches) {
      patch.appendChild(el('div', 'muted gf-note', 'split needs a wider screen: showing inline'));
      mode = 'inline';
    }
    let sideText = null;
    if (mode === 'inline') patch.appendChild(renderUnified(parsed, true));
    else if (mode === 'split') patch.appendChild(renderSplit(file));
    else sideText = await showSide(patch, ctx, f, file, mode, current, whole, () => showDiff(ctx, i, true, fullContext));
    if (!current()) return;
    await blameDiff(head, box, patch, ctx, f, d, mode, current, sideText);
    if (ctx.query?.get('path') === f.path && /^\d+$/.test(ctx.query.get('line') || '')) {
      const line = ctx.query.get('line');
      patch.querySelector('[data-new="' + line + '"], [data-line="' + line + '"]')?.scrollIntoView({block: 'center'});
    }
    try {
      const evidence = await getJSON('/api/git/findings?' + new URLSearchParams({base: d.base, head: d.head}));
      if (!current()) return;
      for (const failure of evidence.read_failures) box.appendChild(el('div', 'err', failure));
      drawDiffFindings(patch, evidence.findings, f, mode);
      if (ctx.query) focusFinding(patch, ctx.query.get('finding'));
    } catch (e) { if (current()) box.appendChild(el('div', 'err', 'Findings: ' + e.message)); }
  };
  const blameDiff = async (head, box, patch, ctx, f, d, mode, current, sideText) => {
    const button = el('button', 'mini diff-blame', 'blame');
    head.appendChild(button);
    let request = 0;
    const load = async () => {
      const n = ++request;
      clearBlame(patch);
      box.querySelectorAll('.blame-note').forEach(e => e.remove());
      button.setAttribute('aria-pressed', String(blameOn)); button.classList.toggle('active', blameOn);
      if (!blameOn) return;
      const notice = el('div', 'muted blame-note', 'loading blame…'); box.insertBefore(notice, patch);
      const unavailable = reason => { notice.classList.add('err'); notice.textContent = 'Blame unavailable: ' + reason; };
      if (ctx.panel) { unavailable('recorded tree comparisons do not identify blame commits.'); return; }
      if (d.binary) { unavailable('binary file.'); return; }
      const rows = diffBlameRows(patch, mode, f);
      const sides = [...new Set(rows.filter(r => r.number).map(r => r.side))];
      if (!sides.length) { unavailable('no displayed text lines.'); return; }
      for (const side of sides) {
        if (!d[side + '_digest']) { unavailable(side + ' content has no digest (not a regular file or over the read limit).'); return; }
        if (sideText && sideText.digest !== d[side + '_digest']) { unavailable(side + ' file changed since the patch was read; reopen the diff.'); return; }
      }
      try {
        const results = await Promise.all(sides.map(async side => {
          const data = await getJSON('/api/git/blame?' + new URLSearchParams({
            rev: d[side], path: side === 'base' ? f.old_path || f.path : f.path, whole: '1',
          }));
          return [side, data];
        }));
        if (!current() || n !== request) return;
        const maps = new Map();
        for (const [side, data] of results) {
          if (data.digest !== d[side + '_digest']) { unavailable(side + ' file changed since the diff was opened; reopen it.'); return; }
          maps.set(side, new Map(data.lines.map(line => [line.line, line])));
        }
        for (const row of rows) {
          row.line = maps.get(row.side)?.get(row.number);
          if (row.number && !row.line) { unavailable('the requested range was not attributed.'); return; }
        }
        notice.remove();
        renderBlame(rows);
      } catch (e) { if (current() && n === request) unavailable(e.message); }
    };
    button.onclick = () => { blameOn = !blameOn; return load(); };
    await load();
  };
  // The file at one side of a diff, its changed lines marked: the deletions on the old
  // side, the additions on the new. The text comes highlighted from /api/git/text.
  const showSide = async (host, ctx, f, file, side, current, whole, openWhole) => {
    if ((side === 'old' && f.status === 'A') || (side === 'new' && f.status === 'D')) {
      host.appendChild(el('div', 'muted gf-note', 'no ' + side + ' side: the file ' + (side === 'old' ? 'was added' : 'was deleted')));
      return;
    }
    const rev = side === 'old' ? ctx.base : ctx.head, path = side === 'old' ? (f.old_path || f.path) : f.path;
    host.appendChild(el('div', 'muted', 'loading…'));
    let t;
    try { t = await getJSON(ctx.panel ? '/api/reviews/' + encodeURIComponent(ctx.panel) + '/file?' + new URLSearchParams({path, side: side === 'old' ? 'base' : 'head', ...(whole ? {whole: '1'} : {})}) : '/api/git/text?rev=' + encodeURIComponent(rev) + '&path=' + encodeURIComponent(path) + (whole ? '&whole=1' : '')); }
    catch (e) { if (current()) { host.innerHTML = ''; host.appendChild(el('div', 'err', e.message)); } return; }
    if (!current() || !host.isConnected) return;
    host.innerHTML = '';
    if (t.binary) { host.appendChild(el('div', 'muted gf-note', 'binary file: no text')); return; }
    if (t.truncated) {
      const note = el('div', 'err gf-note', 'the file is cut at ' + Math.round(t.max_bytes / 1024) + ' KiB ');
      const all = el('button', 'mini', 'Show the whole file'); all.onclick = openWhole;
      note.appendChild(all); host.appendChild(note);
    }
    const marks = new Set();
    for (const h of file.hunks) for (const l of h.lines) {
      if (side === 'old' && l.kind === 'del') marks.add(l.old);
      if (side === 'new' && l.kind === 'add') marks.add(l.new);
    }
    host.appendChild(renderLines(splitLines(t.text), t.spans, marks));
    return t;
  };

  return {closeDiff, showDiff};
}
