// A file diff owns its request lifetime and its exact-comparison finding anchors.
function gitDiffViewer(body) {
  // -- a file's diff: it takes the whole body; the list is one step back --------------
  const diffView = el('div', 'git-diff'); body.appendChild(diffView);
  const diffs = new Map();
  let showing = 0;
  const closeDiff = () => { showing++; body.classList.remove('diffing'); diffView.innerHTML = ''; delete diffView.dataset.key; };
  const showDiff = async (ctx, i, whole) => {
    const request = ++showing;
    const f = ctx.files[i];
    const current = () => request === showing && diffView.isConnected;
    const key = ctx.base + '|' + ctx.head + '|' + f.path + (whole ? '|whole' : '');
    diffView.dataset.key = key;
    diffView.innerHTML = '';
    body.classList.add('diffing');
    const head = el('div', 'git-diff-head row');
    const back = el('button', 'mini', '‹ files'); back.onclick = closeDiff;
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
    let mode = recall(DIFF_MODE_KEY, 'inline');
    if (!DIFF_MODES.includes(mode)) mode = 'inline';
    const modes = el('span', 'gview');
    for (const m of DIFF_MODES) {
      const b = el('button', 'mini' + (m === mode ? ' active' : ''), m);
      b.title = { inline: 'the unified patch', split: 'old and new side by side', old: 'the file at the base, deletions marked', new: 'the file at the head, additions marked' }[m];
      b.onclick = () => { remember(DIFF_MODE_KEY, m); showDiff(ctx, i, whole); };
      modes.appendChild(b);
    }
    head.appendChild(modes);
    diffView.appendChild(head);
    const box = el('div', 'git-diff-body'); diffView.appendChild(box);
    let d = diffs.get(key);
    if (!d) {
      box.appendChild(el('div', 'muted', 'loading…'));
      try {
        const q = '?base=' + encodeURIComponent(ctx.base) + '&head=' + encodeURIComponent(ctx.head)
          + '&path=' + encodeURIComponent(f.path) + '&old_path=' + encodeURIComponent(f.old_path || '')
          + (whole ? '&whole=1' : '');
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
      const all = el('button', 'mini', 'Show the whole patch'); all.onclick = () => showDiff(ctx, i, true);
      note.appendChild(all); box.appendChild(note);
    }
    const patch = el('div', 'gf-patch'); box.appendChild(patch);
    const parsed = parseUnifiedDiff(d.patch || '');
    const file = parsed[0] || { preamble: [], hunks: [], binary: d.binary };
    if (mode === 'split' && window.matchMedia('(max-width: 780px)').matches) {
      patch.appendChild(el('div', 'muted gf-note', 'split needs a wider screen: showing inline'));
      mode = 'inline';
    }
    if (mode === 'inline') patch.appendChild(renderUnified(parsed, true));
    else if (mode === 'split') patch.appendChild(renderSplit(file));
    else await showSide(patch, ctx, f, file, mode, current, whole, () => showDiff(ctx, i, true));
    try {
      const evidence = await getJSON('/api/git/findings?' + new URLSearchParams({base: d.base, head: d.head}));
      if (!current()) return;
      for (const failure of evidence.read_failures) box.appendChild(el('div', 'err', failure));
      drawDiffFindings(patch, evidence.findings, f, mode);
      if (ctx.query) focusFinding(patch, ctx.query.get('finding'));
    } catch (e) { if (current()) box.appendChild(el('div', 'err', 'Findings: ' + e.message)); }
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
  };

  return {closeDiff, showDiff};
}
