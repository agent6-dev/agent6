function fileContentSearch(revision) {
  const form = el('form', 'row files-search'); form.setAttribute('role', 'search');
  const field = el('input', 'field'); field.type = 'search'; field.required = true;
  field.placeholder = 'Search file contents'; field.setAttribute('aria-label', field.placeholder);
  form.append(field, el('button', null, 'Search'));
  form.onsubmit = async e => {
    e.preventDefault();
    const rev = revision(), q = field.value;
    const dialog = openDialog('File content matches', 'wide files-matches');
    if (!dialog) return;
    const where = rev === GIT_WORKTREE ? 'working tree (ignored files excluded)' : rev;
    dialog.appendChild(el('p', null, `"${q}" in ${where}`));
    const note = el('p', 'muted', 'Searching…'); note.setAttribute('role', 'status'); dialog.appendChild(note);
    try {
      const d = await getJSON('/api/git/grep?' + new URLSearchParams({rev, q}));
      if (!dialog.open) return;
      const n = d.hits.length;
      note.textContent = d.truncated ? `${n} shown. Results are cut.` : n === 0 ? 'No matches.' : n === 1 ? '1 match' : `${n} matches`;
      const list = el('ul'); dialog.appendChild(list);
      for (const hit of d.hits) {
        const item = list.appendChild(el('li')), label = `${hit.path}:${hit.line}`;
        const linked = hit.path.isWellFormed();
        const row = item.appendChild(el(linked ? 'a' : 'span', null, label));
        if (linked) {
          row.href = fileHash(d.rev, hit.path) + '?line=' + hit.line;
          row.onclick = event => {
            if (!plainFileClick(event)) return;
            dialog.close();
            if (location.hash === row.hash) { event.preventDefault(); route(); }
          };
        } else row.append(' (path is not UTF-8)');
        row.appendChild(el('code', null, hit.text));
      }
    } catch (err) { if (dialog.open) { note.className = 'err'; note.textContent = err.message; } }
  };
  return form;
}
