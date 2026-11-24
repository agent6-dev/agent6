// The task graph keeps the worker's cursor separate from the operator's choice.
function taskTree(host, base) {
  const note = el('div', 'sub muted'), tree = el('div', 'tree');
  const empty = el('div', 'muted', 'no task graph yet');
  const retire = el('button', 'mini', 'Retire task');
  host.append(note, tree, empty, retire);
  const choice = () => tree.querySelector('input:checked');
  runOnce(retire, async () => {
    const task = choice().taskNode;
    if (!await confirmDialog(`Retire task ${task.short_id}: ${task.title}?`,
      'This marks the task obsolete at the next step. It does not stop a worker.', 'Retire task', true)) return;
    if (!host.isConnected || choice()?.value !== task.id) { toast('The selected task changed. Nothing retired.', true); return; }
    try {
      const d = await postJSON(base + '/steer', {text: '/retire ' + task.id});
      toast(d.message);
    } catch (e) { toast(e.message, true); }
  }, () => !!choice());
  tree.onchange = retire.sync;
  return (tasks, live, asOf) => {
    const selected = choice()?.value;
    const rows = tasks.map(t => {
      const pick = live && (t.status === 'pending' || t.status === 'in_progress');
      return {key: t.id, data: [t, pick], paint: () => {
        const line = el(pick ? 'label' : 'div', 'node' + (t.is_cursor ? ' cursor' : ''));
        if (pick) {
          const radio = el('input'); radio.type = 'radio'; radio.name = 'retire-task';
          radio.taskNode = t; radio.value = t.id; radio.checked = t.id === selected;
          line.appendChild(radio);
        }
        line.appendChild(el('span', 'muted', String(t.short_id || '').padStart(3) + ' '));
        line.appendChild(el('span', 'st-' + t.status, '  '.repeat(t.depth) + (t.glyph || '·') + ' '));
        const task = line.appendChild(el('span', 'task'));
        task.appendChild(el('span', 'title', t.title));
        if (t.note) task.appendChild(el('span', 'muted', ' ' + t.note));
        return line;
      }};
    });
    patchRows(tree, rows);
    note.hidden = !asOf; note.textContent = asOf;
    empty.hidden = !!tasks.length;
    retire.hidden = !tree.querySelector('input');
    if (retire.hidden) { tree.removeAttribute('role'); tree.removeAttribute('aria-label'); }
    else { tree.setAttribute('role', 'radiogroup'); tree.setAttribute('aria-label', 'Task to retire'); }
    retire.sync();
  };
}
