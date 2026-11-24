// A fetch can finish over an open picker; never put a second modal above it.
let dialogCount = 0;
const CLOSE_ICON = '<svg class="ic" viewBox="0 0 24 24" stroke-width="1.75" stroke-linecap="round" fill="none" stroke="currentColor" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';
function closeButton(label, onclick) {
  const b = el('button', 'quiet x-btn');
  b.setAttribute('aria-label', label); b.title = label;
  b.innerHTML = CLOSE_ICON;
  b.onclick = onclick;
  return b;
}
function openDialog(title, cls) {
  if (document.querySelector('dialog[open]')) return null;
  const dialog = el('dialog', cls ? 'card ' + cls : 'card');
  // Chrome reads a visible word-break opportunity as a space.
  const [first, ...rest] = title.split(/(?<=\.)/);
  const heading = el('h2', null, first);
  for (const part of rest) { heading.appendChild(el('wbr')).setAttribute('aria-hidden', 'true'); heading.appendChild(el('span', null, part)); }
  heading.id = 'dialog-title-' + ++dialogCount;
  dialog.setAttribute('aria-labelledby', heading.id);
  dialog.setAttribute('closedby', 'any');
  // Touch can report both clicks of the opening double tap with a click count of one.
  let own = false, stray = false;
  const opened = performance.now();
  dialog.addEventListener('pointerdown', e => {
    stray = !own && e.pointerType === 'touch' && e.timeStamp - opened < 500;
  }, true);
  dialog.addEventListener('keydown', () => { stray = false; }, true);
  dialog.addEventListener('mousedown', e => {
    if (e.detail > 1 && !own) stray = true;
    if (stray) e.preventDefault(); else own = true;
  }, true);
  dialog.addEventListener('click', e => { if (stray) { e.stopPropagation(); e.preventDefault(); } }, true);
  dialog.addEventListener('cancel', e => { if (stray) e.preventDefault(); });
  dialog.tabIndex = -1; // no Tab stop: Firefox stops on a scrolling dialog, which draws no ring
  const head = el('div', 'dialog-head');
  head.appendChild(heading); head.appendChild(closeButton('Close', () => dialog.close()));
  dialog.appendChild(head);
  // Keep messages above the modal backdrop and inside its focus trap.
  const toasts = document.getElementById('toasts');
  // Back on the page before the dialog closes: `close` fires a task later, and they would be hidden until it ran.
  dialog.addEventListener('beforetoggle', e => { if (e.newState === 'closed' && toasts.parentNode === dialog) document.body.append(toasts); });
  dialog.addEventListener('close', () => dialog.remove());
  const root = document.documentElement;
  root.classList.toggle('scrollbar', innerWidth > root.clientWidth);
  document.body.appendChild(dialog);
  dialog.showModal();
  queueMicrotask(() => dialog.append(toasts)); // after what the caller adds, so they come last in its Tab order
  dialog.focus(); // not its Close button, which the browser would pick; a caller's field takes it next
  return dialog;
}
async function confirmDialog(title, text, action, danger) {
  const dialog = openDialog(title, 'confirm');
  if (!dialog) return false;
  dialog.setAttribute('role', 'alertdialog');
  const consequence = dialog.appendChild(el('p'));
  consequence.id = 'dialog-text-' + dialogCount;
  consequence.appendChild(withCode(text));
  dialog.setAttribute('aria-describedby', consequence.id);
  const row = dialog.appendChild(el('div', 'form-row'));
  const yes = row.appendChild(el('button', danger ? 'danger' : 'primary', action));
  const no = row.appendChild(el('button', null, 'Cancel'));
  // An Enter held while opening the dialog must not confirm it.
  dialog.onkeydown = (e) => { if (e.key === 'Enter' && e.repeat) e.preventDefault(); };
  (danger ? no : yes).focus();
  return new Promise((resolve) => {
    yes.onclick = () => { resolve(true); dialog.close(); };
    no.onclick = () => dialog.close();
    dialog.addEventListener('close', () => resolve(false));
  });
}
