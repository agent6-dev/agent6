// Notifications belong to the open page; the service worker handles no push.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(()=>{}); });
}
function osNotifyOn() { return 'Notification' in window && Notification.permission === 'granted'; }
function osNotify(title, body) {
  try { if (osNotifyOn()) new Notification(title, { body: body || '', icon: '/icon.svg' }); } catch (_) {}
}
// Only the hub raises an OS notification, so an open session tab cannot ring twice.
const LIVE_WORDS = ['in review', 'running', 'starting', 'waiting'];
function seedStatuses(rows) { const m = {}; for (const r of rows || []) m[r.session_id] = r.status; return m; }
function announceSessions(prev, rows, notify = true) {
  for (const r of rows || []) {
    const before = prev[r.session_id];
    if (before === undefined || before === r.status) continue;
    let title = '';
    if (r.status === 'waiting') title = `${r.session_id} needs you`;
    else if (LIVE_WORDS.includes(before) && !LIVE_WORDS.includes(r.status)) title = `${r.session_id} ${r.status_label}`;
    if (!title) continue;
    toast(title);
    if (notify) osNotify('agent6: ' + title, firstLine(r.task, 120));
  }
}

function toast(msg, level) {
  const kind = level === true || level === 'error' ? 'bad' : level === 'warn' ? 'warn' : level === 'hold' ? 'hold' : '';
  const host = document.getElementById('toasts');
  const said = kind && kind + '\n' + msg;
  const same = said && [...host.children].find(t => t.said === said);
  if (same) { host.prepend(same); return same; }
  const t = el('div', kind ? 'toast ' + kind : 'toast');
  t.appendChild(el('div')).append(withCode(msg));
  if (kind) {
    t.said = said;
    t.setAttribute('role', kind === 'hold' ? 'status' : 'alert');
    t.appendChild(closeButton('Dismiss', () => t.remove()));
  } else {
    let timer = 0;
    const fade = () => { timer = setTimeout(() => t.remove(), 4000); };
    t.onpointerenter = () => clearTimeout(timer);
    t.onpointerleave = fade;
    fade();
  }
  host.prepend(t);
  return t;
}
function newestHeld() { return document.querySelector('#toasts > .toast:is(.bad, .warn, .hold)'); }

function notificationsButton(cls) {
  const b = el('button', cls);
  const supported = 'Notification' in window;
  const paint = () => {
    const p = supported ? Notification.permission : '';
    b.textContent = { granted: 'Notifications on', denied: 'Notifications blocked', default: 'Turn on notifications' }[p] || 'Notifications not supported';
    b.title = p === 'denied' ? "allow notifications for this page in the browser's site settings" : '';
    b.disabled = !supported;
  };
  // Only browser settings can revoke a notification permission.
  b.onclick = async () => {
    await Notification.requestPermission();
    paint();
    // The stored permission can differ from the value returned by the prompt.
    const p = Notification.permission;
    if (p === 'granted') toast("notifications are on: the browser's site settings turn them off");
    else if (p === 'denied') toast("notifications are blocked: allow them for this page in the browser's site settings", true);
  };
  paint();
  return b;
}
