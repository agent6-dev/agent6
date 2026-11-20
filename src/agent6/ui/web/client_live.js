let live = null;
// The heartbeat keeps ticking between server frames, including silent turns.
let hbState = { active: false, role: 'worker', last: 0, spin: 0 };
// A dead worker can leave an unfinished journal.
function notLive(s) { return typeof s.live === 'boolean' ? !s.live : !!s.finished; }
let hbTimer = null;
let hubTimer = null;
let hubVisWake = null;
const HUB_POLL_MS = 4000;
const HB_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
function hbTick() {
  const line = document.getElementById('hb-line');
  if (!line || !hbState.active) return;
  const secs = Math.floor((Date.now() - hbState.last) / 1000);
  const glyph = HB_FRAMES[hbState.spin % HB_FRAMES.length];
  line.textContent = `${glyph} ${hbState.role} working… ${secs}s`;
}

// Keep the last good paint during a disconnect.
const staleLine = el('div', 'muted card stale');
staleLine.setAttribute('role', 'status'); // in the page while empty, so its text is read out when it comes
let answeredAt = 0;
function watchLive() { view.prepend(staleLine); answered(); }
function answered() { answeredAt = Date.now(); if (staleLine.textContent) staleLine.textContent = ''; }
function unanswered(reason, retryMs) {
  const said = `${reason} · not updated since ${new Date(answeredAt).toTimeString().slice(0, 5)} · retrying every ${retryMs / 1000} s`;
  if (staleLine.textContent !== said) staleLine.textContent = said; // a repeat is not read out again
  hbState.active = false; // a heartbeat would count time the page cannot see
}
// EventSource cannot send the sign-in header.
const STREAM_RETRY_MS = 1500;
function openStream(url, onData) {
  const stream = {
    closed: false, ctrl: null, timer: null,
    close() { stream.closed = true; if (stream.ctrl) stream.ctrl.abort(); if (stream.timer) clearTimeout(stream.timer); },
  };
  const connect = async () => {
    stream.ctrl = new AbortController();
    let r = null, reason = '';
    try { r = await serverFetch(url, { signal: stream.ctrl.signal }); } catch (e) { reason = e.message; }
    if (stream.closed) return;
    if (r && r.ok && r.body) {
      answered();
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done || stream.closed) break;
          buf += dec.decode(value, { stream: true });
          let cut;
          while ((cut = buf.indexOf('\n\n')) >= 0) {
            const frame = buf.slice(0, cut);
            buf = buf.slice(cut + 2);
            const data = frame.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trimStart()).join('\n');
            if (data) onData(data);
          }
        }
      } catch (_) { /* aborted, or the connection dropped */ }
      if (!stream.closed) answered(); // the paint was current until the connection dropped
    } else if (r) {
      reason = (await r.json().catch(() => ({}))).error || r.statusText || 'HTTP ' + r.status; // HTTP/2 has no reason phrase
    }
    if (stream.closed) return;
    if (reason) unanswered(reason, STREAM_RETRY_MS);
    stream.timer = setTimeout(connect, STREAM_RETRY_MS);
  };
  watchLive();
  connect();
  return stream;
}
function closeLive(keepFilePreview = false) {
  if (!keepFilePreview) closeFilePreview();
  staleLine.remove();
  if (live) { live.close(); live = null; }
  if (hubTimer) { clearInterval(hubTimer); hubTimer = null; }
  if (hubVisWake) { document.removeEventListener('visibilitychange', hubVisWake); hubVisWake = null; }
  if (hbTimer) { clearInterval(hbTimer); hbTimer = null; }
  hbState.active = false;
}
