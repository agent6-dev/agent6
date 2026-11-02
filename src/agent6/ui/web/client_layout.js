// CSS keeps each pane within its container when the viewport or sidebar changes.
function resizeHandle(pane, { label, key, property, min, max }) {
  const h = el('div', 'resize-handle');
  h.tabIndex = 0;
  h.title = 'drag or use arrow keys to resize · double-click to reset';
  h.setAttribute('role', 'separator');
  h.setAttribute('aria-label', label);
  h.setAttribute('aria-orientation', 'vertical');
  const describe = () => {
    const width = Math.round(pane.getBoundingClientRect().width);
    h.setAttribute('aria-valuemin', String(min));
    h.setAttribute('aria-valuemax', String(Math.round(max())));
    h.setAttribute('aria-valuenow', String(width));
    h.setAttribute('aria-valuetext', width + ' pixels');
  };
  const resize = width => {
    const w = Math.round(Math.min(max(), Math.max(min, width)));
    pane.style.setProperty(property, w + 'px');
    remember(key, w + 'px');
    describe();
  };
  const saved = parseFloat(recall(key, ''));
  if (Number.isFinite(saved)) pane.style.setProperty(property, saved + 'px');
  describe();
  h.onfocus = describe;
  h.onkeydown = e => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    resize(pane.getBoundingClientRect().width + (e.key === 'ArrowLeft' ? -20 : 20));
  };
  h.onpointerdown = e => {
    if (e.button !== 0 || h.classList.contains('dragging')) return;
    e.preventDefault();
    h.focus();
    h.setPointerCapture(e.pointerId);
    h.classList.add('dragging');
    const startX = e.clientX, startW = pane.getBoundingClientRect().width;
    h.onpointermove = ev => resize(startW + ev.clientX - startX);
  };
  const stop = () => { h.classList.remove('dragging'); h.onpointermove = null; };
  h.onpointerup = e => { h.releasePointerCapture(e.pointerId); stop(); };
  h.onpointercancel = stop;
  h.onlostpointercapture = stop;
  h.ondblclick = () => {
    pane.style.removeProperty(property);
    remember(key, '');
    describe();
  };
  return h;
}
