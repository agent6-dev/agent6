const DETAIL_CYCLE = { collapsed: 'expanded', expanded: 'hidden', hidden: 'collapsed' };
function tailStr(s, n) { return s.length <= n ? s : '…' + s.slice(-n); }

// On phones the page scrolls the conversation instead of its box.
const phoneLayout = matchMedia('(width <= 48.75em)');
function makeConv(url, box, body, empty = {}) {
  const conv = {
    items: [], open: new Map(),
    detail: recall('a6-detail', '') || 'collapsed',
    timer: null,
    finished: false,
  };
  const itemsHost = el('div', 'conv');
  const liveHost = el('div', 'conv conv-live');
  liveHost.style.display = 'none';
  conv.prompts = el('div', 'prompts'); // paintPrompts' boxes, after the last line, so new lines land above them
  body.append(itemsHost, liveHost, conv.prompts);
  const scroller = () => phoneLayout.matches ? document.scrollingElement : box;
  if (!phoneLayout.matches) box.classList.add('tail');
  let from = scroller(), top = 0;
  const mark = () => {
    const s = scroller();
    // The page and box offsets are unrelated after a layout switch.
    if (s !== from) { from = s; top = s.scrollTop; if (s === box) conv.toTail(); }
    if (s === box || s.scrollHeight > s.clientHeight) {
      // Scroll events arrive a frame late; appended lines can move the tail in the meantime.
      if (s.scrollTop + s.clientHeight >= s.scrollHeight - 40) box.classList.add('tail');
      else if (s.scrollTop < top) box.classList.remove('tail');
    }
    top = s.scrollTop;
    box.classList.toggle('scrolled', box.scrollTop > 0);
  };
  box.addEventListener('scroll', mark);
  const pageScroll = () => box.isConnected ? mark() : removeEventListener('scroll', pageScroll);
  addEventListener('scroll', pageScroll);
  const toEnd = () => { const s = scroller(); s.scrollTop = s.scrollHeight; };
  const follow = new ResizeObserver(() => { if (box.classList.contains('tail') && box.checkVisibility()) toEnd(); mark(); });
  follow.observe(box); follow.observe(body);
  // Removing lines during a paint can clamp the offset without a user scroll.
  new MutationObserver(() => { top = Math.min(top, scroller().scrollTop); }).observe(body, { childList: true, subtree: true });
  conv.toTail = () => { box.classList.add('tail'); toEnd(); };
  const emptyNote = () => conv.finished
    ? empty.done || 'this session made no conversation'
    : empty.live || 'no conversation yet; it appears as the session streams';

  const paintItems = () => {
    const refocus = itemsHost.contains(document.activeElement);
    itemsHost.innerHTML = '';
    let shown = 0;
    conv.items.forEach((it, i) => {
      if ((it.kind === 'thinking' || it.kind === 'tool') && conv.detail === 'hidden' && !conv.open.get(i)) return;
      const expanded = conv.open.get(i) ?? (conv.detail === 'expanded');
      const lines = expanded && it.full ? it.full : it.lines;
      const div = el('div', 'ci' + (it.full ? ' exp' : ''));
      div.dataset.i = i;
      for (const line of lines) {
        const ln = el('div');
        for (const [text, style] of line) {
          // The commit subject after the receipt counts is free text.
          const label = style === 'done-detail' && box.closest('.card-conv')
            ? text.replace(/( · \d+ tools? · \d+ )(commits?)(?= · |$)/, '$1checkpoint $2 from this execution')
            : text;
          ln.appendChild(el('span', 's-' + style, label));
        }
        if (!line.length) ln.appendChild(document.createTextNode(' '));
        const hang = /^\s*(?:[\p{P}\p{S}]\s+)?/u.exec(ln.textContent)[0].length;
        if (hang) ln.style.setProperty('--hang', hang + 'ch');
        div.appendChild(ln);
      }
      if (it.full) {
        const toggle = () => { conv.open.set(i, !expanded); paintItems(); };
        const head = div.firstElementChild;
        head.setAttribute('role', 'button');
        head.setAttribute('aria-expanded', String(expanded));
        head.tabIndex = -1;
        head.onfocus = () => {
          conv.stop = i;
          for (const other of itemsHost.querySelectorAll('[role=button]')) other.tabIndex = other === head ? 0 : -1;
        };
        head.onkeydown = (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); return; }
          const heads = [...itemsHost.querySelectorAll('[role=button]')], at = heads.indexOf(head);
          const next = { ArrowUp: heads[at - 1], ArrowDown: heads[at + 1], Home: heads[0], End: heads.at(-1) }[e.key];
          if (next) { e.preventDefault(); next.focus(); }
        };
        div.title = expanded ? 'click to collapse' : 'click to expand';
        div.onclick = () => {
          // A drag to copy must not toggle the item.
          const picked = getSelection();
          if (!picked.isCollapsed && div.contains(picked.anchorNode)) return;
          toggle();
        };
      }
      itemsHost.appendChild(div); shown++;
    });
    if (!shown && liveHost.style.display === 'none') {
      itemsHost.appendChild(el('div', 'muted conv-empty', emptyNote()));
    }
    const heads = itemsHost.querySelectorAll('[role=button]');
    const stop = itemsHost.querySelector('.ci[data-i="' + conv.stop + '"] > [role=button]') || heads[heads.length - 1];
    if (stop) { stop.tabIndex = 0; if (refocus) stop.focus({ preventScroll: true }); }
  };

  conv.refresh = async () => {
    if (conv.closed || !box.isConnected) return;
    let data; try { data = await getJSON(url); } catch (_) { return; }
    if (!box.isConnected) return;
    // Repainting unchanged text would discard a selection on every frame.
    const key = JSON.stringify(data.items || []);
    if (key === conv.itemsKey) return;
    conv.itemsKey = key;
    conv.items = data.items || [];
    paintItems();
  };
  conv.poke = () => {
    if (conv.timer) return;
    conv.timer = setTimeout(() => { conv.timer = null; conv.refresh(); }, 900);
  };
  conv.close = () => { conv.closed = true; if (conv.timer) { clearTimeout(conv.timer); conv.timer = null; } };
  conv.setLive = (s) => {
    conv.finished = notLive(s);
    const r = s.last_model_call;
    liveHost.innerHTML = '';
    const note = itemsHost.querySelector('.conv-empty');
    if (notLive(s)) {
      const dead = s.dead_state || '';
      liveHost.style.display = dead ? '' : 'none';
      if (dead) liveHost.appendChild(el('div', 'muted', dead));
      if (note) { note.textContent = emptyNote(); note.style.display = dead ? 'none' : ''; }
      return;
    }
    // An unanswered prompt means neither thinking nor working.
    if (s.operator_blocked) {
      const waiting = s.status === 'waiting';
      liveHost.style.display = waiting ? '' : 'none';
      if (note) note.style.display = 'none';
      if (waiting) liveHost.appendChild(el('div', 'muted', '· ' + runState(s)));
      return;
    }
    if (!r) {
      liveHost.style.display = 'none';
      if (note) { note.textContent = emptyNote(); note.style.display = ''; }
      return;
    }
    const think = r.streamed_thinking, text = r.streamed_text;
    liveHost.style.display = '';
    if (note) note.style.display = 'none';
    if (think || text) {
      if (think) {
        const line = el('div');
        line.appendChild(el('span', 'lt', '· thinking… '));
        if (conv.detail === 'expanded') line.appendChild(el('span', 's-thinking', tailStr(think, 1600)));
        liveHost.appendChild(line);
      }
      if (text) liveHost.appendChild(el('div', null, tailStr(text, 1600)));
    } else {
      const hb = el('div', 'muted'); hb.id = 'hb-line'; liveHost.appendChild(hb); hbTick();
    }
  };
  conv.detailButton = () => {
    // Reserve room for all three labels so the detail control stays still.
    const b = el('button', 'mini');
    const cycle = b.appendChild(el('span', 'cycle'));
    const labels = {};
    for (const level in DETAIL_CYCLE) labels[level] = cycle.appendChild(el('span', null, 'detail: ' + level));
    const show = () => { for (const level in labels) labels[level].classList.toggle('on', level === conv.detail); };
    show();
    b.onclick = () => {
      conv.detail = DETAIL_CYCLE[conv.detail];
      remember('a6-detail', conv.detail);
      show();
      conv.open.clear();
      paintItems();
    };
    return b;
  };
  return conv;
}

function convCard(url, title, cls, empty) {
  const card = el('div', 'card conv-card ' + (cls || ''));
  const hrow = el('div', 'card-head-row');
  hrow.appendChild(el('h2', null, title));
  const box = el('div', 'conv-box');
  const body = el('div');
  box.appendChild(body);
  const conv = makeConv(url, box, body, empty);
  hrow.appendChild(conv.detailButton());
  card.appendChild(hrow); card.appendChild(box);
  return { card, conv, box };
}
