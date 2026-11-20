let booted = false;
// After each await, a superseded render must leave the new page and its stream alone.
let routeGen = 0;
function showProblem(heading, reason, failed, back) {
  view.innerHTML = '';
  const card = el('div', 'card problem');
  card.appendChild(el('h2', null, heading));
  card.appendChild(el('p', failed ? 'err' : null)).append(withCode(reason));
  const ways = el('div', 'row');
  if (failed) { const again = el('button', null, 'Try again'); again.onclick = route; ways.appendChild(again); }
  if (back) { const link = el('a', null, 'Back to ' + back[0]); link.href = back[1]; ways.appendChild(link); }
  card.appendChild(ways);
  view.appendChild(card);
}
const BACK = { session: ['Sessions', '#/'], machine: ['State machines', '#/machines'], draft: ['State machines', '#/machines'] };

let shownHash = null;
function setViewHash(hash, replace = false) { if (location.hash !== hash) history[replace ? 'replaceState' : 'pushState'](null, '', hash); shownHash = location.hash; }
async function route() {
  const gen = ++routeGen;
  const held = document.activeElement;
  const repaint = location.hash === shownHash; // a navigation matches nothing: a control of the same name on the new page acts on something else
  closeLive(true);
  document.querySelector('dialog[open]')?.close();
  document.querySelectorAll('.wmenu-btn').forEach(b => b.remove());
  // Keep the old page readable while the replacement loads.
  for (const old of view.children) if (!old.classList.contains('empty')) old.setAttribute('aria-busy', 'true');
  const h = location.hash.replace(/^#/, '') || '/';
  // An explicit hash takes precedence over the CLI target.
  if (!booted) {
    booted = true;
    if (h === '/') {
      try {
        const meta = await getJSON('/api/meta');
        if (gen !== routeGen) return;
        if (meta.target && meta.target_kind) {
          location.hash = '#/' + meta.target_kind + '/' + encodeURIComponent(meta.target);
          return; // the hashchange re-enters route()
        }
      } catch (_) {
        if (gen !== routeGen) return;
      }
    }
  }
  const [routePath, queryText = ''] = h.split('?');
  const query = new URLSearchParams(queryText);
  const parts = routePath.split('/').filter(Boolean);
  if (parts[0] !== 'files' || query.has('panel')) closeFilePreview();
  try {
    if (parts.length === 0) { setTab('hub'); await renderHub(undefined, gen); }
    else if (parts[0] === 'machines') { setTab('machines'); await renderHub('machines', gen); }
    else if (parts[0] === 'config') { setTab('config'); await renderConfig(gen); }
    else if (parts[0] === 'git') { setTab('git'); await renderGit(gen, parts.slice(1).map(decodeURIComponent), query); }
    else if (parts[0] === 'files') { setTab('files'); await renderFiles(gen, decodeURIComponent(parts[1] || ''), parts.slice(2).map(decodeURIComponent).join('/'), query); }
    else if (parts[0] === 'session' && parts[1]) { setTab('hub'); await renderRun(decodeURIComponent(parts[1]), undefined, gen); }
    else if (parts[0] === 'machine' && parts[1]) { setTab('machines'); await renderMachine(decodeURIComponent(parts[1]), gen); }
    else if (parts[0] === 'draft' && parts[1]) { const n = decodeURIComponent(parts[1]); setTab('machines'); await renderRun(n, { base: '/api/draft/' + encodeURIComponent(n), readOnly: true, title: 'Machine draft', crumb: 'draft ' + n }, gen); }
    else { setTab(''); setCrumb('Not found'); showProblem('Not found', 'No page lives at #' + h + '.', false, BACK.session); }
  } catch (e) {
    if (gen !== routeGen) return;
    if (e.status === 404) showProblem('Not found', e.message, false, BACK[parts[0]]);
    else showProblem('Could not load this page', e.message, true, BACK[parts[0]]);
  }
  if (gen !== routeGen) return;
  for (const kept of view.children) kept.removeAttribute('aria-busy');
  shownHash = location.hash;
  refocus(held, repaint ? document.body : null, crumb);
}
window.addEventListener('hashchange', route);
