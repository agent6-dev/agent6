// Origin-scoped storage keeps the token off requests to other local servers.
const TOKEN_KEY = 'a6-token';
let loginToken = ''; // an in-memory sign-in when browser storage is unavailable
function authHeaders(extra) {
  const h = Object.assign({}, extra || {});
  const t = loginToken || recall(TOKEN_KEY, '');
  if (t) h['Authorization'] = 'Bearer ' + t;
  return h;
}
function saveToken(login) { loginToken = login.token; remember(TOKEN_KEY, loginToken); }
// A page with blocked or full storage still works for this visit.
function recall(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch (_) { return fallback; } }
function remember(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
function forget(key) { try { localStorage.removeItem(key); } catch (_) {} }
async function serverFetch(url, init = {}) {
  let r;
  try { r = await fetch(url, { ...init, headers: authHeaders(init.headers) }); }
  catch (e) { throw new Error(`no answer from the agent6 web server at ${location.host} (${e.message})`); }
  const packWarning = r.headers.get('X-Agent6-Pack-Warning');
  if (packWarning) toast(decodeURIComponent(packWarning), 'warn');
  if (r.status !== 401) return r;
  r.body?.cancel(); // nothing reads it, and an unread body holds its request open
  loginToken = '';
  forget(TOKEN_KEY);
  showLogin();
  return new Promise(() => {}); // never settles: nothing paints an error over the form, and a sign-in reruns the route
}
async function getResponse(url) {
  const r = await serverFetch(url);
  if (r.ok) return r;
  const reason = (await r.json().catch(() => ({}))).error || r.statusText || 'HTTP ' + r.status;
  throw Object.assign(new Error(reason), { status: r.status });
}
async function getJSON(url) { return (await getResponse(url)).json(); }
async function getBlob(url) { return (await getResponse(url)).blob(); }
async function postJSON(url, body, {keepalive = false} = {}) {
  const r = await serverFetch(url, { method: 'POST', keepalive, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) throw Object.assign(new Error(data.error || r.statusText || 'HTTP ' + r.status), { data });
  return data;
}

// A second 401 must not replace the secret the operator is typing.
function showLogin(error) {
  routeGen++; // a render in flight must not paint over the form
  closeLive();
  document.querySelector('dialog[open]')?.close();
  setTab('');
  setCrumb('Sign in');
  let form = document.getElementById('login');
  if (!form) {
    view.innerHTML = '';
    form = el('form', 'card login');
    form.id = 'login';
    form.appendChild(el('h2', null, 'Sign in'));
    form.appendChild(el('div', 'muted')).append(withCode('Type the secret `agent6 web` printed in its terminal.'));
    const row = el('div', 'form-row');
    const input = el('input', 'field');
    input.type = 'text';
    // The phone keyboard must not correct or remember the secret.
    const attrs = { 'aria-label': 'secret', 'aria-describedby': 'login-err', autocapitalize: 'none', autocorrect: 'off', spellcheck: 'false', autocomplete: 'off' };
    for (const name in attrs) input.setAttribute(name, attrs[name]);
    input.oninput = () => input.removeAttribute('aria-invalid');
    row.appendChild(input);
    row.appendChild(el('button', 'primary', 'Sign in'));
    form.appendChild(row);
    const refusal = form.appendChild(el('div', 'err'));
    refusal.id = 'login-err';
    refusal.setAttribute('role', 'alert');
    form.onsubmit = async (e) => {
      e.preventDefault();
      try { saveToken(await postJSON('/api/login', { secret: input.value })); route(); }
      catch (err) { showLogin(err.message); }
    };
    view.appendChild(form);
    input.focus();
  }
  form.removeAttribute('aria-busy');
  if (error) {
    document.getElementById('login-err').replaceChildren(withCode(error));
    form.querySelector('input').setAttribute('aria-invalid', 'true');
  }
}
async function boot() {
  const m = /^#secret=(.+)$/.exec(location.hash);
  if (m) {
    let refused = '';
    try { saveToken(await postJSON('/api/login', { secret: decodeURIComponent(m[1]) })); }
    catch (e) { refused = e.message; }
    history.replaceState(null, '', location.pathname + location.search);
    if (refused) { showLogin(refused); return; }
  }
  route();
}
