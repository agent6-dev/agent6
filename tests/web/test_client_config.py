from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import threading
from importlib import resources

import pytest

from agent6.ui.web import server
from tests.web import signin


def test_entry_forms_and_removal_use_the_live_registry(tmp_path: pathlib.Path) -> None:
    """Rendered entry actions write and remove all three tables through the live HTTP routes."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise the web config forms")
    resource = resources.files("agent6.ui.web").joinpath("client_config.js")
    client = resource.read_text(encoding="utf-8")
    scenario = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag, cls, text = '') {
    Object.assign(this, {tag, className: cls, textContent: text, children: [], style: {},
                        value: '', checked: false, disabled: false, isConnected: true});
  }
  appendChild(child) {
    this.children.push(child); child.parent = this;
    if (this.tag === 'select' && this.children.length === 1) this.value = child.value;
    return child;
  }
  set innerHTML(value) { this.children = []; }
  setAttribute(name, value) { this[name] = value; }
  getAttribute(name) { return this[name]; }
  focus() { document.activeElement = this; this.onfocus?.(); }
  removeAttribute(name) { delete this[name]; }
  scrollIntoView() {}
  remove() {
    this.parent.children = this.parent.children.filter(child => child !== this);
    this.isConnected = false;
  }
}
function el(tag, cls, text) { return new Element(tag, cls, text); }
const document = {
  body: el('body'), activeElement: null,
  createDocumentFragment: () => el('fragment'), createTextNode: text => el('text', '', text),
  addEventListener() {}, removeEventListener() {},
};
const window = {scrollX: 0, scrollY: 0, scrollTo(x, y) { this.scrollX = x; this.scrollY = y; }};
const view = document.body.appendChild(el('main'));
let activeOverlayClose = null, routeGen = 0, token = '', accept = false;
const writes = [], confirmations = [], notices = [];
function boot() {}
function setCrumb() {}
function esc(text) { return text; }
function actionable(node, fn) { node.onclick = fn; }
function toast(text, error) { notices.push({text, error}); }
function confirm(text) { confirmations.push(text); return accept; }
async function request(path, body) {
  const response = await fetch(base + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {'Content-Type': 'application/json', Authorization: 'Bearer ' + token},
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}
let getJSON = path => request(path);
const postJSON = (path, body) => { writes.push({path, body}); return request(path, body); };
const all = root => [root, ...root.children.flatMap(all)];
const button = (text, root = document.body) => {
  const found = all(root).find(e => e.tag === 'button' && e.textContent === text);
  assert.ok(found, 'missing button: ' + text); return found;
};
const overlay = () => all(document.body).find(e => e.className === 'overlay');
const field = name => {
  const found = all(overlay()).find(e => e.name === name);
  assert.ok(found, 'missing field: ' + name); return found;
};
const checkbox = () => all(overlay()).find(e => e.type === 'checkbox');
const row = key => all(view).find(e => e.tag === 'tr' && e.children[0]?.textContent === key);
async function run() {
  token = (await request('/api/login', {secret})).token;
  await renderConfig();
  const bar = all(view).find(e => e.className === 'row cfg-bar');
  assert.deepEqual(bar.children.map(e => e.textContent), [
    '', 'Modified only', 'Add provider…', 'Add MCP server…', 'Add skill…', 'Add reviewer…',
  ]);
  assert.equal(bar.children[0].placeholder, 'filter settings…');
  bar.children[0].value = 'providers.';
  bar.children[0].oninput();

  await button('Add provider…').onclick();
  assert.equal(field('backend').tag, 'select');
  assert.equal(field('deployment').value, '');
  assert.equal(field('base_url').tag, 'input');
  assert.match(field('base_url').title, /endpoint/);
  field('name').value = 'bad.name';
  await button('Add', overlay()).onclick();
  assert.equal(notices.at(-1).text,
    "provider name 'bad.name' is not a valid TOML bare key" +
    " (use only ASCII letters, digits, '-', '_').");
  assert.equal(button('Add', overlay()).disabled, false);
  field('name').value = 'openrouter'; field('name').oninput();
  assert.equal(field('backend').value, 'openai');
  assert.equal(field('base_url').value, 'https://openrouter.ai/api/v1');
  field('name').value = 'custom'; field('name').oninput();
  assert.equal(field('base_url').value, '');
  field('base_url').value = 'https://custom.example/v1';
  field('name').value = 'openrouter'; field('name').oninput();
  assert.equal(field('base_url').value, 'https://custom.example/v1');
  field('base_url').value = ''; field('name').oninput();
  await button('Add', overlay()).onclick();
  assert.equal(writes.at(-1).path, '/api/config/entry');
  assert.equal(writes.at(-1).body.repo, false);
  assert.ok(row('providers.openrouter.base_url'));
  const refreshedBar = all(view).find(e => e.className === 'row cfg-bar');
  assert.equal(refreshedBar.children[0].value, 'providers.');
  assert.equal(row('sandbox.network').style.display, 'none');
  refreshedBar.children[0].value = ''; refreshedBar.children[0].oninput();

  await button('Add MCP server…').onclick();
  assert.equal(field('command').tag, 'input');
  assert.match(field('command').title, /argv/);
  assert.deepEqual(field('enabled').children.map(e => e.value), ['', 'true', 'false']);
  assert.deepEqual(field('approve').children.map(e => e.value), ['', 'ask', 'allow']);
  await button('Add', overlay()).onclick();
  assert.match(notices.at(-1).text, /name is required/);
  assert.equal(button('Add', overlay()).disabled, false);
  field('name').value = 'local'; field('command').value = 'server "two words"';
  field('enabled').value = 'false'; field('approve').value = 'ask'; checkbox().checked = true;
  await button('Add', overlay()).onclick();
  assert.equal(writes.at(-1).body.table, 'mcp.servers');
  assert.equal(writes.at(-1).body.repo, true);
  const config = await getJSON('/api/config');
  assert.deepEqual(config['mcp.servers.local.command'].value, ['server', 'two words']);
  assert.equal(config['mcp.servers.local.enabled'].value, false);

  for (const name of ['alpha', 'beta', 'org.some.skill']) {
    await button('Add skill…').onclick();
    assert.deepEqual(field('state').children.map(e => e.value), ['enabled', 'disabled', 'always']);
    field('name').value = name; field('state').value = 'always';
    await button('Add', overlay()).onclick();
    assert.ok(row('skills.state.' + (name === 'org.some.skill' ? '"org.some.skill"' : name)));
  }
  row('providers.openrouter.base_url').onclick();
  await button('Add provider…', overlay()).onclick();
  assert.equal(overlay().children[0]['aria-label'], 'add provider');  // the popup hands over to the add form
  button('Cancel', overlay()).onclick();
  for (const [key, label, table, name, repo] of [
    ['mcp.servers.local.url', 'MCP server', 'mcp.servers', 'local', true],
    ['providers.openrouter.base_url', 'provider', 'providers', 'openrouter', false],
    ['skills.state.alpha', 'skill', 'skills.state', 'alpha', false],
    ['skills.state."org.some.skill"', 'skill', 'skills.state', 'org.some.skill', false],
  ]) {
    row(key).onclick();
    const remove = button('Remove ' + label, overlay());
    accept = false;
    const before = writes.length;
    await remove.onclick();
    assert.equal(writes.length, before);
    assert.ok(overlay());
    assert.ok(confirmations.at(-1).includes(label + " '" + name + "'"));
    if (name === 'org.some.skill') {
      assert.ok(!confirmations.at(-1).includes('skills.state.org.some.skill'));
    }
    assert.ok(confirmations.at(-1).includes(repo ? 'repo' : 'global'));
    accept = true;
    await remove.onclick();
    assert.deepEqual(writes.at(-1), {path: '/api/config/entry/remove', body: {table, name, repo}});
    assert.equal(row(key), undefined);
  }
  assert.ok(row('skills.state.beta'));
  row('sandbox.network').onclick();
  const network = all(overlay()).find(e => e.tag === 'select');
  assert.deepEqual(network.children.map(e => e.value), ['auto', 'session', 'host']);
  assert.equal(network.children[1].textContent,
    'session: Run-private network without outside access');
  assert.ok(!all(overlay()).some(e => e.tag === 'input' && e.type !== 'checkbox'));
  assert.ok(!all(overlay()).some(e => e.tag === 'button' && e.textContent.startsWith('Remove ')));
  await button('Cancel', overlay()).onclick();
  await button('Add provider…').onclick();
  field('name').value = 'masked'; field('backend').value = 'openai';
  field('base_url').value = 'https://written.test/v1';
  await button('Add', overlay()).onclick();
  assert.ok(notices.at(-1).text.includes(
    'note: --config ' + flagPath + ' overrides providers.masked.base_url while that flag is used.'
  ));
  const search = () => all(view).find(e => e.className === 'filter');
  search().value = 'sandbox.'; search().oninput();
  await postJSON('/api/config', {key: 'sandbox.run_commands', value: '"ask"'});
  await renderConfig();
  search().value = 'sandbox.'; search().oninput();
  button('Modified only').onclick();
  assert.equal(row('sandbox.run_commands').style.display, '');
  assert.equal(row('sandbox.network').style.display, 'none');
  assert.equal(row('providers.masked.base_url').style.display, 'none');
  button('Modified only').onclick();
  window.scrollY = 140;
  row('sandbox.run_commands').onclick();
  all(overlay()).find(e => e.tag === 'select').value = 'allow';
  await button('Save', overlay()).onclick();
  assert.equal(row('sandbox.run_commands').children[1].textContent, '"allow"');
  assert.equal(search().value, 'sandbox.');
  assert.equal(document.activeElement, row('sandbox.run_commands'));
  assert.equal(window.scrollY, 140);
  row('sandbox.run_commands').onclick();
  await button('Unset', overlay()).onclick();
  assert.equal(row('sandbox.run_commands').children[1].textContent, '"ask"');
  assert.equal(search().value, 'sandbox.');
  assert.equal(document.activeElement, row('sandbox.run_commands'));
  await postJSON('/api/config/entry', {
    table: 'providers', name: 'b', fields: {backend: 'openai', base_url: 'http://localhost:2'},
  });
  await postJSON('/api/config', {
    key: 'models.worker',
    value: '{model = "masked/old", effort = "high", temperature = 0.4}',
  });
  await renderConfig();
  getJSON = path => path.includes('/api/config/suggest/')
    ? Promise.resolve({values: ['b/alpha', 'b/beta']}) : request(path);
  row('models.worker.model').onclick();
  assert.ok(!all(overlay()).some(e => e.id === 'cfg-provider'));
  const model = all(overlay()).find(e => e.role === 'combobox');
  assert.equal(all(overlay()).filter(e => e.tag === 'input' && e.type !== 'checkbox').length, 1);
  assert.equal(model.value, 'masked/old');
  model.focus();
  await Promise.resolve(); await Promise.resolve();
  const options = () => all(overlay()).filter(e => e.role === 'option');
  assert.deepEqual(options().map(e => e.textContent), ['b/alpha', 'b/beta']);
  const press = key => model.onkeydown({key, preventDefault() {}, stopPropagation() {}});
  const count = writes.length;
  press('ArrowDown'); press('ArrowDown'); press('ArrowUp'); press('Enter');
  assert.equal(model.value, 'b/alpha'); assert.equal(writes.length, count);
  model.value = 'b/outside'; model.oninput();
  assert.equal(options().length, 0);
  assert.ok(all(overlay()).some(e => e.role === 'status' && e.textContent.includes('No matches')));
  press('Escape'); assert.equal(model['aria-expanded'], 'false');
  await button('Save', overlay()).onclick();
  const updated = await getJSON('/api/config');
  assert.ok(!('models.worker.provider' in updated));
  assert.equal(updated['models.worker.model'].value, 'b/outside');
  assert.equal(writes.at(-1).body.key, 'models.worker.model');
  assert.equal(updated['models.worker.effort'].value, 'high');
  assert.equal(updated['models.worker.temperature'].value, 0.4);
  getJSON = path => path.includes('/api/config/suggest/')
    ? Promise.resolve({values: []}) : request(path);
  row('models.worker.model').onclick();
  all(overlay()).find(e => e.role === 'combobox').focus();
  await Promise.resolve(); await Promise.resolve();
  assert.ok(all(overlay()).some(e =>
    e.role === 'status' && e.textContent.includes('Catalog empty or unavailable')));
  await button('Cancel', overlay()).onclick();
  getJSON = path => path.includes('/api/config/suggest/')
    ? Promise.reject(new Error('offline')) : request(path);
  row('models.worker.model').onclick();
  all(overlay()).find(e => e.role === 'combobox').focus();
  await Promise.resolve(); await Promise.resolve();
  assert.ok(all(overlay()).some(e =>
    e.role === 'status' && e.textContent.includes('Catalog unavailable')));
  await button('Cancel', overlay()).onclick();
  row('sandbox.run_commands').onclick();
  const saving = button('Save', overlay()).onclick();
  routeGen++; activeOverlayClose();
  view.innerHTML = ''; const other = view.appendChild(el('div', '', 'another page'));
  await saving;
  assert.equal(view.children[0], other);
}
run().catch(error => { console.error(error); process.exitCode = 1; });
"""
    flag = tmp_path / "flag.toml"
    flag.write_text('[providers.masked]\nbackend = "openai"\nbase_url = "https://mask.test/v1"\n')
    srv = server.WebServer(("127.0.0.1", 0), tmp_path, "", (flag,), secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        setup = (
            f"const base = 'http://127.0.0.1:{srv.server_port}', "
            f"secret = {json.dumps(signin.SECRET)}, flagPath = {json.dumps(str(flag))};\n"
        )
        result = subprocess.run(
            [node, "-e", setup + client + scenario],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
