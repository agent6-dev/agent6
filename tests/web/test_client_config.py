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
  focus() { document.activeElement = this; }
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
const getJSON = path => request(path);
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
    '', 'Add provider…', 'Add MCP server…', 'Add skill…', 'Add reviewer…',
  ]);
  assert.equal(bar.children[0].placeholder, 'filter settings…');

  await button('Add provider…').onclick();
  assert.equal(field('api_format').tag, 'select');
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
  assert.equal(field('api_format').value, 'openai');
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
  assert.ok(!all(overlay()).some(e => e.tag === 'button' && e.textContent.startsWith('Remove ')));
  await button('Cancel', overlay()).onclick();
  await button('Add provider…').onclick();
  field('name').value = 'masked'; field('api_format').value = 'openai';
  field('base_url').value = 'https://written.test/v1';
  await button('Add', overlay()).onclick();
  assert.ok(notices.at(-1).text.includes(
    'note: --config ' + flagPath + ' overrides providers.masked.base_url while that flag is used.'
  ));
}
run().catch(error => { console.error(error); process.exitCode = 1; });
"""
    flag = tmp_path / "flag.toml"
    flag.write_text(
        '[providers.masked]\napi_format = "openai"\nbase_url = "https://mask.test/v1"\n'
    )
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
            timeout=30,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
