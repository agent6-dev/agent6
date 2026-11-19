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
    web = resources.files("agent6.ui.web")
    core = web.joinpath("client.js").read_text(encoding="utf-8")
    dialogs = core[core.index("// --- dialogs") : core.index("// --- sign-in")]
    client = web.joinpath("client_config.js").read_text(encoding="utf-8")
    scenario = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag, cls, text = '') {
    Object.assign(this, {tag, className: cls, textContent: text, children: [], style: {},
                        value: '', checked: false, disabled: false, isConnected: true,
                        listeners: {}});
  }
  // An element's text is its own, then its children's, as in the DOM: a cell holds text nodes and
  // <wbr>s, a dialog's title its first part and a span after each dot.
  get textContent() { return (this.text ?? '') + this.children.map(c => c.textContent).join(''); }
  set textContent(text) { this.text = text; this.children = []; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  // A modal dialog hands the focus back to its opener when it closes.
  showModal() { this.open = true; this.opener = document.activeElement; }
  close() {
    if (!this.open) return;
    this.open = false; this.opener?.focus();
    (this.listeners.close || []).forEach(fn => fn());
  }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  appendChild(child) {
    this.children.push(child); child.parent = this;
    if (this.tag === 'select' && this.children.length === 1) this.value = child.value;
    return child;
  }
  set innerHTML(value) { this.text = ''; this.children = []; }
  replaceChildren(...nodes) { this.textContent = ''; nodes.forEach(n => this.appendChild(n)); }
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
  // openDialog's one question: is a dialog open?
  querySelector(selector) { assert.equal(selector, 'dialog[open]'); return dialog() ?? null; },
  getElementById: () => el('div'),  // the messages, which ride in an open dialog
  documentElement: {clientWidth: 0, classList: {toggle() {}}},
};
const window = {scrollX: 0, scrollY: 0, scrollTo(x, y) { this.scrollX = x; this.scrollY = y; }};
const innerWidth = 0;
const view = document.body.appendChild(el('main'));
let routeGen = 0, token = '';
const writes = [], notices = [];
function boot() {}
function setCrumb() {}
function esc(text) { return text; }
function composing(e) { return e.isComposing || e.keyCode === 229; }
function toast(text, error) { notices.push({text, error}); }
function withCode(text) { return el('text', '', text); }
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
const dialog = () => all(document.body).find(e => e.tag === 'dialog' && e.open);
const alert = () => all(dialog()).find(e => e.role === 'alert');
const field = name => {
  const found = all(dialog()).find(e => e.name === name);
  assert.ok(found, 'missing field: ' + name); return found;
};
// The line a field's aria-describedby names: an Add dialog field's description.
const described = name => all(dialog()).find(e => e.id === field(name)['aria-describedby']);
const checkbox = () => all(dialog()).find(e => e.type === 'checkbox');
const row = key => all(view).find(e => e.tag === 'tr' && e.children[0]?.textContent === key);
const edit = key => row(key).children[0].children[0];  // the key cell's button
async function run() {
  token = (await request('/api/login', {secret})).token;
  await renderConfig();
  const bar = all(view).find(e => e.className === 'toolbar cfg-bar');
  assert.deepEqual(bar.children.map(e => e.textContent), [
    '', 'Modified only', 'Preset…', 'Add preset…',
    'Add provider…', 'Add MCP server…', 'Add skill…', 'Add reviewer…', 'Remove preset',
  ]);
  assert.equal(button('Remove preset').disabled, true);  // preset = none
  assert.equal(bar.children[0].placeholder, 'filter settings…');
  bar.children[0].value = 'providers.';
  bar.children[0].oninput();

  await button('Add provider…').onclick();
  assert.equal(field('backend').tag, 'select');
  assert.equal(field('deployment').value, '');
  assert.equal(field('base_url').tag, 'input');
  assert.match(described('base_url').textContent, /endpoint/);
  field('name').value = 'bad.name';
  const toasted = notices.length;
  await button('Add', dialog()).onclick();
  // The refusal stays in its dialog: a toast would sit under the backdrop.
  assert.equal(alert().textContent,
    "provider name 'bad.name' is not a valid TOML bare key" +
    " (use only ASCII letters, digits, '-', '_').");
  assert.equal(notices.length, toasted);
  assert.equal(button('Add', dialog()).disabled, false);
  field('name').value = 'openrouter'; field('name').oninput();
  assert.equal(field('backend').value, 'openai');
  assert.equal(field('base_url').value, 'https://openrouter.ai/api/v1');
  field('name').value = 'custom'; field('name').oninput();
  assert.equal(field('base_url').value, '');
  field('base_url').value = 'https://custom.example/v1';
  field('name').value = 'openrouter'; field('name').oninput();
  assert.equal(field('base_url').value, 'https://custom.example/v1');
  field('base_url').value = ''; field('name').oninput();
  await button('Add', dialog()).onclick();
  assert.equal(writes.at(-1).path, '/api/config/entry');
  assert.equal(writes.at(-1).body.repo, false);
  assert.ok(row('providers.openrouter.base_url'));
  const refreshedBar = all(view).find(e => e.className === 'toolbar cfg-bar');
  assert.equal(refreshedBar.children[0].value, 'providers.');
  assert.equal(row('sandbox.network').style.display, 'none');
  refreshedBar.children[0].value = ''; refreshedBar.children[0].oninput();

  await button('Add MCP server…').onclick();
  assert.equal(field('command').tag, 'input');
  assert.match(described('command').textContent, /argv/);
  assert.deepEqual(field('enabled').children.map(e => e.value), ['', 'true', 'false']);
  assert.deepEqual(field('approve').children.map(e => e.value), ['', 'ask', 'allow']);
  await button('Add', dialog()).onclick();
  assert.match(alert().textContent, /name is required/);
  assert.equal(button('Add', dialog()).disabled, false);
  field('name').value = 'local'; field('command').value = 'server "two words"';
  field('enabled').value = 'false'; field('approve').value = 'ask'; checkbox().checked = true;
  await button('Add', dialog()).onclick();
  assert.equal(writes.at(-1).body.table, 'mcp.servers');
  assert.equal(writes.at(-1).body.repo, true);
  const config = await getJSON('/api/config');
  assert.deepEqual(config['mcp.servers.local.command'].value, ['server', 'two words']);
  assert.equal(config['mcp.servers.local.enabled'].value, false);

  for (const name of ['alpha', 'beta', 'org.some.skill']) {
    await button('Add skill…').onclick();
    assert.deepEqual(field('state').children.map(e => e.value), ['enabled', 'disabled', 'always']);
    field('name').value = name; field('state').value = 'always';
    await button('Add', dialog()).onclick();
    assert.ok(row('skills.state.' + (name === 'org.some.skill' ? '"org.some.skill"' : name)));
  }
  row('providers.openrouter.base_url').onclick();
  await button('Add provider…', dialog()).onclick();
  // handed to the add form
  assert.equal(all(dialog()).find(e => e.tag === 'h2').textContent, 'Add provider');
  button('Cancel', dialog()).onclick();
  for (const [key, label, table, name, repo] of [
    ['mcp.servers.local.url', 'MCP server', 'mcp.servers', 'local', true],
    ['providers.openrouter.base_url', 'provider', 'providers', 'openrouter', false],
    ['skills.state.alpha', 'skill', 'skills.state', 'alpha', false],
    ['skills.state."org.some.skill"', 'skill', 'skills.state', 'org.some.skill', false],
  ]) {
    // The question takes the editor's place; its Cancel writes nothing, its Remove removes.
    for (const answer of ['Cancel', 'Remove']) {
      row(key).onclick();
      const before = writes.length;
      const removing = button('Remove ' + label, dialog()).onclick();
      const question = dialog();
      assert.equal(question.role, 'alertdialog');
      assert.ok(question.textContent.includes(label + " '" + name + "'"));
      if (name === 'org.some.skill') {
        assert.ok(!question.textContent.includes('skills.state.org.some.skill'));
      }
      assert.ok(question.textContent.includes(repo ? 'repo' : 'global'));
      button(answer, question).onclick();
      await removing;
      assert.equal(dialog(), undefined);
      if (answer === 'Cancel') assert.equal(writes.length, before);
    }
    assert.deepEqual(writes.at(-1), {path: '/api/config/entry/remove', body: {table, name, repo}});
    assert.equal(row(key), undefined);
  }
  assert.ok(row('skills.state.beta'));
  row('sandbox.network').onclick();
  const network = all(dialog()).find(e => e.tag === 'select');
  assert.deepEqual(network.children.map(e => e.value), ['auto', 'session', 'host']);
  assert.equal(network.children[1].textContent, 'session');
  const meaning = () => all(dialog()).find(e => e.id === 'cfg-meaning').textContent;
  assert.equal(network['aria-describedby'], 'cfg-desc cfg-meaning');
  network.value = 'session'; network.oninput();
  assert.equal(meaning(), 'Run-private network without outside access');
  assert.ok(!all(dialog()).some(e => e.tag === 'input' && e.type !== 'checkbox'));
  assert.ok(!all(dialog()).some(e => e.tag === 'button' && e.textContent.startsWith('Remove ')));
  await button('Cancel', dialog()).onclick();
  await button('Add provider…').onclick();
  field('name').value = 'masked'; field('backend').value = 'openai';
  field('base_url').value = 'https://written.test/v1';
  await button('Add', dialog()).onclick();
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
  all(dialog()).find(e => e.tag === 'select').value = 'allow';
  await button('Save', dialog()).onclick();
  assert.equal(row('sandbox.run_commands').children[1].textContent, '"allow"');
  assert.equal(search().value, 'sandbox.');
  assert.equal(document.activeElement, edit('sandbox.run_commands'));
  assert.equal(window.scrollY, 140);
  row('sandbox.run_commands').onclick();
  await button('Unset', dialog()).onclick();
  assert.equal(row('sandbox.run_commands').children[1].textContent, '"ask"');
  assert.equal(search().value, 'sandbox.');
  assert.equal(document.activeElement, edit('sandbox.run_commands'));
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
  assert.ok(!all(dialog()).some(e => e.id === 'cfg-provider'));
  const model = all(dialog()).find(e => e.role === 'combobox');
  assert.equal(all(dialog()).filter(e => e.tag === 'input' && e.type !== 'checkbox').length, 1);
  assert.equal(model.value, 'masked/old');
  model.focus();
  await Promise.resolve(); await Promise.resolve();
  const options = () => all(dialog()).filter(e => e.role === 'option');
  assert.deepEqual(options().map(e => e.textContent), ['b/alpha', 'b/beta']);
  const press = key => model.onkeydown({key, preventDefault() {}, stopPropagation() {}});
  const count = writes.length;
  press('ArrowDown'); press('ArrowDown'); press('ArrowUp'); press('Enter');
  assert.equal(model.value, 'b/alpha'); assert.equal(writes.length, count);
  model.value = 'b/outside'; model.oninput();
  assert.equal(options().length, 0);
  assert.ok(all(dialog()).some(e => e.role === 'status' && e.textContent.includes('No matches')));
  await button('Save', dialog()).onclick();
  const updated = await getJSON('/api/config');
  assert.ok(!('models.worker.provider' in updated));
  assert.equal(updated['models.worker.model'].value, 'b/outside');
  assert.equal(writes.at(-1).body.key, 'models.worker.model');
  assert.equal(updated['models.worker.effort'].value, 'high');
  assert.equal(updated['models.worker.temperature'].value, 0.4);
  getJSON = path => path.includes('/api/config/suggest/')
    ? Promise.resolve({values: []}) : request(path);
  row('models.worker.model').onclick();
  all(dialog()).find(e => e.role === 'combobox').focus();
  await Promise.resolve(); await Promise.resolve();
  assert.ok(all(dialog()).some(e =>
    e.role === 'status' && e.textContent.includes('Catalog empty or unavailable')));
  await button('Cancel', dialog()).onclick();
  getJSON = path => path.includes('/api/config/suggest/')
    ? Promise.reject(new Error('offline')) : request(path);
  row('models.worker.model').onclick();
  all(dialog()).find(e => e.role === 'combobox').focus();
  await Promise.resolve(); await Promise.resolve();
  assert.ok(all(dialog()).some(e =>
    e.role === 'status' && e.textContent.includes('Catalog unavailable')));
  await button('Cancel', dialog()).onclick();
  // Add preset selects the new preset; the page then edits it: a save lands in its table.
  await postJSON('/api/config/preset', {name: 'fast'});
  await renderConfig();
  assert.equal(button('Remove preset').disabled, false);
  assert.equal(button('Add provider…').disabled, false);
  button('Preset…').onclick();
  assert.equal(all(dialog()).find(e => e.tag === 'select').value, 'fast');
  await button('Cancel', dialog()).onclick();
  row('sandbox.network').onclick();
  all(dialog()).find(e => e.tag === 'select').value = 'host';
  await button('Save', dialog()).onclick();
  assert.deepEqual(writes.at(-1), {path: '/api/config',
    body: {key: 'sandbox.network', value: 'host', repo: false}});
  const refetched = await getJSON('/api/config');
  assert.equal(refetched['sandbox.network'].source, 'preset');
  await postJSON('/api/config/preset/remove', {name: 'fast'});
  await renderConfig();
  assert.equal(button('Remove preset').disabled, true);
  row('sandbox.run_commands').onclick();
  const saving = button('Save', dialog()).onclick();
  routeGen++; dialog().close();
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
            [node, "-"],
            input=setup + dialogs + client + scenario,
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
