"""Exercise Files request lifetimes through the client script without a browser."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
from importlib import resources

import pytest

_DOM = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag, cls = '', text = '') {
    this.tag = tag; this.cls = new Set((cls || '').split(' ')); this.text = text;
    this.children = []; this.dataset = {}; this.value = ''; this.parent = null;
    this.style = {setProperty() {}};
    this.classList = {
      add: c => this.cls.add(c), remove: c => this.cls.delete(c),
      contains: c => this.cls.has(c),
      toggle: (c, yes) => yes ? this.cls.add(c) : this.cls.delete(c),
    };
  }
  set innerHTML(_) {
    this.children.forEach(c => c.parent = null); this.children = []; this.text = '';
  }
  get firstChild() { return this.children[0] || null; }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  set textContent(t) { this.innerHTML = ''; this.text = t; }
  get isConnected() { return this === view || !!this.parent?.isConnected; }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  insertBefore(c, before) {
    this.children.splice(this.children.indexOf(before), 0, c); c.parent = this;
  }
  remove() {
    this.parent.children = this.parent.children.filter(c => c !== this); this.parent = null;
  }
  setAttribute(k, v) { this[k] = v; }
  scrollIntoView() { this.scrolled = true; }
  querySelectorAll(selector) {
    const matches = n => selector.startsWith('.')
      ? selector.slice(1).split('.').every(c => n.cls.has(c))
      : selector === '[data-line]' && n.dataset.line;
    return this.children.flatMap(c => [
      ...(matches(c) ? [c] : []), ...c.querySelectorAll(selector)
    ]);
  }
  querySelector(s) { return this.querySelectorAll(s)[0] || null; }
}
const el = (tag, cls, text = '') => new Element(tag, cls, text);
const document = {
  createTextNode: text => el('text', '', text),
  createElementNS: (_, tag) => el(tag),
  querySelectorAll: s => view.querySelectorAll(s),
};
const location = {hash: '#/'};
const history = {
  pushState: (_, title, hash) => location.hash = hash,
  replaceState: (_, title, hash) => location.hash = hash,
};
const window = {addEventListener() {}};
let activeTab = '';
const setTab = tab => activeTab = tab;
const closeLive = () => {};
const closeOverlay = () => {};
const recall = (_, fallback) => fallback;
const remember = () => {};
const toast = message => { throw new Error(message); };
const view = el('main');
const setCrumb = () => {};
const boot = () => {};
const esc = x => x;
const actionable = (node, fn) => node.onclick = fn;
const tick = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
let chooseRevision;
const picker = (_, value, choose) => {
  chooseRevision = choose;
  return {root: el('input'), set() {}};
};
const pending = [];
let getJSON = async url => {
  if (url === '/api/git/refs') return {head: {sha: 'tip'}, refs: []};
  if (url.startsWith('/api/git/tree')) return {rev: url.includes('new') ? 'new' : 'old',
    files: [{path: 'a.txt', kind: 'file', size: 5}], truncated: false};
  return new Promise((resolve, reject) => pending.push({url, resolve, reject}));
};
"""


def _run(scenario: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise the Files client")
    web = resources.files("agent6.ui.web")
    client = web.joinpath("client.js").read_text(encoding="utf-8")
    router = client[client.index("// --- router") : client.index("// --- hub")]
    source = (
        router
        + "\n"
        + "\n".join(
            web.joinpath(name).read_text(encoding="utf-8")
            for name in ("client_source.js", "client_git.js", "client_files.js")
        )
    )
    script = (
        _DOM
        + source
        + "\ngitPicker = picker; routeGen = 1;\nlet finished = false;\n(async () => {\n"
        + scenario
        + "\nfinished = true;\n})().catch(e => { console.error(e); process.exitCode = 1; });\n"
        + "process.on('beforeExit', () => { if (!finished && !process.exitCode) {"
        + " console.error('the scenario did not finish'); process.exitCode = 1; } });"
    )
    result = subprocess.run([node, "-e", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failed", [False, True])
def test_revision_change_discards_pending_content_and_errors(failed: bool) -> None:
    """A late response cannot restore the file cleared by a revision change."""
    reply = (
        "reject(new Error('stale error'))"
        if failed
        else "resolve({text: 'stale content', lines: 1, spans: []})"
    )
    _run(
        """
await renderFiles(1);
const opening = view.querySelector('.gf').onclick();
await tick();
chooseRevision({rev: 'new', label: 'new'});
await tick();
assert.equal(pending.length, 1);
pending[0]."""
        + reply
        + """;
await opening;
assert.ok(!view.querySelector('.files-view').textContent.includes('stale'), 'old response painted');
assert.ok(!view.querySelector('.files-body').classList.contains('viewing'), 'old file stayed open');
"""
    )


def test_file_links_decode_the_path_and_publish_selection() -> None:
    """The hash opens an encoded file path and a click publishes the resolved revision."""
    _run("""
location.hash = '#/files/HEAD/dir/a%20%23%25.txt';
const opening = route();
await tick();
assert.equal(activeTab, 'files');
assert.equal(pending.length, 1, 'deep link did not open its file');
assert.ok(pending[0].url.includes('path=dir%2Fa%20%23%25.txt'));
pending[0].resolve({text: 'linked content', lines: 1, spans: []});
await opening;
assert.ok(view.querySelector('.files-view').textContent.includes('linked content'));
const next = view.querySelector('.gf').onclick();
await tick();
assert.equal(location.hash, '#/files/old/a.txt');
pending[1].reject(new Error('no file'));
await next;
assert.ok(view.querySelector('.files-view').textContent.includes('no file'));
""")


def test_git_links_open_commit_compare_and_unknown_revision_errors() -> None:
    """The router opens and publishes commit and compare selections without replacing the graph."""
    _run("""
const sha = 'a'.repeat(40), base = 'b'.repeat(40);
const commit = {sha, short: sha.slice(0,7), subject: 'linked commit', parents: [], refs: [],
  author: {name:'t', email:'t@t', date:'2026-01-01'},
  committer: {name:'t', email:'t@t', date:'2026-01-01'}, files: [], base};
getJSON = async url => {
  if (url === '/api/git/refs') return {head: {sha}, refs: []};
  if (url === '/api/git/worktree') return {files: []};
  if (url.startsWith('/api/git/graph')) return {commits: [commit], tips:[sha], has_more:false};
  if (url === '/api/git/commit/' + sha) return commit;
  if (url.startsWith('/api/git/compare')) return {
    base:{sha:base,short:'bbbbbbb',subject:'base'},
    head:{sha,short:'aaaaaaa',subject:'head'}, files:[]};
  throw new Error('unknown revision');
};
location.hash = '#/git/commit/' + sha;
await route(); await tick();
assert.ok(view.querySelector('.git-side').textContent.includes('linked commit'),
  'commit link ignored');
assert.ok(view.querySelector('.gc-row').scrolled);
location.hash = '#/git/compare/' + base + '..' + sha;
await route(); await tick();
assert.ok(view.querySelector('.gcmp-result').textContent.includes('bbbbbbb'),
  'compare link ignored');
location.hash = '#/git';
await route(); await tick();
assert.equal(view.querySelector('.gcmp-result'), null, 'a bare #/git restored the comparison');
const graph = view.querySelector('.git-graph');
await view.querySelector('.gc-row').onclick();
assert.equal(location.hash, '#/git/commit/' + sha);
assert.equal(view.querySelector('.git-graph'), graph, 'selection replaced the graph');
location.hash = '#/git/commit/missing';
await route(); await tick();
assert.ok(view.querySelector('.git-side').textContent.includes('unknown revision'));
""")


def test_blame_toggles_a_column_on_existing_numbered_lines() -> None:
    """Blame attaches by data-line without rebuilding highlights and links to its commit."""
    _run(
        r"""
await renderFiles(1);
const opening = view.querySelector('.gf').onclick();
await tick();
pending[0].resolve({text: 'one\ntwo\n', lines: 2, spans: [[[0,3,'keyword']],[]]});
await opening;
const source = view.querySelector('.src');
const first = source.querySelector('.sl');
const toggle = view.querySelector('.files-blame');
assert.ok(toggle, 'blame toggle missing');
const loading = toggle.onclick();
await tick();
assert.ok(pending[1].url.startsWith('/api/git/blame?rev=old&path=a.txt'));
pending[1].resolve({lines:[
  {line:1,sha:'a'.repeat(40),author:'writer',date:'2026-01-02T00:00:00+00:00'},
  {line:2,sha:'',author:'not committed',date:''}
]});
await loading;
assert.equal(source.querySelector('.sl'), first);
assert.equal(source.querySelectorAll('.hl-keyword').length, 1);
assert.equal(source.querySelectorAll('.blame-cell').length, 2);
assert.ok(first.firstChild.classList.contains('blame-cell'));
assert.ok(first.textContent.includes('aaaaaaa'));
assert.ok(first.textContent.includes('writer'));
assert.ok(first.textContent.includes('2026-01-02'));
assert.equal(first.querySelector('.blame-commit').href, '#/git/commit/' + 'a'.repeat(40));
assert.ok(source.querySelectorAll('.blame-cell')[1].textContent.includes('not committed'));
await toggle.onclick();
assert.equal(source.querySelectorAll('.blame-cell').length, 0);
assert.equal(source.querySelector('.sl'), first);
const late = toggle.onclick(); await tick();
await toggle.onclick();
pending[2].reject(new Error('stale blame error'));
await late;
assert.ok(!view.textContent.includes('stale blame error'));
"""
    )


def test_client_javascript_parses() -> None:
    """The composed source parses without a browser or network."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to parse the client")
    web = resources.files("agent6.ui.web")
    for asset in web.iterdir():
        if asset.name.endswith(".js"):
            result = subprocess.run(
                [node, "--check", str(pathlib.Path(str(asset)))],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr


def test_a_scenario_that_never_finishes_fails() -> None:
    """A dangling await ends Node before its assertions ran; the harness reads that as red."""
    with pytest.raises(AssertionError, match="did not finish"):
        _run("await new Promise(() => {});")


def test_blame_of_other_bytes_than_the_text_shown_is_not_attached() -> None:
    """Blame rows whose digest differs from the text's are refused with a note, not attached."""
    _run(
        r"""
await renderFiles(1);
const opening = view.querySelector('.gf').onclick();
await tick();
pending[0].resolve({text: 'one\ntwo\n', lines: 2, spans: [[], []], digest: 'd1'});
await opening;
const toggle = view.querySelector('.files-blame');
const loading = toggle.onclick();
await tick();
pending[1].resolve({digest: 'd2', lines: [
  {line: 1, sha: 'a'.repeat(40), author: 'writer', date: '2026-01-02T00:00:00+00:00'}
]});
await loading;
assert.equal(view.querySelectorAll('.blame-cell').length, 0, 'blame attached to other bytes');
assert.ok(view.querySelector('.blame-note').textContent.includes('changed since'),
  'no note of the change');
"""
    )


def test_a_stale_commit_response_leaves_another_tabs_url_alone() -> None:
    """A commit response arriving after the user moved to Files rewrote the Files URL."""
    _run("""
const sha = 'a'.repeat(40);
const commit = {sha, short: sha.slice(0,7), subject: 'late commit', parents: [], refs: [],
  author: {name:'t', email:'t@t', date:'2026-01-01'},
  committer: {name:'t', email:'t@t', date:'2026-01-01'}, files: [], base: sha};
let holdRefs = false, releaseRefs;
getJSON = async url => {
  if (url === '/api/git/refs') {
    if (!holdRefs) return {head: {sha}, refs: []};
    return new Promise(resolve => { releaseRefs = () => resolve({head: {sha}, refs: []}); });
  }
  if (url === '/api/git/worktree') return {files: []};
  if (url.startsWith('/api/git/graph')) return {commits: [commit], tips: [sha], has_more: false};
  if (url === '/api/git/commit/' + sha) return new Promise(resolve => pending.push({url, resolve}));
  if (url.startsWith('/api/git/tree')) return {rev: 'HEAD', files: [], truncated: false};
  throw new Error('unexpected ' + url);
};
location.hash = '#/git';
await route(); await tick();
holdRefs = true;
const opening = view.querySelector('.gc-row').onclick();
await tick();
assert.equal(location.hash, '#/git/commit/' + sha);
assert.equal(pending.length, 1, 'the commit request did not go out');
location.hash = '#/files';
const files = route();
await tick();
pending[0].resolve(commit);
await opening; await tick();
assert.equal(location.hash, '#/files', 'a stale commit response rewrote the Files URL');
releaseRefs(); await files; await tick();
assert.equal(activeTab, 'files');
""")
