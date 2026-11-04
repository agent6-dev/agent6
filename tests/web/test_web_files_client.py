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
    this.style = {setProperty(k, v) { this[k] = v; }, removeProperty(k) { delete this[k]; }};
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
  addEventListener(name, fn) { this['on' + name] = fn; }
  removeEventListener(name) { this['on' + name] = null; }
  getBoundingClientRect() { return {width: this.cls.has('files-body') ? 1000 : 360}; }
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
const window = {addEventListener() {}, matchMedia: () => ({matches:false})};
let activeTab = '';
const setTab = tab => activeTab = tab;
const closeLive = () => {};
const closeOverlay = () => {};
const saved = new Map();
const recall = (key, fallback) => saved.get(key) ?? fallback;
const remember = (key, value) => saved.set(key, value);
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
            for name in (
                "client_layout.js",
                "client_source.js",
                "client_review.js",
                "client_git_diff.js",
                "client_git.js",
                "client_files.js",
            )
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


def test_commit_route_selects_its_historical_file_and_back_restores_the_list() -> None:
    """A commit path query opens that file rather than stopping at the commit header."""
    _run("""
const sha = 'a'.repeat(40), base = 'b'.repeat(40);
const file = {path:'dir/a #.txt', status:'A', additions:1, deletions:0};
const commit = {sha, short:sha.slice(0,7), subject:'root', parents:[], refs:[], base,
  author:{name:'writer',email:'w@w',date:'2026'}, committer:{name:'writer',date:'2026'},
  files:[file]};
getJSON = async url => {
  if (url === '/api/git/refs') return {head:{sha},refs:[]};
  if (url === '/api/git/worktree') return {files:[]};
  if (url.startsWith('/api/git/graph')) return {commits:[],tips:[],has_more:false};
  if (url === '/api/git/commit/' + sha) return commit;
  if (url.startsWith('/api/git/diff')) return {base,head:sha,patch:'@@ -0,0 +1 @@\\n+text\\n'};
  if (url.startsWith('/api/git/findings')) return {findings:[],read_failures:[]};
  throw new Error(url);
};
location.hash = '#/git/commit/' + sha + '?path=dir%2Fa+%23.txt&line=1';
await route(); await tick();
assert.ok(view.querySelector('.git-body').classList.contains('diffing'));
assert.ok(view.querySelector('.git-diff').textContent.includes('text'));
assert.ok(location.hash.includes('path=dir%2Fa+%23.txt'));
view.querySelector('.git-diff-head').children[0].onclick();
assert.ok(!view.querySelector('.git-body').classList.contains('diffing'));
assert.equal(location.hash, '#/git/commit/' + sha);
assert.ok(view.querySelector('.gfiles').textContent.includes(file.path));
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
  {line:1,sha:'a'.repeat(40),author:'writer',date:'2026-01-02T00:00:00+00:00',filename:'a.txt',original_line:1},
  {line:2,sha:'',author:'not committed',date:''}
]});
await loading;
assert.equal(source.querySelector('.sl'), first);
assert.equal(source.querySelectorAll('.hl-keyword').length, 1);
assert.equal(source.querySelectorAll('.blame-cell').length, 2);
assert.ok(!first.firstChild.classList.contains('blame-cell'));
assert.equal(first.querySelectorAll('.blame-label').length, 1);
assert.ok(first.textContent.includes('aaaaaaa'));
assert.ok(first.textContent.includes('writer'));
assert.ok(first.textContent.includes('2026-01-02'));
assert.equal(first.querySelector('.blame-commit').href,
  '#/git/commit/' + 'a'.repeat(40) + '?path=a.txt&line=1');
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


def test_files_tree_controls_resize_collapse_and_restore() -> None:
    """The Files controls change and retain width and visibility through a render."""
    _run("""
await renderFiles(1);
let tree = view.querySelector('.files-tree');
let handle = view.querySelector('.resize-handle');
let toggle = view.querySelector('.files-tree-toggle');
assert.ok(handle, 'file tree has no divider');
assert.ok(toggle, 'file tree has no collapse control');
handle.onkeydown({key: 'ArrowRight', preventDefault() {}});
assert.equal(tree.style['--files-tree-w'], '380px');
assert.equal(saved.get('a6-files-tree-w'), '380px');
toggle.onclick();
assert.ok(view.querySelector('.files-body').classList.contains('tree-closed'));
assert.equal(toggle['aria-expanded'], 'false');
await renderFiles(1);
tree = view.querySelector('.files-tree');
toggle = view.querySelector('.files-tree-toggle');
assert.equal(tree.style['--files-tree-w'], '380px');
assert.ok(view.querySelector('.files-body').classList.contains('tree-closed'));
toggle.onclick();
assert.ok(!view.querySelector('.files-body').classList.contains('tree-closed'));
assert.equal(toggle['aria-expanded'], 'true');
""")


def test_ignored_toggle_is_worktree_only_and_remembers_the_listing_and_read_option() -> None:
    """Ignored rows need a remembered opt-in; the same choice reaches file reads and badges."""
    _run(r"""
const requests = [];
getJSON = async url => {
  requests.push(url);
  if (url === '/api/git/refs') return {head:{sha:'tip'},refs:[]};
  if (url.startsWith('/api/git/tree')) {
    const ignored = url.includes('ignored=1'), worktree = url.includes('rev=WORKTREE');
    return {rev:worktree ? 'WORKTREE' : 'tip',ignored,
      files:ignored ? [{path:'plan.md',kind:'file',size:7,state:'ignored'}] : [],truncated:false};
  }
  return {text:'# Plan\n',lines:1,spans:[],rendered_html:'<h1>Plan</h1>',
    binary:false,truncated:false};
};
await renderFiles(1, 'HEAD');
assert.equal(view.querySelector('.files-ignored').hidden, true);
await renderFiles(1, 'WORKTREE');
let toggle = view.querySelector('.files-ignored');
assert.equal(toggle.hidden, false);
assert.equal(toggle['aria-pressed'], 'false');
assert.equal(view.querySelector('.gf'), null);
await toggle.onclick();
assert.equal(toggle['aria-pressed'], 'true');
assert.equal(view.querySelector('.gf-st').textContent, 'I');
assert.equal(view.querySelector('.gf-st').title, '1 ignored');
await view.querySelector('.gf').onclick();
assert.ok(requests.some(url => url.startsWith('/api/git/text?') && url.includes('ignored=1')));
assert.ok(view.querySelector('.md-view'));
await renderFiles(1, 'WORKTREE');
toggle = view.querySelector('.files-ignored');
assert.equal(toggle['aria-pressed'], 'true');
assert.equal(view.querySelector('.gf-st').textContent, 'I');
await toggle.onclick();
assert.equal(view.querySelector('.gf'), null);
await renderFiles(1, 'HEAD');
assert.equal(view.querySelector('.files-ignored').hidden, true);
assert.ok(!requests[requests.length-1].includes('ignored=1'));
""")


def test_markdown_mode_switches_content_and_blame_and_survives_reopening() -> None:
    """The remembered markdown mode changes visibility without losing source or blame state."""
    _run(r"""
const answer = {text:'# Heading\n',lines:1,spans:[],digest:'same',
  rendered_html:'<h1 id="heading">Heading</h1>\n',binary:false,truncated:false};
async function open() {
  const opening = renderFiles(1, 'HEAD', 'README.md'); await tick();
  pending[pending.length - 1].resolve(answer); await opening;
}
await open();
let modes = view.querySelector('.files-md-mode');
assert.ok(modes, 'markdown toggle missing');
let [rendered, source] = modes.children;
assert.equal(rendered['aria-pressed'], 'true');
assert.equal(source['aria-pressed'], 'false');
assert.equal(view.querySelector('.md-view').hidden, false);
assert.equal(view.querySelector('.files-source').hidden, true);
assert.equal(view.querySelector('.files-blame').hidden, true);
source.onclick();
assert.equal(source['aria-pressed'], 'true');
assert.equal(view.querySelector('.md-view').hidden, true);
assert.equal(view.querySelector('.files-source').hidden, false);
assert.ok(view.querySelector('.src').textContent.includes('# Heading'));
assert.equal(view.querySelector('.files-blame').hidden, false);
const blaming = view.querySelector('.files-blame').onclick(); await tick();
rendered.onclick();
pending[pending.length - 1].resolve({digest:'same',lines:[]}); await blaming;
assert.equal(view.querySelector('.files-source').hidden, true);
source.onclick();
await open();
modes = view.querySelector('.files-md-mode');
[rendered, source] = modes.children;
assert.equal(source['aria-pressed'], 'true', 'source choice lost on reopening');
assert.equal(view.querySelector('.md-view').hidden, true);
rendered.onclick();
await open();
assert.equal(view.querySelector('.files-md-mode').children[0]['aria-pressed'], 'true',
  'rendered choice lost on reopening');
assert.equal(view.querySelector('.md-view').hidden, false);
""")


def test_markdown_images_use_authenticated_blob_reads_and_opaque_data_urls() -> None:
    """An inline API image cannot send a bearer token; the client fetches its bytes first."""
    _run(r"""
const url = '/api/git/blob?rev=old&path=logo.png';
const image = el('img'); image.dataset.blobSrc = url;
const setHTML = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML').set;
const queryAll = Element.prototype.querySelectorAll;
Object.defineProperty(Element.prototype, 'innerHTML', {set(html) {
  setHTML.call(this, html);
  if (this.cls.has('md-view')) this.appendChild(image);
}});
Element.prototype.querySelectorAll = function(selector) {
  return selector === 'img[data-blob-src]' ? this.children.filter(c => c === image)
    : queryAll.call(this, selector);
};
let fetched = '';
globalThis.getBlob = async request => { fetched = request; return new Blob(['pixels']); };
globalThis.FileReader = class {
  readAsDataURL(blob) {
    blob.text().then(text => {
      this.result = 'data:image/png;base64,' + Buffer.from(text).toString('base64');
      this.onload();
    });
  }
};
const opening = renderFiles(1, 'HEAD', 'README.md'); await tick();
pending[0].resolve({text:'![Logo](logo.png)',lines:1,spans:[],
  rendered_html:'<p><img data-blob-src="' + url + '" alt="Logo"></p>',
  binary:false,truncated:false});
await opening; await new Promise(setImmediate);
assert.equal(fetched, url, 'image bytes did not use getBlob');
assert.equal(image.src, 'data:image/png;base64,cGl4ZWxz', 'image lacks an opaque data URL');
""")


def test_markdown_anchor_query_survives_the_resolved_file_selection() -> None:
    """Resolving HEAD to a commit must not discard a table-of-contents destination."""
    _run(r"""
remember(MARKDOWN_MODE_KEY, 'source');
const opening = renderFiles(1, 'HEAD', 'README.md', new URLSearchParams('anchor=heading'));
await tick();
pending[0].resolve({text:'# Heading\n',lines:1,spans:[],
  rendered_html:'<h1 id="heading">Heading</h1>\n',binary:false,truncated:false});
await opening;
assert.equal(location.hash, '#/files/old/README.md?anchor=heading');
assert.equal(view.querySelector('.files-md-mode').children[1]['aria-pressed'], 'true',
  'an anchor discarded the remembered source choice');
""")


def test_blame_groups_stop_at_sides_paths_gaps_and_absent_cells() -> None:
    """Grouping never joins unrelated displayed lines, even when the sha matches."""
    _run("""
const source = renderLines(['a', 'b', 'c', 'd', 'e', 'f', 'g'], [], null);
view.appendChild(source);
const nodes = source.querySelectorAll('.sl');
const rows = nodes.map((node, i) => ({node, side:'head', segment:0, path:'new.txt',
  line:{line:i+1, sha:'a'.repeat(40), filename:'old.txt', original_line:i+1,
    author:'writer', date:'2026-01-01'}}));
rows[2].side = 'base';
rows[3].line.filename = 'another.txt';
rows[4].segment = 1;
rows[5].node = null;
renderBlame(rows);
assert.equal(source.querySelectorAll('.blame-label').length, 5);
assert.equal(nodes[1].querySelector('.blame-cell').textContent, '');
assert.equal(source.querySelectorAll('.blame-cell').length, 6);
assert.equal(pending.length, 0, 'commit details fetched before hover');
""")


def test_diff_controls_keep_paths_revisions_and_full_context_in_the_cache_key() -> None:
    """Files links preserve head identity, and full context makes a distinct request."""
    _run("""
const body = el('div'); view.appendChild(body);
const viewer = gitDiffViewer(body), requests = [];
getJSON = async url => {
  requests.push(url);
  return url.startsWith('/api/git/findings') ? {findings:[],read_failures:[]}
    : {base:'base',head:'head',patch:'@@ -1 +1 @@\\n-old\\n+new\\n'};
};
const f = {path:'dir/a #.txt',old_path:'old.txt',status:'M',additions:1,deletions:1};
await viewer.showDiff({base:'base',head:'WORKTREE',files:[f]}, 0);
assert.equal(body.querySelector('.diff-open-file').href, '#/files/WORKTREE/dir/a%20%23.txt');
await body.querySelector('.diff-context').onclick();
assert.ok(requests.some(url => url.includes('full_context=1')));
assert.equal(body.querySelector('.diff-context').textContent, 'Show hunks only');
await viewer.showDiff({base:'base',head:'head',files:[{...f,status:'D'}]}, 0);
assert.equal(body.querySelector('.diff-open-file').disabled, true);
assert.equal(body.querySelector('.diff-open-base').href, '#/files/base/old.txt');
await viewer.showDiff({base:'tree1',head:'tree2',panel:'p',files:[f],
  query:new URLSearchParams('finding=keep')}, 0);
assert.equal(body.querySelector('.diff-open-file').href,
  '#/files/tree2/dir/a%20%23.txt?finding=keep&panel=p&side=head');
assert.equal(body.querySelector('.diff-context').disabled, true);
""")


@pytest.mark.parametrize("mode", ["inline", "split", "old", "new"])
def test_diff_blame_uses_each_displayed_lines_side(mode: str) -> None:
    """Equal context text can carry different attribution on each side."""
    _run(
        "remember(DIFF_MODE_KEY, "
        + repr(mode)
        + ");\n"
        + r"""
const body = el('div'); view.appendChild(body);
const viewer = gitDiffViewer(body), requests = [];
const sha = {base:'a'.repeat(40),head:'b'.repeat(40)};
getJSON = async url => {
  requests.push(url);
  const q = new URLSearchParams(url.split('?')[1]), side = q.get('rev');
  if (url.startsWith('/api/git/findings')) return {findings:[],read_failures:[]};
  if (url.startsWith('/api/git/text')) return {digest:side, text:'same\nsame\n'};
  if (url.startsWith('/api/git/blame')) return {digest:side,lines:[1,2].map(line => ({
    line,sha:sha[side],author:side,date:'2026-01-01',filename:'f.txt',original_line:line}))};
  return {base:'base',head:'head',base_digest:'base',head_digest:'head',
    patch:'@@ -1,2 +1,2 @@\n-old\n+new\n same\n'};
};
await viewer.showDiff({base:'base',head:'head',files:[
  {path:'f.txt',status:'M',additions:1,deletions:1}]}, 0);
const toggle = body.querySelector('.diff-blame');
assert.ok(toggle, 'missing diff blame toggle');
await toggle.onclick();
const mode = recall(DIFF_MODE_KEY);
const cells = body.querySelectorAll('.blame-cell');
assert.equal(cells.length, mode === 'split' ? 4 : mode === 'inline' ? 3 : 2);
for (const cell of cells) assert.equal(cell.dataset.sha, sha[cell.dataset.side]);
const sides = new Set(cells.map(c => c.dataset.side));
assert.deepEqual([...sides].sort(), mode === 'old' ? ['base'] : mode === 'new'
  ? ['head'] : ['base', 'head']);
assert.ok(requests.filter(u => u.startsWith('/api/git/blame')).every(u => u.includes('whole=1')));
await toggle.onclick();
assert.equal(body.querySelectorAll('.blame-cell').length, 0);
"""
    )


def test_diff_blame_refuses_other_bytes_and_ignores_late_responses() -> None:
    """Worktree attribution cannot attach to a changed file or a superseded mode."""
    _run(r"""
const body = el('div'); view.appendChild(body);
const viewer = gitDiffViewer(body), waiters = [];
getJSON = async url => {
  if (url.startsWith('/api/git/findings')) return {findings:[],read_failures:[]};
  if (url.startsWith('/api/git/blame')) return new Promise(resolve => waiters.push(resolve));
  return {base:'base',head:'WORKTREE',base_digest:'base',head_digest:'shown',
    patch:'@@ -0,0 +1 @@\n+new\n'};
};
const ctx = {base:'base',head:'WORKTREE',files:[
  {path:'f.txt',status:'A',additions:1,deletions:0}]};
await viewer.showDiff(ctx, 0);
let loading = body.querySelector('.diff-blame').onclick(); await tick();
waiters[0]({digest:'changed',lines:[]}); await loading;
assert.ok(body.querySelector('.blame-note').textContent.includes('changed'));
assert.equal(body.querySelectorAll('.blame-cell').length, 0);
await body.querySelector('.diff-blame').onclick();
loading = body.querySelector('.diff-blame').onclick(); await tick();
viewer.closeDiff();
waiters[1]({digest:'shown',lines:[{line:1,sha:'a'.repeat(40)}]}); await loading;
assert.equal(body.querySelectorAll('.blame-cell').length, 0);
await viewer.showDiff({...ctx, panel:'recorded'}, 0);
assert.ok(body.querySelector('.blame-note').textContent.includes('recorded'));
assert.equal(waiters.length, 2, 'recorded trees fell back to HEAD');
""")


def test_diff_blame_ignores_responses_after_a_mode_or_route_change() -> None:
    """A stale request cannot add gutters while the old DOM is still connected."""
    _run(r"""
const body = el('div'); view.appendChild(body);
const viewer = gitDiffViewer(body), waiters = [];
getJSON = async url => {
  if (url.startsWith('/api/git/findings')) return {findings:[],read_failures:[]};
  if (url.startsWith('/api/git/blame')) return new Promise(resolve => waiters.push(resolve));
  return {base:'base',head:'head',head_digest:'shown',patch:'@@ -0,0 +1 @@\n+new\n'};
};
const ctx = {base:'base',head:'head',files:[{path:'f.txt',status:'A'}]};
const answer = {digest:'shown',lines:[{line:1,sha:'a'.repeat(40),
  filename:'f.txt',original_line:1,author:'a',date:'2026'}]};
await viewer.showDiff(ctx, 0);
const old = body.querySelector('.diff-blame').onclick(); await tick();
remember(DIFF_MODE_KEY, 'split');
const next = viewer.showDiff(ctx, 0); await tick();
waiters[0](answer); await old;
assert.equal(body.querySelectorAll('.blame-cell').length, 0);
routeGen++;
waiters[1](answer); await next;
assert.equal(body.querySelectorAll('.blame-cell').length, 0);
""")


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
