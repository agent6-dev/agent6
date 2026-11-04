from __future__ import annotations

import shutil
import subprocess
from importlib import resources

import pytest

from tests.web import test_client_review, test_web_files_client


@pytest.mark.parametrize("mode", ["inline", "split"])
def test_diff_renderers_use_side_lines_and_keep_markers_outside_spans(mode: str) -> None:
    """Markers cannot shift keywords, and renamed sides keep their absolute line indexes."""
    test_web_files_client._run(
        r"""
const patch = 'diff --git a/old.py b/new.py\n--- a/old.py\n+++ b/new.py\n'
  + '@@ -5,2 +7,2 @@\n-def old():\n+class New:\n     pass\n\\ No newline at end of file\n';
const files = parseUnifiedDiff(patch);
const old = {'old.py': [[], [], [], [], [[0, 3, 'keyword']], [[4, 8, 'keyword']]]};
const news = {'new.py': [[], [], [], [], [], [], [[0, 5, 'keyword']], [[4, 8, 'keyword']]]};
const box = MODE === 'inline' ? renderUnified(files, true, old, news)
  : renderSplit(files[0], old, news);
view.appendChild(box);
assert.deepEqual(box.querySelectorAll('.hl-keyword').map(n => n.textContent),
  MODE === 'inline' ? ['def', 'class', 'pass'] : ['def', 'class', 'pass', 'pass']);
assert.equal(box.querySelectorAll('.hunk')[0].querySelectorAll('.hl-keyword').length, 0);
assert.equal(box.querySelectorAll('.meta')[0].querySelectorAll('.hl-keyword').length, 0);
const refs = diffBlameRows(box, MODE, {path: 'new.py', old_path: 'old.py'});
assert.ok(refs.some(r => r.side === 'base' && r.number === 5 && r.path === 'old.py'));
assert.ok(refs.some(r => r.side === 'head' && r.number === 7 && r.path === 'new.py'));
if (MODE === 'inline') {
  const del = box.querySelector('.del');
  assert.equal(del.querySelector('.diff-marker').textContent, '-');
  assert.equal(del.querySelector('.source-code').textContent, '-def old():');
} else {
  assert.equal(box.querySelector('.del').querySelector('.source-code').textContent, 'def old():');
}
assert.equal(renderUnified(files, true).querySelectorAll('.hl-keyword').length, 0);
""".replace("MODE", repr(mode))
    )


def test_diff_spans_decode_git_paths_and_preserve_utf16_and_literal_text() -> None:
    """Quoted UTF-8 paths select the right map, and source markup stays inert text."""
    test_web_files_client._run(r"""
const patch = 'diff --git "a/\\303\\251\\t.py" "b/\\303\\251\\t.py"\n'
  + '--- "a/\\303\\251\\t.py"\n+++ "b/\\303\\251\\t.py"\n'
  + '@@ -1 +1 @@\n-s = "😀"\n+s = "<img>"\n';
const files = parseUnifiedDiff(patch);
const box = renderUnified(files, true,
  {'é\t.py': [[[4, 8, 'string']]]}, {'é\t.py': [[[4, 11, 'string']]]});
assert.deepEqual(box.querySelectorAll('.hl-string').map(n => n.textContent), ['"😀"', '"<img>"']);
assert.equal(box.querySelector('.add').querySelector('.source-code').textContent, '+s = "<img>"');
assert.ok(box.querySelectorAll('.hl-string').every(n => !n.children.length));
""")


@pytest.mark.parametrize("mode", ["inline", "split"])
@pytest.mark.parametrize("panel", ["", "panel"])
def test_git_and_recorded_diff_pass_spans_through_the_shared_viewer(mode: str, panel: str) -> None:
    """Both diff routes retain highlighting and the finding's side and line anchor."""
    test_client_review._run(
        "const mode = "
        + repr(mode)
        + "; const panel = "
        + repr(panel)
        + ";"
        + r"""
remember(DIFF_MODE_KEY, mode);
const host = el('div'); view.appendChild(host);
const e = entry('base', 1, 'old.py');
getJSON = async url => {
  if (url.startsWith('/api/git/findings')) return {findings: [e], read_failures: []};
  assert.ok(url.startsWith(panel ? '/api/reviews/panel/diff?' : '/api/git/diff?'));
  return {base: 'base', head: 'head', path: 'new.py', old_path: 'old.py',
    patch: '--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-def old():\n+class New:\n',
    old_spans: {'old.py': [[[0, 3, 'keyword']]]}, new_spans: {'new.py': [[[0, 5, 'keyword']]]}};
};
await gitDiffViewer(host).showDiff({base: 'base', head: 'head', panel,
  files: [{path: 'new.py', old_path: 'old.py', status: 'R'}]}, 0);
assert.deepEqual(host.querySelectorAll('.hl-keyword').map(n => n.textContent), ['def', 'class']);
const row = host.querySelector('[data-old="1"]');
assert.ok(row.querySelector('.finding'));
assert.equal(row.querySelector('.hl-keyword').textContent, 'def');
"""
    )


def test_latest_commit_fetches_highlighted_payload_once_per_commit() -> None:
    """Live repaints reuse the selected commit's payload and reject superseded responses."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise the run diff")
    web = resources.files("agent6.ui.web")
    run = web.joinpath("client_run.js").read_text()
    start = run.index("  cards.diff.innerHTML = '';")
    block = "function paintDiff(cards, s) {\n" + run[start:]
    script = (
        test_web_files_client._DOM
        + web.joinpath("client_source.js").read_text()
        + block
        + r"""
Object.defineProperty(Element.prototype, 'parentElement', {get() { return this.parent; }});
document.createElement = tag => el(tag);
const Option = function(text, value) {
  const o = el('option', '', text); o.value = value; return o;
};
let detailState = null;
const paintDetails = (_, s) => detailState = s;
const card = el('div'); view.appendChild(card);
const cards = {_base: '/api/session/run', diff: card.appendChild(el('div'))};
const patch = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-pass\n+return\n';
const payload = {patch, cumulative: false,
  old_spans: {'a.py': [[[0, 4, 'keyword']]]}, new_spans: {'a.py': [[[0, 6, 'keyword']]]}};
const state = {mode: 'run', git_control: 'agent6',
  steps: [{sha: 'abcdef1', label: 'one'}], latest_diff: patch};
const calls = [];
getJSON = async url => { calls.push(url); return payload; };
let finished = false;
(async () => {
  paintDiff(cards, state); await tick();
  assert.deepEqual(calls, ['/api/session/run/diff?sha=abcdef1&cumulative=0']);
  assert.equal(cards.diff.querySelectorAll('.hl-keyword').length, 2);
  paintDiff(cards, state); await tick();
  assert.equal(calls.length, 1, 'live repaint fetched the same commit again');
  const waiters = [];
  getJSON = url => new Promise(resolve => waiters.push({url, resolve}));
  paintDiff(cards, {...state, steps: [{sha: 'abcdef2', label: 'two'}]});
  paintDiff(cards, state); await tick();
  waiters[0].resolve({...payload, patch: patch.replace('return', 'stale!')}); await tick();
  assert.ok(!cards.diff.textContent.includes('stale!'));
  assert.ok(cards.diff.textContent.includes('+return'));
  assert.equal(detailState, state);
  finished = true;
})().catch(e => { console.error(e); process.exitCode = 1; });
process.on('beforeExit', () => {
  if (!finished && !process.exitCode) {
    console.error('scenario did not finish'); process.exitCode = 1;
  }
});
"""
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
