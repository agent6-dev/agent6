from __future__ import annotations

import shutil
import subprocess
from importlib import resources

import pytest

from tests.web.test_web_files_client import _DOM


def _run(scenario: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is needed to exercise findings")
    web = resources.files("agent6.ui.web")
    review = web.joinpath("client_review.js")
    client = web.joinpath("client.js").read_text()
    scripts = client[client.index("// --- router") : client.index("// --- hub")]
    scripts += web.joinpath("client_source.js").read_text()
    if review.is_file():
        scripts += review.read_text()
    for name in ("client_git_diff.js", "client_git.js", "client_files.js"):
        scripts += web.joinpath(name).read_text()
    dom = (
        _DOM.replace(
            "const toast = message => { throw new Error(message); };",
            "const toast = message => notices.push(message);",
        ).replace(
            "const recall = (_, fallback) => fallback;",
            "let diffMode = 'inline'; const recall = (key, fallback) => "
            "key === 'a6-diff-mode' ? diffMode : fallback;",
        )
        + r"""
Element.prototype.querySelectorAll = function(s) {
  const match = n => s.startsWith('.') ? s.slice(1).split('.').every(c => n.cls.has(c))
    : s.startsWith('[data-') ? (() => {
        const m = /^\[data-([^=\]]+)(?:="([^"]*)")?\]$/.exec(s);
        return m && n.dataset[m[1]] != null
          && (m[2] === undefined || String(n.dataset[m[1]]) === m[2]);
      })() : n.tag === s;
  return this.children.flatMap(c => [...(match(c) ? [c] : []), ...c.querySelectorAll(s)]);
};
Object.defineProperty(Element.prototype, 'parentElement', {get() { return this.parent; }});
const notLive = s => s.finished;
let confirmations = 0, notices = [], posts = [];
const confirm = () => { confirmations++; return true; };
const crypto = {getRandomValues: array => array.fill(1)};
let postJSON = async (url, body) => {
  posts.push({url, body}); return {message: 'steer submitted'};
};
const paintRun = () => {};
Element.prototype.focus = function() { this.focused = true; };
window.matchMedia = () => ({matches: false});
const entry = (side = 'base', line = 2, path = 'old.txt') => ({
  session_id: 'run', version: 1, matches: true,
  panel: {panel_id: 'panel', target: {base_tree: 'bt', head_tree: 'ht'}},
  finding: {finding_id: 'f1', status: 'open', actions: [], fix_state: 'none'},
  observation: {panel_id: 'panel', finding: {severity: 'block', category: 'correctness',
    title: '<img src=x onerror=alert(1)>', detail: '<script>hostile()</script>',
    location: {path, side, start_line: line, end_line: line}}},
});
const patch = parseUnifiedDiff('diff --git a/old.txt b/new.txt\n'
  + '@@ -1,2 +1,2 @@\n first\n-old\n+new\n');
"""
    )
    result = subprocess.run(
        [
            node,
            "-e",
            dom
            + scripts
            + "\nlet finished = false; (async () => {\n"
            + scenario
            + "\nfinished = true; })().catch(e => { console.error(e); process.exitCode = 1; });"
            + "process.on('beforeExit', () => { if (!finished && !process.exitCode) {"
            + " console.error('the scenario did not finish'); process.exitCode = 1; } });",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_line_origin_badge_uses_the_observation_and_links_immutable_base_evidence() -> None:
    _run("""
const e = entry('head', 8, 'new.txt');
e.observation.line_origin = {kind: 'pre_existing',
  reason: 'The cited lines exist at the base. This does not date the defect.',
  base_location: {path: 'old name.txt', side: 'base', start_line: 2, end_line: 3},
  commits: ['abc1234']};
e.finding.line_origin = {kind: 'changed'};
const card = findingCard(e), badge = card.querySelector('.finding-origin');
assert.ok(badge, 'pre-existing badge missing');
assert.equal(badge.textContent, 'Pre-existing line');
assert.ok(badge.title.includes('exist at the base') && badge.title.includes('does not date'));
assert.ok(badge.href.startsWith('#/files/bt/old%20name.txt?'));
const q = new URLSearchParams(badge.href.split('?')[1]);
assert.equal(q.get('side'), 'base');
assert.equal(q.get('panel'), 'panel');
assert.equal(q.get('line'), '2');
assert.equal(q.get('end_line'), '3');
assert.ok(card.querySelectorAll('a').some(a => a.href === '#/git/commit/abc1234'));
assert.ok(card.textContent.includes('block · open'));
for (const kind of ['changed', 'unknown']) {
  e.observation.line_origin = {kind};
  assert.equal(findingCard(e).querySelector('.finding-origin'), null);
}
""")


@pytest.mark.parametrize("mode", ["inline", "split", "old", "new"])
def test_findings_attach_to_the_correct_side_without_interpreting_html(mode: str) -> None:
    _run(
        """
assert.equal(typeof drawDiffFindings, 'function', 'findings renderer is missing');
const mode = MODE;
const host = el('div'); view.appendChild(host);
host.appendChild(mode === 'inline' ? renderUnified(patch, true) : mode === 'split'
  ? renderSplit(patch[0]) : renderLines(['first', 'second'], []));
const e = entry(mode === 'new' ? 'head' : 'base', 2, mode === 'new' ? 'new.txt' : 'old.txt');
drawDiffFindings(host, [e], {path: 'new.txt', old_path: 'old.txt'}, mode);
const card = host.querySelector('.finding');
assert.ok(card, 'the finding disappeared');
assert.ok(card.textContent.includes('<script>hostile()</script>'));
assert.equal(card.querySelectorAll('script').length, 0);
const row = mode === 'split' ? card.parentElement.parentElement : card.parentElement;
assert.equal(Number(row.dataset[mode === 'old' || mode === 'new' ? 'line' : 'old']), 2);
if (mode === 'split') assert.equal(card.parentElement, row.children[0], 'wrong side');
""".replace("MODE", repr(mode))
    )


def test_actions_keep_versions_confirmation_reasons_and_history() -> None:
    _run("""
assert.equal(typeof paintFindings, 'function', 'Findings card is missing');
const e = entry();
e.finding.observations = [e.observation];
e.finding.actions = [{type: 'loop.review.finding.resolved', reason: '<b>operator</b>',
  resolution: {disposition: 'fixed'}}];
const state = {finished: true, review: {panels: [e.panel], findings: [e.finding],
  version: 1, open_counts: {block: 1}, merge_guard: {open_block_ids: ['f1']}}};
const cards = {_id: 'run', _base: '/api/session/run', findings: el('div')};
view.appendChild(cards.findings);
paintFindings(cards, state);
assert.ok(cards.findings.textContent.includes('<b>operator</b>'));
assert.equal(cards.findings.querySelectorAll('b').length, 0);
assert.ok(reviewMergeReason(state.review).includes('f1'));
const button = name => cards.findings.querySelectorAll('button').find(b => b.textContent === name);
getJSON = async () => state;
await button('Resume to fix').onclick();
assert.equal(confirmations, 1);
assert.equal(posts[0].url, '/api/session/run/findings/f1/fix');
assert.equal(posts[0].body.expected_version, 1);
assert.equal(posts[0].body.resume, true);
button('Resolve').onclick();
const form = cards.findings.querySelector('form');
const reason = form.querySelector('textarea'); reason.value = '<b>intentional</b>';
form.querySelector('select').value = 'intentional';
await form.onsubmit({preventDefault() {}});
assert.equal(posts[1].url, '/api/session/run/findings/f1/resolve');
assert.equal(posts[1].body.resolution.reason, '<b>intentional</b>');
assert.equal(reason.maxLength, 2000);
paintFindings(cards, state);
assert.equal(cards.findings.querySelector('form'), form, 'SSE wiped an in-progress form');
e.finding.status = 'resolved'; state.review.version = 2;
cards._finding_status = 'all'; paintFindings(cards, state);
button('Reopen').onclick();
const reopen = cards.findings.querySelector('form');
reopen.querySelector('textarea').value = 'Still reproducible';
await reopen.onsubmit({preventDefault() {}});
assert.equal(posts[2].url, '/api/session/run/findings/f1/reopen');
assert.equal(posts[2].body.reason, 'Still reproducible');
assert.equal(posts[2].body.expected_version, 2);
""")


@pytest.mark.parametrize("mode", ["inline", "split", "old", "new", "narrow", "truncated"])
def test_review_deep_link_reload_opens_diff_then_exact_file(mode: str) -> None:
    _run(
        "const mode = "
        + repr(mode)
        + ";"
        + """
diffMode = ['narrow', 'truncated'].includes(mode) ? 'split' : mode;
window.matchMedia = () => ({matches: mode === 'narrow'});
const e = entry(mode === 'new' ? 'head' : 'base', 2, mode === 'new' ? 'new.txt' : 'old.txt');
e.finding.observations = [e.observation];
const reviewed = {session_id: 'run', panel: e.panel,
  review: {panels: [e.panel], findings: [e.finding], version: 1},
  files: [{path: 'new.txt', old_path: 'old.txt', status: 'R', additions: 1, deletions: 1}]};
const requests = [];
getJSON = async url => {
  requests.push(url);
  if (url === '/api/git/refs') return {refs: [], head: {sha: 'hc'}};
  if (url === '/api/git/worktree') return {base: 'bc', files: []};
  if (url.startsWith('/api/git/graph')) return {commits: [], tips: [], has_more: false};
  if (url === '/api/reviews/panel') return reviewed;
  if (url.startsWith('/api/reviews/panel/diff')) return {
    base: 'bt', head: 'ht', truncated: mode === 'truncated', max_bytes: 1024,
    patch: mode === 'truncated' ? '@@ -1,2 +1,2 @@\\n first\\n'
      : '@@ -1,2 +1,2 @@\\n first\\n-old\\n+new\\n'};
  if (url.startsWith('/api/git/findings')) return {findings: [e], read_failures: []};
  if (url.startsWith('/api/reviews/panel/file')) return {
    rev: url.includes('side=head') ? 'ht' : 'bt', text: 'first\\nreviewed content\\n'};
  throw new Error('unexpected request ' + url);
};
location.hash = reviewLink(e);
await route();
assert.ok(view.querySelector('.finding'), view.textContent);
assert.equal(activeTab, 'git');
assert.ok(location.hash.includes('panel=panel') && location.hash.includes('path='));
if (mode === 'narrow') assert.ok(view.textContent.includes('showing inline'));
if (mode === 'truncated') assert.ok(view.textContent.includes('outside the displayed lines'));
await route();
assert.ok(view.querySelector('.finding').scrolled, 'reloaded link did not focus its finding');
location.hash = reviewFileLink(e);
await route();
assert.equal(activeTab, 'files');
assert.ok(view.querySelector('.files-text').textContent.includes('reviewed content'));
await route();
assert.ok(view.querySelector('.files-text').textContent.includes(
  'Reviewed tree: ' + (mode === 'new' ? 'ht' : 'bt')));
assert.ok(requests.some(url => url.includes('/file?path=' + e.observation.finding.location.path)
  && url.includes('side=' + e.observation.finding.location.side)));
"""
    )


def test_unpaired_deletion_stays_on_the_old_side() -> None:
    _run("""
assert.equal(typeof drawDiffFindings, 'function', 'findings renderer is missing');
const host = el('div'); view.appendChild(host);
const deleted = parseUnifiedDiff('@@ -2 +1,0 @@\\n-gone\\n');
host.appendChild(renderSplit(deleted[0]));
drawDiffFindings(host, [entry()], {path: 'old.txt'}, 'split');
const card = host.querySelector('.finding');
assert.equal(card.parentElement, host.querySelector('[data-old="2"]').children[0]);
assert.equal(host.querySelector('[data-new="2"]'), null);
""")


def test_file_level_missing_line_and_wrong_comparison_keep_reviewed_links() -> None:
    _run("""
assert.equal(typeof drawDiffFindings, 'function', 'findings renderer is missing');
const host = el('div'); view.appendChild(host); host.appendChild(renderUnified(patch, true));
const wrong = entry(); wrong.matches = false;
const file = entry('head', null, 'new.txt'); file.finding.finding_id = 'file';
const cut = entry('head', 200, 'new.txt'); cut.finding.finding_id = 'cut';
drawDiffFindings(host, [wrong, file, cut], {path: 'new.txt', old_path: 'old.txt'}, 'inline');
assert.equal(host.querySelectorAll('.finding').length, 3);
for (const card of host.querySelectorAll('.finding')) assert.equal(card.parentElement, host);
assert.ok(host.textContent.includes('different comparison'));
assert.ok(host.textContent.includes('outside the displayed lines'));
const links = host.querySelectorAll('a');
assert.ok(links.some(a => a.href.includes('panel=panel') && a.href.includes('side=base')));
assert.ok(links.some(a => a.href.startsWith('#/files/') && a.href.includes('side=head')));
""")
