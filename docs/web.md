# Web UI

`agent6 web` serves a browser front-end for driving agent6 from a desktop or a phone: start and watch runs, steer them, answer their prompts, run state machines, and browse the repository's commits and files.

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-desktop.webm" type="video/webm">
</video>

The same UI on a phone (single column, bottom nav):

<video controls muted loop playsinline preload="metadata" class="no-lightbox"
       style="max-width: 390px">
  <source src="/screenshots/out/web-phone.webm" type="video/webm">
</video>

## Start the server

```bash
agent6 web              # serve the hub on http://127.0.0.1:7658
agent6 web <session-id> # open a session on load
agent6 web <machine>    # open a machine instance on load
agent6 web <draft>      # open a `machine create` draft on load (read-only)
```

`--host` / `--port` override the [`[web]`](config.md#web) config for one invocation.
Stop it with Ctrl-C.

## Sign in

The server requires a secret and prints it when it starts:

```text
agent6 web: serving on http://127.0.0.1:7658/#secret=amber-acorn-agile-alder-alert-arrow  (Ctrl-C to stop)
agent6 web: secret: amber-acorn-agile-alder-alert-arrow (type it once per browser; --secret or AGENT6_WEB_SECRET sets your own)
```

- Open the printed URL: the page signs in with the secret in the address, then removes it from the address bar.
- On another device, open the server's address and type the six words into the sign-in form.
  Capitals, spaces and hyphens between the words do not matter, so `Amber acorn agile alder alert arrow` from a phone keyboard signs in.
- With browser storage available, a browser signs in once and stays signed in.
  It keeps a keyed digest of the secret, never the secret, in storage scoped to this server's address and port.
  Without browser storage, sign in again after reloading.
- Each launch mints a new secret, and every browser signs in again.

Set your own secret, at least 12 characters after lowercasing and joining words with single hyphens, to keep browsers signed in across restarts:

```bash
export AGENT6_WEB_SECRET='a phrase of your own'   # read when the server starts
agent6 web
agent6 web --secret 'a phrase of your own'        # the flag wins over the variable
```

- With your own secret the printed URL carries no secret, and the terminal names where it was set, never its value.
- A `--secret` value shows in the process list (`ps`); `AGENT6_WEB_SECRET` keeps it out.
- Anyone who can read the terminal, or your environment, has the secret.

## Pages

Every page with a composer docks it at the bottom, like a chat: Enter sends, Shift+Enter inserts a newline.
Buttons do what the matching CLI command does, and prompts are answered inline on every page that shows a run.
A question the model queued, or an ended run's, keeps its box: the answer reaches the run at its next turn, or at its resume.

- **Sessions page**: every session (mode, status, last activity, cost), a fan-out's lanes folded under its row, in [review queue](usage.md#the-review-queue) order under group headings with each row's reason; **Activity** orders by last activity instead, a choice the browser remembers
    - the composer starts a run, plan or ask under a chosen preset and model (the model box shows what the config resolves; a pick overrides it)
    - `more…` holds **Prune merged runs** (`sessions prune`, or with `--delete-squashed` after a confirm) and **Clear saved asks**
- **Machines page**: instances, `machine create` drafts, and buttons that run an authored machine file; the composer creates a new one
- **Session view** (live over SSE): the conversation, the same folded transcript the CLI and TUI render, with the turn in flight streaming underneath; a detail toggle cycles collapsed, expanded, hidden
    - each task and steer expands to its full recorded text; every tool item expands to all arguments without a length cap
    - each verification attempt has one marker: `verify: waiting for approval`, then `verify: running <command>`, then its result
    - a resizable drawer holds the run's context: overview, plan.md, run review, findings, task graph, budget, tool calls, background shells, latest commit diff, event log
    - the Latest commit widget selects any per-step commit (cumulative toggle) and the budget and task graph follow it; a run under `git.control = "model"` has no chain and says so
    - the composer steers a live run or resumes an ended one under the preset and model picked above it, and takes the [steer directives](usage.md#follow-and-steer-a-run); Ctrl+Enter sends a live steer as `/now`, Ctrl-R searches past messages
    - buttons: stop now or after the step, compact, fork (a new run at the latest checkpoint, started from its composer), merge, delete history, run a finished plan (`run --from`), and review (`sessions review`, minutes; refused while the run is live)
    - "Allow all" on a prompt appears only where it would grant more than the one call
- **Machine view**: the state overview, the path taken and the current agent state's conversation; the entry submits as **Steer** (into the current state) or **Message** (a `poke` a waiting machine's next tool reads); **Stop** parks the instance at its next transition and `machine run` resumes it
- **Config page**: every setting with value and source, filterable; click a row to set it, with the choices an enum or a provider's model listing offers; secrets never shown
    - **Modified only** combines with the text filter to show settings a config layer set, including values equal to their defaults.
    - The **Model** editor pairs provider and model for each role, accepts model ids outside the live suggestions, and saves both without replacing the role's other settings.
    - **Add provider…** adds or updates a `[providers.<name>]` entry, with format and URL prefilled for known provider names.
    - **Add MCP server…** adds or updates a `[mcp.servers.<name>]` entry.
      Enter either a URL or a command line; shell quoting splits the command into argv, without running a shell.
    - **Add skill…** sets `skills.state.<name>` to `enabled`, `disabled` or `always`; it does not install the skill.
    - Entry forms save to global config by default and offer repo config as a target.
      A blank field keeps an existing value or uses the schema default.
    - **Remove provider**, **Remove MCP server** or **Remove skill** in an entry row's editor confirms its name and config layer, then removes the whole entry, including its subtables.
      Removal targets the highest layer holding the entry and accepts only global or repo config.
      Values from lower layers may become visible.
    - Skill states have individual rows, so Remove drops only the selected skill's state.
      **Unset** still drops only the selected leaf.
    - **Add reviewer…** appends to the ordered `review.reviewers` list through a form for `focus`, `provider`, `model`, `effort` and `temperature`.
      Focus is required; set provider and model together or leave both blank.
    - The `review.reviewers` edit overlay lists reviewers by position and focus, with a **Remove** button for each.
      Removal confirms the position and saves the remaining roster to its source layer (global or repo).
      Removing the last reviewer writes `[]`, so a lower layer's roster stays hidden.

- **Git page**: the repository's commit graph, a commit's files and diffs, and a compare of any two revisions; it reads the repository `agent6 web` was started in and writes nothing
    - the graph walks every branch, remote-tracking branch and tag plus HEAD, newest first in topological order, 200 commits a page with **Load more** for the next page from the same resolved tips, even if refs move; lanes are drawn per commit and merges join their parents
    - **Refs ▾** filters the walk: tick a whole group (branches, remotes, tags, agent6 run refs) or single refs; the `refs/agent6/*` chain refs are off until ticked
    - a **Working tree · N uncommitted files** row tops the graph while the tree is dirty: tracked changes against HEAD, staged or not, plus untracked files `.gitignore` does not exclude; click it for the list, each file opening its diff like a commit's
    - the page reads the repository once per visit; **↻** re-reads the refs, the graph and the working-tree row, and nothing polls
    - click a commit for its message, author, committer, parents and the files it changed from its first parent (a root commit shows its files against the empty tree), each file with its status (A, M, D, R `old → new`) and line counts, or `binary`
    - **list | tree** above every file list: the tree groups the files under their folders, a folder row showing how many files it holds with their summed counts and folding them on a click; the choice is kept per browser
    - click a file and its unified diff fills the page: **‹ files** returns to the list, **‹** and **›** step through the files; a patch over 256 KiB is cut at that size and offers the whole patch on a click, a binary file says so
    - the diff header offers four modes, the choice kept per browser: **inline** (the unified patch), **split** (old and new side by side, a deletion beside the addition that replaces it, context on both sides; at phone width it shows inline and says so), **old** and **new** (the file at the base or the head, highlighted, with its deleted or added lines marked); every line carries its line numbers
    - **Open in Files** opens the head revision and path; a deletion disables it with a reason and offers **Open base file in Files** instead
    - **Show full context** in inline and split modes includes unchanged lines; **Show hunks only** restores the compact patch, and a change without text hunks shows its file with a note
    - **blame** uses each line's own side: removed lines use the base, added and inline context lines use the head, split cells use their column, and old/new views use their selected side
    - diff blame checks both content digests and says why it is unavailable when content changed or a side has no digest
    - recorded review comparisons preserve their tree identities and Files links; full context and commit blame are unavailable there
    - an image (`png`, `jpg`, `jpeg`, `gif`, `webp`, `svg`, `bmp`, `ico`, `avif`) shows **before** and **after** side by side above git's binary line, each with its size and dimensions; at phone width the two stack
    - **Compare** takes two revisions, base and head, each from a picker that searches branch and tag names, commit shas and commit messages as you type (Enter keeps free text as the revision for the server to verify); the head picker offers **working tree** first; **⇄** swaps them
    - the compare lists the files that differ between the two, as `git diff base head` does, so a commit against its parent reads like that commit's own diff; a commit's **Set as base** and **Set as head** put it in the pickers
    - limits: 500 commits a page at most, 2000 files a list, 256 KiB a patch unless opened whole, 20 MiB an image, 30 refs and 30 commits a search; each list says when it was cut
- **Files page**: the files at one revision; it reads the repository `agent6 web` was started in and writes nothing
    - on desktop, drag the divider or use its arrow keys to resize the tree, and use **Hide tree** / **Show tree** to collapse or reopen it; the browser remembers both choices
    - the revision picker at the top is the compare pickers' search, with **working tree** as its first choice; HEAD to start (the working tree in a repository with no commits yet)
    - the files as a folder tree with a filter box: a folder row shows how many files it holds and their size and folds them on a click; a tree over 200 files starts folded; at a commit the committed files, in the working tree the tracked files plus untracked ones `.gitignore` does not exclude
    - working-tree badges use `H` for tracked, `A` for untracked and `M` for tracked files with uncommitted changes; a folder shows `M` if any child is modified, else `A` if any is untracked, else `H`, with the counts in its tooltip
    - commit revisions have no state badges
    - click a file: text with line numbers and highlighting (cut at 1 MiB, the whole file on a click), an image as itself, a binary file as its size; a symlink or submodule entry says so
    - **blame** beside a text file's title adds a faded right gutter shared with diffs, with one commit, author and UTC date per consecutive group and a bar on its other lines
    - hovering or focusing a group darkens it and opens a card with the full sha, author, date, commit subject and historical path when different
    - a group's sha link opens that historical file's diff in its commit; the card's subject opens the commit's file list
    - working-tree blame includes staged and unstaged content; lines without a commit read **not committed**
    - a file that changed since it was opened says so instead of attaching blame to other bytes
    - limits: 20000 files a tree, 1 MiB of text and matching blame lines unless opened whole, 20 MiB a file; the tree says when it was cut

Start a machine on the Machines page and watch the current state stream, answering its approvals and questions in place:

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-machine.webm" type="video/webm">
</video>

## Receipts

The Findings card's **Receipt** button opens Testing, Review, Risk and How to revert from the [recorded session evidence](usage.md#session-receipts).
Expand stdout or stderr to read a verification attempt's retained output.
**Ask a run to revert** fills the Sessions page's new-work composer with a proposed task; it does not submit the task or run a git command.
`GET /api/session/<id>/receipt` returns the same derived receipt as JSON.

## Links

- `#/git/commit/<sha>` opens a commit and scrolls to its graph row when loaded; `?path=<historical path>&line=<original line>` opens and scrolls to that file's diff
- `#/git/compare/<base>..<head>` opens a comparison
- `#/git/worktree` opens the working tree's uncommitted changes
- `#/files/<rev>/<path>` opens a file, with `WORKTREE` for the working tree
- revisions and path components are URL-encoded; selecting a commit, a comparison, the working tree or a file updates the hash for copying, Back restores the selection, and a bare `#/git` shows the graph alone
- an unknown revision or path shows an error on the page

## Highlighting

Source text on the Git and Files pages is highlighted with the tree-sitter language pack agent6 ships (372 languages).
A language's grammar downloads into `~/.cache/tree-sitter-language-pack/` the first time a file in it is shown, as the outline tools do; offline, a file in a language not yet cached shows as plain text.
A plain text file, a file over 512 KiB, or one whose parse runs past a second shows uncoloured.

## Layout

- desktop: a nav rail, the run view a fixed pane with the drawer and conversation scrolling inside it
- phone: a top bar (theme toggle), bottom tab nav, the composer docked above it, and the run view one widget at a time, switched from the top-bar menu

## Notifications and installing (PWA)

The page installs as an app (phone home-screen icon or desktop window).
**🔔 Notifications** on a machine view grants permission; `machine.notify` and a machine's end then pop OS notifications on desktop.
For a phone not open on the page, point [`[machine.notify].on_event`](config.md#machinenotify-optional) at a push service.

## The HTTP API

The page reads the same wire form as `agent6 attach --json`.
[Formats](formats.md#web-api) owns the request and response field lists.
Sign in once for a token, then send it as a bearer header with every call:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"secret": "amber-acorn-agile-alder-alert-arrow"}' localhost:7658/api/login
# {"ok": true, "token": "<64 hex characters>"}
A='Authorization: Bearer <token>'
curl -s -H "$A" localhost:7658/api/hub                       # hub state, sessions in review-queue order
curl -s -H "$A" "localhost:7658/api/hub?sort=activity"       # sessions by last activity
curl -s -H "$A" localhost:7658/api/session/<id>              # a run's state, as JSON
curl -s -H "$A" localhost:7658/api/session/<id>/conversation # the folded conversation
curl -s -H "$A" localhost:7658/api/machine/<name>            # a machine's state, as JSON
curl -s -H "$A" localhost:7658/api/config                    # effective config
curl -sN -H "$A" localhost:7658/api/session/<id>/events      # SSE: a snapshot per change
```

- sign-in: `POST` `/api/login` with `{"secret": "<secret>"}` answers the token; a wrong secret answers `403`
- every other `/api/...` route answers `401` with `{"error": ..., "login": true}` without the token
- `GET /metrics` answers Prometheus text when `web.metrics_route = "on"` and `telemetry.metrics = "prometheus"`, with the web secret as the bearer, not the sign-in token ([telemetry export](usage.md#telemetry-export))
- reads: `/api/meta`, `/api/hub`, `/api/session/<id>/{receipt,findings,resume_defaults}`, `/api/reviews/<panel_id>` with `/diff` and `/file`, `/api/routes?mode=&preset=`, `/api/config` with `/suggest/<key>?provider=<name>` (the model editor's configured provider), `/entry_tables` and `/reviewer_choices`, `/api/session/<id>` (what `attach --json` prints; `?step=<sha>` folds up to that commit) with `/conversation`, `/restate`, `/diff` (`?sha=&cumulative=1`) and `/events`, `/api/machine/<name>` with `/reasoning`, `/conversation` and `/events`, `/api/draft/<name>` with `/conversation`, `/diff` and `/events`
- git reads, all of the repository the server runs in: `/api/git/refs`, `/api/git/graph` (`?refs=<full ref name or resolved sha>` repeated, `&limit=&skip=`), `/api/git/commit/<rev>`, `/api/git/compare?base=&head=`, `/api/git/diff?base=&head=&path=&old_path=`, `/api/git/search?q=`, `/api/git/worktree`
    - graph pages return `tips`, the resolved shas walked; pass these as `refs` with the next `skip`, while **↻** starts from live refs
    - a revision is a branch, tag, full ref name, sha prefix or expression git resolves to a commit; the empty tree sha is accepted as a `base`
    - `WORKTREE` names the working tree as the `head` of a compare or diff (an untracked file diffs against nothing); as a `base` or a commit it answers `422`
    - `/api/git/diff` accepts `full_context=1` for all unchanged lines, independently of `whole=1`, which lifts the patch byte cap
    - [Diff payloads](formats.md#uiwebgit_modeldiff_payload) include checked content identities for blame.
    - regular worktree patches and their digests use the same checked bytes, copied privately before Git compares them; diff blame requests `whole=1` to cover hunks beyond the default text range
    - `/api/git/worktree` lists uncommitted files against HEAD: tracked changes first, then untracked files, with `.gitignore` honoured
    - `/api/git/tree?rev=` lists files at a revision or in the working tree, capped at 20000 files; the working tree includes the index and untracked files `.gitignore` does not exclude, omitting tracked files gone from disk; only mode `160000` entries are submodules, including unpopulated gitlinks
    - `/api/git/text?rev=&path=` returns highlighted file text, cut at 1 MiB on a line boundary unless `whole=1`; binary content has no text, and a file over 20 MiB answers `413`
    - `/api/git/blob?rev=&path=` serves a file's bytes at a revision, or from disk for `WORKTREE` (a regular file git lists, never through a symlink); an image is typed by extension, anything else is `application/octet-stream` with `Content-Disposition: attachment`; every blob response carries `Content-Security-Policy: sandbox; default-src 'none'; frame-ancestors 'none'` and `X-Content-Type-Options: nosniff`, so a repository file opened directly runs under no origin; a file over 20 MiB answers `413`
    - `/api/git/blame?rev=&path=` returns per-line attribution; its caps and refusals match `/api/git/text`, including `whole=1`
    - blame preserves historical paths and line numbers, including before a rename
    - blame dates are UTC ISO 8601; working-tree blame uses checked file bytes against HEAD, with an empty `sha` and `date` and author `not committed` for uncommitted lines
    - blob, text and blame share a working-tree reader that refuses ancestor and leaf symlinks, paths resolving outside the repository, and non-regular files
    - a revision or path starting with `-`, an unknown revision, a graph ref neither listed in `/api/git/refs` nor a verified full sha, or a path in neither tree answers `422`
- writes, small JSON `POST`s: `/api/new`, `/api/session/<id>/{steer,approve,answer,merge,undo,fork,resume,run_plan,review,stop,compact,rm}`, `/api/session/<id>/findings/<fid>/{fix,resolve,reopen}`, `/api/machine/<name>/{poke,stop,steer,approve,answer}`, `/api/sessions/{prune,rm_asks}`, `/api/config`, `/api/config/entry`, `/api/config/entry/remove`, `/api/config/reviewer`, `/api/config/reviewer/remove`, `/api/machine/{create,run}`
- session rows, session snapshots and machine snapshots list their fields, the operator's words and the spend among them, on [Formats](formats.md#cli-json)
- roster writes use the [typed request bodies](formats.md#reviewerbody) for additions and [removals](formats.md#removereviewerbody)
- writes use the shared config writer or the same spawn and answer-file contracts as the CLI; machine names and answer ids validate to one path component
- the page and its PWA assets, served without the token (they hold no state): `/`, `/manifest.webmanifest`, `/sw.js`, `/icon.svg`, `/favicon.svg`

Session snapshots and streams use the hub's status label, including the mode and the `unmerged` mark on ended runs.
Idle session streams refresh disk-backed state (shell results, the merge mark) on each heartbeat and send a frame when it changed.
Heartbeats run every 15 seconds, including after the run ends.

A run or draft `/diff` without `sha` returns its latest journaled commit, or 422 with `no commit yet`.
With `cumulative=1`, the patch covers the chain through that commit.

## Finding actions API

`GET /api/session/<id>/findings` returns the shared review evidence and its `version`, with open finding rows.
`?status=all` includes resolved rows.

The run's **Findings** card shows open severity counts, status and severity filters, and resolution history.
Fix submits stored evidence through the steer bridge; **Resume to fix** asks for confirmation before resuming a stopped session.
Resolve and Reopen require a reason.
The card shows the panel's worker-finish decision separately from the open-block merge rule.
Merge is disabled for open block findings or unreadable evidence, and the server enforces the same rule on direct requests.

`POST /api/session/<id>/findings/<finding-id>/{fix,resolve,reopen}` calls the same application path as the [CLI finding actions](usage.md#session-findings).
The [finding action format](formats.md#findingaction) defines concurrency checks and idempotency.

- Fix can resume a stopped session only after the operator confirms **Resume to fix**.
- Resolve records the operator’s conclusion and reason.
- Reopen records why the finding remains reproducible.
- A completed action returns the [action result](formats.md#actionresult).
  A queued live action returns 202; repeat its request ID to read the result without resending it.
- Unknown records return 404, stale versions and conflicting actions return 409, and invalid action fields return 422.

`GET /api/git/findings?base=<revision>&head=<revision>` returns recorded findings and whether both displayed trees match each observation.
Working-tree comparisons match no observation.

`GET /api/reviews/<panel_id>` returns the recorded target and changed files.
Its `/diff?path=<path>` and `/file?side=<base-or-head>&path=<path>` routes read those trees, not the checkout.
The diff route accepts either path of a rename.
Both content routes mark truncation and accept `whole=1` with the Git page's size limits.
An unknown panel or file returns 404; invalid paths and sides return 422; unreadable panel evidence returns 409.

The Git page places findings on their recorded base or head line in inline, split, old and new modes.
A rename's base finding uses its old path.
File-level findings appear beside the patch; locations outside a truncated patch link to the full reviewed file.
A different comparison shows a link to the reviewed comparison instead of an inline anchor.
Compare links retain `panel`, `finding`, `path` and `side` query parameters across reloads.
**Reviewed file** opens the Files page at the recorded tree, including deleted files on the base side.

The **Pre-existing line** badge opens the mapped lines in the recorded base file.
The adjacent **Blame** links open their attribution commits.
The badge's tooltip describes line evidence, not the age of the defect; the [severity and merge rules](usage.md#session-findings) stay the same.
Finding routes include `line_origin` on each observation and the latest evidence on each finding.
Finding text and operator reasons render as text, never HTML.

## Code-review rule sources

The Findings card and reviewed comparison show rule completeness, source paths, blob IDs and supplied byte counts.
A source link opens the recorded base-tree file, not the current checkout.
See [Code-review rules](usage.md#code-review-rules) for discovery and limits.

## Remote access (Tailscale)

The server binds `127.0.0.1` by default and speaks plain HTTP.
For remote access, put [Tailscale](https://tailscale.com) in front of the loopback bind:

```bash
agent6 web                # keep it on 127.0.0.1:7658
tailscale serve --bg 7658 # HTTPS + WireGuard, reachable on your tailnet
```

- the tailnet decides who reaches the server and `tailscale serve` terminates HTTPS; each browser still [signs in](#sign-in) with the secret
- a non-loopback bind sends the secret and the token over plain HTTP, where anyone on the network path can read them and then drive the agent, so it refuses without an opt-in: [`[web].allow_non_loopback = true`](config.md#web), or `--allow-non-loopback` for a host given with `--host`
