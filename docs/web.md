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

Buttons do what the matching CLI command does, and prompts are answered inline on every page that shows a run.
A question the model queued, or an ended run's, keeps its box: the answer reaches the run at its next turn, or at its resume.

- **Sessions page**: every session (mode, status, last activity, cost), a fan-out's lanes folded under its row, in [review queue](usage.md#the-review-queue) order under group headings with each row's reason; **Activity** orders by last activity instead, a choice the browser remembers
    - the composer starts a run, plan or ask under a chosen preset and model (the model box shows what the config resolves; a pick overrides it)
- **Machines page**: instances, `machine create` drafts, and buttons that run an authored machine file; the composer creates a new one
- **Session view** (live over SSE): the conversation, the same folded transcript the CLI and TUI render, with the turn in flight streaming underneath; a detail toggle cycles collapsed, expanded, hidden
    - the composer steers a live run or resumes an ended one under the preset and model picked above it, and takes the [steer directives](usage.md#follow-and-steer-a-run); Ctrl+Enter sends a live steer as `/now`, Ctrl-R searches past messages
    - a click on a line number in the diff card quotes that line into the composer, with its path, side and commit; write your words beneath the quote and send it
- **Machine view**: the state overview, the path taken and the current agent state's conversation; the entry submits as **Steer** (into the current state) or **Message** (a `poke` a waiting machine's next tool reads); **Stop** parks the instance at its next transition and `machine run` resumes it
- **Config page**: every setting with value and source, one collapsible section per table in the order every surface shares (the top-level `preset`, `[providers]`, `[models]`, the rest by name, `[agent6]` last), filterable; click a row to set it, with the choices an enum or a provider's model listing offers, or to add or remove an entry of its table (a provider, an MCP server, a skill, a reviewer seat) from the same popup; a preset picker shows and edits one preset ([Presets](config.md#presets)); secrets never shown

- **Git page**: the repository's commit graph, a commit's files and diffs, and a compare of any two revisions; it reads the repository `agent6 web` was started in and writes nothing
    - **blame** uses each line's own side: removed lines use the base, added and inline context lines use the head, split cells use their column, and old/new views use their selected side
    - recorded review comparisons preserve their tree identities and Files links; full context and commit blame are unavailable there
    - a click on a line number in a diff quotes that line, with its path, side and revision, for the new-work composer on the Sessions page; several lines queue up before you write and start the run
- **Files page**: the files at one revision; a symlink shows its target; it reads the repository `agent6 web` was started in and writes nothing
    - a markdown file opens rendered; **source** shows the text; **show ignored** lists the files every exclude rule hides, badged `I`

Start a machine on the Machines page and watch the current state stream, answering its approvals and questions in place:

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-machine.webm" type="video/webm">
</video>

## Receipts

The Findings card's **Receipt** button opens Testing, Review, Risk and How to revert from the [recorded session evidence](usage.md#session-receipts).
**Ask a run to revert** fills the Sessions page's new-work composer with a proposed task; it does not submit the task or run a git command.
`GET /api/session/<id>/receipt` returns the same derived receipt as JSON.

## Links

- `#/git/commit/<sha>` opens a commit and scrolls to its graph row when loaded; `?path=<historical path>&line=<original line>` opens and scrolls to that file's diff
- `#/git/compare/<base>..<head>` opens a comparison
- `#/git/worktree` opens the working tree's uncommitted changes
- `#/files/<rev>/<path>` opens a file, with `WORKTREE` for the working tree
- revisions and path components are URL-encoded; selecting a commit, a comparison, the working tree or a file updates the hash for copying, Back restores the selection, and a bare `#/git` shows the graph alone

## Highlighting

Source text on the Git and Files pages is highlighted with the tree-sitter language pack agent6 ships (372 languages).
A language's grammar downloads into `~/.cache/tree-sitter-language-pack/` the first time a file in it is shown, as the outline tools do; offline, a file in a language not yet cached shows as plain text.
A plain text file, a file over 512 KiB, or one whose parse runs past a second shows uncoloured.

## Notifications and installing (PWA)

The page installs as an app (phone home-screen icon or desktop window).
**🔔 Notifications** on a machine view grants permission; `machine.notify` and a machine's end then pop OS notifications on desktop.
For a phone not open on the page, point [`[machine.notify].on_event`](config.md#machinenotify-optional) at a push service.

## The HTTP API

The page reads the same wire form as `agent6 attach --json`.
[Formats](formats.md#web-api) owns the request and response field lists.

<details markdown="1">
<summary>HTTP endpoints and finding actions</summary>

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
- `GET /metrics` answers Prometheus text when `web.metrics_route = "on"` and `telemetry.metrics = "prometheus"`, with the web secret as the bearer, not the sign-in token ([telemetry export](cli.md#telemetry-export))
- reads: `/api/meta`, `/api/hub`, `/api/session/<id>/{receipt,findings,resume_defaults}`, `/api/reviews/<panel_id>` with `/diff` and `/file`, `/api/routes?mode=&preset=`, `/api/config` with `/suggest/<key>?provider=<name>` (an optional provider filter), `/entry_tables` and `/reviewer_choices`, `/api/session/<id>` (what `attach --json` prints; `?commit=<sha>` folds up to that commit) with `/conversation`, `/restate`, `/diff` (`?sha=&cumulative=1`) and `/events`, `/api/machine/<name>` with `/reasoning`, `/conversation` and `/events`, `/api/draft/<name>` with `/conversation`, `/diff` and `/events`
- git reads, all of the repository the server runs in: `/api/git/refs`, `/api/git/graph` (`?refs=<full ref name or resolved sha>` repeated, `&limit=&skip=`), `/api/git/commit/<rev>`, `/api/git/compare?base=&head=`, `/api/git/diff?base=&head=&path=&old_path=`, `/api/git/search?q=`, `/api/git/worktree`
    - graph pages return `tips`, the resolved shas walked; pass these as `refs` with the next `skip`, while **↻** starts from live refs
    - a revision is a branch, tag, full ref name, sha prefix or expression git resolves to a commit; the empty tree sha is accepted as a `base`
    - `WORKTREE` names the working tree as the `head` of a compare or diff (an untracked file diffs against nothing); as a `base` or a commit it answers `422`
    - `/api/git/diff` accepts `full_context=1` for all unchanged lines, independently of `whole=1`, which lifts the patch byte cap
    - [Diff payloads](formats.md#apigitdiff-payload) include checked content identities for blame.
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

`POST /api/session/<id>/findings/<finding-id>/{fix,resolve,reopen}` calls the same application path as the [CLI finding actions](usage.md#session-findings).
The [finding action format](formats.md#findingaction) defines concurrency checks and idempotency.

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

Compare links retain `panel`, `finding`, `path` and `side` query parameters across reloads.

Finding routes include `line_origin` on each observation and the latest evidence on each finding.

</details>

## Code-review rule sources

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
