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
A session started from the page keeps running after the server stops.

## Browsers

The page targets current Chrome and Firefox and uses only features both ship.
Safari and iPhone browsers are untested.

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

Above 780px wide a left rail lists the pages: by name from 1024px wide, as icons below that.
The page follows the browser's font size and zoom: text, controls, spacing and these widths (given at the default 16px) scale together.
The rail's last button folds it to its icons, then hides it; the menu button at the top bar's left brings it back, and the browser remembers the choice.

Buttons do what the matching CLI command does, and prompts are answered inline on every page that shows a run.
A run's approvals and questions wait after its conversation's last line, over the composer, and scroll with the conversation.
On a phone, a run's or a machine's page opens at the conversation's end while a prompt waits there.
While you stay at the page's end, it scrolls on as new lines and prompts land.
A run's page on a phone shows one card at a time, picked from the menu at the top bar's right end.
The Conversation card opens at its end, every other card at its top.
While a prompt waits and another card shows, a dot marks the menu button and its Conversation entry.
In a composer, Enter sends and Shift+Enter types a new line; on a touch screen, Enter types the new line, and a button or Ctrl+Enter sends.
In the Sessions or run composer, type `/` for commands and use the arrow keys to scroll through the choices.
Type `@` there for the workspace's files: Tab, Enter on a chosen row or a click inserts the path as a [file reference](usage.md#file-references).
A question the model queued, or an ended run's, keeps its box: the answer reaches the run at its next turn, or at its resume.
A failure, a warning and Prune's undelete commands stay on screen until dismissed, the newest by Esc; any other message fades 4 seconds after it shows or after the pointer leaves it.
When the server stops answering, the Sessions and State machines lists, a session and a machine keep what they last showed, under a line that names the reason and the time of the last update, until the next answer.

- **Sessions page**: every session (mode, status, last activity, usage), newest activity first, a group's children folded under its row, each row carrying its [merge note](usage.md#a-rows-merge-note)
    - a run's row names its worktree and its branch while the branch exists; **Review** fills the composer with `/review <branch>`, and **Merge** merges the branch as `sessions merge` does when the run holds unmerged commits, after a confirm that names the unmerged runs that changed some of the same paths; on a conflict in a run with its own worktree it offers **Resume and resolve**, the `agent6 resume <id> --merge <branch>` that merges the base into the run's work for the run to resolve
    - the list is one Tab stop: ↑ ↓, Home and End move between rows, and Tab reaches the focused row's buttons; a Ctrl, Cmd, Shift or middle click on a row opens it as one on its title does, in a new tab or window
    - the composer starts a run, plan or ask under a chosen preset and model (the model box shows what the config resolves; a pick overrides it); **worktree** starts the run in a new git worktree (`run --worktree`); **advanced** holds the plan to run from (`run --from`) and a spend cap (`--max-usd`): a plain number such as `2.5`; while the cap holds anything else, Start waits and **advanced** stays open
    - on a phone the pickers fold behind one line that names the mode, the preset, the model and any run option set; a tap on it opens them
    - on a screen under 576px tall at the default text size (a zoomed window, a phone on its side), the composer sits below the list instead of over it and messages show under the top bar, here and on the State machines page
    - **Run this plan** on a plan's view opens this composer with the plan chosen; Enter on an empty task runs the plan as written
    - on a phone, **Maintenance** folds the notification, prune and clear-saved-asks controls
- **State machines page**: instances, `machine create` drafts, and authored machine files listed by name and path with a **Run** button; the composer creates a new one; its lists take the same keys and clicks as the Sessions list
    - the composer keeps the machine description when you leave the page and clears it after a successful create
- **Session view** (live over SSE): the conversation, the same folded transcript the CLI and TUI render, with the turn in flight streaming underneath; a detail toggle cycles collapsed, expanded, hidden, and a click on an item with more to show, or Enter on its first line, opens or closes it; the conversation is one Tab stop, and the arrow keys, Home and End move between those items
    - the conversation footer counts only the checkpoint commits recorded by that execution, not its children's commits or earlier executions' checkpoints
    - on a phone, **Stop now** stays visible and **More** holds the other run actions. Press Escape, click outside More or choose an action to close it.
    - **Tool calls** rows open to selectable arguments and recorded output
    - **Budget** shows labelled meters for capped usage and plain text for uncapped usage
    - In a live **Task graph**, select a pending or in-progress task and choose **Retire task**.
      Confirm to mark it obsolete at the next step without stopping a worker.
      A graph shown as of a commit is read-only.
    - In **Findings**, choose **Resolve**, pick a disposition and give a reason.
      The form keeps your reason and focus while the finding still accepts the same action.
    - **Children** folds to one line in Details, such as `children: 1 merged, 1 ranked, 1 running`.
      Each child counts once by its recorded outcome, or by its status if it has no outcome.
      On a phone, choose **Children** from **Widgets**.
      Open the card to see each task, its status, and its recorded model, tool, turn, tokens, spend and elapsed time.
      The row shows the child's status before the parent's outcome, such as `answered · no changes`.
      A merged child names the parent and the commit on its chain.
      A stopped child shows the commits its ref holds; a model-owned child names its retained ref.
      A work child shows the measured number of commits beyond its start, including while it runs.
      A child that never started shows the reason.
      A reviewer shows its verdict when available.
      Open a row to read the prompt and recent conversation in place.
      A long prompt shows its first lines.
      Open **Full prompt** to read the rest.
      **Send** messages the live child.
      **Stop** asks before stopping that child and its live descendants.
      These controls also work on a live reviewer.
    - the composer steers a live run or resumes an ended one under the preset and model picked above it (on a phone, behind one line that names them), and takes the [steer directives](usage.md#follow-and-steer-a-run); its button, **Send** on a live run and **Resume** on an ended one, does what Enter does, and a phone shows it as a send icon; Ctrl+Enter sends a live steer as `/now`, Ctrl+R searches past messages
    - an ended child, and an ended run whose worktree is gone or whose manifest cannot be read, closes the composer; the line under it says why, as `agent6 sessions show` does, and **Fork** continues a child in a worktree of its own
    - an ended run whose manifest cannot be read also dims **Fork** and **Merge**, whose tooltips say why
    - **?** outside a field lists the keyboard shortcuts; `>` typed into Ctrl+K's box opens the command palette: the pages, the switcher, the theme, new work and the controls the current page shows, each by name
    - Type `/` in Ctrl+K on any page, then part of a path to find a file in the working tree.
      Enter opens the highlighted file on the Files page.
      If the tree listing is cut, the picker searches only the listed files and says so.
    - Ctrl+K on any page opens the session switcher: live and recent sessions newest first, narrowed as you type by id, task, branch or worktree; each row names its checkout and the ref holding its commits; Enter opens the highlighted one, **Review** and **Merge** act as the Sessions page's row buttons do; a draft typed in a composer waits for the way back, the new-work composer's with its picks
    - a click on a line number in the diff card quotes that line into the composer, with its path, side and commit; Tab reaches the numbers, ↑ and ↓ move through them, and Enter or Space quotes; write your words beneath the quote and send it
- **Machine view**: the state overview with the status and its reason, as the list words them, the path taken and when a parked machine wakes, and the conversation of the agent state the machine is in, or last ran; the entry submits as **Steer** (into the current state) or **Message** (a `poke` a waiting machine's next tool reads), and Enter sends the filled one; **Stop** parks the instance at its next transition and `machine run` resumes it
    - as in a run, the conversation shows its detail button only when it has detail to hide or reveal
    - an ended machine replaces its composer controls with the reason it ended
- **Config page**: every setting with value and source, one collapsible section per table in the order every surface shares (`[agent6]`, `[providers]`, `[models]`, then the rest by name), filterable; click a row to set it, with the choices an enum or a provider's model listing offers, or to add or remove an entry of its table (a provider, an MCP server, a skill, a reviewer) from the same popup; **Preset: name** sets `preset`, and the page edits the selected preset's table, as a line under the bar says ([Presets](config.md#presets)); secrets never shown

    - Tab and Shift+Tab keep the focused setting below the sticky toolbar when its buttons wrap.
    - Removing an entry or reviewer asks inside the editor and names the target layer.
      Cancel or Escape returns to the draft and its selected layer.
      After you confirm, Config reloads even if the reply is lost.
      A removal cannot switch to a different preset or config layer.
      If the reviewer roster no longer matches your confirmation, nothing is removed.

- **Git page**: the repository's commit graph, a commit's files and diffs, and a compare of any two revisions, over the repository `agent6 web` was started in
    - follow a graph column down to a commit's first parent; a side line leads to another parent of a merge
    - **Compare** counts the commits between the base and head. If the histories diverge, it shows how many commits the head is ahead and behind.
      A working-tree head counts to `HEAD` and checks for uncommitted changes at the time of the comparison.
      An empty tree or recorded review tree has no commit count.
    - **Compared commits** lists the commits reachable from just one side, including commits outside the selected refs.
      **Show all commits** returns to the graph for your selected refs without closing Compare.
      Closing Compare also restores that graph.
    - your own hands, file by file: the working-tree panel stages and unstages a file, discards one tracked file's unstaged edits after a confirm (restored from the index; no untracked file is ever deleted), and commits the index with the message you type under the repository's own identity; the toolbar switches branches, creates one at HEAD, and **Review** fills the new-work composer with `/review <branch>`
    - each working-tree row says when its changes are staged or partly staged, and an untracked file reads `U`
    - every write is refused while a session is live in this checkout (a worktree run, a fork or a child works elsewhere and does not block); push, pull, rebase, amend and reset stay in your terminal
    - **Fetch** brings the chosen remote's branches into `<remote>/<branch>` and prunes deleted branches there
      `HEAD` and `pr/` are reserved for the default branch and pull requests.
      Fetch refuses a branch that collides with either.
      Fetch refuses overlapping remote names, such as `team` and `team/mirror`, because they share tracking refs.
    - **Fetch PR** brings one pull request into `<remote>/pr/<number>` (the remote's `refs/pull/<number>/head`, as GitHub, Gitea and Forgejo name it), left selected for **Review**; nothing is merged, and no branch, tag or checkout moves
        - the fetch goes to the URL shown beside the buttons, over https, ssh or a local path, and reads [none of the repository's own git config](security.md#7-git)
    - **blame** uses each line's own side: removed lines use the base, added and inline context lines use the head, split cells use their column, and old/new views use their selected side
    - recorded review comparisons preserve their tree identities and Files links; full context and commit blame are unavailable there
    - an open diff hides the repository toolbar, and the unselected graph has no empty details pane
    - On a narrow screen the side panel takes the graph's place in the Tab order. **Graph** brings the graph and toolbar back.
    - split diffs fall back to inline when the diff pane is too narrow, keeping the saved split choice
    - a click on a line number in a diff quotes that line, with its path, side and revision, for the new-work composer on the Sessions page; Tab reaches the numbers, ↑ and ↓ move through them, and Enter or Space quotes; several lines queue up before you write and start the run
- **Files page**: opens on the working tree of the checkout where you started `agent6 web`.
    - choose a commit in the revision box to read its files instead
    - **Search file contents** finds literal, case-sensitive text at the selected revision when you press Enter or choose **Search**.
      A working-tree search includes untracked files but excludes ignored files.
      Open a match to read its source line.
      The results say when the search was cut.
    - a symlink shows its target as a link when the revision lists it and the path is UTF-8
    - a markdown file opens rendered in a sandboxed frame, with any raw HTML shown as its source; **Source** shows the text; **Show ignored** lists the files every exclude rule hides, badged `I`
    - an HTML file opens as **Source**; **Preview** shows it in a sandboxed frame, with repository CSS and images at the listed revision
    - HTML scripts are off by default; previews need HTTPS or localhost and refuse direct tabs
    - **Run scripts** asks each time the page opens; a page with scripts can read every file under its folder and send them to any server through WebRTC; for a page at the top of the repository that is every file Files shows
    - closing the preview ends the grant
    - a working-tree HTML form can submit answers with `<form method="post">` and no `action`
    - **Save form answers?** shows the sibling file, byte count and whether Save creates or replaces it
    - Save writes `<page name>.answers.json` instead of editing the HTML; Cancel writes nothing
    - one field value becomes a string; repeated names become lists in submission order
    - **Source**, another file, navigation away and tab exit close the preview; an abandoned preview expires after 30 seconds without the UI's renewal
    - markdown remote images, video and audio stay blocked until allowed for that open file; the bar inside the frame lists every full address, and **Load** opens a signed-in confirmation ([browser network limits](security.md#5-network))
    - the confirmation lists the same addresses and warns what loading them sends; Cancel leaves them blocked, and closing the preview discards the allowance
    - an image line naming a repository `.mp4`, `.webm`, `.mp3`, `.wav` or `.ogg` file shows the browser's player, with no autoplay; media keeps the selected revision and the 20 MiB file limit
    - **Checkboxes:** click a `- [ ]` or `- [x]` item in working-tree markdown to change its mark.
      The dialog shows the file and byte count.
      **Save** changes only that mark if the file still matches the displayed version.
      Checkboxes at a commit are read-only.
    - **Edit** opens working-tree text in a dialog with **Save** and **Cancel**; it loads the complete file, up to 20 MiB, even when the source preview is cut
    - the editor checks the disk every two seconds and the server checks again at Save
    - a conflict keeps your draft and shows the disk version.
      The footer offers **Load new file**, **Overwrite with mine**, and **Save as new file**.
      Enter a new name to save a copy beside the original.
    - Overwrite acknowledges only the disk version shown; another change refuses it again; Save as new file never replaces an existing file
    - hidden paths, symlinks, hardlinks, binary files and invalid UTF-8 cannot be edited; saves preserve CRLF line endings and neither stage nor commit
    - the tree keeps the folders you open and opens the ones holding a linked file; ↑ and ↓ move through it, → and ← open and close a folder
    - each file row links to the file at the selected revision, so you can open it in another browser tab
      A path that is not UTF-8 has no Files URL, so its row shows the reason instead.
    - Back and Forward move between the files opened, and the tree keeps its filter and folds
    - **Refresh** rereads the tree and the open file at the selected revision, keeping the filter and folder folds

Start a machine on the State machines page and watch the current state stream, answering its approvals and questions in place:

<video controls muted loop playsinline preload="metadata" class="no-lightbox">
  <source src="/screenshots/out/web-machine.webm" type="video/webm">
</video>

## Receipts

The Findings card's **Receipt** button opens a wide dialog with Testing, Review, Children and How to revert from the [recorded session evidence](usage.md#session-receipts); each child named links to its own view.
**Ask a run to revert** fills the Sessions page's new-work composer with a proposed task; it does not submit the task or run a git command.
`GET /api/session/<id>/receipt` returns the same derived receipt as JSON.

## Links

- `#/git/commit/<sha>` opens a commit and scrolls to its graph row when loaded; `?path=<historical path>&line=<original line>` opens and scrolls to that file's diff
- `#/git/compare/<base>..<head>` opens a comparison
- `#/git/worktree` opens the working tree's uncommitted changes
- `#/files/<rev>/<path>` opens a file, with `WORKTREE` for the working tree; the revision is the one picked, so a link to `HEAD` or a branch follows it
- `?line=<number>` on a file link opens Source at that line, counted from 1.
  Search hits at a commit link to its resolved SHA.
- revisions and path components are URL-encoded; selecting a commit, a comparison, the working tree or a file updates the hash for copying, Back restores the selection, and a bare `#/git` shows the graph alone

## Highlighting

Source text on the Git and Files pages is highlighted with the tree-sitter language pack agent6 ships (372 languages).
A language's grammar downloads into `~/.cache/tree-sitter-language-pack/` the first time a file in it is shown, as the outline tools do; offline, a file in a language not yet cached shows as plain text.
A plain text file, a file over 512 KiB, or one whose parse runs past a second shows uncoloured.

## Notifications and installing (PWA)

The page installs as an app (phone home-screen icon or desktop window).
**Turn on notifications** on the Sessions page or a machine view asks the browser for permission; a session that starts waiting on an answer or ends, `machine.notify` and a machine's end then pop OS notifications on desktop, and the hub keeps watching in a background tab.
The button then reads **Notifications on**, or **Notifications blocked** when the browser refuses; the browser's site settings turn them off or allow them again.
The hub, a session view and a machine view show the same notice as a toast either way.
For a phone not open on the page, point [`[notify].on_event`](config.md#notify-optional) at a push service.

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
curl -s -H "$A" localhost:7658/api/hub                       # hub state, sessions by last activity
curl -s -H "$A" localhost:7658/api/session/<id>              # a run's state, as JSON
curl -s -H "$A" localhost:7658/api/session/<id>/conversation # the folded conversation
curl -s -H "$A" localhost:7658/api/machine/<name>            # a machine's state, as JSON
curl -s -H "$A" localhost:7658/api/config                    # effective config
curl -sN -H "$A" localhost:7658/api/session/<id>/events      # SSE: a snapshot per change
```

- sign-in: `POST` `/api/login` with `{"secret": "<secret>"}` answers the token; a wrong secret answers `403`
- every other `/api/...` route answers `401` with `{"error": ..., "login": true}` without the token
- `GET /metrics` answers Prometheus text when `web.metrics_route = true`, with the web secret as the bearer, not the sign-in token ([telemetry export](cli.md#telemetry-export))
- reads: `/api/meta`, `/api/hub`, `/api/session/<id>/{receipt,findings,merge,resume_defaults}` (`merge` is what the Merge confirm states), `/api/reviews/<panel_id>` with `/diff` and `/file`, `/api/routes?mode=&preset=`, `/api/refs?q=&session=` (the files an `@path` completes to, in the session's workspace or the repository), `/api/config` with `/suggest/<key>?provider=<name>` (an optional provider filter), `/entry_tables` and `/reviewer_choices`, `/api/session/<id>` (what `attach --json` prints; `?commit=<sha>` folds up to that commit) with `/conversation` (`?recent=1` reads the live conversation tail), `/restate`, `/diff` (`?sha=&cumulative=1`) and `/events`, `/api/machine/<name>` with `/reasoning`, `/conversation` and `/events`, `/api/draft/<name>` with `/conversation`, `/diff` and `/events`
- git reads, all of the repository the server runs in: `/api/git/refs`, `/api/git/graph` (`?refs=<full ref name or resolved sha>` repeated, `&limit=&skip=`), `/api/git/commit/<rev>`, `/api/git/compare?base=&head=`, `/api/git/diff?base=&head=&path=&old_path=`, `/api/git/search?q=`, `/api/git/worktree`
    - ref graph pages return `tips`, the resolved shas to pass as `refs` with the next `skip`. **Refresh** starts from live refs.
    - `/api/git/graph?base=<commit>&head=<commit>` walks the symmetric difference instead of `refs`.
      Pass the two returned `tips` as `base` and `head` with the next `skip`.
      **Refresh** keeps the commit pair shown in Compare.
    - a revision is a branch, tag, full ref name, sha prefix or expression git resolves to a commit; the empty tree sha is accepted as a `base`
    - `WORKTREE` names the working tree as the `head` of a compare or diff (an untracked file diffs against nothing); as a `base` or a commit it answers `422`
    - `/api/git/diff` accepts `full_context=1` for all unchanged lines, independently of `whole=1`, which lifts the patch byte cap
    - [Diff payloads](formats.md#apigitdiff-payload) include checked content identities for blame.
    - regular worktree patches and their digests use the same checked bytes, copied privately before Git compares them; diff blame requests `whole=1` to cover hunks beyond the default text range
    - `/api/git/worktree` lists uncommitted files against HEAD: tracked changes first, then untracked files, with `.gitignore` honoured
    - `/api/git/tree?rev=` lists files at a revision or in the working tree, capped at 20000 files; the working tree includes the index and untracked files `.gitignore` does not exclude, omitting tracked files gone from disk; only mode `160000` entries are submodules, including unpopulated gitlinks
    - `/api/git/grep?rev=&q=` searches nonbinary files for one nonempty line of literal text (up to 4096 UTF-8 bytes, no NUL).
      The [result](formats.md#apigitgrep-payload) marks truncation at 100 hits or 256 KiB of git output.
      A grep that takes over three seconds is killed and answers `422`.
    - `/api/git/text?rev=&path=` returns highlighted file text, cut at 1 MiB on a line boundary unless `whole=1`; binary content has no text, and a file over 20 MiB answers `413`
    - `/api/git/blob?rev=&path=` serves a file's bytes at a revision, or from disk for `WORKTREE` (a regular file git lists, never through a symlink); an image is typed by extension, anything else is `application/octet-stream` with `Content-Disposition: attachment`; every blob response carries `Content-Security-Policy: sandbox; default-src 'none'; frame-ancestors 'none'` and `X-Content-Type-Options: nosniff`, so a repository file opened directly runs under no origin; a file over 20 MiB answers `413`
    - `/api/git/blame?rev=&path=` returns per-line attribution; its caps and refusals match `/api/git/text`, including `whole=1`
    - blame preserves historical paths and line numbers, including before a rename
    - a blame date is ISO 8601 at the author's own offset, as `git log --format=%aI` prints it; working-tree blame uses checked file bytes against HEAD, with an empty `sha` and `date` and author `not committed` for uncommitted lines
    - blob, text and blame share a working-tree reader that refuses ancestor and leaf symlinks, paths resolving outside the repository, and non-regular files
    - a revision or path starting with `-`, an unknown revision, a graph ref neither listed in `/api/git/refs` nor a verified full sha, or a path in neither tree answers `422`
- working-tree text: `GET /api/files/text?path=` returns the complete UTF-8 text and its SHA-256 `digest`, with both null when the file is absent
    - `POST` `/api/files/text` accepts `{path, text, digest}`; a matching digest saves the text, and null creates only if absent
    - a mismatch returns 409; no force flag bypasses this check
    - both routes require the sign-in header and refuse hidden path components, symlinks, hardlinks, special files, binary content and invalid UTF-8; files are limited to 20 MiB
    - saves preserve CRLF line endings and create no directories; these routes neither stage nor commit
- preview capabilities: signed-in `POST /api/files/preview` accepts `{rev, path, ignored, digest}` and returns a random `key`, a frame `url`, the resolved `rev`, `kind` and full remote addresses in `images` and `media`
    - markdown requires the digest from `/api/git/text`; a mismatch returns 409, and the server keeps the rendered snapshot until close
    - `POST /api/files/preview/renew` and `/close` take `{key}` and require the sign-in header; a key expires after 30 seconds without renewal
    - `GET /preview/<key>/<path>` reads through the Files blob reader with the opened revision and ignored-file permission; query parameters cannot change either
    - HTML and markdown documents require `Sec-Fetch-Dest: iframe`; the reader checks resource destinations and media types
    - missing metadata, standalone documents, objects and embeds receive a plain 403; browser previews need HTTPS or localhost for Fetch Metadata
    - every preview response carries a sandbox CSP and `Referrer-Policy: no-referrer`
    - signed-in `POST /api/files/preview/scripts` takes `{key}` and grants inline and repository scripts to the opened HTML document until close
    - a granted key serves files only from the page's folder and below
    - `POST /preview/<key>/<page>` accepts URL-encoded fields from the opened working-tree HTML frame, up to 1 MiB and 1024 fields
    - this submission holds one proposal in memory and writes nothing
    - signed-in `POST /api/files/preview/answers` takes `{key}` and returns `{pending}` once per proposal
    - `pending` is null or the Edit route's `{path, text, digest}` for the sibling answers file
    - confirmation sends the displayed proposal to `/api/files/text`; a disk change refuses the save
    - markdown runs only its nonce-authorized owned script and escapes raw HTML
    - signed-in `POST /api/files/preview/remote` accepts only `{key}` and permits the snapshot's remote image origins in `img-src` and video/audio origins in `media-src` until close; it accepts no host list
    - other resource types gain no remote permission; scheme-relative image addresses use HTTPS
- composer preview: `POST` `/api/parallel-preview` reads the draft and returns its tasks and attempt counts without starting a run
- writes, small JSON `POST`s: `/api/git/{stage,unstage,discard,commit,branch,switch}` (409 while a session is live in the checkout, 422 for git's own refusal), `/api/git/fetch`, `/api/new`, `/api/session/<id>/{steer,approve,answer,merge,undo,fork,resume,review,stop,compact,rm}`, `/api/session/<id>/findings/<fid>/{fix,resolve,reopen}`, `/api/session/<id>/findings/fix`, `/api/machine/<name>/{poke,stop,steer,approve,answer}`, `/api/sessions/{prune,rm_asks}`, `/api/config`, `/api/config/entry`, `/api/config/entry/remove`, `/api/config/reviewer`, `/api/config/reviewer/remove`, `/api/machine/{create,run}`
- session rows, session snapshots and machine snapshots list their fields, the operator's words and the spend among them, on [Formats](formats.md#cli-json)
- roster writes use the [typed request bodies](formats.md#reviewerbody) for additions and [removals](formats.md#removereviewerbody)
- writes use the shared config writer or the same spawn and answer-file contracts as the CLI; machine names and answer ids validate to one path component
- the page and its PWA assets, served without the token (they hold no state): `/`, `/manifest.webmanifest`, `/sw.js`, `/icon.svg`, `/favicon.svg`

Session snapshots and streams use the hub's status label, including the mode and the merge state on ended runs.
A child's status names the parent's chain when the parent merge covers its work.
A child with later unmerged commits reads `unmerged`.
A session stream's first frame folds the whole log as it stands at connect.
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
- `POST /api/session/<id>/findings/fix` takes a [batch Fix body](formats.md#fixmanybody): the finding IDs, or `all_open` for every open finding without a queued Fix.
  It answers one result per finding under `results`, 202 while any is queued, and 409 when no finding qualifies.
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
