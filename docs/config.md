<!-- Generated from docs/config_template.md by docs/gen_config.py; edit those, then regenerate. -->
# Configuration

Every field has a default and the security-sensitive ones default to the safe value, so you set only what you want to change.
This is the field reference; the [security model](security.md) covers what `[sandbox]` and `[git]` enforce.

## Where config lives

The layers, lowest precedence first:

| Layer | Path | Set with |
|---|---|---|
| built-in defaults | (none) | (secure defaults, always present) |
| global *(default location)* | `$XDG_CONFIG_HOME/agent6/config.toml` | `agent6 connect`, `agent6 model` |
| per-repo *(override)* | `<state-dir>/<repo-id>/config.toml` | `agent6 init`, `agent6 config set --repo` |
| flag (FILE) | `--config FILE` | `agent6 run --config FILE` |
| machine overlay | a machine file's `[config]` table | `agent6 config set --machine-file FILE` |

`--config FILE` may repeat; later files win, and each file is layered after the repo config.
In the `flag (FILE)` source `config show` prints, FILE is the path as given on the command line.

A selected preset is inserted above the config layer that selected it; [Presets](#presets) gives the precedence rules.
The per-repo config lives in the state dir, out of the workspace: per-machine, never committed.
It can be empty or absent when the global config supplies a provider and model; `harness.verify_command` is inferred per run when unset.

## Creating and inspecting

- `agent6 connect`: add a provider and, when needed, credentials (stored `0600`), global by default; `--repo` writes the provider entry to the repo config while credentials stay global.
  The prompts offer the values a run here reads and say when a layer above the target sets one.
  Enter writes the shown value unless the target already holds it.
  An existing block's format wins over a preset's.
- `agent6 model <role> [provider/]model [--effort off|low|medium|high|xhigh|max]`: a model id alone keeps the role's provider; a configured provider name alone opens a model picker on a terminal, or lists its models without one.
- `agent6 init`: optional setup wizard (per-repo config, inferred `verify_command`, `.gitignore`, `AGENTS.md`); every step asks first unless `--yes` skips the prompts.
- `agent6 config show`: every effective value and which layer set it.
  `--descriptions` adds each value's meaning under its row; `config show <key>...` prints the named keys (or sections) untruncated, meaning included.
  Non-empty mappings show one row per entry with its origin.
  Empty mappings keep a `{}` row.
  Mapping entry values are masked except for fields marked display-safe (`skills.state`).
- `agent6 config set|unset|add|remove <dotted.key> [value]` (`--repo`, or `--machine-file FILE` for a machine `[config]` overlay); `agent6 config get <dotted.key>` prints the effective value and its layer.
  A key component that is not a bare TOML key (`[A-Za-z0-9_-]+`) is written quoted, as in `skills.state."org.some.skill"` or `providers.demo.extra_body."a.b"` (keep those quotes inside a shell-quoted argument).
  `get` accepts a whole mapping or one entry and prints each entry with its own origin.
  `set` and `unset` accept a whole mapping or one entry, such as `skills.state.alpha`.
  Every edit is re-validated; an invalid edit is rolled back when the config lock is held.
  A sibling pair that must move together is set as one inline table: `agent6 config set context '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'`.
- Writes are atomic; failure to take the edit lock does not block the write, but concurrent edits can lose updates.
  An invalid edit without the lock is reported as "kept as written", since rolling it back could erase another writer's update.
  A symlinked config file is followed only when you own the target, or its nearest existing parent when the target does not exist.
- The [TUI config page](terminal.md#configuration) adds and removes named provider, MCP server and skill state entries through forms.
- `config set`, `config unset`, and the web and TUI config editors name the preset or higher config layer that overrides a value they write or remove.
- `agent6 config fill`: materialize defaults + global config into the global file; pass `--force` to replace an existing file.
  The repo layer and any selected preset are left as-is.
- `agent6 config fix`: drop auto-removable invalid entries (unknown keys, stale values) from every layer and preset, a global value a repo override masks included, naming each, and report entries that need manual repair; `--machine-file FILE` repairs an overlay instead.
- `agent6 check`: validate config + sandbox + provider keys without starting an agent run.
  `config show` prints what an `auto` knob resolved to on this host, tagged `(adaptive)`; `check` adds why a level fell short, live jail probes, and each MCP server's network and `approve`.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version; only `1` is accepted. |

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`.
Provider names must match `[A-Za-z0-9_-]+`: one or more ASCII letters, digits, hyphens or underscores.
The schema and config editors use the same rule, so `config set providers.<name>.<leaf>` can address each entry.
Three orthogonal choices describe an HTTP backend: **`api_format`** (the wire dialect, the only field that selects code), **`deployment`** (URL/placement quirks of where it is hosted), and **auth** (`auth_style` + `api_key_env` or `token_command`).
A minimal block is just `api_format` (plus `base_url` for a non-default host).

| Field | Default | Meaning |
|---|---|---|
| `api_format` | *(required)* | The wire format: `anthropic` (the Messages API), `openai` (Chat Completions: OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp, Gemini's OpenAI endpoint), `chatgpt` (the ChatGPT-subscription Codex backend, Responses API), or `claude_code` (the installed, signed-in Claude Code binary on a Claude subscription; no HTTP endpoint, no key). |
| `deployment` | `"direct"` | `direct`, `vertex` (Google Vertex AI), or `azure` (Azure OpenAI; `openai` format only): the URL shape and where the model name and API version go. |
| `base_url` | per (format, deployment) | provider API endpoint's host and path prefix (`https://api.anthropic.com/v1`); required for `deployment = "vertex"` or `"azure"`; ChatGPT sign-in also connected to its fixed OAuth authority |
| `auth_style` | per (format, deployment) | How the key is sent: `x_api_key` (Anthropic), `bearer` (`Authorization: Bearer`, the OpenAI style), `api_key_header` (Azure), or `none` (an unauthenticated local endpoint). |
| `api_key_env` | none | environment variable holding the API key; precedence over `secrets.toml`; unset for a key stored by `agent6 connect` or an unauthenticated local endpoint |
| `token_command` | `[]` | argv command printing a short-lived bearer token to stdout; re-run after `token_command_ttl_s` and once after a `401` or `403`; precedence over `api_key_env` |
| `token_command_ttl_s` | `300.0` | Seconds a `token_command` token is reused before the command runs again. |
| `extra_headers` | `{}` | extra HTTP headers on every provider request; no secrets, config file not `0600` |
| `extra_body` | `{}` | provider-specific JSON merged last into every request body; precedence for tuning keys (`max_tokens`, `temperature`); agent6-owned structural keys filtered out (messages, model, stream, tools, tool choice, response shape); JSON-shaped values only, TOML dates and times refused; OpenRouter routing options here |
| `extra_query` | `{}` | Extra URL query parameters on every request (Azure's `api-version`). |
| `http_timeout_s` | `600.0` | Seconds one HTTP call may take to read or write; the connect phase is capped at 20 s. |
| `prompt_caching` | `true` | prompt caching for `api_format = "anthropic"` only; system prompt, tools and growing conversation re-read at 0.1x the input price |
| `binary` | `"claude"` | executable for `api_format = "claude_code"` only; a name on PATH or an absolute path |

### Deployments

```toml
# Anthropic direct (default): equivalent to a bare api_format = "anthropic"
[providers.anthropic]
api_format = "anthropic"

# Gemini on Vertex (OpenAI-compatible endpoint)
[providers.vertex-gemini]
api_format = "openai"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/endpoints/openapi"
token_command = ["gcloud", "auth", "print-access-token"]

# Claude on Vertex (model in the URL, anthropic_version in the body: handled
# by deployment = "vertex")
[providers.vertex-claude]
api_format = "anthropic"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/publishers/anthropic/models"
token_command = ["gcloud", "auth", "print-access-token"]

# Azure OpenAI (the model id is the deployment name; api-version required)
[providers.azure]
api_format = "openai"
deployment = "azure"
base_url = "https://RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
extra_query = { "api-version" = "2024-06-01" }
```

### ChatGPT subscription (`api_format = "chatgpt"`)

Uses a ChatGPT plan (Plus/Pro/Business) instead of an API key, over the Codex Responses backend.

```bash
agent6 connect chatgpt      # browser sign-in (paste fallback when headless)
agent6 model worker chatgpt/gpt-5.6-sol
```

- `agent6 connect chatgpt` runs a PKCE OAuth sign-in against OpenAI's fixed OAuth authority (`https://auth.openai.com`, a constant, not config) and stores the tokens in `secrets.toml` (0600); they refresh automatically.
- Usage draws on the plan's own limits; cost meters show an authoritative $0 and the tokens meter nothing, so `[budget].max_plan_points` is the ledger that bounds these calls.
- Past the included window, calls draw on purchased credits (real money; auto top-up can buy more).
  `[budget].allow_plan_overage = false` (the default) is a circuit breaker: a usage preflight before the first call and every response's headers report both windows and the credit state, and once a window is exhausted with credits present the run stops at its next boundary; a call already in flight completes, so a boundary-crossing call can spend before the stop.
- Whether these conversations train OpenAI's models follows the ChatGPT account's own data controls (Settings > Data controls > "Improve the model for everyone"); agent6 cannot change that setting.
  agent6 never calls the feedback/rating endpoints, which would opt the rated turns into training regardless of it; there is no rating surface.
- Model names complete from the backend's own listing for the signed-in plan (fetched like other providers' catalogs, never a static list), and its context windows size compaction.
- An `unsupported_value` refusal naming `reasoning.effort` and listing accepted levels, over HTTP 400 or a stream error frame, silently retries once at the lowest listed effort.
  Later calls on that provider instance keep the accepted levels; they are not persisted across machine attempts.
- `agent6 connect chatgpt --logout` signs out: the grant is revoked at the OAuth authority (best effort) and the tokens leave `secrets.toml`.
- Spend is plan-metered, not dollar-metered: every response carries the account's rate-limit window, surfaces show the fullest window as `plan usage (<entry>): N% of the 7-day window`; the duration is whatever the provider reports (`5-hour window`), or `window` when unknown.
  `[budget].max_plan_points` caps the points one run may consume (`--max-plan-points` per run).
  Dollar figures stay an authoritative $0.

### Claude Code subscription (`api_format = "claude_code"`)

Runs the worker inside the installed, signed-in Claude Code binary: a Claude subscription (Pro/Max) instead of an API key.

```bash
claude auth login                        # once, in Claude Code itself
agent6 connect claude                    # writes [providers.claude], checks the sign-in
agent6 model worker claude/claude-sonnet-4-5
```

- agent6's own loop, tools, jail, verify gate, and approvals drive the run; the binary supplies the model.
  Inside it every Claude Code capability is off: built-in tools, hooks, settings, CLAUDE.md, MCP servers, skills, slash commands, session files, auto-memory, auto-compaction.
- `binary` names the executable (default `claude`, resolved on PATH).
  agent6 never reads the login under `~/.claude`; a `CLAUDE_CONFIG_DIR` in the environment selects a relocated one.
- Spend is plan-metered, not dollar-metered: every round reports the account's 5-hour and 7-day windows.
  The end-of-run summary and fan-out report show the fuller one as `plan usage (<entry>): N% of the 7-day window` (or `5-hour window`).
  Window labels show the reported duration (or `window` when unknown), adding the distinguishing key suffix with the common prefix removed when reported windows share a duration.
  `[budget].max_plan_points` caps the points one run may consume (`--max-plan-points` per run).
  Dollar figures stay an authoritative $0; the binary's own list-price estimate is not recorded.
- Ignored: `[models.<role>].temperature` and the loop's per-call output-token cap (the binary owns sampling).
  Refused: `effort = "off"` (`claude --effort` has no off value; use `low`).
- Side roles keep their own providers; route one here explicitly (`agent6 model reviewer claude/claude-haiku-4-5`).
  Each side call is one short-lived `claude` process.
- One `claude` process serves a worker execution.
  It restarts, replaying the conversation as one text message, on resume, fork, `/undo`, a steer or stop mid-turn, a tier-2 context restart, and when the live context nears the window.
  Tier-1 compaction shrinks the model's context at that next restart, not before.
- Claude Code appends the account email to every system prompt it sends; agent6 replaces it with `<operator-email>` in the model's returned text.
- Tool results are capped at 34,000 bytes of UTF-8 for this provider (60,000 elsewhere): Claude Code writes a result over 50,000 bytes under `~/.claude/projects` and hands the model a 2 KB preview of it, and the turn's notices (a verify tail, a review critique, the nudges) ride in the same payload, with 16,000 bytes kept for them.
- Use a full model id (`claude-sonnet-4-5`, not `sonnet`) so the context window is known for compaction sizing.
- The bundled context-window table tries exact and date/tag-normalized ids first, then falls back from `claude-<family>-<major>-<minor>` to `claude-<family>-<major>` when that key exists; the compaction log names the family key when used.
- `agent6 connect claude --logout` is refused: agent6 stores no Claude Code credentials; `claude auth logout` signs out.

### OpenRouter routing and caching (`extra_body`)

OpenRouter's default routing is not deterministic, so prompt caching may or may not engage call-to-call.
Pin it with `extra_body.provider` ([routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
api_format = "openai"
base_url = "https://openrouter.ai/api/v1"
extra_body = { provider = { sort = "throughput" } }  # prefer fast backends
# Alternatives: { order = ["DeepInfra"], allow_fallbacks = true }
#               { max_price = { prompt = 1, completion = 2 } }
```

```bash
agent6 config set providers.openrouter.extra_body \
  '{ provider = { sort = "throughput" } }'
```

Caching matters more than payload size: the large per-call input is the same prefix every turn.
Watch `cache_r` in the cost summary to confirm it engages.

### Short-lived bearer tokens (`token_command`)

For endpoints authenticated by a refreshed bearer rather than a static key (Vertex OAuth, OIDC/STS gateways): point `token_command` at anything that prints a current token, e.g. `["gcloud", "auth", "print-access-token"]` or `["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"]`.
Cached `token_command_ttl_s` seconds, re-run once on `401`/`403`.
It runs in agent6's own process with your environment (operator-only, same trust as an MCP `command`); non-zero exit, timeout, or empty output surfaces as a provider error.

## `[models.<role>]`

Role routing. **`worker`** drives `run`/`resume`; **`planner`** drives `plan`; **`reviewer`** drives `review`, the in-loop review panel, the context summariser and gister, and the prompt reviser.
`planner`/`reviewer` fall back to `worker`.
Cross-vendor mixes are fine.

<!-- the three roles are the same shape, so the table is rendered once -->
| Field | Default | Meaning |
|---|---|---|
| `provider` | *(required)* | A `[providers.<name>]` entry, by name. |
| `model` | *(required)* | Model id as that provider names it (`agent6 model` lists them). |
| `temperature` | `0.0` | sampling temperature pinned on every call, `temperature = 0.0` to `2.0`; stable tool use at `temperature = 0.0`; TOML omission: `temperature = 0.0`; provider default only through `temperature=None` in the Python API |
| `effort` | none | reasoning effort via `effort`: `off`, `low`, `medium`, `high`, `xhigh` or `max`; top tiers where offered by the model, collapsed to Anthropic's highest; unset: `effort = "low"` on openai-compatible reasoning models, no thinking on Anthropic, as printed by `agent6 config show`; no effort sent with `api_format = "chatgpt"` or `"claude_code"`, leaving the model's own default |

## `[sandbox]`

The field summary; the model is in security.md: [Sandbox](security.md#2-sandbox) and [Network](security.md#5-network).
For Go's compiled-package cache, see [Go build cache](usage.md#go-build-cache).

`extra_read_paths`, `extra_write_paths` and `hide_paths` take absolute paths with no `..` segment, and `extra_device_paths` only paths under `/dev`; anything else refuses at config load.

| Field | Default | Meaning |
|---|---|---|
| `isolation` | `"auto"` | jailed-command confinement; `sandbox.isolation = "full"`: user and mount namespaces, seccomp, Landlock where supported by the kernel; `sandbox.isolation = "partial"`: Landlock and seccomp, no namespaces; `sandbox.isolation = "none"`: unconfined, also via `--disable-sandbox` or `AGENT6_DISABLE_SANDBOX=1`; `sandbox.isolation = "auto"`: strongest host-supported level, warning below full isolation; explicit full or partial isolation refused on an unsupported host |
| `network` | `"auto"` | jailed-command network; `sandbox.network = "session"`: private to the run, commands mutually reachable, no off-box access or outside ingress, refused if unenforceable; `sandbox.network = "host"`: machine's network; `sandbox.network = "auto"`: session network with `sandbox.isolation = "full"`, host network with a warning under partial or no isolation; one launcher shared by the run's commands, no per-command network isolation; machine-state override in its `config` table, never wider than this setting |
| `run_commands` | `"ask"` | one permission for `run_command`, `run_verify_command`, `run_metric_command` and `stop_background`; `sandbox.run_commands = "allow"`: commands allowed; `sandbox.run_commands = "deny"`: tools and verify gate withheld; `sandbox.run_commands = "ask"`: per-call prompt with allow-for-this-session answers, startup refused without someone to answer; allow clamped to ask in `ask` and `plan` modes; per-invocation flags: `--auto-approve` (never over configured deny), `--no-commands` |
| `fetch_approved_hosts` | `[]` | hosts readable by `fetch` without approval; other hosts: prompt, refusal without an operator; empty: every fetch prompts; `sandbox.fetch_approved_hosts = ["*"]`: any host; leading dot: subdomains allowed (`.readthedocs.io`); host entries only, never URL prefixes; fixed fetch limits: https only, 1 MiB, redirects returned without following; tool hidden when jailed commands have host networking (`sandbox.network = "host"` or `sandbox.isolation` other than `"full"`); tool withheld from machine and agent states |
| `protect_git` | `"auto"` | `.git/` write protection against jailed commands planting filters for agent6's host-side commits to execute; `sandbox.protect_git = "on"`: read-only bind-remount, requires `sandbox.isolation = "full"`, startup refused elsewhere or with `git.control = "model"`; `sandbox.protect_git = "auto"`: bind under full isolation, warning below it, off under model git control; `sandbox.protect_git = "off"`: writable; in-process `.git` writes refused at every isolation level with `sandbox.protect_git = "on"` or `"auto"` |
| `home` | `"auto"` | home directory (HOME) for jailed commands; `sandbox.home = "tmp"`: `/tmp/agent6-home` in the run's private tmpfs, removed with the run, requires `sandbox.isolation = "full"`, startup refused elsewhere; `sandbox.home = "cache"`: persistent `$XDG_CACHE_HOME/agent6/homes/<repo-id>`, one directory per repo, created `0700`, refused if loosened, read-write bind at its real path; `sandbox.home = "auto"`: tmp under full isolation, cache under partial or no isolation (no private tmpfs); persistent HOME: cross-run channel inside the jail, poisoned caches or `~/.gitconfig` aliases reaching later jailed runs in the same repo, never other repos or the operator's tools; cached HOME for Go builds: compiled-package cache retained across runs without a host-cache grant |
| `memory_limit_mib` | `0` (off) | per-process `RLIMIT_DATA` data-segment cap in MiB (1024 * 1024 bytes) for jailed processes and their children; `sandbox.memory_limit_mib = 0`: no cap; explicit cap to bound a task; cap exceeded: ordinary command error |
| `extra_read_paths` | `[]` | absolute paths outside the repo readable and executable at their real locations (toolchains, conda/Go/Rust/Node interpreters, shared data dirs); mounted for jailed commands, readable by in-process tools; wider sandbox, only paths needed by the build |
| `extra_write_paths` | `[]` | absolute paths outside the repo readable and writable at their real locations (build caches, output dirs, sibling checkouts edited by the task); read access included with write access; wider sandbox, only paths written by the task |
| `extra_device_paths` | `[]` | device nodes under `/dev` exposed read-write (GPU compute: `/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm`); default device access only to `/dev/null`, `/dev/zero`, `/dev/urandom`, `/dev/random` and `/dev/full`; existing character or block devices required at run start, otherwise startup refused; wider sandbox, direct hardware access |
| `hide_paths` | `[]` | absolute paths never readable or writable by the run, even under a broader grant; agent6 config dir and state base always hidden, `secrets.toml` and run history excluded even from `sandbox.extra_read_paths` granting `$HOME`; data dir and cache readable for installed skills; in-process refusal at every isolation level; jailed-command masking: empty dirs and files, mount namespace required; under `sandbox.isolation = "partial"`: run refused for an unmaskable entry, loud warning instead for a grant exposing always-hidden dirs |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `dirty_tree` | `"ask"` | handling of tracked files' uncommitted changes at run start; `git.dirty_tree = "ask"`: an `ask_user` choice of stash for the run, include in its commits, or cancel (park for later resume), startup refused without someone to answer; `git.dirty_tree = "stash"`: stash without asking, restored at the end per `git.auto_stash_pop`, otherwise a printed restore command naming the stash by sha; worktree restore command: its directory, no branch switch; `git.dirty_tree = "include"`: start without asking, changes in the first commit, an unmerged run marked merged on landing if its work matches those changes exactly; untracked files never counted or committed; `--parallel` allowed with `git.dirty_tree = "stash"` or `"include"`, refused with `"ask"`; same policy on resume for files changed since the last execution, stash refused if the run also changed one of them |
| `auto_stash_pop` | `false` | pre-run stash applied over the working tree at run or resume end, even over the run's uncommitted work; worktree sessions restored in their own directory without a branch switch; stash dropped only after a clean apply; failed apply: stash preserved, restore command printed, conflicts visible in `git status`; no `reset --hard`; requires `git.dirty_tree = "stash"` |
| `control` | `"agent6"` | git ownership during a run; `git.control = "agent6"`: each editing turn on the run's own commit chain and branch, HEAD untouched; `git.control = "model"`: model-managed commits and branches, no per-turn chain or run branch, `sessions diff`/`merge`, `/undo` and `fork` refused; with model control: `sandbox.protect_git = "auto"` disabled, `sandbox.protect_git = "on"` refused |
| `branch_per_run` | `true` | visible `agent6/<run-id>` branch advanced to the run's chain tip; `git.branch_per_run = false`: only the hidden `refs/agent6/<run-id>/head` ref; forced on for `--parallel` lanes, whose work is imported by branch |
| `commit_per_turn` | `true` | each editing turn committed to the run's detached chain through a temp index; HEAD never moved; index and working tree touched only with the run's own branch checked out; `git.commit_per_turn = false`: no agent6 commits, work only in the worktree, degraded resume-from-git, `sessions diff`/`merge` and `/parallel` dispatch from a changed tree |
| `merge_method` | `"squash"` | method for `agent6 sessions merge` to land a run on its base; `git.merge_method = "squash"`: one commit; `git.merge_method = "merge"`: a merge commit keeping per-step history; `git.merge_method = "ff_only"`: fast-forward only; consolidation only, per-turn commits always on the run's chain |
| `auto_merge` | `false` | automatic merge into the base branch only for end reason `finish_session` with a passed or absent verify gate; other ends: no auto-merge, reason printed; hidden chain ref merged with `git.branch_per_run = false`; conflict: nothing moved, instructions printed |
| `auto_prune` | `false` | run branch deleted after auto-merge only when `git branch -d` allows it (`git.merge_method = "merge"` or `"ff_only"`); squash-merged branch: `-D` command printed, never force-deleted; requires `git.auto_merge = true`; no action without a run branch |
| `run_repo_hooks` | `false` | repo's `.git/hooks/*` during agent6 git operations; `git.run_repo_hooks = false`: hooks skipped to prevent repo-controlled code running on the host; `core.fsmonitor` and `diff.external` always neutralized |
| `run_repo_drivers` | `false` | repo's filter and merge drivers (`filter.<name>.clean/smudge/process`, `merge.<name>.driver`) during agent6 git operations; `git.run_repo_drivers = false`: each driver neutralized by name to prevent repo-controlled `.git/config` code running on the host at every commit; `git.run_repo_drivers = true` required for Git LFS clean/smudge filters |

`agent6 sessions merge --method <method>` overrides `git.merge_method` for that merge.

`git.control = "agent6"` is enforced by the sandbox; without it, an explicit setting refuses to run and the default warns that a commit the model runs lands on the checkout's branch.

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` | none | author and committer name for agent6 commits; also passed as `GIT_AUTHOR_NAME` and `GIT_COMMITTER_NAME` to jailed `git commit` under `git.control = "model"`; repo's own `git config` identity when unset; startup refused without a resolvable identity |
| `email` | none | author and committer email for agent6 commits; also passed as `GIT_AUTHOR_EMAIL` and `GIT_COMMITTER_EMAIL` to jailed `git commit` under `git.control = "model"`; repo's own `git config` identity when unset; startup refused without a resolvable identity |
| `trailer` | `""` | git trailer line (`Key: value`) on every agent6 commit, such as `"Assisted-by: agent6:{model}"` or `"Co-authored-by: agent6:{model} <noreply@agent6.dev>"`; `{model}`: the code-writing model, several joined with `, `; no trailer when empty |

### `[git.commit.checkpoint]` and `[git.commit.squash]`

| Field | Default | Meaning |
|---|---|---|
| `checkpoint.message` | `"agent6"` | The message of each per-turn commit: `agent6` (`agent6 turn N: <summary>`), `conventional` (a `type(scope): subject` derived from the diff, no model call), or `model` (the model writes it from the git facts, falling back to `agent6` with a warning on any failure). |
| `squash.message` | `"agent6"` | The message of the one commit a squash merge produces: `agent6` (the task's first clause as the subject, the finish summary as the body, else the steps as bullets), `conventional` (the same body under a `type(scope): subject` derived from the diff, no model call), `combine` (git's own squash message: the per-step log concatenated), or `model` (model-written from up to 64 KiB of the base-to-tip diff, marked when truncated, with the task and execution finish summaries as context and commit conventions from AGENTS.md; falls back to `agent6` with a warning on any failure). |

## `preset` (top-level)

| Field | Default | Meaning |
|---|---|---|
| `preset` | `"none"` | selected preset; `preset = "none"`: no preset, stop preset lookup; `preset = "no_review"`: panel off; `preset = "review"`: three seats able to block the finish; `preset = "deep_review"`: five repo-scope seats able to block the finish; custom preset: `[presets.<name>]`; overrides at the selecting layer; `--preset`: per run; `resume --preset`: per resumed execution; empty string refused |

## `[harness]`

| Field | Default | Meaning |
|---|---|---|
| `verify_command` | `[]` | step-success command as argv, no shell (pipeline form: `["sh", "-c", "a && b"]`); explicit value: pinned gate; unset: inferred and printed per run; inference order: AGENTS.md `## Verify command` block or inline `Verify:` line, root `verify.sh`, repo manifests and loose `test_*.py` files, then a model call over those manifests; AGENTS.md commands joined with `&&`, one per line, no multiline `for` or `if`; commands with a first word absent from sandbox PATH skipped and named; files under `sandbox.hide_paths` treated as absent; a plain-command package.json test script run directly when its runner is on sandbox PATH but `npm` is not; no inferred gate: gateless start, adoption of the first gate from a recognizable project created mid-run; `$ ` and `> ` prompts stripped; in a fence containing a `$ ` prompt or opening with a `> ` prompt: only prompted lines and their continuations treated as commands |
| `verify_infer` | `true` | verify-command inference when `harness.verify_command` is unset (AGENTS.md fence, repo signals, model call); mid-run adoption for a gateless run creating a recognizable project; adopted gate dropped and never re-adopted if unable to run (exit 127 or missing module named by `-m`); `harness.verify_infer = false`: no inference or adoption, run stays gateless; explicit `harness.verify_command` unaffected |
| `verify_timeout_s` | `600.0` | seconds per `harness.verify_command` or `harness.metric.command` call before kill and failure; run ending over a gate killed here: `finished · gate timed out`; pathless pytest gate over budget (harness call or model's `run_verify_command`): retry on test files nearest the run's diff, harness gates scoped until a full gate passes; `finish_session` over scoped green: `passed · scoped gate`; model's `run_command` unbounded (see `harness.command_checkin_s`); cap on each gate call, not cached work; `go test` may pass from a cache filled by the model's `run_command`; Go test cache disabled by `go test -count=1` |
| `max_turns` | `200` | loop turns per execution before stop reason `max_turns`; `harness.max_turns = -1`: unlimited; fresh allowance on resume |
| `went_quiet_max_nudges` | `4` | Empty turns (no text, no tool call) re-asked per streak, reasoning-starvation bursts included; 0 ends the run on the first. |
| `loop_guard_kill_threshold` | `10` | The same (tool, args) call this many times in a row ends the run as `loop_guard_killed` (the notice fires from three, every other turn); 0 leaves the notice alone. |
| `verify_settled_stop_after` | `12` | consecutive turns with no edit, commit or memory write and an unchanged tree before a plain run ends after green verify or a gateless editing step; verify runs neutral; streak reset by progress or operator steer; `harness.verify_settled_stop_after = 0`: stop disabled, nudge at three idle turns retained |
| `stagnation_notice_after_s` | `300.0` | wall-clock seconds without the run's own edit or verify before one notice (recall spirals: few calls, long reasoning); `harness.stagnation_notice_after_s = 0`: disabled; clock from the first turn after the start gate, excluding operator wait time |
| `memory_advisory` | `true` | one notice per run at the first red-to-green verify with no repo-memory write, inviting a durable repo fact; cost about two turns on a short fix; `harness.memory_advisory = false`: no notice, memory still readable and writable by the model |
| `command_checkin_s` | `900.0` | seconds before the model's `run_command` returns as a background job; no timeout or kill, command still running; response: `returncode: null`, `still_running: true`, `background_id`; model options: wait with `read_background`, stop the job or carry on; `harness.command_checkin_s = 0`: hand-back disabled |
| `standing_patience` | `-1` | consecutive fruitless standing-goal re-entries (no executed tool call) before soft ends are honoured; `harness.standing_patience = -1`: no end from fruitless rounds alone, only budget, turn cap or operator stop; `harness.standing_patience = 0`: end on the first fruitless round; `harness.standing_patience = N`: escalating nudges for N fruitless re-entries, then soft ends honoured; streak reset by a round landing work |
| `verify_when` | `"finish"` | automatic `harness.verify_command` timing; `harness.verify_when = "finish"`: `finish_session` after a tree change since the last green run; `harness.verify_when = "turn"`: also after each editing turn; `harness.verify_when = "never"`: only the model's own `run_verify_command` calls; tool available in every mode; no gate without a verify command |
| `verify_at_start` | `true` | one `harness.verify_command` call before a fresh run's first turn to record pre-existing gate failure; red output in the first request; finish leaving that same red: `finished · gate was already red`; cost: one gate call per fresh run; `harness.verify_at_start = false`: skip for a gate too slow to run twice; no start gate with `harness.verify_when = "never"` or `sandbox.run_commands = "deny"`; recorded baseline kept on resume |
| `verify_retries` | `2` | red finish certifications returned to the model with gate output before the finish stands as `finished · gate red`, never passed; `harness.verify_retries = 0`: end on the first red; pre-existing red not returned unless made green during this run |

The table was `[harness]` from 0.0.35; a config that still says `[workflow]` is refused with the table named.

## `[review]`

| Field | Default | Meaning |
|---|---|---|
| `trigger` | `"off"` | in-loop panel over the diff so far, findings sent to the model as a message; `review.trigger = "off"`: never; `review.trigger = "on_verify_fail"`: after each failed verify; `review.trigger = "before_finish"`: on `finish_session`, finish rejectable by a gating `review.block_on`; `review.trigger = "periodic"`: every `review.interval_turns` turns; no `review.reviewers`: one seat on `[models.reviewer]`, also used by `agent6 review` |
| `interval_turns` | `10` | Turns between panels when `trigger = "periodic"`. |
| `block_on` | `"none"` | effect of panel `block` verdicts; `review.block_on = "none"`: guidance only; `review.block_on = "any"`: finish rejected by one blocking seat; `review.block_on = "model_quorum"`: `review.min_blocking_models` distinct provider/model routes required to block; `review.block_on = "all"`: every seat required to block, abstentions counted as no block; gating only with `review.trigger = "before_finish"`, other triggers advisory |
| `min_blocking_models` | `2` | distinct provider/model routes required to block with `review.block_on = "model_quorum"`; two seats on one route counted once, even with different focus or effort |
| `max_review_rejections` | `4` | ends rejectable by a gating panel per run (`finish_session`, silent finish, settled or plateau stop); count preserved across resumes, never reset; at the cap: panels and findings retained, every end accepted, no indefinite panel stall |
| `min_remaining_budget_fraction` | `0.25` | remaining-budget fraction below which the panel is skipped and the finish accepted; `review.min_remaining_budget_fraction = 0.25`: no panel in the last quarter |
| `reviewers` | `[]` | ordered panel roster of tables with `focus` and optional `provider`, `model`, `temperature` and `effort`; routing omitted: inherited `[models.reviewer]`; pinned route: role-model defaults, `provider` and `model` required together; empty roster: built-in focuses cycled through `security`, `correctness`, `tests`, `over-engineering`, `edge-cases`, one seat when `review.trigger` is on; one-off roster via `agent6 review --reviewers N --focus ...` |
| `concurrency` | `1` | simultaneous in-loop panel seats; `review.concurrency = 1`: one after another; `agent6 review`: all seats always parallel |
| `scope` | `"diff"` | How much a seat reads: `diff` (one call over the diff, the task, and the verify result) or `repo` (a read-only tool-using reviewer that also reads the repo around the diff to catch cross-file impact; several calls per seat, and it reads the checkout, so the reviewed head must be checked out with a clean tree). |
| `repo_max_turns` | `20` | provider-call cap for a `review.scope = "repo"` seat, tool calls included; no verdict within the cap: abstention, reported by the panel |

The [TUI](terminal.md#configuration) and [web](web.md#pages) config pages edit the ordered roster with **Add reviewer…** and positional removal.
Each edit writes the whole list to the chosen layer, including `[]` after the last removal.

Each `reviewers` entry has these fields:

| Field | Default | Meaning |
|---|---|---|
| `focus` | *(required)* | A short review stance. |
| `provider` | none | A provider name; requires `model` too. |
| `model` | none | A model id; requires `provider` too. |
| `temperature` | none | sampling temperature; route's value when unset; `temperature = 0.0` an explicit override |
| `effort` | none | reasoning effort; route's value when unset; `effort = "off"` an explicit override |

Set `provider` and `model` together to pin a route; omit both to inherit the resolved reviewer role.
An unpinned entry inherits the role's sampling settings; a pinned entry defaults to `temperature = 0.0` and unset effort.
Explicit `effort = "off"` and `temperature = 0.0` override inherited values.
`model_quorum` requires at least `min_blocking_models` distinct resolved provider/model routes at config validation.
Different focuses or effort levels on one route still count as one blocking model.

```toml
[review]
trigger = "before_finish"
scope = "diff"
repo_max_turns = 20
interval_turns = 10
min_remaining_budget_fraction = 0.25
block_on = "any"
min_blocking_models = 2
reviewers = [
  { focus = "security" },
  { focus = "correctness", provider = "openrouter", model = "moonshotai/kimi-k2", effort = "high" },
]
```

A `block` gates only when its `file:line` is in the diff and its category is one of `security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`, `verify-uncovered-correctness`; every other finding is advisory and cannot stall the run.
Within one execution, findings already injected neither repeat nor count toward a later panel's gate.
A resume starts with no prior findings.

## `[context]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

| Field | Default | Meaning |
|---|---|---|
| `drop_at_chars` | _adaptive_ | tier-1 threshold in accumulated tool-result characters (about 4 per token); above it: oldest results replaced by short, re-fetchable placeholders; unset: about 45% of the model's context window; explicit thresholds: both `context.drop_at_chars` and `context.summarise_at_chars` required |
| `summarise_at_chars` | _adaptive_ | tier-2 threshold in whole-context characters; above it: elided history summarized, conversation restarted on the summary, task DAG preserved; unset: model's window minus a 16k-token reserve; required above `context.drop_at_chars` |
| `keep_recent_chars` | `80000` | characters of newest history kept verbatim after the summary on a tier-2 restart; `context.keep_recent_chars = 0`: none |
| `keep_thinking_turns` | `-1` | thinking retained at tier 1 for the newest N assistant turns with `context.keep_thinking_turns = N` (`N >= 1`); `context.keep_thinking_turns = -1`: all thinking; `context.keep_thinking_turns = 0`: invalid, newest signed thinking block needed by an in-flight tool call; less replay for Anthropic's signed blocks and ChatGPT's reasoning items; no thinking re-sent on the OpenAI wire |
| `summary_max_tokens` | `2048` | output-token cap for a tier-2 summary or gist distillation; smaller caps overridden by a reasoning model's per-call floor for reasoning tokens; no cap accepted by the chatgpt backend |
| `elision_gists` | `true` | large `read_file` results replaced at tier 1 by model-written gists before bare placeholders; gists also dropped under continued pressure to keep the byte bound; `context.elision_gists = false`: bare placeholders directly |

## `[prompt]`

| Field | Default | Meaning |
|---|---|---|
| `system_prompt_file` | `""` | file path replacing run mode's built-in base system prompt; dynamic blocks still appended; tool contracts supplied by the operator; startup warning for missing core tool names; empty: built-in base; assembled prompt, tool definitions and first message shown by `agent6 prompt show` |
| `revise_prompt` | `"off"` | one task rewrite by the reviewer model before the loop; `prompt.revise_prompt = "off"`: no rewrite; `prompt.revise_prompt = "auto"`: revision used as written; `prompt.revise_prompt = "interactive"`: accept, keep original, edit or quit (stop the run), terminal required, skipped under TUI, ACP or a spawned lane; same pass for `/task` queued into a live run, handled as `prompt.revise_prompt = "auto"` |
| `decompose` | `"auto"` | task decomposition before editing in run mode, ordered DAG subtasks worked one at a time; `prompt.decompose = "on"`: help for small models under-finishing multi-part tasks (measured on mistral-small), 2-4x overhead for capable models; `prompt.decompose = "off"`: disabled; `prompt.decompose = "auto"`: per-worker-model capability registry, resolved value in `config show`; forced for one run by `--decompose` |

## `[skills]`

Operator-installed SKILL.md packs (the agentskills.io format, shared with Claude Code), under `$XDG_DATA_HOME/agent6/skills/<name>/`.

A skill name is 1 to 64 characters of lowercase ASCII letters, digits and hyphens, with no leading, trailing or doubled hyphen, and equals its skill directory name.
Discovery skips a skill that breaks this rule with a warning.

- `agent6 skills install <url>` takes a SKILL.md URL, a git repo (every `skills/*/SKILL.md`) or a local path; `extra_dirs` points at an existing collection (`~/.claude/skills`, …) instead of copying
- installed means enabled: an index in the system prompt, on-demand content via `use_skill`, a `/<name>` pause-menu command, and `run --skill <name>`
- repo-local skill dirs are not discovered (repo content is not config); the trust model is in [security.md](security.md)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch for skills: `false` means no skill index in the prompt, no `use_skill` tool, and no slash commands. |
| `extra_dirs` | `[]` | Additional directories scanned for skills, before the installed skills dir; a skill of the same name in an earlier dir wins. |
| `state` | `{}` | per-skill state by name; `skills.state.<name> = "enabled"`: indexed, loaded on `use_skill`; `skills.state.<name> = "disabled"`: dropped; `skills.state.<name> = "always"`: full text in the system prompt; layers merged key by key; written by `agent6 skills enable [--repo]` or `agent6 skills disable [--repo]` |

Measured (2026-07): small and frontier open models alike almost never invoke a skill organically from the passive index, and no prompt lever made it reliable.
When a skill must apply, use `always`, `/name`, or `--skill`.

### Presets

A preset fills many settings at once.
`agent6 config presets` lists them; select with `--preset <name>`, `agent6 config set preset <name>` (`--repo`), or the preset picker of the TUI and web new-task composers.
A preset overrides config at the layer that selected it (most-specific source wins, presets never stack); a more-specific config layer, `--config FILE`, or an individual flag still beats it.
A `--config FILE` or machine overlay cannot select one.

The default `preset = "none"` selects no preset.
Set `preset = "none"` in the repo config to stop lookup before a global preset, or pass `--preset none` to override both layers.
This keeps the layers' own settings; it does not reset them to defaults.
An empty string is refused.
The name `none` is reserved, so `[presets.none]` is refused.

| Preset | For | Sets |
|---|---|---|
| `no_review` | turns off the review panel | `review.trigger = "off"` |
| `review` | three review seats that can block the finish | `review.trigger = "before_finish"`, `review.block_on = "any"`, `review.reviewers = [{ focus = "security" }, { focus = "correctness" }, { focus = "tests" }]`, `review.concurrency = 3` |
| `deep_review` | five repo-scope review seats that can block the finish | `review.trigger = "before_finish"`, `review.block_on = "any"`, `review.scope = "repo"`, `review.reviewers = [{ focus = "security" }, { focus = "correctness" }, { focus = "tests" }, { focus = "edge-cases" }, { focus = "over-engineering" }]`, `review.concurrency = 5` |

Define your own with a `[presets.<name>]` table (a partial config); a built-in's name replaces that built-in wholesale:

```toml
preset = "myteam"

[presets.myteam.review]
trigger = "before_finish"
block_on = "any"
reviewers = [
  { focus = "security", provider = "anthropic", model = "claude-opus-4-8" },
  { focus = "correctness", provider = "openrouter", model = "moonshotai/kimi-k2", effort = "high" },
]
```

### `[harness.metric]` (optional)

A continuous score for measurable goals; `command` runs in the jail like `verify_command`.
Set its required fields together: `agent6 config set --repo harness.metric '{ command = ["python3", "bench.py"], pattern = "score: ([0-9.]+)", goal = "minimize" }'`.

| Field | Default | Meaning |
|---|---|---|
| `command` | *(required)* | score-printing command as argv, no shell; run after every committed edit and on the model's `run_metric_command` call |
| `pattern` | *(required)* | A regular expression over the command's output; its first capture group is the number, e.g. `"score: ([0-9.]+)"`. |
| `goal` | *(required)* | score direction; `harness.metric.goal = "minimize"`: smaller wins; `harness.metric.goal = "maximize"`: larger wins; trajectory reported; finish allowed once a verified edit only ties the best |

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost, else cached price × tokens) count against `max_usd`; a call carrying a plan-usage reading counts percentage points against `max_plan_points`; the rest count input+output tokens against `max_tokens_fallback`.
All three: `-1` unlimited, `0` refuse that ledger up front, `> 0` the cap.

| Field | Default | Meaning |
|---|---|---|
| `max_usd` | `10.0` | cap on the metered spend of one run (provider-reported cost, else price times tokens at the model's fetched rates, cache-aware); at the cap: resumable end (`budget_exhausted`); fresh budget per resumed execution; `budget.max_usd = -1`: unlimited; `budget.max_usd = 0`: every metered call refused; per-run override: `--max-usd` |
| `max_tokens_fallback` | `2000000` | input-plus-output token cap for unpriceable calls (local models, models without price data); `budget.max_tokens_fallback = -1`: unlimited; `budget.max_tokens_fallback = 0`: unmeterable models refused; per-run override: `--max-tokens-fallback` |
| `max_plan_points` | `-1.0` | subscription-plan percentage-point cap per run; rise in the account's reported used-percent, summed across window resets (values above 100 meaningful), largest movement across multiple windows; a window counts as reset when its reported reset time moves later by more than a minute; a lower reading whose reset time has not moved counts nothing; one more reading at the end of the run; whole-percent readings, each tick counted as a full point; early stop at the cap, never late; in-flight call still completed; account-global readings include other runs' spend; `budget.max_plan_points = -1`: unlimited; `budget.max_plan_points = 0`: plan-metered calls refused; per-run override: `--max-plan-points` |
| `allow_plan_overage` | `false` | plan overage for `api_format = "chatgpt"` or `"claude_code"`: purchased credits or extra usage after the included window, possible auto top-up through the saved payment method; `budget.allow_plan_overage = false`: circuit breaker, not a guarantee; account windows and credit state read at chatgpt preflight and from every response's headers, or from every claude_code round's rate-limit event; with `budget.allow_plan_overage = false`, exhausted window and credits present: stop at the next boundary, in-flight call still completed; `budget.allow_plan_overage = true`: chatgpt credit-balance drop read as dollars against `budget.max_usd`; no claude_code credit-balance reading, extra usage not metered by `budget.max_usd`; included-plan usage unaffected |

`--max-usd` / `--max-tokens-fallback` / `--max-plan-points` override per run.
A flag's value is recorded with the run: a resume without that flag runs under it again, and a flag on the resume replaces it for later resumes too.
A worker with no price data is not metered by `max_usd` at all and runs under `max_tokens_fallback`, with a note at startup.
Included-plan calls are bounded by `max_plan_points`, not by `max_usd` or `max_tokens_fallback`.
Prices come from provider listings (OpenRouter's; cached under `$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its OpenRouter listing.

## `[machine]`

| Field | Default | Meaning |
|---|---|---|
| `snapshot_keep` | `5` | blackboard snapshots kept per machine instance; latest read by `machine status`, journal folded for recovery and `machine replay`; `machine.snapshot_keep = 0`: all snapshots |
| `state_log_keep` | `50` | per-state log dirs kept under `<instance>/states/`, watchable logs of each state's execution; full transition history in the journal regardless; `machine.state_log_keep = 0`: all logs |
| `pass_env` | `[]` | operator environment variable names allowed in a machine `tool` state's own `pass_env`; startup refused for a state naming an unlisted variable; global/repo config only, machine `[config]` overlay refused; provider `api_key_env` variables never allowed |

### `[machine.notify]` (optional)

Operator hook on every `machine.notify` and the terminal `machine.end`: an operator argv on the host with a minimal env (PATH/HOME/locale/desktop-bus + `AGENT6_MACHINE_ID/DIR/EVENT/STATE/MESSAGE/LEVEL`), never your full environment.
Global/repo config only (a machine `[config]` overlay setting it is rejected).
Fan out to ntfy/Pushover/email/Telegram yourself.

| Field | Default | Meaning |
|---|---|---|
| `on_event` | `[]` | argv command on every machine notify event and at machine end, no shell; event in `AGENT6_MACHINE_*` variables; no hook when empty |
| `timeout_s` | `30.0` | Seconds the hook may run before it is killed. |

## `[notify]` (optional)

Runs after `run`/`resume` with the same minimal env plus `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`.
`OK=1` means the agent stopped deliberately; `VERIFIED` is what the gate said; a hook that wants "green" reads the second.

| Field | Default | Meaning |
|---|---|---|
| `on_complete` | `[]` | argv command at run or resume end, no shell; environment with `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`; no hook when empty |
| `timeout_s` | `30.0` | Seconds the hook may run before it is killed. |

## `[web]`

Bind for `agent6 web` ([the web UI](web.md)).
Loopback only by default; remote access is expected behind `tailscale serve`.
Every browser [signs in](web.md#sign-in) with the server's secret.

| Field | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Address `agent6 web` binds; a non-loopback address also needs `allow_non_loopback = true`. |
| `port` | `7658` | Port `agent6 web` listens on. |
| `allow_non_loopback` | `false` | Allow `host` to be a non-loopback address, so a typo can never silently expose the write surface (approvals, steers, config writes) beyond this machine. |

## `[telemetry]`

What `agent6 telemetry export` sends, and where ([Telemetry export](usage.md#telemetry-export)).
Off by default: no signal is selected, nothing is dialled, no socket is opened.
The exporter is a separate process the operator starts; a run never exports its own events.
Header values live in `secrets.toml` under `[telemetry.headers]`, never in config.

| Field | Default | Meaning |
|---|---|---|
| `traces` | `"none"` | Where spans go: `otlp` posts them to `endpoint`. |
| `metrics` | `"none"` | Where metrics go: `otlp` posts them to `endpoint`; `prometheus` writes `prometheus_file` for node_exporter's textfile collector. |
| `logs` | `"none"` | Where the significant events go as log records: `otlp` posts them to `endpoint`. |
| `endpoint` | `""` | The OTLP/HTTP base URL; `/v1/traces`, `/v1/metrics` and `/v1/logs` are appended; `https://`, or `http://` to a loopback host |
| `headers` | `[]` | Header names sent with every export; each value comes from `[telemetry.headers]` in `secrets.toml`. |
| `capture` | `"metadata"` | `metadata` exports outcomes, counts, timings and ids; `content` adds capped excerpts of the task, steers, tool arguments, the gate command, model errors, finding titles and degradation details. |
| `prometheus_file` | `""` | The absolute path `metrics = "prometheus"` writes atomically, in a directory node_exporter's textfile collector reads. |

## `[parallel]`

Lane limit for `run --parallel N|[provider/]model,...`.
Each lane is a disposable clone running its own `agent6/<id>` branch; lanes are imported and ranked, nothing merges for you.
A lane runs under the settings you set, at their effective values; a setting left at its default is a default in the lane too, so a sandbox default this host cannot honour degrades with a warning there as it does in a single run, and one you set refuses.
`--max-usd` is per lane (the total is printed before spawning); `--auto-approve` forwards to every lane.
A live run dispatches lanes the same way via `/parallel [SPEC --] TASK` ([grammar and examples](usage.md#parallel-directive)).
A lane never fans out; headless surfaces without a dispatcher answer "not available".

| Field | Default | Meaning |
|---|---|---|
| `max_lanes` | `4` | The most lanes one `--parallel` fan-out may run, `1` to `1024`; a spec asking for more is refused before anything is cloned. |

## `[workspace]`

Placement and setup for lane clones, machine state clones, forks and `run --worktree` worktrees.
An empty `base_dir` uses `<cache_dir>/parallel`.

| Field | Default | Meaning |
|---|---|---|
| `base_dir` | `""` | base directory for lane clones, machine-state clones, forks and `run --worktree` worktrees, grouped by repo; empty: `<cache_dir>/parallel`; lane clone removed after work import; fork and `run --worktree` worktrees removed by `sessions prune` once merged or holding no work |
| `setup_command` | `[]` | environment-setup command for a new isolated workspace as argv, no shell (pipeline form: `["sh", "-c", "a && b"]`); `run --worktree` and fork worktrees, lane and machine-state clones: tracked files only, no untracked `.venv` or `node_modules`; command run once in each workspace before the first turn, under the run's command sandbox (same isolation, network and grants); nonzero exit or 600-second timeout: session ended before any turn; unset: AGENTS.md `## Setup command` fenced block, parsed like `## Verify command`, command printed; no configured or inferred command: nothing run |

## `[mcp]` and `[mcp.servers.<name>]` (optional)

agent6 as an MCP CLIENT; for the other direction (`agent6 mcp serve`, agent6 as a server) see [Editor integration](acp.md#as-an-mcp-server).

MCP servers, spawned (`command`) or connected (`url`); tools appear as `mcp__<name>__<tool>` in run mode only (plan, ask, machine and agent sessions never offer them: agent6 cannot classify an external tool as read-only).
A spawned server runs as a jailed child by default (its own `[mcp.servers.<name>.sandbox]` policy; `unconfined = true` opts out) with a curated env (never your provider keys; `pass_env` adds named vars); a `url` server is a process you run and confine yourself.
The model chooses the arguments, so each call is approved like a command (`approve`); audit each server like a `run_command` allow-list.
`agent6 mcp add` saves a spawned server unprobed with a warning when `sandbox.isolation` resolves to `none`; otherwise it requires a successful probe before writing the entry.
During a run, a server that does not start is skipped with an `mcp.server_unavailable` journal event, never fatal.
`agent6 check mcp` probes under the run's sandbox with the repository bound read-only and skips a server it cannot hold that way ([the rule](security.md#2-sandbox)).

```
agent6 mcp add files -- npx -y @modelcontextprotocol/server-filesystem .
agent6 mcp add browser --url http://127.0.0.1:8931/mcp --token-env PW_TOKEN
agent6 mcp list
agent6 mcp remove files
```

`agent6 mcp remove <name>` drops the entry from the global config (`--repo` for the per-repo one): `config unset` cannot, because the entry is a table rather than a leaf, and dropping its `command`/`url` is refused since an entry needs exactly one of them.

A spawned server's `[sandbox]` block names what it needs on top of the sandbox a `run_command` gets.

```toml
[mcp.servers.notes]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "~/notes"]

[mcp.servers.notes.sandbox]
extra_read_paths  = ["~/notes"]   # additive; the system and tool dirs are already there
extra_write_paths = ["~/notes"]
```

A confined spawn also loses the desktop-session addresses (`DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `DISPLAY`, `WAYLAND_DISPLAY`): Landlock does not gate unix-socket `connect()`, and an unconfined session daemon would act on the server's behalf.
Anything else that reaches an unconfined process is still a way out, so name the narrowest paths that work.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch for MCP servers: `false` means no `mcp__*` tools reach the model, whatever `[mcp.servers]` lists. |
| `servers.<name>.command` | `[]` | argv of a spawned stdio MCP server; jailed like a command, plus `sandbox` grants; exactly one of `command` or `url` required |
| `servers.<name>.url` | `""` | operator-run http(s) MCP endpoint; connection only, environment and confinement outside agent6's control; exactly one of `command` or `url` required |
| `servers.<name>.token_env` | `""` | bearer-token environment variable for a `url` server, named rather than inlined, never logged; token readable on the wire over plaintext `http://` to a non-loopback host; approval in `mcp add`, warning on every run starting it |
| `servers.<name>.enabled` | `true` | `false` withholds this server's tools from the model without deleting the entry. |
| `servers.<name>.pass_env` | `[]` | Environment variables a spawned server needs, by name; everything else is agent6's curated base environment. |
| `servers.<name>.approve` | `"ask"` | approval policy for this server's tools; `approve = "ask"`: prompt before each call with the model's arguments; `approve = "allow"`: no prompt; session answer "allow all": this server only for the run, not command tools or sibling servers; "deny all": tools withdrawn from the next turn; `--auto-approve`: `approve = "allow"` for the run; no deny mode, tools withheld with `enabled = false` |
| `servers.<name>.startup_timeout_s` | `10.0` | Seconds the server gets to answer `initialize` and `tools/list` before it is given up on. |
| `servers.<name>.call_timeout_s` | `60.0` | Seconds one `tools/call` may take before it fails; a spawned server is restarted after a timeout. |
| `servers.<name>.http_trust_env` | `false` | ambient `HTTP(S)_PROXY`, `.netrc` and `SSL_CERT_FILE` for a `url` server; `http_trust_env = false`: no proxy routing of a local server's bearer token; opt-in for a server reachable only through the environment's proxy |

### `[mcp.servers.<name>.sandbox]`

| Field | Default | Meaning |
|---|---|---|
| `extra_read_paths` | `[]` | read and execute paths for this server beyond the run's sandbox (absolute or `~`); workspace, system dirs, tool dirs and writable `HOME` included without extra grants; only the server's own data needed here |
| `extra_write_paths` | `[]` | Paths this server may write beyond the run's sandbox (absolute or `~`). |
| `network` | `"auto"` | server network; `network = "auto"`: private namespace where supported, otherwise host network with a warning; `network = "none"`: private network or refusal, never a connected fallback; `network = "session"`: run's network, background dev servers reachable by this server (such as a browser testing the app), no off-box access; `network = "host"`: machine's network |
| `unconfined` | `false` | no sandbox for a server needing arbitrary host access; refused alongside any other field in this block, no partial application |

---

## Reaching a run's network

A run's commands share one network with no route off the box, so a dev server the agent starts is reachable only through these operator-only commands:

```
agent6 sessions show <id>        # what it is serving, and the command to open it (the TUI dashboard and web run headers say the same)
agent6 forward <id>              # list the ports it is listening on
agent6 forward <id> 3000         # bridge that port to one on this machine
agent6 exec <id> -- curl ...     # run a command in the run's sandbox
```

`exec` runs in the whole sandbox (the run's recorded isolation and network, with mounts from your current config), so what you see is what the agent sees.
None of them is reachable by the model.

---

## Environment variables

| Variable | Effect |
|---|---|
| `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME` | The XDG base directories; agent6 lives in an `agent6/` dir under each (config and secrets; per-repo state; the cache; installed skills). |
| `AGENT6_DETACHED_AWAY` | approval and question handling without a terminal or editor; `AGENT6_DETACHED_AWAY=wait`: both parked for a front-end; `AGENT6_DETACHED_AWAY=deny`: approval refused; `AGENT6_DETACHED_AWAY=approve`: approval granted; deny or approve: questions answered with empty answers and a note to decide alone, counted by `sessions show`; hub and machine spawns: `AGENT6_DETACHED_AWAY=wait`; `--parallel` lanes: coordinator's marker, otherwise wait with a terminal to attach from, deny without one |
| `AGENT6_AUTO_APPROVE` | `1` grants every command approval to a machine's agent states, as `--auto-approve` does (a configured `deny` stays deny). |
| `AGENT6_NO_COMMANDS` | `1` withholds every command tool from a machine's agent states, as `--no-commands` does. |
| `AGENT6_JAIL_BIN` | Path to a specific `agent6-jail` binary (else bundled). |
| `AGENT6_DISABLE_SANDBOX` | `1` forces `sandbox.isolation = "none"` for one invocation (same as `--disable-sandbox`). |
| `AGENT6_ALLOW_ROOT` | `1` permits running as root (same as `--allow-root`). |
| `AGENT6_WEB_SECRET` | browser sign-in secret for `agent6 web`; at least 12 characters after lowercasing and joining words with single hyphens; overridden by `--secret`; unset: fresh secret generated and printed per launch |

A provider's `api_key_env` names the env var supplying its key; omit it to read `secrets.toml`.
The bench and development switches are listed in [architecture.md](architecture.md#bench-and-development-switches).
