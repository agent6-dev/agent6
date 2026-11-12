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
| per-repo *(override)* | `$XDG_STATE_HOME/agent6/<repo-id>/config.toml` | `agent6 init`, `agent6 config set --repo` |
| flag (FILE) | `--config FILE` | `agent6 run --config FILE` |
| machine overlay | a machine file's `[config]` table | `agent6 config set --machine-file FILE` |

`--config FILE` may repeat; later files win.

A selected preset is inserted above the config layer that selected it; [Presets](#presets) gives the precedence rules.
The per-repo config lives in the state dir, out of the workspace: per-machine, never committed.
It can be empty or absent when the global config supplies a provider and model.

## Creating and inspecting

- `agent6 connect`: add a provider and, when needed, credentials (stored `0600`), global by default; `--repo` writes the provider entry to the repo config while credentials stay global.
  An existing block's format wins over a preset's.
- `agent6 model <role> [provider/]model [--effort off|low|medium|high|xhigh|max]`: a model id alone keeps the role's provider; a configured provider name alone opens a model picker on a terminal, or lists its models without one.
  A model change preserves the target layer's other role settings; only an explicit `--effort` replaces effort.
- `agent6 init`: optional setup wizard (per-repo config, inferred `harness.verify.command`, `.gitignore`, `AGENTS.md`); every step asks first unless `--yes` skips the prompts.
- `agent6 config show`: every effective value and which layer set it.
  `--descriptions` adds each value's meaning under its row; `config show <key>...` prints the named keys (or sections), meaning included.
  Text values on the CLI, TUI and web use TOML literals without truncation, including quoted strings and inline tables, so an unmasked value pastes into `config set` unchanged.
  `(unset)` marks the absence of a value; TOML has no null literal.
  Mapping entry values are masked except for fields marked display-safe (`skills.state`, `review.reviewers`).
- `agent6 config set|unset|add|remove <dotted.key> [value]` (`--repo`, or `--machine-file FILE` for a machine `[config]` overlay); `agent6 config get <dotted.key>` prints the effective value and its layer.
  A key component that is not a bare TOML key (`[A-Za-z0-9_-]+`) is written quoted, as in `providers.demo.extra_body."a.b"` (keep those quotes inside a shell-quoted argument).
  `get` accepts a whole mapping or one entry and prints each entry with its own origin.
  `set` and `unset` accept a whole mapping or one entry, such as `skills.state.alpha`.
  Every edit is re-validated; an invalid edit is rolled back when the config lock is held.
  A sibling pair that must move together is set as one inline table: `agent6 config set compaction '{ drop_at_chars = 200000, summarise_at_chars = 400000 }'`.
- Writes are atomic; failure to take the config lock does not block the write, but concurrent edits can lose updates.
  An invalid edit without the lock is reported as "kept as written".
  A symlinked config file is followed only when you own the target, or its nearest existing parent when the target does not exist.
- The [TUI config page](terminal.md#configuration) adds and removes named provider, MCP server and skill state entries through forms.
- `agent6 config set --secret telemetry.headers.<name> <value>` saves a literal header value in the private global `secrets.toml`; `agent6 config unset --secret telemetry.headers.<name>` removes it.
  This target accepts only telemetry header keys and cannot be combined with `--repo` or `--machine-file`.
  Values are never printed by these commands or `config show`, but command-line values can remain in shell history and the process list ([secrets](security.md#8-secrets-and-connect)).
- `agent6 config fill`: materialize defaults + global config into the global file; pass `--force` to replace an existing file.
  The repo layer and any selected preset are left as-is.
- `agent6 config fix`: drop auto-removable invalid entries (unknown keys, stale values) from the global and repo configs and their presets, a global value a repo override masks included, naming each, and report entries that need manual repair; `--machine-file FILE` repairs an overlay instead.
- `agent6 check`: validate config + sandbox + provider keys without starting an agent run.
  `config show` tags resolved values `(adaptive)`: `auto` settings for `sandbox.isolation`, `sandbox.network` and `harness.prompt.decompose`, each role's unset effort, and adaptive compaction thresholds.

---

## `[agent6]`

| Field | Default | Meaning |
|---|---|---|
| `config_version` | `1` | Config schema version; only `1` is accepted. |
| `preset` | `"none"` | selected preset; `agent6.preset = "none"`: no preset, stop preset lookup; `"no_review"`: panel off; `"review"`: three seats able to block the finish; `"deep_review"`: five seats able to block the finish; custom preset: `[presets.<name>]`; overrides at the selecting layer; `--preset`: per run; `resume --preset`: from that resume on, recorded for later resumes; empty string refused |

### Presets

A preset fills many settings at once.
`agent6 config presets` lists them; select with `--preset <name>`, `agent6 config set agent6.preset <name>` (`--repo`), or the preset picker of the TUI and web new-task composers.
A preset overrides config at the layer that selected it (most-specific source wins, presets never stack); a more-specific config layer, `--config FILE`, or an individual flag still beats it.
A `--config FILE` or machine overlay cannot select one.

The default `agent6.preset = "none"` selects no preset.
Set `agent6.preset = "none"` in the repo config to stop lookup before a global preset, or pass `--preset none` to override both layers.
This keeps the layers' own settings; it does not reset them to defaults.
An empty string is refused.
The name `none` is reserved, so `[presets.none]` is refused.
The TUI and web config pages edit the selected preset in place: with `agent6.preset` set to a name, a save, an unset or an entry change writes `[presets.<name>]` in the chosen layer; with `agent6.preset = "none"`, the global and repo config.
**Preset…** opens the `agent6.preset` setting, **Add preset…** creates a table and selects it, and **Remove preset** removes the selected table and resets `agent6.preset`.

| Preset | For | Sets |
|---|---|---|
| `no_review` | turns off the review panel | `review.when = "never"` |
| `review` | three review seats that can block the finish | `review.when = "finish"`, `review.block_on = "any"`, `review.reviewers = [{ focus = "security" }, { focus = "correctness" }, { focus = "tests" }]`, `review.concurrency = 3` |
| `deep_review` | five review seats that can block the finish | `review.when = "finish"`, `review.block_on = "any"`, `review.reviewers = [{ focus = "security" }, { focus = "correctness" }, { focus = "tests" }, { focus = "edge-cases" }, { focus = "over-engineering" }]`, `review.concurrency = 5` |

Define your own with a `[presets.<name>]` table (a partial config); a built-in's name replaces that built-in wholesale:

```toml
[agent6]
preset = "myteam"

[presets.myteam.review]
when = "finish"
block_on = "any"
reviewers = [
  { focus = "security", model = "anthropic/claude-opus-4-8" },
  { focus = "correctness", model = "openrouter/moonshotai/kimi-k2", effort = "high" },
]
```

The first write under a built-in's name (`agent6 config set presets.review.sandbox.network host`, or a page edit with `agent6.preset = "review"`) copies the built-in's values into the table first, so the rest of it stays.

## `[providers.<name>]`

One backend per block; `<name>` is referenced from `[models.<role>]`.
Provider names must match `[A-Za-z0-9_-]+`: one or more ASCII letters, digits, hyphens or underscores.

Three orthogonal choices describe an HTTP backend: **`backend`** (the wire dialect), **`deployment`** (URL/placement quirks of where it is hosted), and **auth** (`auth_style` + `api_key_env`, `token_command` or `oauth`).
A minimal block is just `backend` (plus `base_url` for a non-default host).

| Field | Default | Meaning |
|---|---|---|
| `providers` | `{}` | provider endpoints by name (`[providers.<name>]`), referenced by `[models.*]` roles; names: one or more ASCII letters, digits, `-` or `_`; written by `agent6 connect` |
| `<name>.backend` | *(required)* | Provider backend; Chat Completions supports OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp and Gemini's OpenAI endpoint; ChatGPT subscriptions use the Codex Responses backend; configured Responses endpoints require `base_url`; Claude Code uses a Claude subscription without an HTTP endpoint or key<ul><li><code>anthropic</code>: Use Anthropic Messages API</li><li><code>openai</code>: Use OpenAI Chat Completions API</li><li><code>chatgpt</code>: Use ChatGPT subscription API</li><li><code>responses</code>: Use OpenAI Responses API</li><li><code>claude_code</code>: Use installed signed-in Claude Code binary</li></ul> |
| `<name>.deployment` | `"direct"` | Provider URL shape and placement of model name and API version; Azure supports only the OpenAI format<ul><li><code>direct</code>: Connect directly to provider</li><li><code>vertex</code>: Route through Google Vertex AI</li><li><code>azure</code>: Route through Microsoft Azure</li></ul> |
| `<name>.base_url` | per (format, deployment) | provider API endpoint's host and path prefix (`https://api.anthropic.com/v1`); required for `deployment = "vertex"` or `"azure"` and for `backend = "responses"`; a ChatGPT sign-in also dials its fixed OAuth authority, and a configured `oauth` table its issuer and exchange endpoints |
| `<name>.auth_style` | per (format, deployment) | Authentication header style; resolved per format and deployment when omitted; Anthropic uses x-api-key, compatible APIs use bearer, and Azure uses api-key<ul><li><code>x_api_key</code>: Send x-api-key authentication header</li><li><code>bearer</code>: Send Authorization Bearer header</li><li><code>api_key_header</code>: Send api-key authentication header</li><li><code>none</code>: Send no authentication header</li></ul> |
| `<name>.api_key_env` | none | environment variable holding the API key; precedence over `secrets.toml`; unset for a key stored by `agent6 connect` or an unauthenticated local endpoint |
| `<name>.token_command` | `[]` | argv command printing a short-lived bearer token to stdout; re-run after `token_command_ttl_s` and once after a `401` or `403`; precedence over `api_key_env` |
| `<name>.token_command_ttl_s` | `300.0` | Seconds a `token_command` token is reused before the command runs again. |
| `<name>.extra_headers` | `{}` | extra HTTP headers: a literal string rides every API request; a template table (`template`, `targets`, `when`) renders from request facts for the request classes it names; no secrets, config file not `0600` |
| `<name>.extra_body` | `{}` | provider-specific JSON merged last into every request body; precedence for tuning keys (`max_tokens`, `temperature`); agent6-owned structural keys filtered out (messages, model, stream, tools, tool choice, response shape); JSON-shaped values only, non-finite floats and TOML dates and times refused; OpenRouter routing options here |
| `<name>.extra_query` | `{}` | Extra URL query parameters on every request (Azure's `api-version`). |
| `<name>.oauth` | none | a configured OAuth issuer and public-client registration: `agent6 connect <name>` signs in there and stores the grant in `secrets.toml`; requires `auth_style = "bearer"`, replaces `api_key_env` and `token_command`; a stored grant is bound to these settings and the entry's transport settings |
| `<name>.http_timeout_s` | `600.0` | Seconds one HTTP call may take to read or write; the connect phase is capped at 20 s. |
| `<name>.prompt_caching` | `true` | prompt caching for `backend = "anthropic"` only; system prompt, tools and growing conversation re-read at 0.1x the input price |
| `<name>.binary` | `"claude"` | executable for `backend = "claude_code"` only; a name on PATH or an absolute path |

### Request headers

`[providers.<name>.extra_headers]` is the one place extra HTTP headers come from.
A literal string (`"x-title" = "agent6"`) rides every API request: model calls, the model listing and the plan reading.
A template table renders from request facts for the request classes it names.
Only a template whose `targets` name `oauth` or `exchange` reaches a configured issuer's credential endpoint.

The facts: `request_id` (fresh per HTTP attempt) and `agent6_version` on every request; `model`, `conversation_id` (an opaque id per role and provider or per review panel and seat, kept across a resume), `initiator` (`user` or `agent`), `is_first_turn`, `is_subagent`, `has_tool_results` and `has_images` on a model call.
`{{` and `}}` are literal braces; a placeholder holds a fact name and nothing else.

The transport owns `authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `cookie`, `host`, `connection`, `content-length`, `transfer-encoding`, `content-type` and `accept`; naming one refuses at config load, as does a control character, two spellings of one header, or a template its target cannot render.

```toml
[providers.work.extra_headers]
"x-title" = "agent6"
"user-agent" = { template = "agent6/{agent6_version}", targets = ["inference", "models"] }
"x-initiator" = { template = "{initiator}" }
"x-vision" = { template = "true", when = "has_images" }
```

| Field | Default | Meaning |
|---|---|---|
| `template` | *(required)* | literal text with `{fact}` placeholders (`{{` and `}}` are literal braces); the facts: `request_id`, `agent6_version` on every request, and on a model call `model`, `conversation_id`, `initiator`, `is_first_turn`, `is_subagent`, `has_tool_results`, `has_images` |
| `targets` | `["inference"]` | Request classes the header reaches; OAuth and exchange targets require an `oauth` setting on the entry<ul><li><code>exchange</code>: Configured second token exchange</li><li><code>inference</code>: Model inference requests</li><li><code>models</code>: Model catalog listing requests</li><li><code>oauth</code>: Device, token and revocation requests</li><li><code>usage</code>: Subscription plan usage requests</li></ul> |
| `when` | none | a Boolean fact; the header is sent only while it is true (model calls only)<ul><li><code>has_images</code>: Request includes image content</li><li><code>has_tool_results</code>: Request includes tool results</li><li><code>is_first_turn</code>: Request starts the first turn</li><li><code>is_subagent</code>: Request comes from a subagent</li></ul> |

### Configured OAuth

`[providers.<name>.oauth]` names an issuer and the public-client registration the operator made there; `agent6 connect <name>` signs in with authorization code and PKCE (`authorization_endpoint`, a loopback redirect or a pasted callback) or the device grant (`device_authorization_endpoint`), and stores the grant in `secrets.toml`.
Every endpoint is https, or plain http on `127.0.0.1` or `[::1]` for a local service; nothing is discovered, no client secret exists, and no redirect is followed.
A stored grant is bound to the entry's transport settings (`backend`, `deployment`, `base_url`, `auth_style`, `extra_query`, `extra_headers`) and its issuer settings: change any of them and the next run refuses until `agent6 connect <name>` runs again.
`[providers.<name>.oauth.exchange]` adds one second exchange when the model endpoint takes a token other than the issuer's: one `GET` or `POST` with the issuer's bearer, whose named top-level fields are the token and its expiry.

| Field | Default | Meaning |
|---|---|---|
| `issuer` | *(required)* | the issuer identifier (https, or http on a loopback literal for a local service); names the sign-in and binds the stored grant; never fetched (no discovery) |
| `client_id` | *(required)* | the public-client registration the operator made at the issuer |
| `scopes` | `[]` | the scopes requested, joined with spaces; none are implied |
| `authorization_endpoint` | none | enables authorization code with PKCE: the URL the browser opens |
| `device_authorization_endpoint` | none | enables the device grant (RFC 8628): the URL that mints the user code |
| `token_endpoint` | *(required)* | the URL for the code exchange, the device polling and the refresh |
| `revocation_endpoint` | none | the URL `agent6 connect <name> --logout` revokes the grant at, when set |
| `redirect_uri` | `"http://127.0.0.1:0/auth/callback"` | the loopback redirect the registration names; port 0 takes a free port before the authorization starts; `http://127.0.0.1` or `http://[::1]`, a path, no query |
| `authorization_params` | `{}` | static, non-secret authorization parameters (an audience, a prompt); never `client_id`, `state`, `redirect_uri`, `response_type`, `scope` or a PKCE parameter |
| `request_encoding` | `"form"` | how the device and token requests are encoded<ul><li><code>form</code>: Encode request as form data</li><li><code>json</code>: Encode request as JSON</li></ul> |
| `refresh_skew_s` | `60.0` | seconds before an expiry the grant is refreshed, capped at half the issued lifetime |
| `exchange` | none | one second exchange of the issuer's bearer for the model endpoint's token |
| `exchange.endpoint` | *(required)* | the exact URL the issuer's bearer is sent to for the model endpoint's token |
| `exchange.method` | `"POST"` | the request method; `GET` sends `parameters` as the query<ul><li><code>GET</code>: Send HTTP GET request</li><li><code>POST</code>: Send HTTP POST request</li></ul> |
| `exchange.request_encoding` | `"form"` | how a `POST` body is encoded<ul><li><code>form</code>: Encode request as form data</li><li><code>json</code>: Encode request as JSON</li></ul> |
| `exchange.parameters` | `{}` | static, non-secret request parameters (query for `GET`, body for `POST`) |
| `exchange.access_token_field` | `"access_token"` | the top-level JSON field holding the token the model endpoint takes |
| `exchange.expires_field` | `"expires_in"` | the top-level JSON field holding that token's expiry; required in the answer |
| `exchange.expires_format` | `"relative_seconds"` | whether `expires_field` is a lifetime in seconds or a Unix time<ul><li><code>relative_seconds</code>: Seconds until token expires</li><li><code>unix_seconds</code>: Absolute Unix expiry timestamp</li></ul> |

```toml
[providers.work]
backend = "openai"
base_url = "https://models.example/v1"

[providers.work.oauth]
issuer = "https://login.example"
client_id = "operator-registered-public-client"
scopes = ["model.invoke"]
authorization_endpoint = "https://login.example/authorize"
device_authorization_endpoint = "https://login.example/device"
token_endpoint = "https://login.example/token"
revocation_endpoint = "https://login.example/revoke"
redirect_uri = "http://127.0.0.1:8765/auth/callback"
```

### Deployments

```toml
# Anthropic direct (default)
[providers.anthropic]
backend = "anthropic"

# Gemini on Vertex (OpenAI-compatible endpoint)
[providers.vertex-gemini]
backend = "openai"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/endpoints/openapi"
token_command = ["gcloud", "auth", "print-access-token"]

# Claude on Vertex (model in the URL, anthropic_version in the body: handled
# by deployment = "vertex")
[providers.vertex-claude]
backend = "anthropic"
deployment = "vertex"
base_url = "https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJ/locations/LOCATION/publishers/anthropic/models"
token_command = ["gcloud", "auth", "print-access-token"]

# Azure OpenAI (the model id is the deployment name; api-version required)
[providers.azure]
backend = "openai"
deployment = "azure"
base_url = "https://RESOURCE.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
extra_query = { "api-version" = "2024-06-01" }
```

### Responses API endpoint (`backend = "responses"`)

The Responses API on an endpoint the operator names: `/responses` is appended to `base_url`, which has no vendor default, and `direct` is the only deployment.
A key (`agent6 connect <name>`), a `token_command` or a configured [`oauth`](#configured-oauth) issuer authenticates it; usage meters tokens like any keyed endpoint, priced from the catalog when the model is known.

```toml
[providers.work]
backend = "responses"
base_url = "https://models.example/v1"

[models.default]
model = "work/reasoning-model"
```

- The system prompt goes as `instructions`, tools as flat function tools, tool results as `function_call_output`, the output cap as `max_output_tokens`; `store = false` always, and encrypted reasoning items are replayed to this endpoint and model only.
- The role's `effort` goes as `reasoning.effort` (`off` sends `none`; unset sends nothing); a `temperature` the endpoint refuses is dropped for the rest of the run.
- `extra_body` never sets `store`, `previous_response_id`, `conversation`, `background`, `max_output_tokens`, `parallel_tool_calls`, `text`, `response_format` or the structural fields.
- Streaming runs when a surface watches the call; a JSON response serves the rest.

### ChatGPT subscription (`backend = "chatgpt"`)

Uses a ChatGPT plan (Plus/Pro/Business) instead of an API key, over the Codex Responses backend.
The `session-id` header and `prompt_cache_key` carry the [shared conversation id](#request-headers).
Calls outside a conversation omit both.

```bash
agent6 connect chatgpt      # browser sign-in (device code, then paste, when headless)
agent6 model default chatgpt/gpt-5.6-sol
```

- `agent6 connect chatgpt` runs a PKCE OAuth sign-in against OpenAI's fixed OAuth authority (`https://auth.openai.com`) and stores the tokens in `secrets.toml` (0600); they refresh automatically.
- Past the included window, calls draw on purchased credits (real money; auto top-up can buy more).
  `[budget].allow_plan_overage = false` (the default) is a circuit breaker: once a window is exhausted with credits present the run stops at its next boundary; a call already in flight completes, so a boundary-crossing call can spend before the stop.
- Whether these conversations train OpenAI's models follows the ChatGPT account's own data controls (Settings > Data controls > "Improve the model for everyone"); agent6 cannot change that setting.
  agent6 never calls the feedback/rating endpoints, which would opt the rated turns into training regardless of it; there is no rating surface.
- Model names complete from the backend's own listing for the signed-in plan (fetched like other providers' catalogs), and its context windows size compaction.
- An `unsupported_value` refusal naming `reasoning.effort` and listing accepted levels, over HTTP 400 or a stream error frame, silently retries once at the lowest listed effort.
  Later calls on that provider instance keep the accepted levels; they are not persisted.
- `agent6 connect chatgpt --logout` signs out: the grant is revoked at the OAuth authority (best effort) and the tokens leave `secrets.toml`.
- Spend is plan-metered, not dollar-metered: every response carries the account's rate-limit windows.
  `[budget].max_plan_points` caps the plan movement observed during one run (`--max-plan-points` per run).
  Dollar figures stay an authoritative $0.

### Claude Code subscription (`backend = "claude_code"`)

Runs model calls inside the installed, signed-in Claude Code binary: a Claude subscription (Pro/Max) instead of an API key.

```bash
claude auth login                        # once, in Claude Code itself
agent6 connect claude                    # writes [providers.claude], checks the sign-in
agent6 model default claude/claude-sonnet-4-5
```

- agent6's own loop, tools, jail, verify gate, and approvals drive the run; the binary supplies the model.
  Inside it every Claude Code capability is off: built-in tools, hooks, settings, CLAUDE.md, MCP servers, skills, slash commands, session files, auto-memory, auto-compaction.
- agent6 never reads the login under `~/.claude`; a `CLAUDE_CONFIG_DIR` in the environment selects a relocated one.
- Spend is plan-metered: every round reports the account's 5-hour and 7-day windows.
  `[budget].max_plan_points` caps the plan movement observed during one run (`--max-plan-points` per run).
  Dollar figures stay an authoritative $0; the binary's own list-price estimate is not recorded.
- Ignored: `[models.<role>].temperature` and the loop's per-call output-token cap (the binary owns sampling).
  Refused: `effort = "off"` (`claude --effort` has no off value; use `low`).
- Optional roles inherit the default; override one with `agent6 model helper claude/claude-haiku-4-5`.
  Each side call without tools is one short-lived `claude` process.
- One `claude` process serves a worker execution.
  It restarts, replaying the conversation as one user message with retained attachments, on resume, fork, `/undo`, a steer or stop mid-turn, a tier-2 context restart, and when the live context nears the window.
  Tier-1 text compaction shrinks the model's context at that next restart, not before.
  Removing or changing a retained media result forces a restart.
- Claude Code appends the account email to every system prompt it sends; agent6 replaces it with `<operator-email>` in the model's returned text.
- Text-only tool results are capped at 34,000 bytes of UTF-8 for this provider (60,000 elsewhere): Claude Code writes a result over 50,000 bytes under `~/.claude/projects` and hands the model a 2 KB preview of it, and the turn's notices (a verify tail, a review critique, the nudges) ride in the same payload, with 16,000 bytes kept for them.
- Image and PDF attachments use MCP image or resource items, outside the text cap.
  The provider checks text plus notices against the 50,000-byte guard and each complete encoded stdin line against 8 MiB.
  See [Media retention](security.md#media-retention) for the child process and upstream boundary.
- Use a full model id (`claude-sonnet-4-5`, not `sonnet`) so the context window is known for compaction sizing.
- `agent6 connect claude --logout` is refused: agent6 stores no Claude Code credentials; `claude auth logout` signs out.

### OpenRouter routing and caching (`extra_body`)

OpenRouter's default routing is not deterministic, so prompt caching may or may not engage call-to-call.
Pin it with `extra_body.provider` ([routing docs](https://openrouter.ai/docs/features/provider-routing)):

```toml
[providers.openrouter]
backend = "openai"
base_url = "https://openrouter.ai/api/v1"
extra_body = { provider = { sort = "throughput" } }  # prefer fast backends
# Alternatives: { order = ["DeepInfra"], allow_fallbacks = true }
#               { max_price = { prompt = 1, completion = 2 } }
```

```bash
agent6 config set providers.openrouter.extra_body \
  '{ provider = { sort = "throughput" } }'
```

Watch `cache_r` in the cost summary to confirm caching engages.

### Short-lived bearer tokens (`token_command`)

For endpoints authenticated by a refreshed bearer rather than a static key (Vertex OAuth, OIDC/STS gateways): point `token_command` at anything that prints a current token, e.g. `["gcloud", "auth", "print-access-token"]` or `["az", "account", "get-access-token", "--query", "accessToken", "-o", "tsv"]`.
It runs outside the jail with your environment minus registered provider-key variables and `AGENT6_WEB_SECRET` (operator-only).
Non-zero exit, timeout, or empty output surfaces as a provider error.

## `[models.<role>]`

Role routing:

| Field | Default | Meaning |
|---|---|---|
| `default` | none | required route for every model call; set with `agent6 model default PROVIDER/MODEL`; the only fallback for each role's unset model, effort and temperature |
| `helper` | none | model for the context summariser, gisting (`compaction.elision_gists`), and verify-command inference; unset fields take `models.default` |
| `worker` | none | model for `agent6 run`, `agent6 ask`, their resumes, machine agent states and the model-written squash message; unset fields take `models.default` |
| `reviewer` | none | model for in-loop review panel seats without a route, the `sessions compare` and `--parallel` judge, and `sessions review`; unset fields take `models.default` |
| `planner` | none | model for `agent6 plan` and plan resumes; unset fields take `models.default` |

Every model call requires a `models.default.model` route.
Unset fields in `worker`, `planner`, `reviewer` and `helper` inherit directly from `models.default`.
A role never inherits from another role.
For example, `[models.planner]` with only `effort = "max"` keeps the default route and temperature.
`agent6 config show` attributes inherited leaves to the layer that supplied the default.
A worker-only config refuses with a remedy; rename `[models.worker]` to `[models.default]` to migrate.
A persisted `model` is fully qualified: `model = "openrouter/openai/gpt-4.1-mini"`.
The first segment names a `[providers.*]` entry; everything after the first slash is the model id verbatim.
`effort` and `temperature` are sibling keys.

| Field | Default | Meaning |
|---|---|---|
| `model` | `""` | Fully qualified provider/model route (`agent6 model` lists them); the first segment names a `[providers.<name>]` entry, the rest is the model id verbatim; required in `models.default` for a model call, inherited from it in other roles. |
| `temperature` | `0.0` | sampling temperature, `temperature = 0.0` to `2.0`, sent where the wire accepts one (not with Anthropic thinking or on `chatgpt` or `claude_code`, dropped for the run after a 400 naming it); `extra_body.temperature` wins; TOML omission: inherits `models.default.temperature` in a role, `temperature = 0.0` in the default |
| `effort` | none | Reasoning effort; top tiers where offered by the model, collapsed to Anthropic's highest; unset in a role: inherits `models.default.effort`; unset in the default: `effort = "low"` on openai-compatible reasoning models, no thinking on Anthropic, none sent with `backend = "chatgpt"`, `"responses"` or `"claude_code"` (the model's default), as printed by `agent6 config show`<ul><li><code>off</code>: Disable reasoning effort</li><li><code>low</code>: Use low reasoning effort</li><li><code>medium</code>: Use medium reasoning effort</li><li><code>high</code>: Use high reasoning effort</li><li><code>xhigh</code>: Use extra-high reasoning effort where supported</li><li><code>max</code>: Use maximum reasoning effort where supported</li></ul> |

## `[sandbox]`

The field summary; the security model is in security.md: [Sandbox](security.md#2-sandbox) and [Network](security.md#5-network).
For Go's compiled-package cache, see [Go build cache](cli.md#go-build-cache).

`extra_read_paths`, `extra_write_paths` and `hide_paths` take absolute paths with no `..` segment, and `extra_device_paths` only paths under `/dev`; anything else refuses at config load.

| Field | Default | Meaning |
|---|---|---|
| `isolation` | `"auto"` | Jailed-command confinement; explicit full or partial isolation refuses unsupported hosts; `--disable-sandbox` or `AGENT6_DISABLE_SANDBOX=1` disables confinement; `container` runs every command inside the `[container]` selection, where the runtime is the boundary and agent6 adds no confinement; filling `[container]` alone changes nothing.<ul><li><code>auto</code>: Strongest supported isolation, warning below full</li><li><code>full</code>: User and mount namespaces, seccomp and supported Landlock</li><li><code>partial</code>: Landlock and seccomp without namespaces</li><li><code>none</code>: Run commands without confinement</li><li><code>container</code>: Run commands inside the selected container</li></ul> |
| `network` | `"auto"` | Jailed-command network; session networking refuses unenforceable isolation; commands share one launcher and can reach each other, with no per-command network isolation; machine-state overrides in its `config` table cannot widen this setting.<ul><li><code>auto</code>: Session under full isolation, otherwise host with warning</li><li><code>session</code>: Run-private network without outside access</li><li><code>host</code>: Use the machine network</li></ul> |
| `run_commands` | `"ask"` | Permission for `run_command`, `run_verify_command`, `run_metric_command` and `stop_background`; asking requires someone to answer and permits session-wide answers; allow is clamped to ask in `ask` and `plan` modes; `--auto-approve` never overrides configured deny; `--no-commands` withholds commands.<ul><li><code>allow</code>: Allow command tools and verification</li><li><code>deny</code>: Withhold command tools and verification</li><li><code>ask</code>: Ask permission before each call</li></ul> |
| `fetch_approved_hosts` | `[]` | hosts readable by `fetch` without approval; other hosts: prompt, refusal without an operator; empty: every fetch prompts; `sandbox.fetch_approved_hosts = ["*"]`: any host; leading dot: subdomains allowed (`.readthedocs.io`); host entries only, never URL prefixes; fixed fetch limits: https only, 1 MiB, redirects returned without following; tool hidden when jailed commands have host networking (`sandbox.network = "host"` or `sandbox.isolation` other than `"full"`); tool withheld from read-only machine agent states (`mode = "read_only"`) |
| `protect_git` | `"auto"` | Protect `.git/` against jailed commands planting filters for host-side commits; explicit protection requires full isolation and refuses model-owned git; automatic protection warns below full isolation; in-process git writes are refused unless protection is off; a linked worktree's recorded repository git directory is always granted read-only; write grants inside it refuse startup.<ul><li><code>auto</code>: Protect under full isolation unless git is model-owned</li><li><code>on</code>: Require read-only git directory bind-remount</li><li><code>off</code>: Leave git directory writable</li></ul> |
| `home` | `"auto"` | HOME for jailed commands; temporary home uses `/tmp/agent6-home` in private tmpfs and requires full isolation; cached home uses `$XDG_CACHE_HOME/agent6/homes/<repo-id>`, created `0700`, refused if loosened, bound read-write at its real path, under no isolation a directory by convention only; host home is the operator's own and refuses under a jail; persistent HOME retains build caches but permits poisoned caches or git aliases to affect later jailed runs in this repo, never other repos or operator tools.<ul><li><code>auto</code>: Each level&#x27;s own: tmp, cache or host</li><li><code>tmp</code>: Private temporary home removed with the run</li><li><code>cache</code>: Persistent private home for this repository</li><li><code>host</code>: The operator&#x27;s own HOME, no isolation only</li></ul> |
| `memory_limit_mib` | `0` (off) | per-process `RLIMIT_DATA` data-segment cap in MiB (1024 * 1024 bytes) for jailed processes and their children; `sandbox.memory_limit_mib = -1`: no cap; explicit cap to bound a task; cap exceeded: ordinary command error |
| `extra_read_paths` | `[]` | absolute paths outside the repo readable and executable at their real locations (toolchains, conda/Go/Rust/Node interpreters, shared data dirs); mounted for jailed commands, readable by in-process tools; wider sandbox, only paths needed by the build |
| `extra_write_paths` | `[]` | absolute paths outside the repo readable and writable at their real locations (build caches, output dirs, sibling checkouts edited by the task); read access included with write access; wider sandbox, only paths written by the task |
| `extra_device_paths` | `[]` | device nodes under `/dev` exposed read-write (GPU compute: `/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm`); default device access only to `/dev/null`, `/dev/zero`, `/dev/urandom`, `/dev/random` and `/dev/full`; existing character or block devices required at run start, otherwise startup refused; wider sandbox, direct hardware access |
| `hide_paths` | `[]` | absolute paths never readable or writable by the run, even under a broader grant; an entry inside the repository stays out of every tree agent6 commits or compares, and is hidden at the same path in a run's own worktree; agent6 config dir and state base always hidden, `secrets.toml` and run history excluded even from `sandbox.extra_read_paths` granting `$HOME`; data dir and cache readable for installed skills; in-process refusal at every isolation level; jailed-command masking: empty dirs and files, mount namespace required; under `sandbox.isolation = "partial"`: run refused for an unmaskable entry, loud warning instead for a grant exposing always-hidden dirs; a grant inside an entry: run refused at every jailed level |

## `[container]`

The container `sandbox.isolation = "container"` runs every command in; the runtime is the boundary and agent6 adds no confinement there.
With `exec` empty, agent6 finds this workspace's running Docker or Podman dev container by its labels ([Commands inside a dev container](cli.md#commands-inside-a-dev-container)).
Filling this table alone changes nothing.

| Field | Default | Meaning |
|---|---|---|
| `exec` | `[]` | argv prefix the launcher invocation is appended to, no shell; empty: find this workspace's running Docker or Podman dev container by its labels; set: run inside a container of your choice, such as `["docker", "exec", "-i", "--user", "vscode", "agent6-dev"]`, `["sudo", "-n", "docker", "exec", "-i", "agent6-dev"]`, `["podman", "exec", "-i", "NAME"]` or `["devcontainer", "exec", "--workspace-folder", "DIR"]`; never rewritten, never given a tty |
| `runtime` | `[]` | argv prefix for the runtime's CLI, used by discovery and by the exec it derives; empty: the `docker`, then `podman`, found on PATH; a host whose docker needs sudo sets `["sudo", "-n", "docker"]` |
| `workspace` | `""` | workspace path inside the container; empty: the discovered bind mount's path, or the host path under `container.exec`; set: that path, which creates no mount |
| `worktree_base_dir` | `".agent6/worktrees"` | base directory, under the project root, for the worktrees agent6 makes while commands run in the container, in place of `git.worktree.base_dir`; kept out of git through the repository's `.git/info/exclude`; an absolute path or one with `..` is refused |
| `start` | `[]` | operator command that starts the container, as argv without a shell; empty: agent6 starts nothing and refuses when no dev container runs; run once when no container runs, before discovery runs again; separate from `git.worktree.setup_command`, which runs inside the prepared workspace |
| `stop` | `[]` | operator command that stops a container the run started, as argv without a shell; runs after the run and every run attached to the container settled their jobs and records; empty: the container is left running; attaching alone grants no stop |

## `[git]`

| Field | Default | Meaning |
|---|---|---|
| `dirty_tree` | `"ask"` | Tracked files' uncommitted changes at run start; asking requires someone to answer, with cancellation parking the run for resume; stashes are restored per `git.auto_stash_pop`, otherwise a printed restore command names the stash by sha and the worktree directory without switching branches; included changes mark an unmerged run merged on landing if its work matches exactly; files already untracked at the start stay out of the run's commits, and files the run creates are its own; `--parallel` refuses asking; a resume applies the same policy to changes since the last execution, the run's own files included, refuses stash if the run also changed a file, and names included files to the model<ul><li><code>ask</code>: Ask whether to stash, include or cancel</li><li><code>stash</code>: Stash tracked changes for the run</li><li><code>include</code>: Include tracked changes in first commit</li></ul> |
| `auto_stash_pop` | `false` | pre-run stash applied over the working tree at run or resume end, even over the run's uncommitted work; worktree sessions restored in their own directory without a branch switch; stash dropped only after a clean apply; failed apply: stash preserved, restore command printed, conflicts visible in `git status`; no `reset --hard`; requires `git.dirty_tree = "stash"` |
| `control` | `"agent6"` | Git ownership during a run; harness control leaves HEAD untouched; model control has no per-turn chain or run branch and refuses `sessions diff`/`merge`, `/undo` and `fork`; automatic git protection is disabled and explicit protection refused.<ul><li><code>agent6</code>: Harness owns per-turn commits and run branch</li><li><code>model</code>: Model owns commits and branches</li></ul> |
| `branch_per_run` | `true` | visible `agent6/<run-id>` branch advanced to the run's chain tip; `git.branch_per_run = false`: only the hidden `refs/agent6/<run-id>/head` ref; forced on for `--parallel` lanes, whose work is imported by branch |
| `commit_per_turn` | `true` | each editing turn committed to the run's detached chain through a temp index; HEAD never moved; index and working tree touched only with the run's own branch checked out; `git.commit_per_turn = false`: no agent6 commits, work only in the worktree, degraded resume-from-git and `sessions diff`/`merge`; `/parallel` dispatch from a changed tree refused; machine run states require true because only the committed chain leaves a clone |
| `merge_method` | `"tasks"` | Method for `agent6 sessions merge` to land a run on its base; a task's commit follows `git.commit.squash_message` under the `model` style with the subtask's title and acceptance as its body, else the subtask's title; checkpoints outside subtasks follow `git.commit.squash_message`, so a run that never decomposed lands as one commit; consolidation only, per-turn commits stay on the run's chain; `sessions merge -m` lands one commit with that message<ul><li><code>tasks</code>: Squash consecutive checkpoints by task</li><li><code>squash</code>: Land as one commit</li><li><code>merge</code>: Merge while preserving per-turn history</li><li><code>ff_only</code>: Require a fast-forward merge</li></ul> |
| `auto_merge` | `false` | automatic merge into the base branch only for end reason `finish_session`, or `lanes_ranked` for a `run --parallel` run that joined its best-ranked lane, with a passed or absent verify gate; other ends: no auto-merge, reason printed; hidden chain ref merged with `git.branch_per_run = false`; conflict: nothing moved, instructions printed |
| `auto_prune` | `false` | run branch deleted after auto-merge only when `git branch -d` allows it (`git.merge_method = "merge"` or `"ff_only"`); squash-merged branch: `-D` command printed, never force-deleted; requires `git.auto_merge = true`; no action without a run branch |
| `run_repo_hooks` | `false` | repo's `.git/hooks/*` during agent6 git operations; `git.run_repo_hooks = false`: hooks skipped to prevent repo-controlled code running on the host; `core.fsmonitor` and `diff.external` always neutralized |
| `run_repo_drivers` | `false` | repo's filter and merge drivers (`filter.<name>.clean/smudge/process`, `merge.<name>.driver`) during agent6 git operations; `git.run_repo_drivers = false`: each driver neutralized by name to prevent repo-controlled `.git/config` code running on the host at every commit; `git.run_repo_drivers = true` required for Git LFS clean/smudge filters |

`agent6 sessions merge --method <method>` overrides `git.merge_method` for that merge.

See [Git](security.md#7-git) for where `sandbox.protect_git` enforces `git.control = "agent6"` and which settings warn.

### `[git.commit]`

| Field | Default | Meaning |
|---|---|---|
| `name` | none | author and committer name for agent6 commits; also passed as `GIT_AUTHOR_NAME` and `GIT_COMMITTER_NAME` to jailed `git commit` under `git.control = "model"`; `GIT_COMMITTER_NAME`, then the repo's own `git config` identity, when unset; startup refused without a resolvable identity |
| `email` | none | author and committer email for agent6 commits; also passed as `GIT_AUTHOR_EMAIL` and `GIT_COMMITTER_EMAIL` to jailed `git commit` under `git.control = "model"`; the `GIT_COMMITTER_*` variables, then the repo's own `git config` identity, when unset; startup refused without a resolvable identity |
| `trailer` | `""` | git trailer line (`Key: value`) on every agent6 commit, such as `"Assisted-by: agent6:{model}"` or `"Co-authored-by: agent6:{model} <noreply@agent6.dev>"`; `{model}`: the code-writing model, several joined with `, `; no trailer when empty |
| `checkpoint_message` | `"agent6"` | Per-turn commit message style; generated messages use `agent6 turn N: <summary>` or `type(scope): subject`; model messages use git facts and fall back to the generated agent6 message with a warning on failure<ul><li><code>agent6</code>: Generate turn number and summary message</li><li><code>conventional</code>: Derive conventional subject from diff without model</li><li><code>model</code>: Ask model to write commit message</li></ul> |
| `squash_message` | `"agent6"` | Squash commit message style; generated messages use the task's first clause, then `continued: <steer>` for a run resumed with `--steer`, with the finish summary as body or commit subjects as bullets; conventional subjects use `type(scope): subject` with the same body; model messages use up to 64 KiB of base-to-tip diff (marked if truncated), the task, continuation and execution finish summaries, and AGENTS.md commit conventions; failure warns and falls back to the generated agent6 message<ul><li><code>agent6</code>: Generate message from task and run summary</li><li><code>conventional</code>: Derive conventional subject from diff without model</li><li><code>combine</code>: Concatenate the git commit log</li><li><code>model</code>: Ask model to write commit message</li></ul> |

### `[git.worktree]`

Placement and setup of the worktrees agent6 makes: lane and machine-state clones, fork and `run --worktree` linked worktrees.

| Field | Default | Meaning |
|---|---|---|
| `base_dir` | `""` | base directory for the worktrees agent6 makes, grouped by repository: lane and machine-state clones, fork and `run --worktree` worktrees; empty: `<cache_dir>/worktrees`; lane clone removed after work import; fork and `run --worktree` worktrees removed by `sessions prune` once merged or holding no work |
| `kind` | `"clone"` | Worktree a fork or a `run --worktree` run works in; a clone runs at any path, in any container, and agent6 copies its commits into the repository as the run makes them; a linked worktree is smaller for a large repository whose base is on another filesystem, commits straight into the repository, and under a container needs the workspace mounted at its host path; lanes and review seats are clones.<ul><li><code>clone</code>: Clone of the repository with its own git data</li><li><code>linked</code>: Linked git worktree sharing the repository&#x27;s git data</li></ul> |
| `setup_command` | `[]` | environment-setup command for a new worktree as argv, no shell (pipeline form: `["sh", "-c", "a && b"]`); `run --worktree` and fork worktrees, lane and machine-state clones: tracked files only, no untracked `.venv` or `node_modules`; command run once in each worktree before the first turn, under the run's command sandbox (same isolation, network and grants); nonzero exit or 600-second timeout: session ended before any turn; unset: AGENTS.md `## Setup command` fenced block, parsed like `## Verify command`, command printed; no configured or inferred command: nothing run |

## `[harness]`

| Field | Default | Meaning |
|---|---|---|
| `max_turns` | `200` | loop turns per execution before stop reason `max_turns`; `harness.max_turns = -1`: unlimited; fresh allowance on resume |
| `went_quiet_max_nudges` | `4` | Empty turns (no text, no tool call) re-asked per streak, reasoning-starvation bursts included; 0 ends the run on the first. |
| `loop_guard_kill_threshold` | `10` | The same (tool, args) call this many times in a row ends the run as `loop_guard_killed` (the notice fires from three, every other turn); `harness.loop_guard_kill_threshold = -1`: the notice alone, no end. |
| `settled_stop_after` | `12` | consecutive turns with no edit, commit or memory write and an unchanged tree before a plain run ends after a green verify or an editing step; verify runs and waits on a live background command neutral; streak reset by progress or operator steer; default 12, measured; `harness.settled_stop_after = -1`: stop disabled, nudge at three idle turns retained |
| `stagnation_notice_after_s` | `300.0` | wall-clock seconds without the run's own edit or verify before one notice (recall spirals: few calls, long reasoning); `harness.stagnation_notice_after_s = 0`: disabled; clock from the first turn after the start gate, excluding operator wait time |
| `memory_advisory` | `true` | one notice per run at the first red-to-green verify with no repo-memory write, inviting a durable repo fact; cost about two turns on a short fix; `harness.memory_advisory = false`: no notice, memory still readable and writable by the model |
| `command_checkin_s` | `900.0` | seconds before the model's `run_command` returns as a background job; no timeout or kill, command still running; response: `returncode: null`, `still_running: true`, `background_id`; model options: wait with `read_background`, stop the job or carry on; `harness.command_checkin_s = 0`: hand-back disabled, command killed at 600 s; also killed at 600 s without background tools (such as in `ask`) |
| `standing_patience` | `-1` | consecutive fruitless standing-goal re-entries (no executed tool call) before soft ends are honoured; `harness.standing_patience = -1`: no end from fruitless rounds alone, only budget, turn cap or operator stop; `harness.standing_patience = 0`: end on the first fruitless round; `harness.standing_patience = N`: escalating nudges for N fruitless re-entries, then soft ends honoured; streak reset by a round landing work |
| `metric` | none | optional score beside the pass/fail gate (benchmark, size or count); run after every committed edit, trend shown to the model; unset: `run_metric_command` still listed, calls refused with `harness.metric` named |

### `[harness.metric]` (optional)

A continuous score for measurable goals; `command` runs in the jail like `harness.verify.command`.
Set its required fields together: `agent6 config set --repo harness.metric '{ command = ["python3", "bench.py"], pattern = "score: ([0-9.]+)", goal = "minimize" }'`.

| Field | Default | Meaning |
|---|---|---|
| `command` | *(required)* | score-printing command as argv, no shell; run after every committed edit and on the model's `run_metric_command` call |
| `pattern` | *(required)* | A regular expression over the command's output; its first capture group is the number, e.g. `"score: ([0-9.]+)"`. |
| `goal` | *(required)* | Score direction; trajectory is reported; finish is allowed once a verified edit only ties the best.<ul><li><code>minimize</code>: Smaller scores are better</li><li><code>maximize</code>: Larger scores are better</li></ul> |

### `[harness.parallel]`

Lane limit for `run --parallel N|selector,...`.
Each lane is a disposable clone running its own `agent6/<id>` branch; lanes are imported and ranked, and the first-ranked lane joins the run's own branch.
`harness.parallel.judges` names who ranks them: each judge ranks once, and the lowest sum of ranks wins.
`git.auto_merge = true` then lands the run as it lands any run; without it, `agent6 sessions merge <run id>` does.
`--max-usd` caps the lanes and the judges together; `--auto-approve` forwards to every lane.
A live run dispatches lanes the same way via `/parallel [SPEC --] TASK` ([grammar and examples](usage.md#parallel-directive)).
`harness.parallel.tool = true` offers a run's model the `parallel` tool, which dispatches a task through the same path ([the parallel tool](usage.md#the-parallel-tool)); it changes nothing for `run --parallel`, the steer or the review panel.
A subordinate model or steer cannot dispatch work; its configured harness panel may create bounded read-only leaves.
Headless surfaces without a dispatcher answer "not available".

| Field | Default | Meaning |
|---|---|---|
| `tool` | `false` | offer a run's model the `parallel` tool, which dispatches a task to isolated lanes (fresh clones of the committed tree, each an ordinary run that may edit) and joins their commits back in order; each lane runs under 200 turns, 30 minutes and the run's remaining budget, which the lanes share; `harness.parallel.tool = false`: no run is offered it; neither value changes `run --parallel`, the `/parallel` steer or the review panel |
| `max_lanes` | `4` | The most lanes one `--parallel` fan-out may run, `1` to `1024`; a spec asking for more is refused before anything is cloned; a review roster longer than this runs in batches of this width, with no seat dropped. |
| `judges` | `["reviewer"]` | who ranks the candidates of a `run --parallel` fan-out and of `sessions compare`: selectors as `--parallel` takes them, a role name or `[provider/]model`; each judges once and the lowest sum of ranks wins, a tie going to the mechanical order (a passed gate first, then the lower cost); a role with no model route judges nothing; `[]`: the mechanical order alone |

### `[harness.prompt]`

| Field | Default | Meaning |
|---|---|---|
| `system_prompt_file` | `""` | file path replacing run mode's built-in base system prompt; dynamic blocks still appended; tool contracts supplied by the operator; startup warning for missing core tool names; empty: built-in base; assembled prompt, tool definitions and first message shown by `agent6 prompt show` |
| `decompose` | `"auto"` | Ordered DAG subtasks worked one at a time before editing in run mode; required decomposition helps small models under-finishing multi-part tasks (measured on mistral-small), with 2-4x overhead for capable models; `config show` prints the resolved value; `--decompose` forces it for one run.<ul><li><code>auto</code>: Choose from worker model capability registry</li><li><code>on</code>: Require ordered task decomposition before editing</li><li><code>off</code>: Do not require task decomposition</li></ul> |

### `[harness.verify]`

| Field | Default | Meaning |
|---|---|---|
| `command` | `[]` | step-success command as argv, no shell (pipeline form: `["sh", "-c", "a && b"]`); explicit value: pinned gate; unset: inferred and printed per run; inference order: AGENTS.md `## Verify command` block or inline `Verify:` line, root `verify.sh`, repo manifests and loose `test_*.py` files, then a model call over those manifests; AGENTS.md commands joined with `&&`, one per line, no multiline `for` or `if`; commands with a first word absent from sandbox PATH skipped and named; files under `sandbox.hide_paths` treated as absent; a plain-command package.json test script run directly when its runner is on sandbox PATH but `npm` is not; no inferred gate: gateless start, adoption of the first gate from a recognizable project created mid-run; `$ ` and `> ` prompts stripped; in a fence containing a `$ ` prompt or opening with a `> ` prompt: only prompted lines and their continuations treated as commands |
| `infer` | `true` | verify-command inference when `harness.verify.command` is unset (AGENTS.md fence, repo signals, model call); mid-run adoption for a gateless run creating a recognizable project; adopted gate dropped and never re-adopted if unable to run (exit 127 or missing module named by `-m`); `harness.verify.infer = false`: no inference or adoption, run stays gateless; explicit `harness.verify.command` unaffected |
| `timeout_s` | `600.0` | seconds per `harness.verify.command` or `harness.metric.command` call before kill and failure; run ending over a gate killed here: `finished · gate timed out`; pathless pytest gate over budget (harness call or model's `run_verify_command`): retry on test files nearest the run's diff, harness gates scoped until a full gate passes; `finish_session` over scoped green: `passed · scoped gate`; model's `run_command` handed back by `harness.command_checkin_s`; cap on each gate call, not cached work; `go test` may pass from a cache filled by the model's `run_command`; Go test cache disabled by `go test -count=1` |
| `when` | `"never"` | When the harness runs `harness.verify.command` itself; the tool remains available in every mode; no gate runs without a command; an uncovered final tree reads unverified; finish-time failures return to the model up to `harness.verify.retries` times.<ul><li><code>finish</code>: Verify uncovered final tree at finish</li><li><code>turn</code>: Also verify after each editing turn</li><li><code>never</code>: Verify only when model calls tool</li></ul> |
| `at_start` | `false` | one `harness.verify.command` call before a fresh run's first turn to record pre-existing gate failure; red output in the first request; a red finish over a red base: `finished · gate red at base` (red at the start and at the finish, the failures not compared); cost: one gate call per fresh run, worth it for a fast gate beside a suite that is already red; needs `harness.verify.when = "finish"` or `"turn"` and `sandbox.run_commands` other than `"deny"`; recorded baseline kept on resume |
| `retries` | `2` | red finish certifications returned to the model with gate output before the finish stands as `finished · gate red`, never passed; `harness.verify.retries = 0`: end on the first red; pre-existing red not returned unless made green during this run |

## `[review]`

| Field | Default | Meaning |
|---|---|---|
| `when` | `"never"` | In-loop panel over the diff so far, with findings sent to the model; only finish-time review can gate finish; periodic review uses `review.interval_turns`; without `review.reviewers`, one seat uses `[models.reviewer]`; each seat runs as a read-only lane with its own session.<ul><li><code>never</code>: Do not run in-loop review</li><li><code>verify_fail</code>: Review after each failed verification</li><li><code>finish</code>: Review when model requests finish</li><li><code>periodic</code>: Review every configured interval of turns</li></ul> |
| `interval_turns` | `10` | Turns between panels when `when = "periodic"`. |
| `block_on` | `"none"` | Effect of panel blocking verdicts; quorum uses `review.min_blocking_models` distinct provider/model routes; abstentions do not block; gating applies only to finish-time review; panels at other moments are advisory.<ul><li><code>none</code>: Treat blocking verdicts as guidance</li><li><code>any</code>: One blocking seat rejects finish</li><li><code>model_quorum</code>: Require configured number of distinct blocking model routes</li><li><code>all</code>: Require every seat to block</li></ul> |
| `min_blocking_models` | `2` | distinct provider/model routes required to block with `review.block_on = "model_quorum"`; two seats on one route counted once, even with different focus or effort |
| `max_rejections` | `4` | ends rejectable by a gating panel per run (`finish_session`, silent finish, settled or plateau stop); count preserved across resumes, never reset; at the cap: panels and findings retained, every end accepted, no indefinite panel stall |
| `min_remaining_budget_fraction` | `0.25` | remaining-budget fraction below which the panel is skipped and the finish accepted; `review.min_remaining_budget_fraction = 0.25`: no panel in the last quarter |
| `reviewers` | `[]` | ordered panel roster of tables with `focus` and optional `model`, `temperature` and `effort`; every omitted field inherits the resolved `[models.reviewer]`, `model` pinned or not; `model` is a provider/model route; empty roster: one `security` seat unless `review.when = "never"` |
| `concurrency` | `1` | simultaneous panel seats, at most `harness.parallel.max_lanes`; a longer roster runs in batches, in roster order; `review.concurrency = 1`: one after another |
| `max_turns` | `20` | model calls a review seat may make, calls that ask for tools included; no verdict within the cap: abstention, reported by the panel |

The [TUI](terminal.md#configuration) and [web](web.md#pages) config pages edit the ordered roster with **Add reviewer…** and positional removal.
Each edit writes the whole list to the global or repo config, including `[]` after the last removal.

Each `reviewers` entry has these fields:

| Field | Default | Meaning |
|---|---|---|
| `focus` | *(required)* | A short review stance. |
| `model` | none | A provider/model route; inherits the reviewer role when unset. |
| `temperature` | none | sampling temperature; route's value when unset; `temperature = 0.0` an explicit override |
| `effort` | none | reasoning effort; route's value when unset; `effort = "off"` an explicit override<ul><li><code>off</code>: Disable reasoning effort</li><li><code>low</code>: Use low reasoning effort</li><li><code>medium</code>: Use medium reasoning effort</li><li><code>high</code>: Use high reasoning effort</li><li><code>xhigh</code>: Use extra-high reasoning effort where supported</li><li><code>max</code>: Use maximum reasoning effort where supported</li></ul> |

Set `model = "provider/model"` to pin a route; omit it to inherit the resolved reviewer role.
Every omitted field inherits the resolved reviewer role's value, `model` pinned or not.
Explicit `effort = "off"` and `temperature = 0.0` override inherited values.
`model_quorum` requires at least `min_blocking_models` distinct resolved provider/model routes at config validation.
Different focuses or effort levels on one route still count as one blocking model.

```toml
[review]
when = "finish"
max_turns = 20
interval_turns = 10
min_remaining_budget_fraction = 0.25
block_on = "any"
min_blocking_models = 2
reviewers = [
  { focus = "security" },
  { focus = "correctness", model = "openrouter/moonshotai/kimi-k2", effort = "high" },
]
```

A finding grounds on the path, side and lines it cites against the diff; an ungrounded finding is dropped.
A `block` holds only when its seat's verdict is `block` and its category is one of `security`, `sandbox-bypass`, `off-topic-edit`, `data-loss`, `verify-uncovered-correctness` (the last only when verify passed); otherwise it reads as `warn`, which cannot stall the run.
Findings injected in an earlier round or execution neither repeat nor count toward a later panel's gate while the blob they cite is unchanged.

## `[compaction]`

Tiered context compaction (approximate chars; tokens ≈ chars/4).

| Field | Default | Meaning |
|---|---|---|
| `drop_at_chars` | _adaptive_ | tier-1 threshold in accumulated tool-result characters (about 4 per token); above it: oldest results replaced by short, re-fetchable placeholders; unset: about 45% of the model's context window; explicit thresholds: both `compaction.drop_at_chars` and `compaction.summarise_at_chars` required |
| `summarise_at_chars` | _adaptive_ | tier-2 threshold in whole-context characters; above it: elided history summarized, conversation restarted on the summary, task DAG preserved; unset: model's window minus a 16k-token reserve; required above `compaction.drop_at_chars` |
| `keep_recent_chars` | `80000` | characters of newest history kept verbatim after the summary on a tier-2 restart; `compaction.keep_recent_chars = 0`: none |
| `keep_thinking_turns` | `-1` | thinking retained at tier 1 for the newest N assistant turns with `compaction.keep_thinking_turns = N` (`N >= 1`); `compaction.keep_thinking_turns = -1`: all thinking; `compaction.keep_thinking_turns = 0`: invalid, newest signed thinking block needed by an in-flight tool call; less replay for Anthropic's signed blocks and ChatGPT's reasoning items; no thinking re-sent on the OpenAI wire |
| `summary_max_tokens` | `2048` | output-token cap for a tier-2 summary or gist distillation; smaller caps overridden by a reasoning model's per-call floor for reasoning tokens; no cap accepted by the chatgpt backend |
| `elision_gists` | `true` | large `read_file` results replaced at tier 1 by model-written gists before bare placeholders; gists also dropped under continued pressure to keep the byte bound; `compaction.elision_gists = false`: bare placeholders directly |

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
| `state` | `{}` | per-skill state by name; layers merged key by key; written by `agent6 skills enable [--repo]` or `agent6 skills disable [--repo]`<ul><li><code>enabled</code>: Index skill and load on request</li><li><code>disabled</code>: Exclude skill from the run</li><li><code>always</code>: Include full skill in system prompt</li></ul> |

When a skill must apply, use `always`, `/name`, or `--skill`.

## `[budget]`

Hard stops; on hit the run ends (exit 3) and is resumable with a fresh budget.
Every call is bounded in exactly one currency: priceable calls (reported cost, else cached price × tokens) count against `max_usd`; a call carrying a plan-usage reading counts percentage points against `max_plan_points`; the rest count input+output tokens against `max_tokens_fallback`.
All three: `-1` unlimited, `0` refuse that ledger up front, `> 0` the cap.

| Field | Default | Meaning |
|---|---|---|
| `max_usd` | `10.0` | cap on the metered spend of one run (provider-reported cost, else price times tokens at the model's fetched rates, cache-aware); at the cap: resumable end (`budget_exhausted`); fresh budget per resumed execution; `budget.max_usd = -1`: unlimited; `budget.max_usd = 0`: every metered call refused; per-run override: `--max-usd` |
| `max_tokens_fallback` | `2000000` | input-plus-output token cap for unpriceable calls (local models, models without price data); `budget.max_tokens_fallback = -1`: unlimited; `budget.max_tokens_fallback = 0`: unmeterable models refused; per-run override: `--max-tokens-fallback` |
| `max_plan_points` | `-1.0` | subscription-plan percentage-point cap per run; rise in the account's reported used-percent, summed across window resets (values above 100 meaningful), largest movement across multiple windows; a window counts as reset when its reported reset time moves later by more than a minute; a lower reading whose reset time has not moved counts nothing; one more reading at the end of the run; whole-percent readings, each tick counted as a full point; early stop at the cap, never late; in-flight call still completed; `budget.max_plan_points = -1`: unlimited; `budget.max_plan_points = 0`: plan-metered calls refused; per-run override: `--max-plan-points`; each concurrent run counts the whole account-window increase toward its own cap, not just its own share of the spend. |
| `allow_plan_overage` | `false` | plan overage for `backend = "chatgpt"` or `"claude_code"`: purchased credits or extra usage after the included window, possible auto top-up through the saved payment method; `budget.allow_plan_overage = false`: circuit breaker, not a guarantee; account windows and credit state read at chatgpt preflight and from every response's headers, or from every claude_code round's rate-limit event; with `budget.allow_plan_overage = false`, exhausted window and credits present: stop at the next boundary, in-flight call still completed; `budget.allow_plan_overage = true`: chatgpt credit-balance drop read as dollars against `budget.max_usd`; no claude_code credit-balance reading, extra usage not metered by `budget.max_usd`; included-plan usage unaffected |

`--max-usd` / `--max-tokens-fallback` / `--max-plan-points` override per run.
A flag's value is recorded with the run: a resume without that flag runs under it again, and a flag on the resume replaces it for later resumes too.
Purchased-credit spend and API cost share the USD ceiling.
Prices come from provider listings (OpenRouter's; cached under `$XDG_CACHE_HOME/agent6/models/`), and a direct-Anthropic id is priced via its OpenRouter listing.

## `[machine]`

| Field | Default | Meaning |
|---|---|---|
| `keep_snapshots` | `5` | blackboard snapshots kept per machine instance; latest read by `machine status`, journal folded for recovery and `machine replay`; `machine.keep_snapshots = -1`: all snapshots |
| `keep_state_logs` | `50` | per-state log dirs kept under `<instance>/states/`, watchable logs of each state's execution; full transition history in the journal regardless; `machine.keep_state_logs = -1`: all logs |
| `pass_env` | `[]` | operator environment variable names allowed in a machine `tool` state's own `pass_env`; startup refused for a state naming an unlisted variable; global/repo config or a `--config FILE` layer; machine `[config]` overlay refused; provider `api_key_env` variables never allowed |

## `[notify]` (optional)

`on_event` is one operator argv on the host for every notification, with a minimal env (PATH/HOME/locale/desktop-bus), never your full environment.
`AGENT6_NOTIFY_EVENT` names the event (`session.waiting` as a prompt opens, `session.stop`, `machine.notify`, `machine.end`), `AGENT6_NOTIFY_MESSAGE` carries the prompt's first line, the end reason or the machine's message, and `AGENT6_NOTIFY_LEVEL` reads `info` for a wait, `ok` or `failed` for an end and `info`, `warn` or `error` for a machine notify.
A session adds `AGENT6_SESSION_ID/DIR/OK/VERIFIED/REASON`: `OK=1` means the agent stopped deliberately; `VERIFIED` is what the gate said; a hook that wants "green" reads the second.
A machine adds `AGENT6_MACHINE_ID/DIR/STATE`.
Global/repo config only (a machine `[config]` overlay setting `on_event` is rejected).
Fan out to ntfy/Pushover/email/Telegram yourself.

| Field | Default | Meaning |
|---|---|---|
| `on_event` | `[]` | argv command on every notification (a session's wait for an answer and its end, a machine's notify and end), no shell; the event in `AGENT6_NOTIFY_EVENT/MESSAGE/LEVEL`, the subject in `AGENT6_SESSION_*` or `AGENT6_MACHINE_*`; no hook when empty |
| `timeout_s` | `30.0` | Seconds the hook may run before it is killed. |

## `[web]`

Bind and optional metrics route for `agent6 web` ([the web UI](web.md)).
Loopback only by default; remote access is expected behind `tailscale serve`.
Every browser [signs in](web.md#sign-in) with the server's secret.

| Field | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Address `agent6 web` binds; a non-loopback address also needs `allow_non_loopback = true`. |
| `port` | `7658` | Port `agent6 web` listens on. |
| `allow_non_loopback` | `false` | Allow `host` to be a non-loopback address, so a typo can never silently expose the write surface (approvals, steers, config writes) beyond this machine. |
| `metrics_route` | `false` | Serve `/metrics` when `telemetry.metrics = "prometheus"`; requires the web secret as `Authorization: Bearer <secret>`, never a cookie or browser sign-in token. |

## `[telemetry]`

What `agent6 telemetry export` sends, and where ([Telemetry export](cli.md#telemetry-export)).
The web scrape route requires `metrics = "prometheus"` and the [`web.metrics_route`](#web) opt-in.
Off by default: no signal is selected, nothing is dialled, no socket is opened.
Header values live in `secrets.toml` under `[telemetry.headers]`, never in config.
The [`--secret` target](#creating-and-inspecting) writes and removes them; `[telemetry].headers` selects which names the exporter sends.

| Field | Default | Meaning |
|---|---|---|
| `traces` | `"none"` | Trace export destination.<ul><li><code>none</code>: Do not export traces</li><li><code>otlp</code>: Post traces to telemetry endpoint</li></ul> |
| `metrics` | `"none"` | Metrics export destination; prometheus writes `prometheus_file` for node_exporter's textfile collector; web scraping also requires `web.metrics_route` enabled.<ul><li><code>none</code>: Do not export metrics</li><li><code>otlp</code>: Post metrics to telemetry endpoint</li><li><code>prometheus</code>: Write textfile metrics and permit authenticated web scraping</li></ul> |
| `logs` | `"none"` | Export destination for significant events as log records.<ul><li><code>none</code>: Do not export log records</li><li><code>otlp</code>: Post log records to telemetry endpoint</li></ul> |
| `endpoint` | `""` | The OTLP/HTTP base URL; `/v1/traces`, `/v1/metrics` and `/v1/logs` are appended; `https://`, or `http://` to a loopback host |
| `headers` | `[]` | Header names sent with every export; each value comes from `[telemetry.headers]` in `secrets.toml`, written by `agent6 config set --secret telemetry.headers.<name> <value>`. |
| `capture` | `"metadata"` | Export detail; content excerpts cover the task, steers, tool arguments, gate command, model errors, finding titles and degradation details.<ul><li><code>metadata</code>: Export outcomes, counts, timings and identifiers</li><li><code>content</code>: Also export capped content excerpts</li></ul> |
| `prometheus_file` | `""` | The absolute path `metrics = "prometheus"` writes atomically, in a directory node_exporter's textfile collector reads. |

## `[mcp]` and `[mcp.servers.<name>]` (optional)

agent6 as an MCP client; for the other direction (`agent6 mcp serve`, agent6 as a server) see [Editor integration](acp.md#as-an-mcp-server).

MCP servers, spawned (`command`) or connected (`url`); tools appear as `mcp__<name>__<tool>` in run mode only (plan, ask, machine and read_only sessions never offer them: agent6 cannot classify an external tool as read-only).
A spawned server runs as a jailed child by default (its own `[mcp.servers.<name>.sandbox]` policy; `unconfined = true` opts out) with a curated env (never your provider keys; `pass_env` adds named vars); a `url` server is a process you run and confine yourself.
The model chooses the arguments, so each call is approved like a command (`approve`); audit each server like a `run_command` allow-list.

`agent6 mcp add` saves a spawned server unprobed with a warning when `sandbox.isolation` resolves to `none` or a run would refuse the config; otherwise it requires a successful probe before writing the entry.
During a run, a server that does not start is skipped with an `mcp.server_unavailable` journal event, never fatal.
`agent6 check mcp` probes under the run's sandbox with the repository bound read-only and skips a server it cannot hold that way ([the rule](security.md#2-sandbox)).

```
agent6 mcp add files -- npx -y @modelcontextprotocol/server-filesystem .
agent6 mcp add browser --url http://127.0.0.1:8931/mcp --token-env PW_TOKEN
agent6 mcp list
agent6 mcp remove files
```

`agent6 mcp remove <name>` drops the entry from the global config (`--repo` for the per-repo one): `config unset` cannot, and dropping its `command`/`url` is refused.

A spawned server's `[sandbox]` block names what it needs on top of the sandbox a `run_command` gets.

```toml
[mcp.servers.notes]
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/you/notes"]

[mcp.servers.notes.sandbox]
extra_write_paths = ["~/notes"]   # additive; includes read access
```

A confined spawn also loses the desktop-session addresses (`DBUS_SESSION_BUS_ADDRESS`, `XDG_RUNTIME_DIR`, `DISPLAY`, `WAYLAND_DISPLAY`): Landlock does not gate unix-socket `connect()`, and an unconfined session daemon would act on the server's behalf.
Anything else that reaches an unconfined process is still a way out, so name the narrowest paths that work.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch for MCP servers: `false` means no `mcp__*` tools reach the model, whatever `[mcp.servers]` lists. |
| `servers` | `{}` | MCP servers by name (`[mcp.servers.<name>]`); their tools reach the model as `mcp__<name>__<tool>` in run mode when `enabled` is on. |
| `servers.<name>.command` | `[]` | argv of a spawned stdio MCP server; jailed like a command, plus `sandbox` grants; exactly one of `command` or `url` required |
| `servers.<name>.url` | `""` | operator-run http(s) MCP endpoint; connection only, environment and confinement outside agent6's control; exactly one of `command` or `url` required |
| `servers.<name>.token_env` | `""` | bearer-token environment variable for a `url` server, named rather than inlined, never logged; token readable on the wire over plaintext `http://` to a non-loopback host; approval in `mcp add`, warning on every run starting it |
| `servers.<name>.enabled` | `true` | `false` withholds this server's tools from the model without deleting the entry. |
| `servers.<name>.pass_env` | `[]` | Environment variables a spawned server needs, by name; everything else is agent6's curated base environment. |
| `servers.<name>.sandbox` | none | extra grants for a spawned server beyond a jailed command's sandbox; unset: exactly that sandbox; `url` server: operator's process, confinement at its launch site |
| `servers.<name>.approve` | `"ask"` | Approval policy for this server's tools; prompts show the model's arguments; session answer "allow all": this server only for the run, not command tools or sibling servers; "deny all": tools withdrawn from the next turn; `--auto-approve`: `approve = "allow"` for the run; no deny mode, tools withheld with `enabled = false`<ul><li><code>ask</code>: Prompt before each server tool call</li><li><code>allow</code>: Call server tools without prompting</li></ul> |
| `servers.<name>.startup_timeout_s` | `10.0` | Seconds the server gets to answer `initialize` and `tools/list` before it is given up on. |
| `servers.<name>.call_timeout_s` | `60.0` | Seconds one `tools/call` may take before it fails; a spawned server is restarted after a timeout. |
| `servers.<name>.http_trust_env` | `false` | ambient `HTTP(S)_PROXY`, `.netrc` and `SSL_CERT_FILE` for a `url` server; `http_trust_env = false`: no proxy routing of a local server's bearer token; opt-in for a server reachable only through the environment's proxy |

### `[mcp.servers.<name>.sandbox]`

| Field | Default | Meaning |
|---|---|---|
| `extra_read_paths` | `[]` | read and execute paths for this server beyond the run's sandbox (absolute or `~`); workspace, system dirs, tool dirs and writable `HOME` included without extra grants; only the server's own data needed here |
| `extra_write_paths` | `[]` | Paths this server may write beyond the run's sandbox (absolute or `~`). |
| `network` | `"auto"` | Server network; automatic mode warns when private networking is unsupported; explicit disconnected networking refuses rather than falling back; session networking permits access to the run's background dev servers.<ul><li><code>auto</code>: Private network where supported, otherwise host with warning</li><li><code>none</code>: Require private disconnected network</li><li><code>session</code>: Join run network without outside access</li><li><code>host</code>: Use the machine network</li></ul> |
| `unconfined` | `false` | no sandbox for a server needing arbitrary host access; refused alongside any other field in this block, no partial application |

---

## Reaching a run's network

When `sandbox.network` resolves to `session`, a run's commands share one network with no route off the box, so a dev server the agent starts is reachable only through these operator-only commands:

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
| `AGENT6_DETACHED_AWAY` | approval and question handling without a terminal or editor; `AGENT6_DETACHED_AWAY=wait`: both parked for a front-end; `AGENT6_DETACHED_AWAY=deny`: approval refused; `AGENT6_DETACHED_AWAY=approve`: approval granted; `deny`: a waiting question is unanswered at once, counted by `sessions show`; `wait`, `approve` or unset: a waiting question waits for a front-end; a queued question (`ask_user` with `wait` false) queues in every mode; hub and machine spawns: `AGENT6_DETACHED_AWAY=wait`; `--parallel` lanes: coordinator's marker, otherwise wait with a terminal to attach from, deny without one |
| `AGENT6_AUTO_APPROVE` | `1` grants every command approval to a machine's agent states, as `--auto-approve` does (a configured `deny` stays deny). |
| `AGENT6_NO_COMMANDS` | `1` withholds every command tool from a machine's agent states, as `--no-commands` does. |
| `AGENT6_JAIL_BIN` | Path to a specific `agent6-jail` binary (else bundled). |
| `AGENT6_DISABLE_SANDBOX` | `1` forces `sandbox.isolation = "none"` for one invocation (same as `--disable-sandbox`). |
| `AGENT6_ALLOW_ROOT` | `1` permits running as root (same as `--allow-root`). |
| `AGENT6_WEB_SECRET` | browser sign-in secret for `agent6 web`; at least 12 characters after lowercasing and joining words with single hyphens; overridden by `--secret`; unset: fresh secret generated and printed per launch |

A provider's `api_key_env` names the env var supplying its key; omit it to read `secrets.toml`.
The bench and development switches are listed in [architecture.md](architecture.md#bench-and-development-switches).
