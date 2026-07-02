# SECURITY

agent6 treats the LLM as untrusted. This document is the layer-by-layer
breakdown of how that assumption is enforced and what the known limits are.

## Reporting

For now: open a GitHub issue prefixed `[security]`, or, for embargoed
issues, email the maintainer listed in `pyproject.toml`. Once agent6 has
a stable distribution, this will move to a private channel + GitHub
Security Advisories.

Please include:

- agent6 version (`agent6 --version`).
- Linux kernel version + distro (`uname -a` + `/etc/os-release`).
- The output of `agent6 check sandbox`.
- Minimal reproduction, ideally as a failing test under `tests/security/`.

## Threat model

Adversary: a fully malicious worker model, or an honest model that has
been prompt-injected by a file in the repository under analysis.

We assume the adversary controls:

- the text of every LLM response,
- the choice of tool calls and their arguments (within the published
  JSON schema),
- the content of any file the agent reads during the run.

We do NOT assume the adversary controls:

- the kernel,
- the agent6 source code (i.e., supply-chain compromise of the agent6
  binary itself is out of scope; sign your releases),
- the Anthropic / OpenAI API endpoints.

Under that adversary, agent6 aims to make the following true:

1. No writes outside the project working directory.
2. No reads of files outside the project working directory (+ read-only
   sibling paths, if explicitly configured).
3. No network egress except to the configured provider endpoints. This is
   enforced structurally for the agent process when `sandbox.agent_network =
   "providers"` (the default; see Defense Layer 1b); `"local"` narrows it to
   loopback providers, `"open"` lifts it. Jailed commands (`run_command`,
   machine `tool` states) are governed separately by `sandbox.tool_network`
   (default `"block"`); see Defense Layer 1b and §8.
4. No `git push`, no `--force`, no history rewrite, no `reset --hard`.
5. No persistence after the run terminates (no daemon, no cron, no
   `.bashrc` mutation; the jail's mount namespace is the
   only place children can write to anyway).

## Defense layers

### 1. Agent-process Landlock (hardened profile)

Applied at the start of `agent6 run`/`resume`, before any provider or
network object is built, **on the `hardened` profile only**. The `strict`
profile does not take this path: it runs every child command in its own
user+mount+pid+net namespace (a strictly stronger boundary than Landlocking
the parent) and confines provider egress with the broker (Defense Layer
1b); applying agent-process Landlock there would additionally break the
jail's `pivot_root(2)`/`mount(2)` on kernels at Landlock ABI ≥ 7. Where it
applies it restricts the Python process itself (irrevocably, inherited by
every child it spawns):

| Landlock rule | Allowed |
|---|---|
| FS read+exec | cwd, `$HOME`, `/usr`, `/etc`, `/tmp`, the system exec dirs `/bin` `/sbin` `/lib` `/lib64` `/dev` (when present), and `/run` + `/proc` when present. The system dirs mirror the jail child's read+exec roots: the launcher opens each from inside this domain to grant the child, so omitting one (notably `/dev` on a merged-`/usr` host) makes the child's execve fail EACCES |
| FS write | cwd, `/tmp`, the `/dev` char devices, and `/proc` when present |
| TCP connect (kernel ≥ 6.7) | the *ports* of configured providers: `443` for each `anthropic` entry, the `base_url` port for each `openai` entry (default `443`) |

Landlock's network hook filters by destination **port only** (it has no
host/IP primitive), so it blocks connections on other ports but does **not**
pin egress to a specific host; for that use `sandbox.agent_network = "providers"`
(Defense Layer 1b). On older kernels (no TCP rules) agent6 warns and runs
FS-only Landlock; don't run there on a host whose UID can read credentials the
agent could exfiltrate.

### 1b. Provider-only egress broker (`sandbox.agent_network = "providers"`)

When enabled (strict profile only; it relies on unprivileged user
namespaces), `agent6 run` confines its own process to host-level egress:

1. While still in the host network namespace (netns) and single-threaded, the
   agent binds one `AF_UNIX` listening socket per allow-listed provider
   `host:port` and `fork()`s a small **broker** child. The broker stays
   in the host netns; for each connection accepted on a given socket it
   dials the single fixed `host:port` that socket represents (resolved
   per-connect, so the allow-list is robust to CDN IP rotation) and
   splices bytes. TLS is end-to-end: the broker only ever sees ciphertext.
2. The agent then `unshare(CLONE_NEWUSER | CLONE_NEWNET)` into a fresh,
   empty network namespace (loopback only: no veth, no default
   route). Its sole path off-host is the set of unix sockets, each of
   which is hard-wired to one provider endpoint chosen by the operator,
   never by the (untrusted) LLM at connect time.

This is **fail-closed**: the kernel network namespace is the real
boundary. A missing route means no connectivity (the agent cannot connect
at all), never a silent leak. Because the upstream of each socket is fixed
at bind time, the egress allow-list is structural rather than a filter the
agent could be tricked into widening. On hosts that only support the
hardened profile agent6 refuses to run rather than execute unconfined.

`sandbox.agent_network = "local"` uses the same broker but pins it to *loopback*
provider endpoints only (local models such as Ollama) and refuses a non-local
provider; `sandbox.agent_network = "open"` skips the broker entirely. For `agent6
machine run`, each `agent` state runs in its own subprocess that performs this
same broker setup for itself (the engine is a thin host-netns supervisor), so a
machine agent's egress is confined exactly as a normal run's is.

Curator and other `AF_UNIX`-based helpers are unaffected (unix sockets
cross the netns boundary). MCP servers that need their own outbound
network access will not have it under `providers`; that is a
deliberate limitation, not a bug.

**`sandbox.allow_urls` (operator-controlled egress additions).** The
allow-list above is, by default, exactly the configured provider
endpoints. An operator may widen it with `sandbox.allow_urls`: a set of
`host` / `host:port` / URL entries that get their own broker sockets
alongside the providers (effective egress = union of provider endpoints
and `allow_urls`). Security properties are unchanged: each added socket is
still hard-wired at bind time to one operator-chosen `host:port`, resolved
per-connect, and the LLM cannot add, widen, or redirect an entry: it is a
static config field, never written from model output. The default is empty
(secure by default), entries are validated at config-load time, and the
field is only consulted under `sandbox.agent_network = "providers"` (ignored
under `local`/`open`). It widens only the agent path, never a jailed command.
Merge is last-overlay-wins: the most-specific config tier that sets
`allow_urls` replaces it wholesale, so a repo or machine overlay cannot
silently *append* to a narrower global allow-list; it must restate the
full set, keeping the effective allow-list auditable via `config show`.


### 2. `agent6-jail` (Rust) for every child command

Every `apply_edit` is in-process, but every `run_verify_command` and
`run_command` is executed by `agent6-jail`. The jail:

- Forks a new user, mount, PID, IPC, UTS, and (in `strict`) network
  namespace.
- Sets up a minimal rootfs of bind mounts under a fresh tmpfs and
  `pivot_root`s into it. The working directory is the only writable
  mount; everything else is `ro,nosuid,nodev`.
- Bind-mounts a curated subset of `/dev`: `null`, `zero`, `urandom`,
  `random`, `full`. `/dev/tty` is not exposed: TTY access lets a
  child write escape sequences to the controlling terminal of the parent.
- Mounts a fresh `proc` (private to the new PID namespace). If that
  fails on the host kernel, `/proc` is left empty inside the jail rather
  than bind-mounting the host `/proc`; the latter would expose host
  process info to the child.
- Applies Landlock (FS + net rules).
- Installs a seccomp filter that allows the syscalls a Linux process
  actually needs (clone, mmap, futex, …) and blocks the dangerous
  remainder (kexec, bpf, ptrace, mount, …).
- Drops all capabilities, sets `NO_NEW_PRIVS`, and applies rlimits
  (CPU, AS, NOFILE, NPROC).
- Then `execve`s the requested binary.

The jail's policy is passed as a JSON document on stdin from
`agent6.sandbox.jail.run_in_jail`. The Rust side validates it against a
strict schema and refuses on any unknown field.

### 2a. Environment setup: sudo, packages, and what the operator provides

The jail is a one-way boundary: the agent works *within* the environment you
give it and cannot expand it. Concretely, what an agent can and cannot do to the
host (verified empirically under both `strict` and `hardened`):

- **`sudo` cannot escalate, even with passwordless sudo.** The jail sets
  `NO_NEW_PRIVS`, so the kernel ignores the setuid bit on `sudo` (and every
  setuid binary). A jailed `sudo -n true` fails with *"the 'no new privileges'
  flag is set, which prevents sudo from running as root"*, whether or not the
  host has a `NOPASSWD` sudoers rule. An agent on a box where *you* can `sudo`
  freely still cannot.
- **Installing system packages from inside the jail is impossible.**
  `apt-get`/`dnf`/`apk` may be present (mounted read-only) but are unusable:
  they need root (blocked above), network to the package mirrors (egress only
  permits your provider endpoints, §1b/§7), and writes to `/usr`, `/var`
  (Landlock denies everything outside the workspace). All three are blocked.
- **Compiling and running code works.** `run_verify_command` and, when
  `sandbox.run_commands` permits, `run_command` execute jailed, so the agent can
  invoke a compiler, test runner, or build tool that is *already installed on
  the host*. It just cannot install new ones, and a build step that needs the
  network is blocked unless `sandbox.tool_network` is loosened.
- **The provisioning model is operator-first.** You install the toolchain,
  create the venv, and fetch dependencies with your own shell and sudo, *before*
  or *outside* agent6; agent6 then works inside the jail with what you provided.
  To widen what a command may read or reach, use config, never sudo:
  `sandbox.extra_read_paths` (extra read mounts), `sandbox.tool_network` (let
  jailed commands reach the network), `[providers.*].base_url` (which hosts
  egress allows). All operator-controlled and visible in `agent6 config show`.

**Running agent6 itself as root is opt-in and weakens the boundary.** Under
`strict` the jail's user namespace maps inside-uid-0 to *the real uid agent6
runs as* (`uid_map "0 <uid> 1"`). As your normal user, the jailed child's
namespaced-root is your unprivileged uid outside, so no real privileges. If you
start agent6 as **root** (`--allow-root` / `AGENT6_ALLOW_ROOT=1`), that
inside-root maps to **real root**, so jailed children run as real root confined
only by Landlock + seccomp + `NO_NEW_PRIVS`: still no write outside the
workspace and no network beyond the provider (so still no package installs),
but a larger blast radius: as root those allowed *reads* include root-only host
files (e.g. `/etc/shadow` under `hardened`; `strict`'s minimal rootfs hides
them). `sudo` adds nothing either way (`NO_NEW_PRIVS`). Run agent6 as your
normal user and pre-provision with your own sudo.

### 3. Profile selection

You *set* the `sandbox.profile` field; it resolves against the host to an
*effective profile*: what actually runs. `auto` is never itself an effective
profile (it is resolved away), and `auto` never resolves to `none` on Linux. No
silent downgrade: an explicit request the host can't satisfy is refused, not
weakened. `none` (unsandboxed) is a deliberate, gated opt-out; see its rows below.

| `sandbox.profile` | Host | Effective profile |
|---|---|---|
| `auto` *(default)* | Linux, user namespaces available | `strict` |
| `auto` | Linux, no user namespaces | `hardened` |
| `auto` | non-Linux (macOS / Windows) | `none` |
| `strict` | Linux, user namespaces available | `strict` |
| `strict` | anything else | ⛔ refuse to run |
| `hardened` | Linux (user namespaces or not) | `hardened` |
| `hardened` | non-Linux (macOS / Windows) | ⛔ refuse to run |
| `none` *(explicit opt-out)* | a detected container | `none` (the container is the boundary) |
| `none` | a bare host | ⛔ refuse unless `AGENT6_ALLOW_NO_SANDBOX=1` |

The `none` opt-out runs commands unsandboxed. It is allowed automatically only
inside a **detected container**, where the container is the blast radius. A
container is proven by a filesystem marker (`/.dockerenv` or
`/run/.containerenv`), not a forgeable env var. On a bare host it is refused unless the operator confirms with
`AGENT6_ALLOW_NO_SANDBOX=1`. Always with a loud startup warning.

The three effective profiles:

- **strict**: full namespaces (user/mount/pid/ipc/uts/net) + `pivot_root` +
  Landlock + seccomp + `capset(0)` + rlimits + `NO_NEW_PRIVS`.
- **hardened**: Landlock + seccomp + `capset(0)` + rlimits + `NO_NEW_PRIVS`,
  but no namespaces (so it works inside default-seccomp Docker, where the
  container blocks the inner `clone(CLONE_NEW*)`; the container is the blast
  radius).
- **none**: *unsandboxed*. Child commands run as plain subprocesses with no
  kernel-enforced confinement; always with a loud warning.

CI should set `profile = "strict"` to fail loudly if the sandbox is weaker than
expected. "User namespaces available" means `unshare -U -r true` succeeds.

### 4. Fixed tool surface

The LLM only ever sees the fixed set declared in
`src/agent6/tools/schema.py` (enumerated in the README): structured
edits, read-only navigation, fixed-argv verify/metric commands, a
terminal `finish_run`, a curator-backed task notepad, and the
capability-gated `run_command`. There is no `shell`, no `write_file`
(writes go through `apply_edit`, an in-process rewriter that refuses
paths outside cwd), no `web_fetch`, and no `eval`. Adding a tool requires
a security review note in the commit message; see [AGENTS.md](https://github.com/agent6-dev/agent6/blob/master/AGENTS.md).

### 5. Git invariants

`src/agent6/git_ops.py` is the only module that invokes `git`. It
exposes typed wrappers for the safe operations (status, add, commit,
diff, branch creation, checkout) and refuses, by construction, to call:

- `git push` (any form, any remote),
- `git reset --hard`,
- `git commit --amend`,
- `git rebase`, `git filter-branch`, `git filter-repo`,
- `git branch -D`, `git branch --force`,
- anything containing `--force` or `-f` on a destructive verb.

`git.allow_push`, `git.allow_force`, and `git.allow_history_rewrite` in
the agent6 config exist for forward compatibility but are currently
ignored; they will stay ignored until there is a concrete review of what
a "safe push" would look like.

Every `git_ops` invocation is also hardened against **repo-controlled host
code execution**: a cloned/poisoned `.git/config` can otherwise run a
command on the host (outside the jail) the moment agent6 runs git in it.
`core.fsmonitor` (fires on index refresh) and `diff.external` (fires on
`git diff`) are always overridden off; the repo's `.git/hooks/*` run only
when `git.run_repo_hooks = true` (default false; `core.hooksPath` is
pointed away from the repo so a `pre-commit` hook can't fire on agent6's
own auto-commit). On strict this complements `protect_git`, which
RO-binds `.git` to stop the worker from *writing* into it. On hardened
the cwd is blanket read-write (no mount namespace to carve, and carving
`.git` read-only would also deny new top-level entries and break
toolchains like cargo/pytest that create `target/` or `.pytest_cache/`),
so `.git` is writable by jailed commands there. That is acceptable: it is
gated by `run_commands` (default `ask`), recoverable (branch-per-run,
commits go through `git_ops`), and the surrounding container is the blast
radius.

### 5b. Secrets, `connect`, and running as root

- **Secrets at rest.** Provider API keys live in
  `$XDG_CONFIG_HOME/agent6/secrets.toml`, created and enforced `0600`
  (owner read/write only). agent6 refuses to read the file if it is
  group/other-accessible or owned by another user, the same posture as
  an SSH private key. Keys may alternatively come from an environment
  variable named by `[providers.<name>].api_key_env`; the env var takes
  precedence. Keys are never written to transcripts, never printed by
  `agent6 config show` (redacted), and never mounted into the jail;
  provider calls happen in agent6's own process, outside the sandbox.
- **`agent6 connect` never executes remote input.** The connect flow only
  prompts locally (key via `getpass`: hidden, or masked with `*` on Python
  3.14+) and writes config/secrets. It does not run any command, URL, or
  script returned by a provider or any remote, by construction. This is a
  deliberate guard against the class of bug where a login flow runs an
  attacker-supplied shell command. After saving a key it makes one read-only
  `GET` to the provider's models/key endpoint to confirm the key authenticates
  (a 401 is caught at setup, not mid-run); the response is only inspected for
  the HTTP status, never executed. Skip it with `--no-verify` for offline or
  local endpoints.
  During a run agent6 opens no listening network socket of any kind (MCP is
  stdio; the egress broker is a private Unix socket). The one accept-side
  socket is the opt-in `agent6 web` front-end, loopback by default (§7).
- **Root.** Running an LLM-driven agent as root is dangerous, so agent6
  refuses unless the operator explicitly opts in with `--allow-root` (or
  `AGENT6_ALLOW_ROOT=1`), and prints a loud banner. When invoked through
  `sudo`, agent6 resolves the *real* user from `SUDO_UID`/`SUDO_GID`/
  `SUDO_USER`, reads that user's config + secrets (not root's), and
  `chown`s anything it writes under the per-repo state dir back to them so
  no root-owned files are left behind. agent6 does not drop privileges
  in-process: under `sudo` the worker's verify/run commands are expected
  to need root and run as root inside the jail, so the jail, not the
  process uid, is the security boundary.

### 6. Curator subprocess and run-state location

The task graph is written by a separate `graph-curator` subprocess. The
main agent process talks to it over a Unix domain socket inside the run
directory and never writes graph data directly, so a bug in the worker /
planner cannot corrupt the on-disk graph; the curator's append-only
`graph.jsonl` is the durable source of truth. The main process writes the
rest of the run state in-process: the resume snapshot (`loop_state.json`),
the event log (`logs.jsonl`), and transcripts.

What keeps the whole run directory safe from jailed commands is its
**location**, not any single writer. Per-repo state (config + run state)
lives out of the workspace under `$XDG_STATE_HOME/agent6/<repo-id>/`
(override with `[agent6].state_dir` or `AGENT6_STATE_HOME`). Jailed
commands run on the repo cwd, and the state dir is outside it, so they
cannot reach it.

### 7. No agent-owned network surface (except opt-in `agent6 web`)

The agent loop does not run an HTTP server, gRPC server, or any other
accept-side socket. The only sockets it opens are:

- outbound HTTPS to the LLM provider,
- a per-run Unix domain socket under the run directory
  (`<state-dir>/<repo-id>/runs/<run-id>/`) with mode `0600` for talking to
  its own curator.

The one accept-side socket agent6 can open is the **`agent6 web`** front-end,
and only when you start it. It binds `127.0.0.1` by default and has no
app-level auth: remote access is meant to run behind `tailscale serve`, so the
tailnet/WireGuard identity is the access control (see [the web UI](web.md)).
Binding a non-loopback host exposes the write surface (spawn runs, answer
prompts) and is refused unless you opt in: a non-loopback `[web].host` needs
`[web].allow_non_loopback = true` (checked at config load) and a `--host`
override needs `--allow-non-loopback`, so a copied config or command cannot
silently expose you. The server only ever renders folded
view-model state and drives the typed front-end contracts, it never serves
secrets and never executes arbitrary input: new-work spawns fixed argv with the
task as one argv element, machine run is allow-listed to the authored files,
answers write only into the addressed run's own answer files (the run id,
answer id, and a machine answer's target state dir are each validated to a
single path component, so a request cannot escape the run's
approvals/questions dir), and merge/prune/config-set are fixed agent6
subcommands.

State-changing POSTs also carry a CSRF guard so another site open in the
operator's browser cannot drive the agent: a POST body must be
`Content-Type: application/json` (a cross-site `fetch` with that type triggers
a CORS preflight the server never answers), and any `Origin` header must match
`Host`. This holds on both the loopback bind and behind `tailscale serve`
(same-origin requests pass either way). It does not cover DNS rebinding, which
would need a Host allow-list incompatible with the tailnet hostname; that
vector stays with the network layer.

The same rules cover the **machine write surface** (`POST
/api/machine/<name>/{poke,answer,approve,steer}`): the machine name goes through
the identical single-path-component guard as a run id, and each answer id is
contained the same way. `poke` writes only the instance signal file (the payload
is inert JSON the next `tool` reads); `answer`/`approve`/`steer` write only into
the current agent state's per-state dir under the per-repo state dir. The
liveness gate registers `frontend.pid` on the instance dir while a browser
streams, so a machine agent state's prompts bridge to the browser exactly as a
run's do. The web PWA assets (manifest, service worker, icon) are static; the
service worker is a no-op passthrough with no Web Push / VAPID and no push
handler.

There is no telemetry, no auto-update, and no remote control plane.

### 8. State-machine egress (the supervisor model) + script bundles

A `agent6 machine run` engine is a thin **supervisor** that stays in the host
network namespace and makes no network calls itself. Each `agent` state runs in
its own subprocess that confines its egress per `sandbox.agent_network` (the
broker, §1b); each `tool` state is jailed by the engine, so a per-tool
`allow_network` decides its netns independently of the agent. This is what lets
a machine confine its agents to the provider API while letting one
deterministic tool reach the network: a `tool` command is fixed and
operator-reviewed (unlike `run_command`, whose argv the LLM chooses), so a
networked tool is not a free exfiltration channel.

Egress is set by `sandbox.agent_network`, `sandbox.tool_network`, and a per-tool
`allow_network`; the effective profile (§3) decides what is *enforceable*. The
tables cover every case; "offline" = no egress.

**Agent-process egress** (the agent's own LLM/provider HTTP), by `sandbox.agent_network`:

| `sandbox.agent_network` | `strict` | `hardened` | `none` |
|---|---|---|---|
| `providers` *(def)* | provider endpoints + `allow_urls`, broker-pinned (§1b) | provider *ports* only (Landlock) | unconfined ⚠ |
| `local` | loopback providers only, broker-pinned (refuse to run if any provider isn't loopback) | ⛔ refuse to run | unconfined ⚠ |
| `open` | unconfined | unconfined | unconfined ⚠ |

**Jailed-command egress** (`run_command` and machine `tool` states), by
`sandbox.tool_network` (columns; cells are the `strict` profile):

| jailed command | `block` *(def)* | `only_explicit_states` | `allow` |
|---|---|---|---|
| `run_command` | offline | offline | host network |
| `tool`, `allow_network = "auto"` (def) / `"block"` | offline | offline | offline |
| `tool`, `allow_network = "allow"` | ⛔ refuse to run | host network | host network |

**Refusals**: these configurations refuse to run (fail-closed):

| Configuration | When |
|---|---|
| `sandbox.tool_network = "allow"` without `sandbox.agent_network = "open"` | config load, any profile ¹ |
| a `tool` sets `allow_network = "allow"` under `sandbox.tool_network = "block"` | machine start, any profile |
| `sandbox.agent_network = "local"` or `sandbox.tool_network = "only_explicit_states"` | run start, `hardened` ² |
| a machine with `tool` states under `sandbox.tool_network = "block"`, or a `tool` with `allow_network = "block"` | machine start, `hardened` ² |

- ⚠ `none` (non-Linux) is **unsandboxed**: nothing above is enforced and nothing
  is refused; the run proceeds with a loud warning.
- ¹ `run_command` runs inside the agent process, so it can't reach the network
  while the agent is confined; hence `sandbox.tool_network = "allow"` needs
  `sandbox.agent_network = "open"`.
- ² `sandbox.tool_network`'s per-command isolation needs a network namespace, so
  it is `strict`-only. On `hardened` (no namespaces) a jailed child instead
  inherits the agent's Landlock and follows `sandbox.agent_network`; the cases
  that would need real per-command isolation are refused rather than mis-confined.

Every surface fails closed:

- **Operator-gated, machine-declared.** `sandbox.agent_network`/
  `sandbox.tool_network` are read only from the operator's global/repo config; a
  machine's `[config]` overlay (possibly LLM-drafted or shared) is rejected at
  load if it declares `[providers.*]`, `[sandbox.*]`, `[profiles.*]`, or
  `git.run_repo_hooks` (a profile preset would otherwise splice that same
  operator-only policy, or a host `[machine.notify]` argv, into the effective
  config, since the operator's selected profile is resolved from every layer
  including the overlay; `run_repo_hooks` would honor the repo's `.git/hooks`,
  running host code outside the jail on a `mode="run"` commit). A `tool` merely
  *declares* `allow_network`; whether `"allow"` is honored is the operator's call
  via `sandbox.tool_network`, and
  every conflict or unenforceable demand is refused at startup naming the state
  (see the rows/notes above), never silently mis-confined.
- **Bundle confinement.** Helper scripts live in an operator-reviewed
  `scripts/` directory beside the `.asm.toml`. `machine check` validates
  that every entry under `scripts/` resolves *inside* the bundle (symlinks
  that escape via `..`/absolute are rejected) and that every static
  `scripts/...` command reference exists and stays inside the bundle, so a
  machine cannot smuggle a path that reads or executes outside its own
  directory. Scripts are drafted at authoring time and reviewed/committed
  by the operator, never fetched or generated from untrusted model output
  at run time. And during a run, the machine's own `.asm.toml` + `scripts/`
  are made read-only in every jail (the same mechanism that RO-binds
  `.git` on strict), so a
  tool or agent state cannot rewrite its own logic, add an `allow_network`
  flag, or alter a bundled script mid-run or for a future run.
- **Notifications.** A machine surfaces attention two ways, neither of which
  widens the agent's surface. Device-present: each front-end renders a
  `machine.notify` (a state's `notify` message) as an ephemeral overlay, and
  `agent6 watch`/the TUI also call `notify-send` with a FIXED argv (exe + two
  positional data arguments, no shell), so a model-authored message is inert
  data, never a command. Out-of-band: the operator hook
  `[machine.notify].on_event` runs an operator-controlled argv on the host,
  outside the jail, on each `machine.notify` and `machine.end`, with only
  `AGENT6_MACHINE_*` env (id, dir, event, state, message, level), mirroring
  `[notify].on_complete`. The hook argv is operator config, never LLM output; a
  machine `[config]` overlay that sets `[machine.notify]` is rejected at load, so
  a shared or LLM-drafted machine cannot inject a host command. There is no Web
  Push / VAPID; the web notification is the foreground Notification API only.

## Prompt-injection resilience

The test suite under [`tests/security/test_prompt_injection.py`](https://github.com/agent6-dev/agent6/blob/master/tests/security/test_prompt_injection.py)
runs a small corpus of adversarial inputs through the planner, worker,
and reviewer prompts and asserts that the agent does not exfiltrate
file content, does not attempt out-of-policy tool calls, and does not
follow embedded instructions to weaken its own constraints.

This is a smoke test, not a proof. The structural defenses above
(sandbox, fixed tool surface, git invariants) are the real mitigation;
prompt-injection corpus tests exist to catch regressions in the prompts,
not to bound what an attacker can do.

## Known limitations

- **Landlock TCP rules** require Linux ≥ 6.7 (ABI ≥ 4). On older kernels
  the *agent process itself* is not network-confined. Children are still
  net-isolated in `strict` via the empty network namespace.
- **User namespaces** must be enabled
  (`kernel.unprivileged_userns_clone = 1`). Some distros disable this
  by default; agent6 detects that and refuses to run `strict`.
- **AppArmor userns restriction** (Ubuntu 24.04+:
  `kernel.apparmor_restrict_unprivileged_userns = 1`) blocks unprivileged
  userns unless the process has an AppArmor profile granting `userns`.
  agent6 ships such a profile, scoped to just the launcher binary; install
  it with `agent6 system apparmor install` (`remove` reverts it) -- the
  surgical fix for strict per-command jailing. agent6's profile detection
  probes the *real launcher binary* (not `/usr/bin/unshare`), so once the
  profile is installed it selects `strict` for per-command jailing; without
  it, `hardened`. One limitation: strict's provider-egress broker
  (`agent_network = "providers"`, the default) needs the *agent process*
  itself to create a userns, which the launcher-only profile does not grant
  -- so a default run downgrades to `hardened` (egress confined by Landlock)
  with a note unless you set the sysctl to 0 host-wide or use
  `agent_network = "open"`.
- **seccomp** is required by the jail; on rare hardened kernels that
  block seccomp from unprivileged callers, the jail fails closed.
- **Devcontainers**: the jail's `hardened` profile is what you get
  inside Docker / VS Code dev containers. The container itself becomes
  the FS blast radius. Network restrictions still apply via the
  agent-process Landlock when the kernel supports it. The XDG state base
  is inside the container and ephemeral (lost on rebuild), so to persist
  run state mount a volume at the state dir or set `[agent6].state_dir` /
  `AGENT6_STATE_HOME` to a persisted out-of-cwd path.
- **Side channels**: agent6 makes no claim about timing, cache, or
  speculative-execution side channels. If your threat model includes
  Spectre-class attacks, do not co-locate agent6 on a host with secrets.
- **Supply chain**: pin your install. The runtime deps are `pydantic`,
  `httpx2`, `argcomplete`, the `tree-sitter` pair (`tree-sitter` +
  `tree-sitter-language-pack`), and `textual` (the live dashboard);
  build-dep is `hatchling`; and the jail pulls a small set of well-known
  Rust crates (`nix`, `libc`, `landlock`, `seccompiler`, `serde`,
  `serde_json`). Verify before upgrading any of them.
