# SECURITY

agent6 treats the LLM as untrusted. This document is the layer-by-layer
breakdown of how that assumption is enforced and what the known limits are.

## Reporting

For now: open a GitHub issue prefixed `[security]`, or — for embargoed
issues — email the maintainer listed in `pyproject.toml`. Once agent6 has
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
   `.bashrc` mutation — the jail's mount namespace is the
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
| FS read | cwd, `$HOME`, `/usr`, `/etc`, `/tmp`, the common `/dev` char devices, and `/run` + `/proc` when present |
| FS write | cwd, `/tmp`, the `/dev` char devices, and `/proc` when present |
| TCP connect (kernel ≥ 6.7) | the *ports* of configured providers — `443` for each `anthropic` entry, the `base_url` port for each `openai` entry (default `443`) |

Landlock's network hook filters by destination **port only** (it has no
host/IP primitive), so it blocks connections on other ports but does **not**
pin egress to a specific host — for that use `sandbox.agent_network = "providers"`
(Defense Layer 1b). On older kernels (no TCP rules) agent6 warns and runs
FS-only Landlock; don't run there on a host whose UID can read credentials the
agent could exfiltrate.

### 1b. Provider-only egress broker (`sandbox.agent_network = "providers"`)

When enabled (strict profile only — it relies on unprivileged user
namespaces), `agent6 run` confines its own process to host-level egress:

1. While still in the host network namespace (netns) and single-threaded, the
   agent binds one `AF_UNIX` listening socket per allow-listed provider
   `host:port` and `fork()`s a small **broker** child. The broker stays
   in the host netns; for each connection accepted on a given socket it
   dials the single fixed `host:port` that socket represents (resolved
   per-connect, so the allow-list is robust to CDN IP rotation) and
   splices bytes. TLS is end-to-end: the broker only ever sees ciphertext.
2. The agent then `unshare(CLONE_NEWUSER | CLONE_NEWNET)` into a fresh,
   **empty** network namespace (loopback only — no veth, no default
   route). Its sole path off-host is the set of unix sockets, each of
   which is hard-wired to one provider endpoint chosen by the operator,
   never by the (untrusted) LLM at connect time.

This is **fail-closed**: the kernel network namespace is the real
boundary. A missing route means no connectivity (the agent cannot connect
at all), never a silent leak. Because the upstream of each socket is fixed
at bind time, the egress allow-list is structural rather than a filter the
agent could be tricked into widening. On hosts that only support the
hardened profile agent6 **refuses to run** rather than execute unconfined.

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
endpoints. An operator may widen it with `sandbox.allow_urls` — a set of
`host` / `host:port` / URL entries that get their own broker sockets
alongside the providers (effective egress = union of provider endpoints
and `allow_urls`). Security properties are unchanged: each added socket is
still hard-wired at bind time to one operator-chosen `host:port`, resolved
per-connect, and the LLM cannot add, widen, or redirect an entry — it is a
static config field, never written from model output. The default is empty
(secure by default), entries are validated at config-load time, and the
field is only consulted under `sandbox.agent_network = "providers"` (ignored
under `local`/`open`). It widens only the agent path, never a jailed command.
Merge is last-overlay-wins: the most-specific config tier that sets
`allow_urls` replaces it wholesale, so a repo or machine overlay cannot
silently *append* to a narrower global allow-list — it must restate the
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
  `random`, `full`. `/dev/tty` is **not** exposed — TTY access lets a
  child write escape sequences to the controlling terminal of the parent.
- Mounts a fresh `proc` (private to the new PID namespace). If that
  fails on the host kernel, `/proc` is left empty inside the jail rather
  than bind-mounting the host `/proc` — the latter would expose host
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

### 3. Profile selection

You *set* the `sandbox.profile` field; it resolves against the host to an
*effective profile* — what actually runs. There is no `none` value (you cannot
ask for it) and no `auto` effective profile (it is resolved away). No silent
downgrade: an explicit request the host can't satisfy is refused, not weakened.

| `sandbox.profile` | Host | Effective profile |
|---|---|---|
| `auto` *(default)* | Linux, user namespaces available | `strict` |
| `auto` | Linux, no user namespaces | `hardened` |
| `auto` | non-Linux (macOS / Windows) | `none` |
| `strict` | Linux, user namespaces available | `strict` |
| `strict` | anything else | ⛔ refuse to run |
| `hardened` | Linux (user namespaces or not) | `hardened` |
| `hardened` | non-Linux (macOS / Windows) | ⛔ refuse to run |

The three effective profiles:

- **strict** — full namespaces (user/mount/pid/ipc/uts/net) + `pivot_root` +
  Landlock + seccomp + `capset(0)` + rlimits + `NO_NEW_PRIVS`.
- **hardened** — Landlock + seccomp + `capset(0)` + rlimits + `NO_NEW_PRIVS`,
  but **no namespaces** (so it works inside default-seccomp Docker, where the
  container blocks the inner `clone(CLONE_NEW*)`; the container is the blast
  radius).
- **none** — *unsandboxed*: child commands run as plain subprocesses with no
  kernel-enforced confinement; always with a loud warning.

CI should set `profile = "strict"` to fail loudly if the sandbox is weaker than
expected. "User namespaces available" means `unshare -U -r true` succeeds.

### 4. Fixed tool surface

The LLM only ever sees the fixed, audited set declared in
`src/agent6/tools/schema.py` (enumerated in the README): structured
edits, read-only navigation, fixed-argv verify/metric commands, a
terminal `finish_run`, a curator-backed task notepad, and the
capability-gated `run_command`. There is no `shell`, no `write_file`
(writes go through `apply_edit`, an in-process rewriter that refuses
paths outside cwd), no `web_fetch`, and no `eval`. Adding a tool requires
a security review note in the commit message — see [AGENTS.md](AGENTS.md).

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
ignored — they will stay ignored until there is a concrete review of what
a "safe push" would look like.

Every `git_ops` invocation is also hardened against **repo-controlled host
code execution**: a cloned/poisoned `.git/config` can otherwise run a
command on the host (outside the jail) the moment agent6 runs git in it.
`core.fsmonitor` (fires on index refresh) and `diff.external` (fires on
`git diff`) are always overridden off; the repo's `.git/hooks/*` run only
when `git.run_repo_hooks = true` (default false — `core.hooksPath` is
pointed away from the repo so a `pre-commit` hook can't fire on agent6's
own auto-commit). This complements `protect_git`, which stops the worker
from *writing* into `.git` in the first place.

### 5b. Secrets, `connect`, and running as root

- **Secrets at rest.** Provider API keys live in
  `$XDG_CONFIG_HOME/agent6/secrets.toml`, created and enforced `0600`
  (owner read/write only). agent6 refuses to read the file if it is
  group/other-accessible or owned by another user — the same posture as
  an SSH private key. Keys may alternatively come from an environment
  variable named by `[providers.<name>].api_key_env`; the env var takes
  precedence. Keys are never written to transcripts, never printed by
  `agent6 config show` (redacted), and never mounted into the jail —
  provider calls happen in agent6's own process, outside the sandbox.
- **`agent6 connect` never executes remote input.** The connect flow only
  prompts locally (key via `getpass` — hidden, or masked with `*` on Python
  3.14+) and writes config/secrets. It does not run any command, URL, or
  script returned by a provider or any remote — by construction. This is a
  deliberate guard against the class of bug where a login flow runs an
  attacker-supplied shell command.
  agent6 also opens no listening network socket of any kind (MCP is
  stdio; the egress broker is a private Unix socket).
- **Root.** Running an LLM-driven agent as root is dangerous, so agent6
  refuses unless the operator explicitly opts in with `--allow-root` (or
  `AGENT6_ALLOW_ROOT=1`), and prints a loud banner. When invoked through
  `sudo`, agent6 resolves the *real* user from `SUDO_UID`/`SUDO_GID`/
  `SUDO_USER`, reads that user's config + secrets (not root's), and
  `chown`s anything it writes under the repo's `.agent6/` back to them so
  no root-owned files are left behind. agent6 does not drop privileges
  in-process: under `sudo` the worker's verify/run commands are expected
  to need root and run as root inside the jail, so the jail — not the
  process uid — is the security boundary.

### 6. Curator subprocess

All writes to `.agent6/runs/<run-id>/` go through a separate
`agent6-curator` subprocess with its own jail policy that allows writes
only to that directory. The main agent process talks to it over a
Unix domain socket inside the run directory and never writes graph data
directly. This means a bug in the worker / planner cannot corrupt the
on-disk task graph, and the curator's append-only `graph.jsonl` is the
durable source of truth.

### 7. No agent-owned network surface

agent6 does not run an HTTP server, gRPC server, or any other accept-side
socket. The only sockets it opens are:

- outbound HTTPS to the LLM provider,
- a per-run Unix domain socket under `.agent6/runs/<run-id>/` with mode
  `0600` for talking to its own curator.

There is no telemetry, no auto-update, no remote control plane, and no
shared state outside the project directory.

### 8. State-machine egress (the supervisor model) + script bundles

A `agent6 machine run` engine is a thin **supervisor** that stays in the host
network namespace and makes no network calls itself. Each `agent` state runs in
its own subprocess that confines its egress per `sandbox.agent_network` (the
broker, §1b); each `tool` state is jailed by the engine, so a per-tool
`allow_network` decides its netns independently of the agent. This is what lets
a machine confine its agents to the provider API while letting one **audited,
deterministic** tool reach the network — a `tool` command is fixed/operator-
reviewed (unlike `run_command`, whose argv the LLM chooses), so a networked
audited tool is not a free exfiltration channel.

Egress is set by `sandbox.agent_network`, `sandbox.tool_network`, and a per-tool
`allow_network`; the effective profile (§3) decides what is *enforceable*. The
tables cover every case; "offline" = no egress.

**Agent-process egress** — the agent's own LLM/provider HTTP, by `sandbox.agent_network`:

| `sandbox.agent_network` | `strict` | `hardened` | `none` |
|---|---|---|---|
| `providers` *(def)* | provider endpoints + `allow_urls`, broker-pinned (§1b) | provider *ports* only (Landlock) | unconfined ⚠ |
| `local` | loopback providers only, broker-pinned (refuse to run if any provider isn't loopback) | ⛔ refuse to run | unconfined ⚠ |
| `open` | unconfined | unconfined | unconfined ⚠ |

**Jailed-command egress** — `run_command` and machine `tool` states, by
`sandbox.tool_network` (columns; cells are the `strict` profile):

| jailed command | `block` *(def)* | `only_explicit_states` | `allow` |
|---|---|---|---|
| `run_command` | offline | offline | host network |
| `tool`, `allow_network = "auto"` (def) / `"block"` | offline | offline | offline |
| `tool`, `allow_network = "allow"` | ⛔ refuse to run | host network | host network |

**Refusals** — these configurations **refuse to run** (fail-closed):

| Configuration | When |
|---|---|
| `sandbox.tool_network = "allow"` without `sandbox.agent_network = "open"` | config load, any profile ¹ |
| a `tool` sets `allow_network = "allow"` under `sandbox.tool_network = "block"` | machine start, any profile |
| `sandbox.agent_network = "local"` or `sandbox.tool_network = "only_explicit_states"` | run start, `hardened` ² |
| a machine with `tool` states under `sandbox.tool_network = "block"`, or a `tool` with `allow_network = "block"` | machine start, `hardened` ² |

- ⚠ `none` (non-Linux) is **unsandboxed**: nothing above is enforced and nothing
  is refused — the run proceeds with a loud warning.
- ¹ `run_command` runs inside the agent process, so it can't reach the network
  while the agent is confined — hence `sandbox.tool_network = "allow"` needs
  `sandbox.agent_network = "open"`.
- ² `sandbox.tool_network`'s per-command isolation needs a network namespace, so
  it is **`strict`-only**. On `hardened` (no namespaces) a jailed child instead
  inherits the agent's Landlock and follows `sandbox.agent_network`; the cases
  that would need real per-command isolation are refused rather than mis-confined.

Every surface fails closed:

- **Operator-gated, machine-declared.** `sandbox.agent_network`/
  `sandbox.tool_network` are read only from the operator's global/repo config — a
  machine's `[config]` overlay (possibly LLM-drafted or shared) is rejected at
  load if it declares `[providers.*]` or `[sandbox.*]`. A `tool` merely
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
  are made **read-only in every jail** (alongside `.git`/`.agent6`), so a
  tool or agent state cannot rewrite its own logic, add an `allow_network`
  flag, or alter an audited script mid-run or for a future run.

## Prompt-injection resilience

The test suite under [`tests/security/test_prompt_injection.py`](tests/security/test_prompt_injection.py)
runs a small corpus of adversarial inputs through the planner, worker,
and reviewer prompts and asserts that the agent does not exfiltrate
file content, does not attempt out-of-policy tool calls, and does not
follow embedded instructions to weaken its own constraints.

This is a smoke test, not a proof. The structural defenses above
(sandbox, fixed tool surface, git invariants) are the real mitigation —
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
  agent6 ships such a profile, scoped to just the launcher binary, in
  [packaging/apparmor/agent6-jail](packaging/apparmor/agent6-jail) — the
  surgical fix (preferred over disabling the sysctl host-wide). agent6's
  profile detection probes the *real launcher binary* (not
  `/usr/bin/unshare`), so once the profile is installed it correctly
  selects `strict`; without it, it uses `hardened` and `agent6 check
  sandbox` prints how to enable `strict`.
- **seccomp** is required by the jail; on rare hardened kernels that
  block seccomp from unprivileged callers, the jail fails closed.
- **Devcontainers**: the jail's `hardened` profile is what you get
  inside Docker / VS Code dev containers. The container itself becomes
  the FS blast radius. Network restrictions still apply via the
  agent-process Landlock when the kernel supports it.
- **Side channels**: agent6 makes no claim about timing, cache, or
  speculative-execution side channels. If your threat model includes
  Spectre-class attacks, do not co-locate agent6 on a host with secrets.
- **Supply chain**: pin your install. The runtime deps are `pydantic`,
  `httpx`, `argcomplete`, the `tree-sitter` pair (`tree-sitter` +
  `tree-sitter-language-pack`), and `textual` (the live dashboard);
  build-dep is `hatchling`; and the jail pulls a small set of well-known
  Rust crates (`nix`, `libc`, `landlock`, `seccompiler`, `serde`,
  `serde_json`). Verify before upgrading any of them.
