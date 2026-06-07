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
   enforced structurally only when `sandbox.network = "provider_only"`
   (see Defense Layer 1b); with `network = "allow"`/`"no"` the agent
   process retains host network access and only the destination *port* is
   constrained by Landlock.
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

- FS read: cwd, `$HOME`, `/usr`, `/etc`, `/tmp`, the common `/dev`
  character devices, and `/run`/`/proc` when present.
- FS write: cwd, `/tmp`, the `/dev` character devices, and `/proc` when
  present.
- TCP connect (kernel ≥ 6.7, Landlock ABI ≥ 4): Landlock's network hook
  filters by **destination port only** — there is no host/IP primitive in
  Landlock. agent6 allows the set of ports used by configured
  `[providers.<name>]` endpoints (`443` for each `kind = "anthropic"`
  entry; the port of `base_url` for each `kind = "openai"` entry — default
  `443`, but any OpenAI-compatible base URL is honoured: OpenRouter,
  Ollama, vLLM, llama.cpp, …). This stops the agent from connecting to
  arbitrary services on other ports, but on its own it does **not** pin
  egress to a specific host: anything reachable on an allowed port is
  permitted. For true host-level egress confinement use
  `sandbox.network = "provider_only"` (Defense Layer 1b).

On older kernels TCP-connect rules are not available; agent6 prints a
loud warning and runs with FS-only Landlock. In that mode the agent
process is still confined for filesystem access, but you should not run
it on a host where the agent's UID has access to sensitive credentials
the agent could open and send over the network.

### 1b. Provider-only egress broker (`sandbox.network = "provider_only"`)

When enabled (strict profile only — it relies on unprivileged user
namespaces), `agent6 run` confines its own process to host-level egress:

1. While still in the host network namespace and single-threaded, the
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
hardened profile the run is **refused** rather than executed unconfined.

Curator and other `AF_UNIX`-based helpers are unaffected (unix sockets
cross the netns boundary). MCP servers that need their own outbound
network access will not have it under `provider_only`; that is a
deliberate limitation, not a bug.


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

`sandbox.profile` ∈ `{auto, strict, hardened}`:

- **strict** — full namespaces + pivot_root + landlock + seccomp +
  capset(0). Requires unprivileged user namespaces. Refuses to start if
  unavailable.
- **hardened** — no namespaces (so it works inside default-seccomp
  Docker, where the container blocks the inner `clone(CLONE_NEW*)`),
  but still landlock + seccomp + capset(0) + rlimits + NO_NEW_PRIVS.
  The container itself is the blast radius.
- **auto** — try strict; if the kernel + container disallow it, fall
  back to hardened. Logs the chosen profile on every run.

CI should use `profile = "strict"` to fail loudly if the sandbox is
weaker than expected.

### 4. Fixed tool surface

The LLM only sees the tools declared in
`src/agent6/tools/schema.py`:

`read_file`, `list_dir`, `grep`, `apply_edit`, `run_verify_command`,
and (capability-gated) `run_command`.

There is no `shell`, no `write_file` (writes go through `apply_edit`,
which is an in-process rewriter that refuses paths outside cwd), no
`web_fetch`, no `eval`, no MCP. Adding a tool requires a security review
note in the commit message — see [AGENTS.md](AGENTS.md).

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
  by default; agent6 will detect that and refuse `strict`.
- **seccomp** is required by the jail; on rare hardened kernels that
  block seccomp from unprivileged callers, the jail fails closed.
- **Devcontainers**: the jail's `hardened` profile is what you get
  inside Docker / VS Code dev containers. The container itself becomes
  the FS blast radius. Network restrictions still apply via the
  agent-process Landlock when the kernel supports it.
- **Side channels**: agent6 makes no claim about timing, cache, or
  speculative-execution side channels. If your threat model includes
  Spectre-class attacks, do not co-locate agent6 on a host with secrets.
- **Supply chain**: pin your install. The runtime dep list is
  `pydantic` and `httpx`; build-dep is `hatchling`; and the jail
  pulls a small set of well-known Rust crates (`nix`, `libc`,
  `landlock`, `seccompiler`, `serde`, `serde_json`). Verify before
  upgrading any of them.
