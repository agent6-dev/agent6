# Installation

## From PyPI

With [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pipx](https://pipx.pypa.io/stable/how-to/install-pipx.html):

=== "uv"

    ```sh
    uv tool install agent6
    ```

=== "pipx"

    ```sh
    pipx install agent6
    ```

Both put the `agent6` entry point in `~/.local/bin`.
If that directory is not on your `PATH`:

=== "uv"

    ```sh
    uv tool update-shell
    ```

=== "pipx"

    ```sh
    pipx ensurepath
    ```

Restart the shell afterwards.

## Shell completion

- rerunning is safe and refreshes it

```sh
agent6 completions          # or: agent6 completions bash|zsh|fish|xonsh
```

- `--print bash` emits the script instead, for `eval` or a dotfiles repo

## Check the install

```sh
agent6 --version
agent6 check                # sandbox (isolation, boundaries, probes), config (provider keys, git version), MCP, verify
```

`agent6 check sandbox` runs the jail through live probes and reports the isolation level a run will use on your kernel.
`agent6 check verify` infers a gate or checks its first executable without running it.
For rustup's `cargo`, follow the [Rust toolchain setup](security.md#rust-toolchains).

## Requirements

- Python 3.12 or newer
- A C compiler on Python 3.15 (`tree-sitter` ships no wheel for it)
- A Rust toolchain where `tree-sitter-language-pack` has no wheel (it ships wheels for glibc 2.34 and newer on x86_64 and aarch64, and builds with maturin elsewhere)
- git 2.47 or newer
- One provider: Anthropic, any OpenAI-compatible endpoint (a local one needs no key), a ChatGPT subscription, or a Claude Code login (`agent6 connect claude`)
- A Linux release as recent as Debian 13 or Ubuntu 26.04 LTS; older ones are not supported
- Linux on x86_64 or aarch64 for the sandbox
- Unprivileged user namespaces for `full` `sandbox.isolation`
- A Rust toolchain to build from source (the PyPI wheels bundle `agent6-jail`)

- On other architectures and on macOS, `sandbox.isolation = "auto"` resolves to `none`: commands run as ordinary subprocesses behind a startup warning, and an explicit `full` or `partial` refuses.
- On Windows, use WSL.
- Unprivileged user namespaces are on by default on Debian
- On Ubuntu, AppArmor blocks them (`kernel.apparmor_restrict_unprivileged_userns = 1`): set it to 0, or `agent6 system apparmor install` (`agent6 system apparmor remove` removes the profile, not the sysctl setting)
- Without user namespaces `sandbox.isolation = "auto"` falls back to `partial`, and to `none` on a kernel with no Landlock.

The [security model](security.md) describes what each isolation level enforces.

## From source

```sh
git clone https://github.com/agent6-dev/agent6
cd agent6
uv sync
uv run agent6 --help
```

`AGENT6_JAIL_BIN=/path/to/agent6-jail` overrides the bundled jail binary.

## Static launcher for the internal container transport

Container transfer requires a static Linux launcher matching the host and target architecture, x86-64 or aarch64.
Published Linux wheels bundle their architecture's static-musl binary.
A native editable build may be dynamically linked and is valid for local execution, but container preparation refuses it.
To build the transferable binary from source on Linux:

```sh
target="$(uname -m)-unknown-linux-musl"
rustup target add "$target"
AGENT6_JAIL_TARGET="$target" uv sync --reinstall-package agent6
```

The target needs `/bin/sh`, `uname`, `mktemp`, `head -c`, `chmod`, executable writable `/tmp` and readable `/proc/self`.
It needs no Python, Node or Rust toolchain.
Preparation never installs dependencies, downloads a launcher or falls back to the host.
The transport is internal; application config does not activate it.
See the [installation trust boundary](security.md#container-transport-boundary).

The opt-in runtime test first attaches to this repository's existing labelled dev container with a non-root user and a different workspace path.
It inspects target evidence and copied logs, then creates a disposable container from the same local image for runtime-client loss and restart checks.
Only that test-created container receives lifecycle commands; the existing dev container stays running.
Set `AGENT6_CONTAINER_TEST_ROOT` if its host checkout differs from the test checkout.
The runtime prefix is explicit and must follow the host's approved CLI policy; agent6 never adds sudo.

```sh
AGENT6_JAIL_BIN=/absolute/path/to/static/agent6-jail \
  AGENT6_CONTAINER_TEST_RUNTIME='["sudo", "-n", "docker"]' \
  uv run pytest -q -s tests/sandbox/test_container_runtime.py
```

Use `["podman"]` instead on a host whose approved runtime is rootless Podman.
An absent matching dev container fails the opted-in test rather than substituting another image.
