# agent6 devcontainer image.
#
# Base: the latest Ubuntu LTS devcontainer image (noble = 24.04). The base
# already sets up the non-root `vscode` user, sudo, common dev tooling, and
# the standard `/workspaces` mount layout, so we deliberately do not duplicate
# that work.
#
# We do NOT use devcontainer "features" — every dependency the project needs
# is installed below so the image is auditable from a single Dockerfile and
# reproducible without the features layer's network behavior.
#
# Note on sandboxing: agent6's strict sandbox profile relies on host kernel
# Landlock + namespace support. Inside a stock devcontainer those primitives
# usually are not available; the agent will fall back to its `auto` profile
# accordingly. We intentionally do NOT add `--privileged`, `--cap-add`,
# `securityOpt: seccomp=unconfined`, or any other escape hatch — losing
# sandboxing inside the dev container is the correct trade-off, weakening
# the host's container boundary is not.
FROM mcr.microsoft.com/devcontainers/base:noble

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System packages:
#   - python3.12 + venv + dev headers for building wheels (pydantic-core, etc.)
#   - build-essential / pkg-config / libssl-dev / curl: needed to build the
#     Rust `agent6-jail` crate and to install rustup + uv
#   - git: required by agent6.git_ops at runtime
#   - ca-certificates: TLS for httpx and rustup downloads
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        pkg-config \
        libssl-dev \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

USER vscode
WORKDIR /home/vscode

# uv: the project's only Python package manager (pyproject.toml + uv.lock).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust toolchain for building the sandbox launcher (`jail/`). Installed as
# the non-root user so cargo's $HOME-resident state stays inside the user's
# home and does not require sudo for routine `cargo build`s.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH="/home/vscode/.local/bin:/home/vscode/.cargo/bin:${PATH}"
