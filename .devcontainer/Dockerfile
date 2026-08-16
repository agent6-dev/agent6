FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
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

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH="/home/vscode/.local/bin:/home/vscode/.cargo/bin:${PATH}"
