FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_NO_SYNC=1
ENV CC=gcc
ENV PYTHONPATH=/workspaces/TorchJD/tests
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY pyproject.toml /tmp/torchjd-install/pyproject.toml
COPY README.md /tmp/torchjd-install/README.md
COPY src/ /tmp/torchjd-install/src/

WORKDIR /tmp/torchjd-install
RUN uv venv /opt/venv \
    && uv pip install --python-version=3.14 -e '.[full]' --group check --group doc --group test --group plot

WORKDIR /workspaces/TorchJD
