FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_NO_SYNC=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV CC=gcc
ENV PYTHONPATH=/workspaces/TorchJD/tests
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /tmp/torchjd-install

COPY pyproject.toml README.md ./
RUN mkdir -p src/torchjd && touch src/torchjd/__init__.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
    && uv pip install --python-version=3.14 -e '.[full]' --group check --group doc --group test --group plot

COPY src/ src/

WORKDIR /workspaces/TorchJD
