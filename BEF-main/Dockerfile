FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev libsecp256k1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY bef_zk bef_zk
COPY backends backends
COPY capsule_bench capsule_bench
COPY scripts scripts
COPY policies policies
COPY server server
COPY docs docs

RUN pip install --upgrade pip && pip install --no-cache-dir .

CMD ["capsule-bench", "--help"]
