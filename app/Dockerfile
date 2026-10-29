FROM python:3.15.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    APP_ENV=production

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libzstd-dev libzstd1 \
    && pip install --no-cache-dir uv \
    && useradd --create-home --uid 10001 appuser

COPY pyproject.toml uv.lock README.md ./
COPY app/ ./app/

RUN uv sync --frozen --no-dev --no-editable \
    && apt-get purge -y --auto-remove build-essential libzstd-dev \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R appuser:appuser /app

USER appuser
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["sh", "-c", "python -c \"import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '8000'), timeout=3)\""]

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
