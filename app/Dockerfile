FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    APP_ENV=production

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY app/ ./app/
COPY migrations/ ./migrations/

RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips \"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
