# Application development

- Python 3.14.x is pinned in `.python-version` and managed with `uv`.
- Run the API locally with `uv run uvicorn app.main:app --reload`.
- Apply database migrations with `uv run alembic upgrade head` before running
  the API locally; Docker applies them on API startup.
- Run the asynchronous ingestion worker with
  `uv run celery -A app.worker.celery_app worker --loglevel=INFO --queues=document-ingestion`.
- Run the durable outbox dispatcher with
  `uv run celery -A app.worker.celery_app beat --loglevel=INFO`.
- Redis and PostgreSQL are required for uploads and ingestion jobs.
- Run tests from the repository root with `uv run --locked pytest`.
- Run live database, broker, and worker coverage using `tests/e2e/compose.yml`
  and `uv run pytest tests/e2e`.
- Keep secrets in environment variables; never commit `.env`.
- Use `app/agent.py` for retrieval, chunking, and RAG-chain behavior.
- Configure database, models, and deployment behavior through `app/config.py`.
- HTTP upload routes live in `app/api/documents.py`; job lifecycle and outbox
  persistence live in `app/database/` and `app/ingestion/`.
