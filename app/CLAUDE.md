# Application development

- Python 3.15.2+ is managed with `uv`.
- Run the API locally with `uv run uvicorn app.main:app --reload`.
- Run tests from the repository root with `uv run pytest`.
- Keep secrets in environment variables; never commit `.env`.
- Use `app/agent.py` for retrieval, chunking, and RAG-chain behavior.
- Configure database, models, and deployment behavior through `app/config.py`.
