# Production RAG starter

A single-collection retrieval-augmented chat API built with FastAPI, LangChain,
an OpenAI-compatible local LLM endpoint, and PostgreSQL/pgvector. It provides
document chunking and ingestion, similarity retrieval, source-aware answers,
API-key protection, rate limiting, health checks, and structured logs. It is
intended as a starting point: add your application's identity, tenancy,
authorization, retention, and evaluation policies before exposing it to users.

## Requirements

- Python 3.15.2+
- PostgreSQL with the `vector` extension (or Docker Compose)
- A local Ollama server with the configured chat and embedding models
- [uv](https://docs.astral.sh/uv/)

## Local development

```powershell
Copy-Item .env.example .env
```

Start the API, pgvector database, and Ollama server:

```powershell
docker compose -f app/docker-compose.yml up --build
```

Download the default local chat and embedding models into the Ollama service.
Model downloads are persisted in a Docker volume:

```powershell
docker compose -f app/docker-compose.yml exec ollama ollama pull llama3.2:1b
docker compose -f app/docker-compose.yml exec ollama ollama pull nomic-embed-text
```

The default settings use Ollama's OpenAI-compatible `/v1` API. No OpenAI
account or API key is needed. The application supplies the compatibility
placeholder expected by the OpenAI client; Ollama does not require
authentication on this local Compose network. If using a secured,
OpenAI-compatible endpoint, set `LLM_API_KEY` as a secret in that environment.

The API is available at `http://localhost:8000`; interactive API documentation
is at `/docs`. Compose maps PostgreSQL to `localhost:5432` for local tools.
The bundled `rag:rag` database credentials are for local development only.

To run the tests:

```powershell
uv sync --group dev
uv run pytest
```

API integration tests use FastAPI's test client and stub external providers, so
they don't need API keys or a running PostgreSQL instance:

```powershell
uv run pytest -m integration
uv run pytest -m "not integration"
```

## Index documents

Set a long, unique `INGEST_API_KEY` in the API environment to enable the
ingestion endpoint. It is disabled when the key is unset. Upload text
documents as JSON:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/documents `
  -Headers @{ "X-Ingest-API-Key" = "your-ingest-key" } `
  -ContentType "application/json" `
  -Body '{"documents":[{"source":"handbook/leave-policy","content":"Employees receive 20 days of annual leave."}]}'
```

Documents are split into overlapping chunks and embedded before being stored
in the configured pgvector collection. Re-ingesting identical source/chunk
content updates the existing vector IDs. This starter has no document deletion
or source-replacement endpoint; rebuild the collection when replacing a corpus,
or add source-scoped deletion for your document lifecycle. Configure the chunk
size, overlap, embedding model, collection, and retrieval count with the
environment settings in `.env.example`.

The chat chain can expand a question into alternative search queries, applies
pgvector maximal-marginal-relevance retrieval, and fuses repeated results
across queries. These defaults help with vocabulary mismatch and redundant
chunks; disable `QUERY_EXPANSION_ENABLED` to avoid its additional model call.
`MAX_CONTEXT_CHARS` bounds the prompt context, and answers without valid source
citations are replaced with a safe abstention. These are quality safeguards,
not guarantees of factual correctness; evaluate them against your corpus.

## Chat API

In development, chat API-key enforcement is optional. Production mode
(`APP_ENV=production`) requires `CHAT_API_KEY` at startup. Send it using the
`X-API-Key` header:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/chat `
  -Headers @{ "X-API-Key" = "your-chat-key" } `
  -ContentType "application/json" `
  -Body '{"message":"How many days of annual leave do employees receive?","thread_id":"demo"}'
```

Responses include source labels, the selected model, cache status, and request
processing time. `thread_id` is a caller-provided correlation value only; this
starter does not persist conversational history. The in-memory response cache
and rate limits are per process and are not shared across replicas. Set
`CACHE_TTL_SECONDS=0` to disable response caching.

## Deployment

The repository includes a Render Blueprint in `render.yaml` and a production
Docker image in `app/Dockerfile`. For a cloud deployment:

1. Provision PostgreSQL with pgvector enabled. Use a managed database with
   backups, TLS, and network restrictions; the local Compose database is not a
   production database.
2. Deploy the repository using the Render Blueprint or build the Docker image.
3. Configure `LLM_BASE_URL`, `DATABASE_URL`, and a strong `CHAT_API_KEY` as
   deployment settings/secrets. `LLM_API_KEY` is only needed if your
   OpenAI-compatible local model endpoint requires authentication. Set
   `INGEST_API_KEY` if remote document ingestion is required. Use the
   provider's pgvector-compatible PostgreSQL URL, for example
   `postgresql+psycopg://...`.
4. Configure database access so only the application can connect. Run the
   service behind a trusted proxy/load balancer and use HTTPS.
5. Review the provider plan, database sizing, backup/recovery, log retention,
   rate limits, and cost controls for the expected workload.

`/health/live` is a process liveness check; `/health` is a readiness check that
also checks PostgreSQL. `/metrics` and `/cache/stats` are process-local starter
metrics, not a replacement for a production metrics backend.

## Production adaptation notes

- This template uses one configured vector collection and has no tenant
  isolation, user identity, or per-user authorization. Add those before
  indexing private or customer-specific content.
- Protect `/documents` with a separate ingestion key and expose it only to
  trusted operators. Do not send untrusted or oversized source data.
- Prompt-injection pattern checks are heuristics, not a security boundary.
  Treat retrieved content as untrusted, validate sources and outputs, and add
  application-specific abuse controls and evaluations.
- Query expansion adds a model call and can increase latency/cost. Tune
  retrieval, chunking, and context limits against a representative evaluation
  set. This implementation does not provide GraphRAG, multimodal indexing, or
  late chunking; add those only when the corpus and evaluation show they help.
- The default cache and rate limiter are in-memory. Use a shared backend and
  identity-aware cache keys when scaling horizontally or serving personalized
  answers.
- LangSmith tracing is not configured by the application. If you enable it
  through LangChain's standard environment variables, review its data
  retention and privacy settings before sending prompts or retrieved content.
