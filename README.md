# Production RAG starter

A single-collection retrieval-augmented chat API built with FastAPI, LangChain,
an OpenAI-compatible local LLM endpoint, and PostgreSQL/pgvector. It provides
asynchronous document uploads, a Redis/Celery ingestion worker, hybrid
keyword/vector retrieval, reranking, source-aware answers, API-key protection,
rate limiting, health checks, and structured logs. It is intended as a
starting point: each deployment uses one server-configured tenant namespace.
Add your application's user identity, per-user authorization, retention, and
evaluation policies before exposing it to users.

## Requirements

- Python 3.14.x (the locked and container-tested runtime)
- PostgreSQL with the `vector` extension (or Docker Compose)
- Redis (included in Docker Compose)
- A local Ollama server with the configured chat and embedding models
- [uv](https://docs.astral.sh/uv/)

The project pins Python to the 3.14 minor line in `.python-version` and
`pyproject.toml`. CI and Docker use the same Python line and pin uv to the same
release; use `uv sync --locked` to install the committed lockfile.

## Local development

For one-click startup, double-click `Start-App.bat` on Windows,
`Start-App.command` on macOS, or `Start-App.desktop` on Linux. On macOS/Linux,
make the launcher scripts executable once from a terminal:

```sh
chmod +x start-app.sh Start-App.command Start-App.desktop
```

Some Linux desktop environments may ask you to trust/allow launching the
`.desktop` file the first time. The launchers build and start the Compose
services, create `.env` from `.env.example` if it does not exist, download the
default chat and embedding models only if missing, wait for the UI and API, and
open the workspace in your browser. Docker Desktop (macOS) or the Docker
service (Linux) must be running; the first model downloads can take several
minutes.

You can also launch from a terminal:

```powershell
.\start-app.ps1
```

```sh
./start-app.sh
```

To start the services manually instead:

```powershell
docker compose -f app/docker-compose.yml up --build -d
docker compose -f app/docker-compose.yml exec ollama ollama pull llama3.2:1b
docker compose -f app/docker-compose.yml exec ollama ollama pull nomic-embed-text
```

The default settings use Ollama's OpenAI-compatible `/v1` API. No OpenAI
account or API key is needed. The application supplies the compatibility
placeholder expected by the OpenAI client; Ollama does not require
authentication on this local Compose network. If using a secured,
OpenAI-compatible endpoint, set `LLM_API_KEY` as a secret in that environment.
For local development, the API defaults to PostgreSQL at
`localhost:5432` with the Compose `rag` credentials, the `local` tenant, and
the ingestion key `local-ingest-key`. Compose overrides the database and model
hosts to the internal `db` and `ollama` services. These credentials and the
default ingestion key are for loopback/local development only.

The browser workspace is a Svelte/TypeScript frontend served by the Vite
development server in its own Docker Compose service. Compose explicitly
builds the Dockerfile's `development` stage, so `http://localhost:5173` retains
hot reload and proxies API requests to FastAPI on the private Compose network.
The default frontend Docker image instead builds the static site and serves it
with Nginx; set `API_PROXY_TARGET` to the backend API root when running that
image. The Render blueprint currently does not deploy a frontend service. The
API remains available at `http://localhost:8000`, with interactive
documentation at `/docs`. Compose maps PostgreSQL to `localhost:5432` for
local tools.
The bundled `rag:rag` database credentials are for local development only.

To run the tests:

```powershell
uv sync --locked --group dev
uv run ruff check .
uv run pytest
```

API integration tests use FastAPI's test client and stub external providers, so
they don't need API keys or a running PostgreSQL instance:

```powershell
uv run pytest -m integration
uv run pytest -m "not integration"
```

The end-to-end tests upload every supported file format (PDF, PPTX, DOCX,
Markdown, and text) through FastAPI, PostgreSQL, Redis, and a real Celery
worker. They require PostgreSQL with pgvector and Redis. Start only those
dependencies, then run the tests:

```powershell
docker compose -f tests/e2e/compose.yml up -d --wait
$env:E2E_DATABASE_URL = "postgresql+psycopg://rag:rag@127.0.0.1:55432/rag_e2e"
$env:E2E_REDIS_URL = "redis://127.0.0.1:6379/0"
uv run pytest tests/e2e
docker compose -f tests/e2e/compose.yml down
```

Each test applies Alembic migrations, uploads a generated document through
FastAPI, dispatches it through Redis, waits for the Celery worker, and verifies
the PostgreSQL job and corpus generation state. It substitutes only the
external LLM/vector indexer, so it does not need a model download. Unit and
deployment tests also cover parser errors, storage failures, response security
headers, Compose/Render topology, image runtime settings, and service outages.

The optional live integration tests exercise a configured OpenAI-compatible
chat/embedding provider, pgvector, and S3-compatible storage. They are
intentionally excluded unless explicitly enabled, to avoid contacting external
services or writing to a deployment by default. Configure a dedicated test
database, model endpoint, and an isolated S3 bucket before opting in:

```powershell
$env:RUN_LIVE_SERVICE_TESTS = "1"
$env:LIVE_DATABASE_URL = "postgresql+psycopg://user:password@host/test_db"
$env:LIVE_LLM_BASE_URL = "https://your-model-endpoint/v1"
$env:LIVE_LLM_API_KEY = "your-test-provider-key"
$env:LIVE_CHAT_MODEL = "your-chat-model"
$env:LIVE_EMBEDDING_MODEL = "your-embedding-model"
$env:LIVE_TENANT_ID = "isolated-live-test-tenant"
$env:LIVE_S3_ENDPOINT_URL = "https://your-object-storage-endpoint"
$env:LIVE_S3_BUCKET = "your-isolated-test-bucket"
$env:LIVE_S3_ACCESS_KEY_ID = "your-test-access-key"
$env:LIVE_S3_SECRET_ACCESS_KEY = "your-test-secret-key"
uv run pytest -m live_services
```

The model/vector test creates and removes a uniquely named vector collection.
Set `LIVE_TENANT_ID` to an isolated tenant used only for these tests.
The S3 test writes a uniquely named object beneath `documents/` and deletes
that object after the round trip. Use least-privilege credentials scoped to
that test bucket and a disposable database; the model check makes real provider
requests and may incur provider charges.

## Upload and index documents

Local development defaults to `INGEST_API_KEY=local-ingest-key`. Replace it
with a long, unique secret for any non-local deployment; uploads are disabled
when the key is unset there. The preferred upload API accepts PDFs, PowerPoint
`.pptx`, Word `.docx`, Markdown, and UTF-8 text files. It stores the file and
queues a job; the request does not perform parsing, embedding, or vector writes:

```powershell
$response = curl.exe --fail --silent --show-error `
  --request POST `
  --header "X-Ingest-API-Key: local-ingest-key" `
  --form "file=@.\handbook.pdf" `
  http://localhost:8000/documents/upload | ConvertFrom-Json
```

The response contains a `document_id`; poll its status (also protected by the
ingestion key) until it is `completed` or `failed`:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/documents/$($response.document_id)" `
  -Headers @{ "X-Ingest-API-Key" = "local-ingest-key" }
```

The worker extracts page/slide text and table rows, chunks it, and indexes it
into pgvector. Scanned PDFs without an embedded text layer are reported as
failed; connect an OCR/layout parser for those documents before production
use. Re-uploading an unchanged source is idempotent. Uploading a changed file
with the same filename replaces its previous chunks after new embeddings are
written. Job status and checksums are stored in PostgreSQL. A transactional
outbox commits each job with the document revision; a Celery Beat dispatcher
retries broker publication and periodically recovers queued messages that were
lost. Workers use late acknowledgements, bounded prefetch, retries, and
per-source advisory locks.
Uploads are size-limited and extensions are allow-listed. PDF signatures and
OOXML ZIP structure are checked; text and Markdown files must decode as UTF-8.
OOXML archives also have entry-count and expansion limits.

The JSON `POST /documents` endpoint remains available for small, already
extracted text batches. It indexes synchronously and enforces
`MAX_INGEST_CHARS`; use `/documents/upload` for large files.

In Compose, uploaded files use a shared volume mounted by both API and worker.
Production startup requires a shared S3-compatible bucket because API and
worker processes have separate filesystems. Configure it with `S3_ENDPOINT_URL`,
`S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, and `S3_REGION`.
Provide private, least-privilege bucket credentials; object storage is not
created automatically by the blueprint. Production also fails fast unless
PostgreSQL, Redis, a non-local HTTPS LLM endpoint, `CHAT_API_KEY`, and a
deployment-specific `TENANT_ID` are configured.

Retrieval combines BM25-ranked PostgreSQL full-text keyword candidates and
pgvector maximal-marginal-relevance candidates using a custom
reciprocal-rank-style score: each result contributes `1 / (60 + rank +
query_index)`, so later expanded queries are slightly discounted. Candidates
are then reranked. Configure `COHERE_API_KEY` to use Cohere reranking; without
it, a local lexical reranker keeps the default Ollama/local-LLM workflow fully
offline. Query expansion can add alternative searches; disable
`QUERY_EXPANSION_ENABLED` to avoid its additional model call. The number of
rewritten queries is capped in code at `QUERY_EXPANSION_COUNT`.
`MAX_CONTEXT_CHARS` bounds the prompt context. Citation checks validate only
that answers contain citation markers within the retrieved-source range; they
do not establish that a source supports a claim. This is a formatting safeguard,
not a grounding evaluation or guarantee of factual correctness; evaluate answers
against your corpus.

The upload pipeline enforces `MAX_UPLOAD_BYTES` (50 MB by default) during
upload, then applies `MAX_DOCUMENT_PAGES` and `MAX_DOCUMENT_CHARS` (5 million
extracted characters by default) in the worker; a file that exceeds a worker
limit is accepted into the queue and later marked failed. The legacy JSON
ingestion is limited to 100 documents and `MAX_INGEST_CHARS` total content
characters per request. Both routes use `INGEST_RATE_LIMIT` (5 requests per
minute by default) and the ingestion key.

## Retrieval evaluation

Populate a JSONL dataset with one `query` and a list of relevant
`expected_sources` per line, then run:

```powershell
uv run python scripts\evaluate_retrieval.py --dataset evaluation\gold_queries.jsonl --k 5
```

The script reports Hit Rate@K and MRR@K over the configured live corpus. The
checked-in examples are a schema/template: replace the source labels and
queries with gold judgments for the corpus being evaluated. It does not call
an LLM or require an external evaluation service.

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
When retrieval finds no documents, or the document-grounded model abstains
because it cannot answer from its sources, the API answers using the model's
general knowledge instead. It labels this answer as unverified against the
uploaded documents and returns no document sources; use it for general
guidance, not as a substitute for a sourced answer.

## Deployment

The repository includes a Render Blueprint in `render.yaml` (API, worker,
outbox dispatcher, and Redis) and a production Docker image in
`app/Dockerfile`. For a cloud deployment:

1. Provision PostgreSQL with pgvector enabled. Use a managed database with
   backups, TLS, and network restrictions; the local Compose database is not a
   production database.
2. Deploy the repository using the Render Blueprint or build the Docker image.
   The API image applies Alembic migrations before serving; the worker and
   dispatcher use the same release and database. The outbox retains jobs when
   Redis is unavailable and the Beat dispatcher retries publication.
3. Configure `LLM_BASE_URL`, `DATABASE_URL`, `REDIS_URL`, a strong
   `CHAT_API_KEY`, and a distinct `TENANT_ID` as deployment settings/secrets.
   The tenant ID is fixed by server configuration (never accepted from a
   request); use the same value for that tenant's API, worker, and dispatcher,
   and a different value for every deployment sharing a database or vector
   collection. Production additionally requires Redis and S3-compatible shared
   object storage. The LLM endpoint must use HTTPS. `LLM_API_KEY` is
   only needed if your OpenAI-compatible model endpoint requires authentication.
   Set `INGEST_API_KEY` to enable document uploads. Use the provider's
   pgvector-compatible PostgreSQL URL, for example
   `postgresql+psycopg://...`.
4. Configure database access so only the application can connect. Run the
   service behind a trusted proxy/load balancer and use HTTPS. Set
   `FORWARDED_ALLOW_IPS` to the proxy's trusted IP address(es) or CIDR ranges;
   the container defaults to trusting only `127.0.0.1`. Do not use `*` unless
   network controls ensure that only your trusted proxy can reach the app.
5. Review the provider plan, database sizing, backup/recovery, log retention,
   rate limits, and cost controls for the expected workload.

`/health/live` is a process liveness check; `/health` is a readiness check that
also checks PostgreSQL. `/metrics` and `/cache/stats` are process-local starter
metrics, not a replacement for a production metrics backend.

## Production adaptation notes

- Retrieval, ingestion, document status, and job dispatch are scoped to the
  server-configured `TENANT_ID`; no tenant value is accepted from callers.
  Production requires a non-`local` tenant ID. This is tenant-level isolation,
  not per-user authorization: callers sharing a deployment's API key have the
  same access. The tenant ID is not a substitute for authenticating individual
  users when a deployment serves multiple users with different permissions.
  Existing indexed vectors and document records are assigned to `local` by
  migration; to serve existing data from production, re-ingest it under that
  deployment's tenant ID.
- Protect `/documents` with a separate ingestion key and expose it only to
  trusted operators. The API limits ingestion request size and rate, but these
  controls do not replace operator authorization or workload monitoring.
- Prompt-injection pattern checks are heuristics, not a security boundary.
  Treat retrieved content as untrusted, validate sources and outputs, and add
  application-specific abuse controls and evaluations.
- Query expansion adds a model call and can increase latency/cost. Tune
  retrieval, chunking, and context limits against a representative evaluation
  set. This implementation does not provide GraphRAG, multimodal indexing, or
  late chunking; add those only when the corpus and evaluation show they help.
- The response cache and rate limiter are in-memory. Cache keys include a
  PostgreSQL corpus generation so successful indexing bypasses stale answers,
  but rate limits are process-local. Use a shared cache and rate-limit backend
  with identity-aware keys when scaling horizontally or serving personalized
  answers. Rate limits use the client address; configure `FORWARDED_ALLOW_IPS`
  with only the deployment's trusted proxy ranges so requests are not all
  attributed to the proxy or spoofed by untrusted forwarded headers.
- LangSmith tracing is not configured by the application. If you enable it
  through LangChain's standard environment variables, review its data
  retention and privacy settings before sending prompts or retrieved content.
