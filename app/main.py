import hmac
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import JSONResponse
from langsmith import tracing_context
from starlette.datastructures import State
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from slowapi.errors import RateLimitExceeded

from app.agent import ProductionAgent
from app.api.documents import router as documents_router
from app.api.limits import limiter
from app.cache import CachedResponse, ResponseCache
from app.config import get_settings
from app.database import DocumentRepository, create_database_engine
from app.models import (
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricResponse,
)
from app.monitoring import MetricsCollector, RequestTimer, get_logger
from app.security import SecurityPipeline


logger = get_logger()


def _security_header_values(
    *,
    production: bool,
    https: bool,
) -> dict[str, str]:
    """Return security headers shared by normal and early-rejected responses."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Cross-Origin-Opener-Policy": "same-origin",
    }
    if production and https:
        headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return headers


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create shared resources at startup and release them during shutdown."""
    settings = get_settings()
    application.state.settings = settings
    application.state.security = SecurityPipeline()
    application.state.cache = ResponseCache(
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )
    application.state.metrics = MetricsCollector()

    application.state.agent = None
    application.state.database_engine = None
    try:
        logger.setLevel(settings.log_level.upper())
        logger.info(
            "Starting RAG API",
            extra={
                "extra_data": {
                    "environment": settings.app_env,
                    "primary_model": settings.primary_model,
                    "vector_collection": settings.vector_collection_name,
                    "langsmith_tracing": settings.langsmith_tracing,
                    "langsmith_project": settings.effective_langsmith_project,
                    "langsmith_sampling_rate": (
                        settings.langsmith_tracing_sampling_rate
                    ),
                    "langsmith_hide_inputs": settings.langsmith_hide_inputs,
                    "langsmith_hide_outputs": settings.langsmith_hide_outputs,
                }
            },
        )
        application.state.agent = ProductionAgent(settings)
        application.state.database_engine = getattr(
            application.state.agent,
            "db_engine",
            None,
        )
        if (
            settings.database_url is not None
            and application.state.database_engine is None
        ):
            application.state.database_engine = create_database_engine(settings)
        ensure_search_indexes = getattr(
            application.state.agent, "ensure_search_indexes", None
        )
        if ensure_search_indexes is not None:
            ensure_search_indexes()
        if application.state.database_engine is not None:
            application.state.document_repository = DocumentRepository(
                application.state.database_engine,
                tenant_id=settings.effective_tenant_id,
            )
        else:
            application.state.document_repository = None
        logger.info("RAG API is ready")
        yield
    finally:
        logger.info(
            "Shutting down RAG API",
            extra={"extra_data": application.state.metrics.get_summary()},
        )
        agent = application.state.agent
        agent_engine = getattr(agent, "db_engine", None)
        if (
            application.state.database_engine is not None
            and application.state.database_engine is not agent_engine
        ):
            application.state.database_engine.dispose()
        close_agent = getattr(agent, "close", None)
        if close_agent is not None:
            close_agent()


app = FastAPI(
    title="Production RAG API",
    description="A pgvector-backed retrieval-augmented chat API.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.include_router(documents_router)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach request correlation and baseline browser security headers."""
    request_id = uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    settings = getattr(request.app.state, "settings", None)
    for name, value in _security_header_values(
        production=bool(settings and settings.is_production),
        https=request.url.scheme == "https",
    ).items():
        response.headers[name] = value
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    _exc: RateLimitExceeded,
) -> JSONResponse:
    """Return a generic 429 response without exposing limiter internals."""
    del _exc
    response = ErrorResponse(
        error="rate_limit_exceeded",
        detail="Too many requests. Please slow down.",
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=429,
        content=response.model_dump(mode="json", exclude_none=True),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Normalize expected HTTP errors without discarding response headers."""
    response = ErrorResponse(
        error="request_error",
        detail=str(exc.detail),
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(mode="json", exclude_none=True),
        headers=exc.headers,
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
def chat(
    request: Request,
    body: ChatRequest,
    x_chat_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatResponse:
    """Validate, cache, and answer a chat request with safe error handling."""
    settings = request.app.state.settings
    security: SecurityPipeline = request.app.state.security
    cache: ResponseCache = request.app.state.cache
    metrics: MetricsCollector = request.app.state.metrics
    agent: ProductionAgent = request.app.state.agent
    configured_chat_key = settings.chat_api_key
    if configured_chat_key is not None and (
        x_chat_api_key is None
        or not hmac.compare_digest(
            configured_chat_key.get_secret_value(),
            x_chat_api_key,
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    timer = RequestTimer()
    cache_hit = False

    try:
        with timer:
            is_allowed, cleaned_message, security_notes = security.check_input(
                body.message
            )
            if not is_allowed:
                logger.warning(
                    "Request blocked by input checks",
                    extra={
                        "extra_data": {
                            "request_id": request.state.request_id,
                        }
                    },
                )
                raise HTTPException(
                    status_code=400,
                    detail="Your message was blocked by the input checks.",
                )

            corpus_version = ""
            version_reader = getattr(agent, "corpus_version", None)
            if version_reader is not None:
                corpus_version = version_reader()
            cache_key = (
                f"{settings.effective_tenant_id}\0"
                f"{settings.vector_collection_name}\0{corpus_version}"
                f"\0{cleaned_message}"
            )
            cached = cache.get(cache_key)
            if cached is not None:
                cache_hit = True
                response_text = cached.response
                sources = list(cached.sources)
                model_used = cached.model_used
            else:
                trace_metadata = {
                    "request_id": request.state.request_id,
                    "environment": settings.app_env,
                    "tenant_id": settings.effective_tenant_id,
                }
                if body.thread_id:
                    trace_metadata["thread_id"] = body.thread_id
                with tracing_context(
                    tags=["api", "chat"],
                    metadata=trace_metadata,
                ):
                    result = agent.invoke(cleaned_message)
                response_text, output_warnings = security.check_output(
                    result["response"]
                )
                security_notes.extend(output_warnings)
                sources = result.get("sources", [])
                model_used = result["model_used"]
                cache.set(
                    cache_key,
                    CachedResponse(
                        response=response_text,
                        sources=tuple(sources),
                        model_used=model_used,
                    ),
                )
    except HTTPException as exc:
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=exc.status_code >= 500,
            cache_hit=cache_hit,
        )
        raise
    except Exception:
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=True,
            cache_hit=cache_hit,
        )
        logger.exception(
            "RAG request failed",
            extra={
                "extra_data": {
                    "request_id": request.state.request_id,
                }
            },
        )
        raise HTTPException(
            status_code=502,
            detail="The RAG service could not process this request.",
        ) from None

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        cache_hit=cache_hit,
    )
    if security_notes:
        logger.info(
            "Input or output was adjusted by safety checks",
            extra={
                "extra_data": {
                    "request_id": request.state.request_id,
                    "notes": security_notes,
                }
            },
        )
    logger.info(
        "RAG request completed",
        extra={
            "extra_data": {
                "request_id": request.state.request_id,
                "model_used": model_used,
                "cached": cache_hit,
                "latency_ms": round(timer.elapsed_ms, 2),
            }
        },
    )
    return ChatResponse(
        response=response_text,
        thread_id=body.thread_id,
        model_used=model_used,
        sources=sources,
        cached=cache_hit,
        processing_time_ms=round(timer.elapsed_ms, 2),
    )


@app.post("/documents", response_model=IngestResponse)
@limiter.limit(get_settings().ingest_rate_limit)
def ingest_documents(
    request: Request,
    body: IngestRequest,
    x_ingest_api_key: str | None = Header(
        default=None,
        alias="X-Ingest-API-Key",
    ),
) -> IngestResponse:
    """Index bounded pre-extracted text and invalidate cached corpus answers."""
    settings = request.app.state.settings
    configured_key = settings.ingest_api_key
    if configured_key is None:
        raise HTTPException(
            status_code=503,
            detail="Document ingestion is disabled.",
        )
    if x_ingest_api_key is None or not hmac.compare_digest(
        configured_key.get_secret_value(),
        x_ingest_api_key,
    ):
        raise HTTPException(status_code=401, detail="Invalid ingestion API key.")
    total_content_chars = sum(len(document.content) for document in body.documents)
    if total_content_chars > settings.max_ingest_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                "The total document content exceeds the "
                f"{settings.max_ingest_chars}-character ingestion limit."
            ),
        )

    try:
        chunks_indexed = request.app.state.agent.ingest_documents(body.documents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "Document ingestion failed",
            extra={
                "extra_data": {
                    "request_id": request.state.request_id,
                }
            },
        )
        raise HTTPException(
            status_code=502,
            detail="The document service could not index this request.",
        ) from None

    repository: DocumentRepository | None = (
        request.app.state.document_repository
    )
    if repository is not None:
        try:
            repository.bump_generation()
        except Exception:
            logger.exception(
                "Could not update the corpus generation after ingestion",
                extra={
                    "extra_data": {
                        "request_id": request.state.request_id,
                    }
                },
            )
            raise HTTPException(
                status_code=502,
                detail="The document service could not finalize this request.",
            ) from None
    request.app.state.cache.clear()
    return IngestResponse(
        documents_received=len(body.documents),
        chunks_indexed=chunks_indexed,
    )


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """Report process liveness without requiring database or model availability."""
    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse)
def readiness(request: Request, response: Response) -> HealthResponse:
    """Report readiness and return 503 when the agent or database is unavailable."""
    agent = getattr(request.app.state, "agent", None)
    database_ready = agent is not None and agent.database_ready()
    checks = {
        "agent": agent is not None,
        "database": database_ready,
    }
    healthy = all(checks.values())
    if not healthy:
        response.status_code = 503
    return HealthResponse(
        status="healthy" if healthy else "degraded",
        environment=get_settings().app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricResponse)
def get_metrics(request: Request) -> MetricResponse:
    """Expose the current process's request metrics."""
    return MetricResponse(**request.app.state.metrics.get_summary())


@app.get("/cache/stats")
def cache_stats(request: Request) -> dict[str, int]:
    """Expose bounded in-memory response-cache statistics."""
    return request.app.state.cache.stats


@app.get("/config", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    """Expose effective deployment limits so clients avoid hardcoded defaults."""
    settings = request.app.state.settings
    return ConfigResponse(max_upload_bytes=settings.max_upload_bytes)


class UploadSizeLimitMiddleware:
    """Enforce an early streaming body limit on multipart upload requests."""

    def __init__(self, application: ASGIApp) -> None:
        """Wrap the ASGI app while preserving access to its state."""
        self.application = application

    @property
    def state(self) -> State:
        """Expose wrapped application state for middleware compatibility."""
        return self.application.state

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Reject oversized requests before multipart parsing consumes the body."""
        if scope["type"] != "http" or scope["path"] != "/documents/upload":
            await self.application(scope, receive, send)
            return

        maximum_body_bytes = (
            self.application.state.settings.max_upload_bytes + 1024 * 1024
        )
        request_id = uuid4().hex
        settings = self.application.state.settings
        security_headers = _security_header_values(
            production=bool(getattr(settings, "is_production", False)),
            https=scope.get("scheme") == "https",
        )
        security_headers["X-Request-ID"] = request_id
        content_length = next(
            (
                value
                for name, value in scope["headers"]
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > maximum_body_bytes:
                    await self._reject_large_upload(send, security_headers)
                    return
            except ValueError:
                await self._reject_large_upload(send, security_headers)
                return

        received_bytes = 0
        exceeded = False
        response_started = False
        response_body_sent = False

        async def limited_receive() -> Message:
            """Count streamed bytes and interrupt request parsing past the limit."""
            nonlocal received_bytes, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > maximum_body_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def limited_send(message: Message) -> None:
            """Replace a partially processed oversized upload with a 413 response."""
            nonlocal response_started, response_body_sent
            if not exceeded:
                await send(message)
                return
            if message["type"] == "http.response.start" and not response_started:
                response_started = True
                body = b'{"detail":"Upload request exceeds the configured limit."}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                            *[
                                (name.lower().encode("ascii"), value.encode("ascii"))
                                for name, value in security_headers.items()
                            ],
                        ],
                    }
                )
            elif message["type"] == "http.response.body" and not response_body_sent:
                response_body_sent = True
                body = b'{"detail":"Upload request exceeds the configured limit."}'
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    }
                )

        await self.application(scope, limited_receive, limited_send)

    @staticmethod
    async def _reject_large_upload(
        send: Send,
        security_headers: dict[str, str],
    ) -> None:
        """Send a bounded JSON 413 response when Content-Length is already too big."""
        body = b'{"detail":"Upload request exceeds the configured limit."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    *[
                        (name.lower().encode("ascii"), value.encode("ascii"))
                        for name, value in security_headers.items()
                    ],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


app = UploadSizeLimitMiddleware(app)
