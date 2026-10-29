import hmac
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import ProductionAgent
from app.cache import CachedResponse, ResponseCache
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricResponse,
)
from app.monitoring import MetricsCollector, RequestTimer, get_logger
from app.security import SecurityPipeline


logger = get_logger()
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    application.state.settings = settings
    application.state.security = SecurityPipeline()
    application.state.cache = ResponseCache(
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )
    application.state.metrics = MetricsCollector()

    logger.setLevel(settings.log_level.upper())
    logger.info(
        "Starting RAG API",
        extra={
            "extra_data": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "vector_collection": settings.vector_collection_name,
            }
        },
    )
    application.state.agent = ProductionAgent(settings)
    logger.info("RAG API is ready")
    try:
        yield
    finally:
        logger.info(
            "Shutting down RAG API",
            extra={"extra_data": application.state.metrics.get_summary()},
        )


app = FastAPI(
    title="Production RAG API",
    description="A pgvector-backed retrieval-augmented chat API.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
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

            cache_key = f"{settings.vector_collection_name}\0{cleaned_message}"
            cached = cache.get(cache_key)
            if cached is not None:
                cache_hit = True
                response_text = cached.response
                sources = list(cached.sources)
                model_used = cached.model_used
            else:
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

    request.app.state.cache.clear()
    return IngestResponse(
        documents_received=len(body.documents),
        chunks_indexed=chunks_indexed,
    )


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health", response_model=HealthResponse)
def readiness(request: Request, response: Response) -> HealthResponse:
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
    return MetricResponse(**request.app.state.metrics.get_summary())


@app.get("/cache/stats")
def cache_stats(request: Request) -> dict[str, int]:
    return request.app.state.cache.stats
