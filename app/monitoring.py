import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            log_entry.update(extra_data)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def get_logger(name: str = "prod-rag", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        error: bool = False,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            self._metrics["requests_total"] += 1
            self._metrics["latency_sum"] += latency_ms
            if error:
                self._metrics["errors_total"] += 1
            if cache_hit:
                self._metrics["cache_hits"] += 1
            else:
                self._metrics["cache_misses"] += 1

    def get_summary(self) -> dict[str, int | float | str]:
        with self._lock:
            requests = self._metrics["requests_total"]
            cache_requests = (
                self._metrics["cache_hits"] + self._metrics["cache_misses"]
            )
            return {
                "total_requests": requests,
                "total_errors": self._metrics["errors_total"],
                "error_rate": (
                    f"{self._metrics['errors_total'] / requests:.2%}"
                    if requests
                    else "0.00%"
                ),
                "avg_latency_ms": round(
                    self._metrics["latency_sum"] / requests, 2
                )
                if requests
                else 0.0,
                "cache_hit_rate": (
                    f"{self._metrics['cache_hits'] / cache_requests:.2%}"
                    if cache_requests
                    else "0.00%"
                ),
            }


class RequestTimer:
    def __enter__(self) -> "RequestTimer":
        self._started_at = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._started_at) * 1000
