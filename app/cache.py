import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CachedResponse:
    response: str
    sources: tuple[str, ...]
    model_used: str


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000):
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must not be negative")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[CachedResponse, float]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    @staticmethod
    def _generate_key(input_data: str) -> str:
        return hashlib.sha256(input_data.encode("utf-8")).hexdigest()

    def set(self, input_data: str, response: CachedResponse) -> None:
        if self.ttl_seconds == 0:
            return

        key = self._generate_key(input_data)
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._entries[key] = (response, expires_at)
            self._entries.move_to_end(key)
            self._remove_expired(time.monotonic())
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get(self, input_data: str) -> CachedResponse | None:
        key = self._generate_key(input_data)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            response, expires_at = entry
            if time.monotonic() >= expires_at:
                del self._entries[key]
                return None

            self._entries.move_to_end(key)
            return response

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _remove_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, (_, expires_at) in self._entries.items()
            if now >= expires_at
        ]
        for key in expired_keys:
            del self._entries[key]

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            self._remove_expired(time.monotonic())
            return {"entries": len(self._entries), "max_entries": self.max_entries}
