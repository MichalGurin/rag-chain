from app.cache import CachedResponse, ResponseCache


def test_cache_returns_unexpired_response_and_evicts_oldest():
    cache = ResponseCache(ttl_seconds=60, max_entries=1)
    first = CachedResponse("answer one", ("source one",), "primary")
    second = CachedResponse("answer two", ("source two",), "fallback")

    cache.set("question one", first)
    assert cache.get("question one") == first

    cache.set("question two", second)
    assert cache.get("question one") is None
    assert cache.get("question two") == second
    assert cache.stats == {"entries": 1, "max_entries": 1}


def test_cache_expires_entries(monkeypatch):
    current_time = [10.0]
    monkeypatch.setattr("app.cache.time.monotonic", lambda: current_time[0])
    cache = ResponseCache(ttl_seconds=2)
    cached = CachedResponse("answer", (), "primary")
    cache.set("question", cached)

    current_time[0] = 12.0
    assert cache.get("question") is None
    assert cache.stats["entries"] == 0


def test_zero_ttl_disables_cache():
    cache = ResponseCache(ttl_seconds=0)
    cache.set("question", CachedResponse("answer", (), "primary"))

    assert cache.get("question") is None


def test_cache_rejects_invalid_capacity():
    try:
        ResponseCache(max_entries=0)
    except ValueError as exc:
        assert "max_entries" in str(exc)
    else:
        raise AssertionError("Expected invalid cache capacity to be rejected")