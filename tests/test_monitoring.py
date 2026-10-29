from app.monitoring import MetricsCollector


def test_metrics_summary_uses_request_and_cache_totals():
    metrics = MetricsCollector()
    metrics.record_request(latency_ms=100, cache_hit=False)
    metrics.record_request(latency_ms=50, error=True, cache_hit=True)

    assert metrics.get_summary() == {
        "total_requests": 2,
        "total_errors": 1,
        "error_rate": "50.00%",
        "avg_latency_ms": 75.0,
        "cache_hit_rate": "50.00%",
    }


def test_empty_metrics_have_zero_rates():
    summary = MetricsCollector().get_summary()

    assert summary["total_requests"] == 0
    assert summary["error_rate"] == "0.00%"
    assert summary["cache_hit_rate"] == "0.00%"
