import pytest

from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration

from app.llm_cost import (
    LLMCostTracker,
    TokenUsageCallback,
    estimate_cost_usd,
    estimate_tokens,
)


def _llm_result(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
) -> LLMResult:
    """Build a chat-model completion result the way the OpenAI SDK reports it."""
    usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    message = AIMessage(content="Answer.", usage_metadata=usage_metadata)
    generation = ChatGeneration(message=message)
    return LLMResult(
        generations=[[generation]],
        llm_output={
            "token_usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "model_name": model,
        },
    )


def test_estimate_cost_usd_uses_model_pricing_per_million_tokens():
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.5)


def test_estimate_cost_usd_estimates_output_tokens_when_unspecified():
    assert estimate_cost_usd("gpt-4o-mini", 1_000) > 0.0


def test_estimate_cost_usd_ignores_models_without_pricing():
    assert estimate_cost_usd("local-llm", 10_000, 5_000) == 0.0


def test_estimate_tokens_rounds_up_small_inputs():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_tracker_allows_calls_within_both_limits():
    tracker = LLMCostTracker(max_estimated_cost_per_request_usd=1.0)

    assert tracker.can_spend(estimate_cost_usd("gpt-4o-mini", 1_000, 1_000)) is True
    assert tracker.is_over_budget() is False


def test_tracker_blocks_estimates_over_the_per_request_limit():
    tracker = LLMCostTracker(max_estimated_cost_per_request_usd=0.001)

    assert tracker.can_spend(estimate_cost_usd("gpt-4o-mini", 100, 50)) is True
    assert tracker.can_spend(estimate_cost_usd("gpt-4o-mini", 10_000, 5_000)) is False


def test_tracker_blocks_calls_after_the_cumulative_budget_is_exhausted():
    tracker = LLMCostTracker(max_total_cost_usd=0.0)

    assert tracker.can_spend(0.0) is True
    tracker.record(
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=100,
        estimated_cost_usd=estimate_cost_usd("gpt-4o-mini", 100, 100),
    )
    assert tracker.is_over_budget() is True
    assert tracker.can_spend(estimate_cost_usd("gpt-4o-mini", 100, 100)) is False


def test_tracker_reserves_estimated_cost_against_the_cumulative_budget():
    tracker = LLMCostTracker(max_total_cost_usd=0.001)

    assert tracker.can_spend(0.0008) is True
    assert tracker.can_spend(0.0008) is False


def test_recorded_usage_does_not_release_another_calls_reservation():
    tracker = LLMCostTracker(max_total_cost_usd=0.001)

    assert tracker.can_spend(0.0004) is True
    tracker.record(
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=100,
        estimated_cost_usd=0.0005,
    )

    assert tracker.can_spend(0.0002) is False
    tracker.release_reservation(0.0004)
    assert tracker.can_spend(0.0002) is True


def test_tracker_accumulates_token_totals_and_cost():
    tracker = LLMCostTracker()
    tracker.record(
        model="gpt-4o-mini",
        input_tokens=1_000,
        output_tokens=200,
        estimated_cost_usd=0.0,
    )

    summary = tracker.get_usage_summary()
    assert summary["llm_calls"] == 1
    assert summary["llm_input_tokens"] == 1_000
    assert summary["llm_output_tokens"] == 200
    assert summary["llm_total_tokens"] == 1_200
    assert summary["llm_estimated_cost_usd"] == 0.0


def test_tracker_resets_running_totals():
    tracker = LLMCostTracker()
    tracker.record(
        model="gpt-4o-mini",
        input_tokens=1_000,
        output_tokens=200,
        estimated_cost_usd=0.0,
    )
    tracker.reset()

    assert tracker.get_usage_summary()["llm_calls"] == 0
    assert tracker.get_usage_summary()["llm_total_tokens"] == 0


def test_token_usage_callback_records_reported_usage():
    tracker = LLMCostTracker()
    TokenUsageCallback(tracker).on_llm_end(_llm_result(1_200, 340))

    summary = tracker.get_usage_summary()
    assert summary["llm_calls"] == 1
    assert summary["llm_input_tokens"] == 1_200
    assert summary["llm_output_tokens"] == 340
    assert summary["llm_total_tokens"] == 1_540
    assert summary["llm_estimated_cost_usd"] > 0.0


def test_token_usage_callback_records_zero_usage_for_empty_runs():
    tracker = LLMCostTracker()
    TokenUsageCallback(tracker).on_llm_end(
        LLMResult(
            generations=[[]],
            llm_output={"model_name": "local-llm"},
        )
    )

    assert tracker.get_usage_summary()["llm_calls"] == 1
