"""LLM token usage tracking and runaway-cost guardrails."""

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Approximate token estimate used before a call: one token per four characters.
ESTIMATED_TOKENS_PER_CHAR = 4.0

# USD of input and output cost per one million tokens, keyed by model prefix.
MODEL_PRICING_USD_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}

DEFAULT_ESTIMATED_OUTPUT_TOKENS = 512
DEFAULT_MAX_ESTIMATED_COST_PER_REQUEST_USD = 0.05


def estimate_tokens(*texts: str) -> int:
    """Estimate token count for free-form text without a tokenizer."""
    char_count = sum(len(text) for text in texts if text)
    return max(1, int(char_count / ESTIMATED_TOKENS_PER_CHAR))


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int | None = None,
) -> float:
    """Estimate the USD spend of a call from per-million-token pricing."""
    price_per_million_input, price_per_million_output = _pricing_for_model(
        model
    )
    if output_tokens is None:
        output_tokens = DEFAULT_ESTIMATED_OUTPUT_TOKENS
    return (
        (input_tokens / 1_000_000) * price_per_million_input
        + (output_tokens / 1_000_000) * price_per_million_output
    )


def _pricing_for_model(
    model: str,
    pricing: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Return (input, output) USD cost per million tokens for a model."""
    table = (
        pricing
        if pricing is not None
        else MODEL_PRICING_USD_PER_MILLION_TOKENS
    )
    for prefix in sorted(table, key=len, reverse=True):
        if model.startswith(prefix):
            return table[prefix]
    return (0.0, 0.0)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts and estimated spend for a single model call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class LLMCostLimitExceeded(RuntimeError):
    """Raised when an LLM call would breach the configured cost guardrails."""


class LLMCostTracker:
    """Thread-safe token/cost totals with per-request and cumulative limits."""

    def __init__(
        self,
        *,
        pricing: Mapping[str, tuple[float, float]] | None = None,
        max_estimated_cost_per_request_usd: float = (
            DEFAULT_MAX_ESTIMATED_COST_PER_REQUEST_USD
        ),
        max_total_cost_usd: float | None = None,
        estimated_output_tokens: int = DEFAULT_ESTIMATED_OUTPUT_TOKENS,
    ) -> None:
        self._pricing = (
            dict(pricing)
            if pricing is not None
            else dict(MODEL_PRICING_USD_PER_MILLION_TOKENS)
        )
        self._max_estimated_cost_per_request_usd = (
            max_estimated_cost_per_request_usd
        )
        self._max_total_cost_usd = max_total_cost_usd
        self._estimated_output_tokens = estimated_output_tokens
        self._lock = threading.Lock()
        self._cost = 0.0
        self._reserved = 0.0
        self._llm_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._over_budget = False

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int | None = None,
    ) -> float:
        """Estimate the spend of a call before it happens."""
        price_per_million_input, price_per_million_output = _pricing_for_model(
            model,
            self._pricing,
        )
        if output_tokens is None:
            output_tokens = self._estimated_output_tokens
        return (
            (input_tokens / 1_000_000) * price_per_million_input
            + (output_tokens / 1_000_000) * price_per_million_output
        )

    def can_spend(self, estimated_cost_usd: float) -> bool:
        """Admit an LLM call, reserving its estimated cost against the budget.

        The budget check and the reservation happen under one lock, so
        concurrent calls cannot each reserve past the cumulative limit before
        usage is reported.
        """
        with self._lock:
            if self._over_budget:
                return False
            if estimated_cost_usd > self._max_estimated_cost_per_request_usd:
                return False
            if (
                self._max_total_cost_usd is not None
                and self._cost + self._reserved + estimated_cost_usd
                > self._max_total_cost_usd
            ):
                return False
            self._reserved += estimated_cost_usd
            return True

    def record(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float | None = None,
    ) -> TokenUsage:
        """Fold a completed call's usage into the running totals."""
        cost = (
            estimated_cost_usd
            if estimated_cost_usd is not None
            else self.estimate_cost(model, input_tokens, output_tokens)
        )
        with self._lock:
            self._cost += cost
            self._llm_calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._total_tokens += input_tokens + output_tokens
            if (
                self._max_total_cost_usd is not None
                and self._cost > self._max_total_cost_usd
            ):
                self._over_budget = True
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
        )

    def release_reservation(self, estimated_cost_usd: float) -> None:
        """Release a call's estimate after it finishes, whether it succeeded or failed."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - estimated_cost_usd)

    def is_over_budget(self) -> bool:
        """Return whether the cumulative budget has been exceeded."""
        with self._lock:
            return self._over_budget

    def get_usage_summary(self) -> dict[str, int | float]:
        """Return process-wide token and spend totals for metrics."""
        with self._lock:
            return {
                "llm_calls": self._llm_calls,
                "llm_input_tokens": self._input_tokens,
                "llm_output_tokens": self._output_tokens,
                "llm_total_tokens": self._total_tokens,
                "llm_estimated_cost_usd": round(self._cost, 8),
            }

    def reset(self) -> None:
        """Clear running totals (for tests and long-lived operators)."""
        with self._lock:
            self._cost = 0.0
            self._reserved = 0.0
            self._llm_calls = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._total_tokens = 0
            self._over_budget = False


class TokenUsageCallback(BaseCallbackHandler):
    """Feed LangChain LLM run events into the cost tracker."""

    def __init__(self, tracker: LLMCostTracker) -> None:
        self.tracker = tracker

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        """Record token usage reported by a finished chat model run."""
        llm_output = response.llm_output or {}
        model = str(llm_output.get("model_name") or "unknown")
        token_usage = llm_output.get("token_usage") or {}
        generations = response.generations or []
        first = generations[0] if generations else None
        generation = first[0] if first else None
        message = getattr(generation, "message", None) if generation else None
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        input_tokens = int(usage_metadata.get("input_tokens") or 0)
        output_tokens = int(usage_metadata.get("output_tokens") or 0)
        if not input_tokens and not output_tokens:
            input_tokens = int(
                token_usage.get("prompt_tokens")
                or token_usage.get("input_tokens")
                or 0
            )
            output_tokens = int(
                token_usage.get("completion_tokens")
                or token_usage.get("output_tokens")
                or 0
            )
        estimated_cost = self.tracker.estimate_cost(
            model,
            input_tokens,
            output_tokens,
        )
        self.tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
        )
