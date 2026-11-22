"""Shared test setup that keeps LangSmith configuration local and offline."""

import os

import pytest
from langsmith import Client
from langsmith import run_trees
from langsmith.schemas import LangSmithInfo

LANGSMITH_ENV_PREFIX = "LANGSMITH_"

# The SDK asks the LangSmith API for server info while it builds or closes a
# client, and importing ``app.main`` already builds one: route decorators such
# as the rate limits call ``get_settings``, which configures tracing once for
# the process. Answering that question locally keeps collection offline even
# when the developer running the suite has a real ``LANGSMITH_API_KEY`` in
# ``.env``.
Client.info = property(lambda self: LangSmithInfo())


def _clear_langsmith_environment() -> None:
    for key in list(os.environ):
        if key.startswith(LANGSMITH_ENV_PREFIX):
            del os.environ[key]


@pytest.fixture(autouse=True)
def isolated_langsmith_state(monkeypatch):
    """Start every test with tracing off and undo what the test changes.

    ``configure_langsmith_tracing`` writes process-wide state, so a test that
    turns tracing on would otherwise leave it on for the rest of the session,
    and a developer's ``.env`` would otherwise decide whether a test traces.
    """
    original_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(LANGSMITH_ENV_PREFIX)
    }
    context = run_trees._context
    original_context = {
        "client": run_trees._CLIENT,
        "enabled": context._TRACING_ENABLED.get(),
        "project_name": context._PROJECT_NAME.get(),
    }
    _clear_langsmith_environment()
    run_trees.configure(client=None, enabled=False, project_name=None)
    # The SDK asks the API for server info while it builds or closes clients.
    monkeypatch.setattr(Client, "info", property(lambda self: LangSmithInfo()))

    yield

    _clear_langsmith_environment()
    os.environ.update(original_environment)
    run_trees.configure(**original_context)


@pytest.fixture
def recorded_traces(monkeypatch):
    """Capture what the SDK would send to LangSmith instead of sending it."""
    runs: list[dict] = []
    feedback: list[dict] = []

    def record(kind: str):
        def handler(self, *args, **kwargs):
            runs.append({"kind": kind, "args": args, **kwargs})

        return handler

    def record_feedback(self, **kwargs):
        feedback.append(
            {
                "trace_id": kwargs["trace_id"],
                "key": kwargs["key"],
                "value": kwargs["value"],
                "score": kwargs["score"],
            }
        )

    monkeypatch.setattr(Client, "create_run", record("create"))
    monkeypatch.setattr(Client, "update_run", record("update"))
    monkeypatch.setattr(Client, "create_feedback", record_feedback)
    return {"runs": runs, "feedback": feedback}
