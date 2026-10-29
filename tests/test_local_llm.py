from unittest.mock import Mock

from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from app import agent as agent_module
from app.config import Settings


def test_agent_configures_chat_and_embeddings_for_local_compatible_endpoint(
    monkeypatch,
):
    chat_calls = []
    embedding_calls = []

    def fake_chat_openai(**kwargs):
        chat_calls.append(kwargs)
        return RunnableLambda(lambda _: "local response")

    def fake_embeddings(**kwargs):
        embedding_calls.append(kwargs)
        return Mock()

    monkeypatch.setattr(agent_module, "ChatOpenAI", fake_chat_openai)
    monkeypatch.setattr(agent_module, "OpenAIEmbeddings", fake_embeddings)
    monkeypatch.setattr(agent_module, "PGVector", Mock())
    settings = Settings(
        _env_file=None,
        database_url=SecretStr("postgresql+psycopg://user:pass@db/rag"),
        llm_base_url="http://ollama:11434/v1",
        primary_model="llama3.2:1b",
        fallback_model="qwen2.5:1.5b",
        embedding_model="nomic-embed-text",
    )

    agent_module.ProductionAgent(settings)

    assert [call["model"] for call in chat_calls] == [
        "llama3.2:1b",
        "qwen2.5:1.5b",
    ]
    assert all(
        call["base_url"] == "http://ollama:11434/v1"
        and call["api_key"] == "ollama"
        for call in chat_calls
    )
    assert embedding_calls == [
        {
            "model": "nomic-embed-text",
            "api_key": "ollama",
            "base_url": "http://ollama:11434/v1",
            "check_embedding_ctx_length": False,
        }
    ]
