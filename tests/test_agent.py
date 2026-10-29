from unittest.mock import Mock

from langchain_core.documents import Document

from app.agent import ProductionAgent
from app.models import DocumentInput
from app.monitoring import get_logger


def make_agent():
    agent = ProductionAgent.__new__(ProductionAgent)
    agent.vector_store = Mock()
    agent.retrieval_k = 2
    agent.retrieval_candidates_k = 3
    agent.max_context_chars = 12000
    agent.query_expansion_enabled = False
    agent.query_expansion_count = 2
    agent.query_rewriter = Mock()
    agent.primary_chain = Mock()
    agent.fallback_chain = Mock()
    agent.logger = get_logger("test-agent")
    return agent


def test_agent_retrieves_context_and_returns_source_labels():
    agent = make_agent()
    document = Document(
        page_content="Employees receive 20 days of annual leave.",
        metadata={"source": "handbook/leave-policy"},
    )
    agent.vector_store.max_marginal_relevance_search.return_value = [document]
    agent.retrieval_k = 3
    agent.primary_chain.invoke.return_value = "Employees receive 20 days [1]."

    result = agent.invoke("How many leave days?")

    assert result == {
        "response": "Employees receive 20 days [1].",
        "model_used": "primary",
        "sources": ["handbook/leave-policy"],
    }
    agent.vector_store.max_marginal_relevance_search.assert_called_once_with(
        "How many leave days?",
        k=3,
        fetch_k=6,
    )
    context = agent.primary_chain.invoke.call_args.args[0]["context"]
    assert "[1] Source: handbook/leave-policy" in context
    assert "Employees receive 20 days" in context
    agent.fallback_chain.invoke.assert_not_called()


def test_agent_uses_fallback_model_after_primary_failure():
    agent = make_agent()
    agent.primary_chain.invoke.side_effect = RuntimeError("primary unavailable")
    agent.fallback_chain.invoke.return_value = "I do not know. [1]"

    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="Known fact.", metadata={"source": "source-a"})
    ]

    result = agent.invoke("What is not in the documents?")

    assert result["model_used"] == "fallback"
    assert result["response"] == "I do not know. [1]"
    assert result["sources"] == ["source-a"]
    agent.fallback_chain.invoke.assert_called_once()


def test_query_expansion_adds_distinct_search_terms():
    assert ProductionAgent._parse_rewritten_queries(
        "How do I cancel?",
        "- How to terminate a subscription?\n"
        "2. How do I cancel?\n"
        "Cancellation policy",
    ) == ["How to terminate a subscription?", "Cancellation policy"]


def test_retrieval_uses_mmr_and_fuses_duplicate_results():
    agent = make_agent()
    first = Document(page_content="First", metadata={"source": "guide"})
    second = Document(page_content="Second", metadata={"source": "guide"})
    third = Document(page_content="Third", metadata={"source": "guide"})
    agent.vector_store.max_marginal_relevance_search.side_effect = [
        [first, second],
        [second, third],
    ]

    results = agent._retrieve_documents(["cancel plan", "terminate plan"])

    assert [document.page_content for document in results] == [
        "Second",
        "First",
    ]
    assert agent.vector_store.max_marginal_relevance_search.call_count == 2


def test_empty_retrieval_abstains_without_calling_the_llm():
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = []

    result = agent.invoke("What does the handbook say?")

    assert result["model_used"] == "none"
    assert result["sources"] == []
    agent.primary_chain.invoke.assert_not_called()


def test_query_expansion_retrieves_with_original_and_rewritten_questions():
    agent = make_agent()
    agent.query_expansion_enabled = True
    agent.query_rewriter.invoke.return_value = "How to terminate a plan?"
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="Cancellation details.", metadata={"source": "faq"})
    ]
    agent.primary_chain.invoke.return_value = "Use the cancellation process [1]."

    result = agent.invoke("How do I cancel?")

    assert result["sources"] == ["faq"]
    assert agent.query_rewriter.invoke.called
    assert agent.vector_store.max_marginal_relevance_search.call_count == 2
    assert [
        call.args[0]
        for call in agent.vector_store.max_marginal_relevance_search.call_args_list
    ] == ["How do I cancel?", "How to terminate a plan?"]


def test_query_expansion_failure_continues_with_original_question():
    agent = make_agent()
    agent.query_expansion_enabled = True
    agent.query_rewriter.invoke.side_effect = RuntimeError("rewrite unavailable")
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="A useful fact.", metadata={"source": "guide"})
    ]
    agent.primary_chain.invoke.return_value = "The fact is supported [1]."

    result = agent.invoke("What is the fact?")

    assert result["model_used"] == "primary"
    agent.vector_store.max_marginal_relevance_search.assert_called_once()
    assert agent.vector_store.max_marginal_relevance_search.call_args.args[0] == (
        "What is the fact?"
    )


def test_invalid_citations_are_replaced_with_abstention():
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(page_content="Known fact.", metadata={"source": "source-a"})
    ]
    agent.primary_chain.invoke.return_value = "Unsupported answer [2]."

    result = agent.invoke("What is the fact?")

    assert result["response"].startswith(
        "I couldn't produce an answer grounded"
    )
    assert result["sources"] == ["source-a"]


def test_retrieval_context_respects_character_budget_and_source_citations():
    agent = make_agent()
    agent.max_context_chars = 50
    documents = [
        Document(page_content="A" * 100, metadata={"source": "guide"}),
        Document(page_content="B" * 100, metadata={"source": "other"}),
    ]

    context, sources = agent._format_context(documents)

    assert len(context) <= 50
    assert sources == ["guide"]
    assert "[1] Source: guide" in context


def test_document_ingestion_assigns_stable_source_chunk_ids():
    agent = make_agent()
    agent.text_splitter = Mock()
    agent.text_splitter.split_documents.side_effect = lambda docs: [
        Document(
            page_content=f"{docs[0].page_content} chunk",
            metadata=dict(docs[0].metadata),
        )
    ]

    first_count = agent.ingest_documents(
        [DocumentInput(source="guide", content="Sample content.")]
    )
    first_ids = agent.vector_store.add_documents.call_args.kwargs["ids"]
    second_count = agent.ingest_documents(
        [DocumentInput(source="guide", content="Sample content.")]
    )
    second_ids = agent.vector_store.add_documents.call_args.kwargs["ids"]

    assert first_count == second_count == 1
    assert first_ids == second_ids
    assert len(first_ids[0]) == 64


def test_database_url_schemes_are_normalized():
    assert ProductionAgent._sqlalchemy_connection_string(
        "postgres://user:pass@localhost/db"
    ) == "postgresql+psycopg://user:pass@localhost/db"
    assert ProductionAgent._sqlalchemy_connection_string(
        "postgresql://user:pass@localhost/db"
    ) == "postgresql+psycopg://user:pass@localhost/db"
