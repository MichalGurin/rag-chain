import hashlib
from unittest.mock import MagicMock, Mock

from langchain_core.documents import Document

from app.agent import ProductionAgent
from app.database.engine import normalize_database_url
from app.models import DocumentInput
from app.monitoring import get_logger


def make_agent():
    agent = ProductionAgent.__new__(ProductionAgent)
    agent.vector_store = Mock()
    agent.tenant_id = "local"
    agent.retrieval_k = 2
    agent.retrieval_candidates_k = 3
    agent.max_context_chars = 12000
    agent.query_expansion_enabled = False
    agent.query_expansion_count = 2
    agent.query_rewriter = Mock()
    agent.primary_chain = Mock()
    agent.fallback_chain = Mock()
    agent.general_chain = Mock()
    agent.general_fallback_chain = Mock()
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
        filter={"tenant_id": "local"},
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


def test_query_expansion_caps_parsed_rewrites():
    assert ProductionAgent._parse_rewritten_queries(
        "How do I cancel?",
        "First alternative\nSecond alternative\nThird alternative",
        max_queries=2,
    ) == ["First alternative", "Second alternative"]


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


def test_bm25_keyword_ranking_accounts_for_term_frequency_and_length():
    agent = make_agent()
    repeated = Document(
        page_content="blue blue blue silver",
        metadata={"source": "repeated"},
    )
    concise = Document(
        page_content="blue silver",
        metadata={"source": "concise"},
    )
    agent._keyword_search = Mock(return_value=[concise, repeated])

    results = agent._bm25_search("blue")

    assert [document.metadata["source"] for document in results] == [
        "repeated",
        "concise",
    ]


def test_keyword_search_scopes_sql_query_to_configured_tenant():
    agent = make_agent()
    agent.tenant_id = "tenant-a"
    agent.db_engine = MagicMock()
    agent.vector_collection_name = "documents"
    agent.retrieval_candidates_k = 3
    agent._fts_index_ready = True
    connection = agent.db_engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.mappings.return_value = [
        {
            "document": "Tenant content.",
            "cmetadata": {"source": "guide", "tenant_id": "tenant-a"},
        }
    ]

    documents = agent._keyword_search("tenant content")

    assert len(documents) == 1
    statement, parameters = connection.execute.call_args.args
    assert "cmetadata->>'tenant_id' = :tenant_id" in str(statement)
    assert parameters["tenant_id"] == "tenant-a"


def test_corpus_version_reads_only_the_configured_tenant():
    agent = ProductionAgent.__new__(ProductionAgent)
    agent.tenant_id = "tenant-a"
    agent.db_engine = MagicMock()
    connection = agent.db_engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.scalar_one.return_value = "7"

    assert agent.corpus_version() == "7"

    statement, parameters = connection.execute.call_args.args
    assert "tenant_id = :tenant_id" in str(statement)
    assert parameters == {"tenant_id": "tenant-a"}


def test_empty_retrieval_answers_from_general_knowledge_without_citations():
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = []
    agent.general_chain.invoke.return_value = "Ottawa is the capital of Canada."

    result = agent.invoke("What is the capital of Canada?")

    assert result["model_used"] == "primary"
    assert result["sources"] == []
    assert result["response"] == (
        "I couldn't find relevant information in your documents. "
        "Here's a general answer (not verified against your documents):\n\n"
        "Ottawa is the capital of Canada."
    )
    agent.general_chain.invoke.assert_called_once_with(
        {"question": "What is the capital of Canada?"}
    )
    agent.primary_chain.invoke.assert_not_called()
    agent.general_fallback_chain.invoke.assert_not_called()


def test_empty_retrieval_uses_general_fallback_after_primary_failure():
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = []
    agent.general_chain.invoke.side_effect = RuntimeError("primary unavailable")
    agent.general_fallback_chain.invoke.return_value = "Ottawa."

    result = agent.invoke("What is the capital of Canada?")

    assert result["model_used"] == "fallback"
    assert result["sources"] == []
    assert result["response"].endswith("Ottawa.")
    agent.general_fallback_chain.invoke.assert_called_once_with(
        {"question": "What is the capital of Canada?"}
    )


def test_source_abstention_falls_back_to_general_knowledge():
    agent = make_agent()
    agent.vector_store.max_marginal_relevance_search.return_value = [
        Document(
            page_content="The office plant is called Fern.",
            metadata={"source": "office-notes"},
        )
    ]
    agent.primary_chain.invoke.return_value = (
        "I don't know based on the provided sources. [1]"
    )
    agent.general_chain.invoke.return_value = (
        "Photosynthesis converts light into chemical energy."
    )

    result = agent.invoke("What is photosynthesis?")

    assert result["model_used"] == "primary"
    assert result["sources"] == []
    assert "general answer" in result["response"]
    assert "not verified against your documents" in result["response"]
    assert "Photosynthesis converts light" in result["response"]
    agent.general_chain.invoke.assert_called_once_with(
        {"question": "What is photosynthesis?"}
    )


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
        "I couldn't produce an answer with valid source citations"
    )
    assert result["sources"] == ["source-a"]


def test_citation_format_does_not_claim_to_verify_grounding():
    assert ProductionAgent._has_valid_citation_format(
        "An unsupported statement [1].",
        source_count=1,
    )


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
    expected_id = hashlib.sha256(
        b"local\0guide\0" b"0\0Sample content. chunk"
    ).hexdigest()
    assert first_ids == [expected_id]


def test_document_ingestion_assigns_server_tenant_metadata():
    agent = make_agent()
    agent.tenant_id = "tenant-a"
    agent.text_splitter = Mock()
    agent.text_splitter.split_documents.side_effect = lambda docs: [
        Document(
            page_content=docs[0].page_content,
            metadata=dict(docs[0].metadata),
        )
    ]

    agent.ingest_documents(
        [DocumentInput(source="guide", content="Tenant content.")]
    )

    indexed = agent.vector_store.add_documents.call_args.kwargs["documents"][0]
    assert indexed.metadata == {
        "source": "guide",
        "tenant_id": "tenant-a",
        "chunk_index": 0,
    }


def test_database_url_schemes_are_normalized():
    assert normalize_database_url(
        "postgres://user:pass@localhost/db"
    ) == "postgresql+psycopg://user:pass@localhost/db"
    assert normalize_database_url(
        "postgresql://user:pass@localhost/db"
    ) == "postgresql+psycopg://user:pass@localhost/db"
