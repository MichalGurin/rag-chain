import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
from docx import Document as WordDocument
from fastapi import HTTPException, UploadFile

from app.api.documents import _persist_upload, _upload_source
from app.document_parser import parse_document
from app.main import UploadSizeLimitMiddleware
from scripts.evaluate_retrieval import score_retrieval


def test_upload_source_keeps_only_the_filename_and_allowed_suffix():
    assert _upload_source(r"..\private\handbook.pdf") == (
        "handbook.pdf",
        ".pdf",
    )
    with pytest.raises(HTTPException) as error:
        _upload_source("payload.exe")
    assert error.value.status_code == 415


def test_upload_rejects_files_over_the_configured_limit(tmp_path):
    upload = UploadFile(filename="policy.txt", file=BytesIO(b"too large"))

    with pytest.raises(ValueError, match="byte limit"):
        _persist_upload(upload, tmp_path / "policy.txt", 4)


def test_upload_checks_pdf_signature(tmp_path):
    upload = UploadFile(filename="policy.pdf", file=BytesIO(b"not a PDF"))

    with pytest.raises(ValueError, match="valid PDF"):
        _persist_upload(upload, tmp_path / "policy.pdf", 1024)


def test_upload_rejects_empty_file(tmp_path):
    upload = UploadFile(filename="empty.txt", file=BytesIO(b""))

    with pytest.raises(ValueError, match="Empty files"):
        _persist_upload(upload, tmp_path / "empty.txt", 1024)


@pytest.mark.parametrize("filename", ["policy.docx", "policy.pptx"])
def test_upload_rejects_invalid_office_archives(tmp_path, filename):
    upload = UploadFile(
        filename=filename,
        file=BytesIO(b"PK\x03\x04not an office archive"),
    )

    with pytest.raises(ValueError, match="Office file is invalid"):
        _persist_upload(upload, tmp_path / filename, 1024)


def test_plain_text_parser_preserves_utf8_content(tmp_path):
    document = tmp_path / "policy.txt"
    document.write_text("Annual leave: 20 days.\n", encoding="utf-8")

    content, page_count = parse_document(str(document), max_pages=5)

    assert content == "Annual leave: 20 days.\n"
    assert page_count == 1


def test_word_parser_preserves_paragraph_and_table_order(tmp_path):
    document = WordDocument()
    document.add_paragraph("Before the table.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Policy"
    table.cell(0, 1).text = "Retention"
    document.add_paragraph("After the table.")
    path = tmp_path / "policy.docx"
    document.save(path)

    content, page_count = parse_document(str(path), max_pages=5)

    assert content.index("Before the table.") < content.index("Policy | Retention")
    assert content.index("Policy | Retention") < content.index("After the table.")
    assert page_count == 1


def test_upload_middleware_rejects_chunked_bodies_over_limit():
    response_messages = []
    request_messages = [
        {"type": "http.request", "body": b"x" * 1_048_577, "more_body": True},
        {"type": "http.request", "body": b"x" * 1_048_577, "more_body": False},
    ]

    async def receive():
        return request_messages.pop(0)

    async def send(message):
        response_messages.append(message)

    class InnerApplication:
        state = SimpleNamespace(
            settings=SimpleNamespace(max_upload_bytes=1_000_000)
        )

        async def __call__(self, _scope, inner_receive, inner_send):
            await inner_receive()
            await inner_receive()
            await inner_send(
                {"type": "http.response.start", "status": 200, "headers": []}
            )
            await inner_send(
                {"type": "http.response.body", "body": b"accepted"}
            )

    limited_app = UploadSizeLimitMiddleware(InnerApplication())

    async def invoke():
        await limited_app(
            {
                "type": "http",
                "path": "/documents/upload",
                "headers": [],
            },
            receive,
            send,
        )

    asyncio.run(invoke())

    assert response_messages[0]["status"] == 413
    assert b"configured limit" in response_messages[1]["body"]
    response_headers = dict(response_messages[0]["headers"])
    assert response_headers[b"x-request-id"]
    assert response_headers[b"x-content-type-options"] == b"nosniff"
    assert response_headers[b"x-frame-options"] == b"DENY"
    assert response_headers[b"referrer-policy"] == b"no-referrer"
    assert response_headers[b"cross-origin-opener-policy"] == b"same-origin"


def test_upload_middleware_declared_oversize_response_has_security_headers():
    response_messages = []

    async def receive():
        raise AssertionError("A declared oversized body must be rejected early.")

    async def send(message):
        response_messages.append(message)

    class InnerApplication:
        state = SimpleNamespace(
            settings=SimpleNamespace(max_upload_bytes=1_000_000)
        )

        async def __call__(self, _scope, _receive, _send):
            raise AssertionError("The wrapped application must not be called.")

    limited_app = UploadSizeLimitMiddleware(InnerApplication())

    async def invoke():
        await limited_app(
            {
                "type": "http",
                "path": "/documents/upload",
                "scheme": "https",
                "headers": [(b"content-length", b"2100000")],
            },
            receive,
            send,
        )

    asyncio.run(invoke())

    assert response_messages[0]["status"] == 413
    response_headers = dict(response_messages[0]["headers"])
    assert response_headers[b"x-request-id"]
    assert response_headers[b"cross-origin-opener-policy"] == b"same-origin"


def test_retrieval_evaluation_calculates_hit_rate_and_mrr():
    examples = [
        {"expected_sources": ["a"]},
        {"expected_sources": ["b"]},
        {"expected_sources": ["c"]},
    ]

    scores = score_retrieval(
        examples,
        [["a", "x"], ["x", "b"], ["x", "y"]],
        k=2,
    )

    assert scores == {
        "hit_rate@2": 2 / 3,
        "mrr@2": 0.5,
        "queries": 3.0,
    }


def test_retrieval_evaluation_rejects_incomplete_results():
    with pytest.raises(ValueError, match="equal non-zero length"):
        score_retrieval([{"expected_sources": ["a"]}], [], k=5)
