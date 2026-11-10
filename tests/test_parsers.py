from pathlib import Path

import pytest

from app.document_parser import parse_document
from tests.document_samples import EXPECTED_TEXT, supported_upload_samples


@pytest.mark.parametrize(
    ("filename", "content", "content_type"),
    supported_upload_samples(),
    ids=[sample[0].split(".")[-1] for sample in supported_upload_samples()],
)
def test_parser_extracts_every_supported_upload_type(
    tmp_path: Path,
    filename: str,
    content: bytes,
    content_type: str,
):
    assert content_type
    document = tmp_path / filename
    document.write_bytes(content)

    extracted, page_count = parse_document(str(document), max_pages=5)

    assert EXPECTED_TEXT in extracted
    assert page_count == 1


def test_pdf_page_limit_is_enforced(tmp_path: Path):
    filename, content, _ = next(
        sample for sample in supported_upload_samples() if sample[0].endswith(".pdf")
    )
    document = tmp_path / filename
    document.write_bytes(content)

    with pytest.raises(ValueError, match="may not exceed 0 pages"):
        parse_document(str(document), max_pages=0)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("broken.pdf", b"%PDF-1.4\nnot a valid PDF"),
        ("broken.docx", b"PK\x03\x04invalid archive"),
        ("broken.pptx", b"PK\x03\x04invalid archive"),
    ],
)
def test_parser_surfaces_corrupt_document_errors(
    tmp_path: Path,
    filename: str,
    content: bytes,
):
    document = tmp_path / filename
    document.write_bytes(content)

    with pytest.raises(Exception):
        parse_document(str(document), max_pages=5)


def test_parser_rejects_unsupported_extensions(tmp_path: Path):
    document = tmp_path / "policy.xlsx"
    document.write_bytes(b"not supported")

    with pytest.raises(ValueError, match="Supported file types"):
        parse_document(str(document), max_pages=5)
