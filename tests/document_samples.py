"""Small in-memory document fixtures shared by parser and live E2E tests."""

from io import BytesIO

from docx import Document as WordDocument
from pptx import Presentation
from pptx.util import Inches


EXPECTED_TEXT = "E2E annual leave policy grants twenty days."


def _pdf_bytes(text: str) -> bytes:
    """Create a minimal, searchable one-page PDF without external tooling."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, content in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(
            f"{index} 0 obj\n".encode("ascii")
            + content
            + b"\nendobj\n"
        )
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(document)


def supported_upload_samples() -> list[tuple[str, bytes, str]]:
    """Return one valid, extractable sample for every supported upload type."""
    word = WordDocument()
    word.add_paragraph(EXPECTED_TEXT)
    word_bytes = BytesIO()
    word.save(word_bytes)

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    text_box = slide.shapes.add_textbox(
        Inches(1),
        Inches(1),
        Inches(8),
        Inches(1),
    )
    text_box.text = EXPECTED_TEXT
    slide_bytes = BytesIO()
    presentation.save(slide_bytes)

    return [
        ("policy.txt", f"{EXPECTED_TEXT}\n".encode(), "text/plain"),
        ("policy.md", f"# Policy\n\n{EXPECTED_TEXT}\n".encode(), "text/markdown"),
        (
            "policy.docx",
            word_bytes.getvalue(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "policy.pptx",
            slide_bytes.getvalue(),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        ("policy.pdf", _pdf_bytes(EXPECTED_TEXT), "application/pdf"),
    ]
