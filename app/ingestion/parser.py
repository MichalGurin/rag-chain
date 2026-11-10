"""Layout-aware extraction for supported office and text document formats."""

from pathlib import Path

from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
import pdfplumber
from pptx import Presentation


def parse_document(path: str, max_pages: int) -> tuple[str, int]:
    """Extract ordered text and tables while rejecting unsupported/scanned files."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8-sig"), 1
    if suffix == ".pdf":
        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            if page_count > max_pages:
                raise ValueError(f"Documents may not exceed {max_pages} pages.")
            pages = []
            for index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text(layout=True) or ""
                tables = page.extract_tables() or []
                table_text = [
                    "\n".join(
                        " | ".join(cell or "" for cell in row)
                        for row in table
                    )
                    for table in tables
                ]
                if page.images and not page_text.strip() and not table_text:
                    raise ValueError(
                        f"Page {index} contains scanned imagery; OCR is required."
                    )
                content = "\n\n".join(
                    part for part in [page_text, *table_text] if part
                )
                pages.append(f"[Page {index}]\n{content}")
        return "\n\n".join(pages), page_count
    if suffix == ".pptx":
        presentation = Presentation(str(file_path))
        if len(presentation.slides) > max_pages:
            raise ValueError(f"Documents may not exceed {max_pages} slides.")
        slides = []
        for index, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text.strip():
                    parts.append(shape.text.strip())
                if shape.has_table:
                    parts.extend(
                        " | ".join(cell.text.strip() for cell in row.cells)
                        for row in shape.table.rows
                    )
            slides.append(f"[Slide {index}]\n" + "\n".join(parts))
        return "\n\n".join(slides), len(presentation.slides)
    if suffix == ".docx":
        document = WordDocument(str(file_path))
        parts = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    parts.append(block.text.strip())
            elif isinstance(block, Table):
                parts.extend(
                    " | ".join(cell.text.strip() for cell in row.cells)
                    for row in block.rows
                )
        return "\n".join(part for part in parts if part), 1
    raise ValueError("Supported file types are PDF, PPTX, DOCX, TXT, and MD.")
