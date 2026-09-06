from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions

from pdf_document_agent import extractor


class FakeDocument:
    def __init__(self, markdown: str, page_count: int = 2) -> None:
        self._markdown = markdown
        self.pages = {page_number: object() for page_number in range(1, page_count + 1)}

    def export_to_markdown(self, *, page_no: int | None = None) -> str:
        if page_no is not None:
            return f"Страница {page_no}: {self._markdown}"
        return self._markdown


class FakeConversionResult:
    def __init__(self, document: FakeDocument) -> None:
        self.document = document


class FakeConverter:
    markdown = "# Заголовок\n\nТекст документа"
    received_source: Path | None = None
    received_format_options: dict | None = None

    def __init__(self, *, format_options: dict | None = None) -> None:
        type(self).received_format_options = format_options

    def convert(self, source: Path) -> FakeConversionResult:
        type(self).received_source = source
        return FakeConversionResult(FakeDocument(self.markdown))


def test_extract_pdf_returns_structured_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "article.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test fixture")
    monkeypatch.setattr(extractor, "DocumentConverter", FakeConverter)

    result = extractor.extract_pdf(pdf_path)

    assert result.source_name == "article.pdf"
    assert result.markdown == "# Заголовок\n\nТекст документа"
    assert result.page_count == 2
    assert [page.number for page in result.pages] == [1, 2]
    assert result.pages[0].markdown.startswith("Страница 1:")
    assert result.pages[1].markdown.startswith("Страница 2:")
    assert FakeConverter.received_source == pdf_path
    assert FakeConverter.received_format_options is not None

    pdf_options = FakeConverter.received_format_options[InputFormat.PDF]
    ocr_options = pdf_options.pipeline_options.ocr_options
    assert isinstance(ocr_options, OcrMacOptions)
    assert ocr_options.lang == ["ru-RU", "en-US"]


def test_extract_pdf_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF не найден"):
        extractor.extract_pdf(tmp_path / "missing.pdf")


def test_extract_pdf_rejects_non_pdf_file(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Требуется файл формата PDF"):
        extractor.extract_pdf(text_path)


def test_extract_pdf_rejects_empty_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test fixture")

    class EmptyConverter(FakeConverter):
        markdown = "   "

    monkeypatch.setattr(extractor, "DocumentConverter", EmptyConverter)

    with pytest.raises(extractor.PdfExtractionError, match="не удалось извлечь содержимое"):
        extractor.extract_pdf(pdf_path)
