from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class PdfExtractionError(RuntimeError):
    """PDF существует, но его содержимое не удалось извлечь."""


@dataclass(frozen=True)
class ExtractedPage:
    """Текст одной страницы с исходным номером из PDF."""

    number: int
    markdown: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Нормализованный результат обработки одного PDF-документа."""

    source_name: str
    markdown: str
    page_count: int
    pages: tuple[ExtractedPage, ...]


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    """Извлечь структурированное содержимое PDF в Markdown через Docling."""
    source = Path(pdf_path).expanduser()

    if source.suffix.lower() != ".pdf":
        raise ValueError("Требуется файл формата PDF.")
    if not source.is_file():
        raise FileNotFoundError(f"PDF не найден: {source}")

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = OcrMacOptions(
            lang=["ru-RU", "en-US"],
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        result = converter.convert(source)
        markdown = result.document.export_to_markdown().strip()
        page_count = len(result.document.pages)
        pages = tuple(
            ExtractedPage(
                number=page_number,
                markdown=result.document.export_to_markdown(
                    page_no=page_number
                ).strip(),
            )
            for page_number in sorted(result.document.pages)
        )
    except Exception as error:
        raise PdfExtractionError(
            f"Не удалось обработать PDF «{source.name}»."
        ) from error

    if not markdown or page_count == 0:
        raise PdfExtractionError(
            f"Из PDF «{source.name}» не удалось извлечь содержимое."
        )

    return ExtractedDocument(
        source_name=source.name,
        markdown=markdown,
        page_count=page_count,
        pages=pages,
    )
