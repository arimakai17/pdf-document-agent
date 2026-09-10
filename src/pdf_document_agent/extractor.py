import math
from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


class PdfExtractionError(RuntimeError):
    """PDF существует, но его содержимое не удалось извлечь."""


# (left, top, right, bottom) в долях [0, 1] относительно размера страницы,
# в системе координат TOPLEFT. Получается ровно один раз на границе extraction.
NormalizedBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class TextRegion:
    """Текстовый регион с реальной нормализованной геометрией из Docling provenance."""

    page_number: int
    text: str
    box: NormalizedBox


@dataclass(frozen=True)
class ExtractedPage:
    """Текст одной страницы с исходным номером из PDF."""

    number: int
    markdown: str
    regions: tuple[TextRegion, ...] = ()


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
        regions_by_page = _collect_text_regions(result.document)
        pages = tuple(
            ExtractedPage(
                number=page_number,
                markdown=result.document.export_to_markdown(
                    page_no=page_number
                ).strip(),
                regions=tuple(regions_by_page.get(page_number, ())),
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


def _collect_text_regions(document) -> dict[int, list[TextRegion]]:
    """Собрать реальную геометрию текста из Docling item provenance.

    Возвращает словарь page_no -> список TextRegion. Координаты нормализуются
    в [0, 1] и приводятся к TOPLEFT ровно один раз здесь, на границе extraction.
    Невалидные/нулевые boxes отбрасываются.
    """
    iterate_items = getattr(document, "iterate_items", None)
    if iterate_items is None:
        return {}
    pages = getattr(document, "pages", {})
    regions: dict[int, list[TextRegion]] = {}
    for item_or_pair in iterate_items():
        if isinstance(item_or_pair, tuple) and len(item_or_pair) == 2:
            item, _level = item_or_pair
        else:
            item = item_or_pair
        text = getattr(item, "text", None)
        if not isinstance(text, str) or not text.strip():
            continue
        for prov in getattr(item, "prov", None) or ():
            box = _normalize_box(prov, pages)
            if box is None:
                continue
            page_no = int(prov.page_no)
            regions.setdefault(page_no, []).append(
                TextRegion(page_number=page_no, text=text, box=box)
            )
    return regions


def _normalize_box(prov, pages) -> NormalizedBox | None:
    """Привести bbox provenance к нормализованному TOPLEFT box или вернуть None."""
    bbox = getattr(prov, "bbox", None)
    if bbox is None:
        return None
    page = pages.get(prov.page_no)
    if page is None:
        return None
    size = page.size
    width = size.width
    height = size.height
    if not (width > 0 and height > 0):
        return None

    top_left = bbox.to_top_left_origin(height)
    left = top_left.l / width
    top = top_left.t / height
    right = top_left.r / width
    bottom = top_left.b / height

    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    left = min(max(left, 0.0), 1.0)
    top = min(max(top, 0.0), 1.0)
    right = min(max(right, 0.0), 1.0)
    bottom = min(max(bottom, 0.0), 1.0)
    if left >= right or top >= bottom:
        return None
    return (left, top, right, bottom)
