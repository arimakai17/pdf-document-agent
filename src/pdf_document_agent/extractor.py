import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMacOptions,
    OcrMode,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


class PdfExtractionError(RuntimeError):
    """PDF существует, но его содержимое не удалось извлечь."""


# (left, top, right, bottom) в долях [0, 1] относительно размера страницы,
# в системе координат TOPLEFT. Получается ровно один раз на границе extraction.
NormalizedBox = tuple[float, float, float, float]

_GATE_VERSION = "wave2-meaningful-text-v1"
_WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_MARKDOWN_NOISE = re.compile(
    r"<!--.*?-->|!\[[^\]]*\]\([^)]*\)|<\/?(?:image|figure)[^>]*>",
    re.IGNORECASE | re.DOTALL,
)
_MAX_MANUAL_OCR_PAGES = 1_000
_MAX_DIAGNOSTIC_LENGTH = 300


def parse_ocr_pages(value: str) -> tuple[int, ...]:
    """Разобрать список страниц для ручного OCR override."""
    if not isinstance(value, str):
        raise ValueError("ocr_pages должен быть строкой страниц")
    if not value.strip():
        return ()

    pages: set[int] = set()
    for component in value.split(","):
        component = component.strip()
        if not component:
            raise ValueError("ocr_pages содержит пустой компонент")
        match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", component)
        if match is None:
            raise ValueError(f"Некорректный компонент ocr_pages: {component}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1:
            raise ValueError("ocr_pages должен содержать только страницы >= 1")
        if end < start:
            raise ValueError("Диапазон ocr_pages должен быть возрастающим")
        if end - start + 1 > _MAX_MANUAL_OCR_PAGES:
            raise ValueError(
                f"ocr_pages поддерживает не более {_MAX_MANUAL_OCR_PAGES} страниц"
            )
        pages.update(range(start, end + 1))
        if len(pages) > _MAX_MANUAL_OCR_PAGES:
            raise ValueError(
                f"ocr_pages поддерживает не более {_MAX_MANUAL_OCR_PAGES} страниц"
            )
    return tuple(sorted(pages))


@dataclass(frozen=True)
class ExtractionConfig:
    """Детерминированная политика page gate и selective OCR."""

    min_meaningful_chars: int = 20
    min_meaningful_words: int = 4
    significant_raster_area: float = 0.15
    ocr_pages: tuple[int, ...] = ()
    gate_version: str = _GATE_VERSION
    ocr_mode: str = OcrMode.FULL_PAGE.value
    ocr_backend: str = "ocrmac"
    ocr_languages: tuple[str, ...] = ("ru-RU", "en-US")

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_meaningful_chars, bool)
            or not isinstance(self.min_meaningful_chars, int)
            or self.min_meaningful_chars < 0
        ):
            raise ValueError("min_meaningful_chars должен быть неотрицательным int")
        if (
            isinstance(self.min_meaningful_words, bool)
            or not isinstance(self.min_meaningful_words, int)
            or self.min_meaningful_words < 0
        ):
            raise ValueError("min_meaningful_words должен быть неотрицательным int")
        if (
            isinstance(self.significant_raster_area, bool)
            or not isinstance(self.significant_raster_area, (int, float))
            or not math.isfinite(self.significant_raster_area)
            or not 0.0 < self.significant_raster_area <= 1.0
        ):
            raise ValueError("significant_raster_area должен быть finite в (0, 1]")
        if not isinstance(self.gate_version, str) or not self.gate_version:
            raise ValueError("gate_version должен быть непустой строкой")
        if self.ocr_backend != "ocrmac":
            raise ValueError("Поддерживается только ocrmac backend")
        if self.ocr_mode != OcrMode.FULL_PAGE.value:
            raise ValueError("Поддерживается только full_page OCR mode")
        if isinstance(self.ocr_languages, (str, bytes)):
            raise ValueError("ocr_languages должен быть последовательностью строк")
        try:
            languages = tuple(self.ocr_languages)
        except TypeError as error:
            raise ValueError("ocr_languages должен быть последовательностью строк") from error
        if not languages or any(
            not isinstance(language, str) or not language for language in languages
        ):
            raise ValueError("ocr_languages должен быть непустым tuple строк")
        object.__setattr__(self, "ocr_languages", languages)

        try:
            pages = tuple(self.ocr_pages)
        except TypeError as error:
            raise ValueError("ocr_pages должен быть последовательностью страниц") from error
        if any(
            isinstance(page, bool) or not isinstance(page, int) or page < 1
            for page in pages
        ):
            raise ValueError("ocr_pages должен содержать только страницы >= 1")
        object.__setattr__(self, "ocr_pages", tuple(sorted(set(pages))))
        object.__setattr__(
            self, "significant_raster_area", float(self.significant_raster_area)
        )

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self)


DEFAULT_EXTRACTION_CONFIG = ExtractionConfig()


def config_fingerprint(config: ExtractionConfig = DEFAULT_EXTRACTION_CONFIG) -> str:
    """Вернуть стабильный SHA-256 fingerprint канонической extraction policy."""
    if not isinstance(config, ExtractionConfig):
        raise TypeError("config должен быть ExtractionConfig")
    payload = {
        "gate_version": config.gate_version,
        "min_meaningful_chars": config.min_meaningful_chars,
        "min_meaningful_words": config.min_meaningful_words,
        "significant_raster_area": config.significant_raster_area,
        "ocr_mode": config.ocr_mode,
        "ocr_backend": config.ocr_backend,
        "ocr_languages": list(config.ocr_languages),
        "ocr_pages": list(config.ocr_pages),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


extraction_config_fingerprint = config_fingerprint
DEFAULT_EXTRACTION_CONFIG_FINGERPRINT = config_fingerprint(DEFAULT_EXTRACTION_CONFIG)


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
    route: str = "docling_text"
    status: str = "ok"
    reason: str = "first-pass text"
    diagnostic: str | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    """Нормализованный результат обработки одного PDF-документа."""

    source_name: str
    markdown: str
    page_count: int
    pages: tuple[ExtractedPage, ...]
    config_fingerprint: str = DEFAULT_EXTRACTION_CONFIG_FINGERPRINT
    warnings: tuple[str, ...] = ()


def extract_pdf(
    pdf_path: str | Path,
    *,
    config: ExtractionConfig = DEFAULT_EXTRACTION_CONFIG,
) -> ExtractedDocument:
    """Извлечь PDF first-pass text и при необходимости отдельные OCR-страницы."""
    source = Path(pdf_path).expanduser()
    if source.suffix.lower() != ".pdf":
        raise ValueError("Требуется файл формата PDF.")
    if not source.is_file():
        raise FileNotFoundError(f"PDF не найден: {source}")
    if not isinstance(config, ExtractionConfig):
        raise TypeError("config должен быть ExtractionConfig")

    first_result = _convert_first_pass(source)
    first_document = first_result.document
    first_pass_page_diagnostics, document_diagnostics = _conversion_diagnostics(
        first_result
    )
    page_numbers = tuple(sorted(first_document.pages))
    page_count = len(page_numbers)
    if page_count == 0:
        raise PdfExtractionError(
            f"Из PDF «{source.name}» не удалось извлечь содержимое."
        )
    invalid_overrides = [page for page in config.ocr_pages if page not in page_numbers]
    if invalid_overrides:
        raise ValueError(
            "ocr_pages содержит отсутствующие страницы: "
            + ", ".join(map(str, invalid_overrides))
        )

    regions_by_page = _collect_text_regions(first_document)
    assembled_raster_areas = _picture_areas(first_document)
    embedded_raster_areas = _embedded_raster_areas(source, page_numbers)
    pages: dict[int, ExtractedPage] = {}
    warnings: list[str] = [
        f"Document: {diagnostic}" for diagnostic in document_diagnostics
    ]
    ocr_targets: dict[int, bool] = {}
    for page_number in page_numbers:
        page, should_ocr, significant_raster = _first_pass_page(
            first_document,
            page_number,
            config,
            regions=tuple(regions_by_page.get(page_number, ())),
            raster_area=max(
                assembled_raster_areas.get(page_number, 0.0),
                embedded_raster_areas.get(page_number, 0.0),
            ),
            conversion_diagnostic=first_pass_page_diagnostics.get(page_number),
        )
        pages[page_number] = page
        if page.diagnostic:
            warnings.append(f"Page {page_number}: {page.diagnostic}")
        if should_ocr:
            ocr_targets[page_number] = significant_raster

    if ocr_targets:
        try:
            ocr_converter = _make_ocr_converter(config)
        except Exception as error:
            diagnostic = str(error) or type(error).__name__
            for page_number in ocr_targets:
                pages[page_number] = _failed_ocr_page(
                    pages[page_number], diagnostic
                )
                warnings.append(f"Page {page_number}: {diagnostic}")
        else:
            for page_number, significant_raster in ocr_targets.items():
                replacement, ocr_warnings = _extract_ocr_page(
                    ocr_converter,
                    source,
                    page_number,
                    pages[page_number],
                    significant_raster=significant_raster,
                )
                pages[page_number] = replacement
                for warning in ocr_warnings:
                    if warning not in warnings:
                        warnings.append(warning)

    for page_number, diagnostic in first_pass_page_diagnostics.items():
        if page_number not in pages:
            warnings.append(f"Page {page_number}: {diagnostic}")

    final_pages = tuple(pages[page_number] for page_number in page_numbers)
    markdown = "\n\n".join(
        page.markdown for page in final_pages if page.markdown
    ).strip()
    return ExtractedDocument(
        source_name=source.name,
        markdown=markdown,
        page_count=page_count,
        pages=final_pages,
        config_fingerprint=config_fingerprint(config),
        warnings=tuple(warnings),
    )


def _convert_first_pass(source: Path):
    try:
        options = ThreadedPdfPipelineOptions()
        options.do_ocr = False
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )
        result = converter.convert(source)
        result.document
        return result
    except Exception as error:
        raise PdfExtractionError(
            f"Не удалось обработать PDF «{source.name}»."
        ) from error


def _conversion_diagnostics(result) -> tuple[dict[int, str], tuple[str, ...]]:
    page_diagnostics: dict[int, list[str]] = {}
    document_diagnostics: list[str] = []
    for error in getattr(result, "errors", ()) or ():
        diagnostic = _docling_error_diagnostic(error)
        page_number = getattr(error, "page_no", None)
        if page_number is None:
            document_diagnostics.append(diagnostic)
        else:
            page_diagnostics.setdefault(int(page_number), []).append(diagnostic)
    return (
        {
            page_number: _join_diagnostics(diagnostics)
            for page_number, diagnostics in page_diagnostics.items()
        },
        tuple(document_diagnostics),
    )


def _docling_error_diagnostic(error) -> str:
    message = _bounded_diagnostic(getattr(error, "error_message", ""))
    component = _bounded_diagnostic(
        getattr(getattr(error, "component_type", None), "value", "")
    )
    module = _bounded_diagnostic(getattr(error, "module_name", ""))
    category = _bounded_diagnostic(
        getattr(getattr(error, "category", None), "value", "")
    )
    context = "/".join(value for value in (component, module, category) if value)
    diagnostic = "Docling conversion error"
    if message:
        diagnostic += f": {message}"
    if context:
        diagnostic += f" ({context})"
    return _bounded_diagnostic(diagnostic)


def _join_diagnostics(diagnostics: list[str]) -> str:
    unique = list(dict.fromkeys(diagnostic for diagnostic in diagnostics if diagnostic))
    return _bounded_diagnostic("; ".join(unique))


def _bounded_diagnostic(value: object) -> str:
    text = str(value) if value is not None else ""
    text = "".join(character if character.isprintable() else " " for character in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_DIAGNOSTIC_LENGTH]


def _first_pass_page(
    document,
    page_number: int,
    config: ExtractionConfig,
    *,
    regions: tuple[TextRegion, ...] = (),
    raster_area: float = 0.0,
    conversion_diagnostic: str | None = None,
) -> tuple[ExtractedPage, bool, bool]:
    try:
        page_object = document.pages[page_number]
        source_page_no = getattr(page_object, "page_no", None)
        if source_page_no is not None and int(source_page_no) != page_number:
            raise ValueError(
                f"first-pass page identity mismatch: expected {page_number}, "
                f"got {source_page_no}"
            )
        markdown = _export_page_markdown(document, page_number)
        meaningful_chars, meaningful_words = _meaningful_metrics(markdown)
        text_sufficient = _text_gate_passes(
            meaningful_chars,
            meaningful_words,
            config,
        )

    except Exception as error:
        return (
            ExtractedPage(
                number=page_number,
                markdown="",
                route="docling_text",
                status="failed",
                reason="first-pass page unavailable",
                diagnostic=_bounded_diagnostic(str(error)) or type(error).__name__,
            ),
            False,
            False,
        )

    manual_override = page_number in config.ocr_pages
    significant_raster = raster_area >= config.significant_raster_area
    if manual_override:
        reason = "manual OCR override"
        should_ocr = True
    elif conversion_diagnostic and significant_raster:
        reason = "first-pass partial conversion; significant raster detected"
        should_ocr = True
    elif text_sufficient:
        should_ocr = False
        if significant_raster:
            reason = (
                "meaningful text gate passed; automatic OCR skipped for mixed page; "
                "manual override available"
            )
        else:
            reason = "meaningful text gate passed"
    elif significant_raster:
        should_ocr = True
        reason = "meaningful text gate failed; significant raster detected"
    else:
        should_ocr = False
        reason = "meaningful text gate failed; no significant raster"

    if conversion_diagnostic:
        status = "ok" if meaningful_chars else "failed"
        reason = f"first-pass partial conversion; {reason}"
    else:
        status = "ok" if meaningful_chars else "empty"

    return (
        ExtractedPage(
            number=page_number,
            markdown=markdown,
            regions=regions,
            route="docling_text",
            status=status,
            reason=reason,
            diagnostic=conversion_diagnostic,
        ),
        should_ocr,
        significant_raster,
    )


def _make_ocr_converter(config: ExtractionConfig):
    options = ThreadedPdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = False
    options.ocr_options = OcrMacOptions(
        mode=OcrMode.FULL_PAGE,
        lang=list(config.ocr_languages),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )


def _extract_ocr_page(
    converter,
    source: Path,
    page_number: int,
    first_page: ExtractedPage,
    *,
    significant_raster: bool,
):
    page_diagnostic: str | None = None
    document_diagnostics: tuple[str, ...] = ()
    try:
        result = converter.convert(source, page_range=(page_number, page_number))
        document = result.document
        page_diagnostics, document_diagnostics = _conversion_diagnostics(result)
        page_diagnostic = page_diagnostics.get(page_number)
        _validate_selective_result(document, page_number)
        markdown = _export_page_markdown(document, page_number)
        regions = tuple(_collect_text_regions(document).get(page_number, ()))
        meaningful_chars, _meaningful_words_count = _meaningful_metrics(markdown)
        if not meaningful_chars:
            if _meaningful_metrics(first_page.markdown)[0]:
                diagnostic = _join_diagnostics(
                    [
                        value
                        for value in (first_page.diagnostic, page_diagnostic)
                        if value
                    ]
                )
                if not page_diagnostic:
                    diagnostic = _bounded_diagnostic("OCR page has no usable text")
                return (
                    _failed_ocr_page(first_page, diagnostic),
                    _ocr_warnings(page_number, page_diagnostic, document_diagnostics),
                )
            if not page_diagnostic and not significant_raster:
                return (
                    ExtractedPage(
                        number=page_number,
                        markdown="",
                        regions=(),
                        route="docling_ocr",
                        status="empty",
                        reason="selective OCR found no text on blank page",
                    ),
                    _ocr_warnings(page_number, None, document_diagnostics),
                )
            raise ValueError("OCR page has no usable text")
        if not regions:
            raise ValueError("OCR text has no usable source provenance")
        return (
            ExtractedPage(
                number=page_number,
                markdown=markdown,
                regions=regions,
                route="docling_ocr",
                status="ok",
                reason=(
                    "selective OCR; partial conversion"
                    if page_diagnostic
                    else "selective OCR"
                ),
                diagnostic=page_diagnostic or first_page.diagnostic,
            ),
            _ocr_warnings(page_number, page_diagnostic, document_diagnostics),
        )
    except Exception as error:
        diagnostic = _bounded_diagnostic(str(error)) or type(error).__name__
        diagnostic = _join_diagnostics(
            [value for value in (page_diagnostic, diagnostic) if value]
        )
        replacement = _failed_ocr_page(first_page, diagnostic)
        return replacement, _ocr_warnings(
            page_number, diagnostic, document_diagnostics
        )


def _ocr_warnings(
    page_number: int,
    page_diagnostic: str | None,
    document_diagnostics: tuple[str, ...],
) -> tuple[str, ...]:
    warnings = []
    if page_diagnostic:
        warnings.append(f"Page {page_number}: {page_diagnostic}")
    warnings.extend(f"Document: {diagnostic}" for diagnostic in document_diagnostics)
    return tuple(warnings)


def _failed_ocr_page(first_page: ExtractedPage, diagnostic: str) -> ExtractedPage:
    diagnostic = _join_diagnostics(
        [value for value in (first_page.diagnostic, diagnostic) if value]
    )
    if _meaningful_metrics(first_page.markdown)[0]:
        return ExtractedPage(
            number=first_page.number,
            markdown=first_page.markdown,
            regions=first_page.regions,
            route="docling_text",
            status="ok",
            reason=first_page.reason,
            diagnostic=diagnostic,
        )
    return ExtractedPage(
        number=first_page.number,
        markdown="",
        regions=(),
        route="docling_ocr",
        status="failed",
        reason="selective OCR failed",
        diagnostic=diagnostic,
    )


def _validate_selective_result(document, page_number: int) -> None:
    pages = getattr(document, "pages", None)
    if not isinstance(pages, dict) or tuple(sorted(pages)) != (page_number,):
        raise ValueError(
            f"selective page identity mismatch: expected only page {page_number}"
        )
    page_object = pages[page_number]
    source_page_no = getattr(page_object, "page_no", None)
    if source_page_no is not None and int(source_page_no) != page_number:
        raise ValueError(
            f"selective page identity mismatch: expected {page_number}, "
            f"got {source_page_no}"
        )
    for provenance_page_no in _provenance_page_numbers(document):
        if provenance_page_no != page_number:
            raise ValueError(
                f"selective provenance mismatch: expected {page_number}, "
                f"got {provenance_page_no}"
            )


def _provenance_page_numbers(document) -> tuple[int, ...]:
    iterate_items = getattr(document, "iterate_items", None)
    if iterate_items is None:
        return ()
    page_numbers: list[int] = []
    for item_or_pair in iterate_items():
        item = item_or_pair[0] if isinstance(item_or_pair, tuple) else item_or_pair
        for prov in getattr(item, "prov", None) or ():
            page_numbers.append(int(prov.page_no))
    return tuple(page_numbers)


def _export_page_markdown(document, page_number: int) -> str:
    value = document.export_to_markdown(page_no=page_number)
    return value.strip() if isinstance(value, str) else ""


def _meaningful_metrics(markdown: str) -> tuple[int, int]:
    cleaned = _MARKDOWN_NOISE.sub(" ", markdown)
    cleaned = re.sub(r"[`*_~#>|]", " ", cleaned)
    chars = len(re.sub(r"\s+", "", cleaned))
    words = len(_WORD_PATTERN.findall(cleaned))
    return chars, words


def _text_gate_passes(chars: int, words: int, config: ExtractionConfig) -> bool:
    return chars >= config.min_meaningful_chars and words >= config.min_meaningful_words


def _is_picture_item(item) -> bool:
    name = type(item).__name__.lower()
    if "picture" in name:
        return True
    label = getattr(item, "label", None)
    label_value = getattr(label, "value", label)
    return isinstance(label_value, str) and label_value.lower() in {"picture", "figure"}


def _picture_areas(document) -> dict[int, float]:
    iterate_items = getattr(document, "iterate_items", None)
    if iterate_items is None:
        return {}
    areas: dict[int, float] = {}
    pages = getattr(document, "pages", {})
    for item_or_pair in iterate_items():
        item = item_or_pair[0] if isinstance(item_or_pair, tuple) else item_or_pair
        if not _is_picture_item(item):
            continue
        for prov in getattr(item, "prov", None) or ():
            box = _normalize_box(prov, pages)
            if box is not None:
                page_number = int(prov.page_no)
                area = (box[2] - box[0]) * (box[3] - box[1])
                areas[page_number] = min(areas.get(page_number, 0.0) + area, 1.0)
    return areas


def _embedded_raster_areas(
    source: Path,
    page_numbers: tuple[int, ...],
) -> dict[int, float]:
    """Measure embedded-image coverage directly at the local PDF boundary.

    Docling can omit full-page pictures from its assembled items even though
    the PDF page contains a large raster. The low-level signal is used only for
    OCR routing; Docling remains the source of extracted text and provenance.
    """
    document = None
    try:
        document = pdfium.PdfDocument(source)
        areas: dict[int, float] = {}
        for page_number in page_numbers:
            if page_number > len(document):
                continue
            page = document[page_number - 1]
            try:
                width, height = page.get_size()
                if not (width > 0 and height > 0):
                    continue
                image_area = 0.0
                for item in page.get_objects(
                    filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]
                ):
                    if not isinstance(item, pdfium.PdfImage):
                        continue
                    left, bottom, right, top = item.get_bounds()
                    clipped_width = max(
                        0.0,
                        min(max(left, right), width) - max(min(left, right), 0.0),
                    )
                    clipped_height = max(
                        0.0,
                        min(max(bottom, top), height) - max(min(bottom, top), 0.0),
                    )
                    image_area += clipped_width * clipped_height
                areas[page_number] = min(image_area / (width * height), 1.0)
            finally:
                page.close()
        return areas
    except Exception:
        return {}
    finally:
        if document is not None:
            document.close()


def _collect_text_regions(document) -> dict[int, list[TextRegion]]:
    """Собрать provenance и нормализовать координаты ровно на extraction boundary."""
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
