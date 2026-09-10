from pathlib import Path

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions
from docling_core.types.doc import BoundingBox, CoordOrigin, ProvenanceItem, Size

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


class _ProvItem:
    def __init__(self, text: str, prov: list[ProvenanceItem]) -> None:
        self.text = text
        self.prov = prov


class _ProvPage:
    def __init__(self, width: float, height: float) -> None:
        self.size = Size(width=width, height=height)


class ProvenanceDocument(FakeDocument):
    def __init__(
        self,
        markdown: str,
        items: list[_ProvItem],
        page_sizes: dict[int, tuple[float, float]],
    ) -> None:
        super().__init__(markdown, page_count=len(page_sizes))
        self._items = items
        self.pages = {
            page_no: _ProvPage(width, height)
            for page_no, (width, height) in page_sizes.items()
        }

    def iterate_items(self):
        for item in self._items:
            yield item, 0


class GeoConverter:
    def __init__(self, *, format_options: dict | None = None) -> None:
        self.format_options = format_options

    def convert(self, source: Path) -> FakeConversionResult:
        return FakeConversionResult(GeoConverter.document)


def _patch_geo_converter(monkeypatch: pytest.MonkeyPatch, document: ProvenanceDocument) -> None:
    GeoConverter.document = document
    monkeypatch.setattr(extractor, "DocumentConverter", GeoConverter)


def test_extract_pdf_attaches_normalized_regions_from_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "geo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test fixture")

    items = [
        _ProvItem(
            "Привет мир",
            [
                ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(
                        l=0, t=700, r=200, b=600, coord_origin=CoordOrigin.BOTTOMLEFT
                    ),
                    charspan=(0, 10),
                )
            ],
        ),
    ]
    document = ProvenanceDocument("Привет мир", items, {1: (400, 800)})
    _patch_geo_converter(monkeypatch, document)

    result = extractor.extract_pdf(pdf_path)

    regions = result.pages[0].regions
    assert len(regions) == 1
    region = regions[0]
    assert region.page_number == 1
    assert region.text == "Привет мир"
    # BOTTOMLEFT (l=0, t=700, r=200, b=600) на 400x800 -> TOPLEFT (t=100, b=200),
    # нормировка: left=0, top=0.125, right=0.5, bottom=0.25.
    assert region.box == (0.0, 0.125, 0.5, 0.25)


def test_extract_pdf_drops_regions_with_invalid_or_missing_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "geo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test fixture")

    items = [
        _ProvItem("без prov", []),
        _ProvItem(
            "нулевой bbox",
            [
                ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(l=10, t=10, r=10, b=20),
                    charspan=(0, 10),
                )
            ],
        ),
        _ProvItem(
            "bbox без страницы",
            [
                ProvenanceItem(
                    page_no=99,
                    bbox=BoundingBox(l=0, t=0, r=100, b=100),
                    charspan=(0, 10),
                )
            ],
        ),
        _ProvItem(
            "валидный",
            [
                ProvenanceItem(
                    page_no=1,
                    bbox=BoundingBox(l=0, t=0, r=100, b=100),
                    charspan=(0, 10),
                )
            ],
        ),
    ]
    document = ProvenanceDocument("текст", items, {1: (400, 800)})
    _patch_geo_converter(monkeypatch, document)

    result = extractor.extract_pdf(pdf_path)

    regions = result.pages[0].regions
    assert [region.text for region in regions] == ["валидный"]
    assert regions[0].box == (0.0, 0.0, 0.25, 0.125)
