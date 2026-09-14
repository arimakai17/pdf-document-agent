from pathlib import Path
from dataclasses import dataclass

import pytest
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions, ThreadedPdfPipelineOptions
from docling_core.types.doc import BoundingBox, CoordOrigin, ProvenanceItem, Size

from pdf_document_agent.eval_fixtures import build_synthetic_corpus
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
    assert result.markdown == (
        "Страница 1: # Заголовок\n\nТекст документа\n\n"
        "Страница 2: # Заголовок\n\nТекст документа"
    )
    assert result.page_count == 2
    assert [page.number for page in result.pages] == [1, 2]
    assert result.pages[0].markdown.startswith("Страница 1:")
    assert result.pages[1].markdown.startswith("Страница 2:")
    assert FakeConverter.received_source == pdf_path
    assert FakeConverter.received_format_options is not None

    pdf_options = FakeConverter.received_format_options[InputFormat.PDF]
    assert isinstance(pdf_options.pipeline_options, ThreadedPdfPipelineOptions)
    assert pdf_options.pipeline_options.do_ocr is False


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

        def convert(self, source: Path) -> FakeConversionResult:
            return FakeConversionResult(FakeDocument(self.markdown, page_count=0))

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


@dataclass
class _PageArtifact:
    size: Size
    page_no: int | None = None


class _PageDocument:
    def __init__(self, pages: dict[int, str], items=(), page_keys=None) -> None:
        keys = page_keys if page_keys is not None else pages.keys()
        self.pages = {
            number: _PageArtifact(Size(width=100, height=100), number)
            for number in keys
        }
        self._pages = pages
        self._items = list(items)

    def export_to_markdown(self, *, page_no: int | None = None) -> str:
        if page_no is None:
            return "\n\n".join(self._pages[number] for number in sorted(self._pages))
        return self._pages.get(page_no, "")

    def iterate_items(self):
        yield from ((item, 0) for item in self._items)


class _PictureItem(_ProvItem):
    pass


def _picture(page_number: int, area: float = 0.4) -> _PictureItem:
    side = area**0.5 * 100
    return _PictureItem(
        "",
        [
            ProvenanceItem(
                page_no=page_number,
                bbox=BoundingBox(l=0, t=0, r=side, b=side),
                charspan=(0, 0),
            )
        ],
    )


class _AdaptiveConverter:
    documents: dict[tuple[bool, tuple[int, int] | None], _PageDocument] = {}
    init_options = []
    convert_calls = []

    def __init__(self, *, format_options=None) -> None:
        assert format_options is not None
        options = format_options[InputFormat.PDF].pipeline_options
        type(self).init_options.append(options)

    def convert(self, source: Path, page_range=None):
        options = type(self).init_options[-1]
        key = (bool(options.do_ocr), page_range)
        type(self).convert_calls.append(key)
        document = type(self).documents[key]
        return FakeConversionResult(document)


def _patch_adaptive_converter(monkeypatch, documents):
    _AdaptiveConverter.documents = documents
    _AdaptiveConverter.init_options = []
    _AdaptiveConverter.convert_calls = []
    monkeypatch.setattr(extractor, "DocumentConverter", _AdaptiveConverter)


def _text_document(pages: dict[int, str], items=()):
    return _PageDocument(pages, items)


def _ocr_document(page_number: int, text: str) -> _PageDocument:
    return _PageDocument(
        {page_number: text},
        [
            _ProvItem(
                text,
                [
                    ProvenanceItem(
                        page_no=page_number,
                        bbox=BoundingBox(
                            l=5,
                            t=20,
                            r=90,
                            b=70,
                            coord_origin=CoordOrigin.TOPLEFT,
                        ),
                        charspan=(0, len(text)),
                    )
                ],
            )
        ],
    )


def test_extraction_config_is_canonical_and_validated() -> None:
    config = extractor.ExtractionConfig(
        min_meaningful_chars=12,
        min_meaningful_words=3,
        significant_raster_area=0.2,
        ocr_pages=(3, 1, 3),
    )

    assert config.ocr_pages == (1, 3)
    assert extractor.config_fingerprint(config) == extractor.config_fingerprint(
        extractor.ExtractionConfig(
            min_meaningful_chars=12,
            min_meaningful_words=3,
            significant_raster_area=0.2,
            ocr_pages=(1, 3),
        )
    )
    with pytest.raises(ValueError):
        extractor.ExtractionConfig(min_meaningful_chars=float("nan"))
    with pytest.raises(ValueError):
        extractor.ExtractionConfig(significant_raster_area=1.1)
    with pytest.raises(ValueError):
        extractor.ExtractionConfig(ocr_pages=(0,))
    with pytest.raises(ValueError):
        extractor.ExtractionConfig(ocr_languages="en-US")  # type: ignore[arg-type]


def test_text_only_never_constructs_ocr_converter(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "text.pdf"
    pdf_path.write_bytes(b"%PDF")
    document = _text_document({1: "достаточно осмысленного текста на странице"})
    _patch_adaptive_converter(monkeypatch, {(False, None): document})

    result = extractor.extract_pdf(pdf_path)

    assert [page.route for page in result.pages] == ["docling_text"]
    assert len(_AdaptiveConverter.init_options) == 1
    assert isinstance(_AdaptiveConverter.init_options[0], ThreadedPdfPipelineOptions)
    assert _AdaptiveConverter.init_options[0].do_ocr is False
    assert all(options.do_ocr is False for options in _AdaptiveConverter.init_options)


def test_scan_ocr_is_selective_and_preserves_source_page(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "", 2: ""}, [_picture(2)])
    ocr = _text_document({2: "OCR page two"}, [
        _ProvItem("OCR page two", [ProvenanceItem(
            page_no=2,
            bbox=BoundingBox(l=0, t=60, r=80, b=80),
            charspan=(0, 12),
        )]),
    ])
    _patch_adaptive_converter(monkeypatch, {
        (False, None): first,
        (True, (2, 2)): ocr,
    })

    result = extractor.extract_pdf(pdf_path)

    assert [(page.number, page.route, page.status) for page in result.pages] == [
        (1, "docling_text", "empty"),
        (2, "docling_ocr", "ok"),
    ]
    assert result.pages[1].markdown == "OCR page two"
    assert result.pages[1].regions[0].page_number == 2
    assert _AdaptiveConverter.convert_calls == [(False, None), (True, (2, 2))]
    assert len(_AdaptiveConverter.init_options) == 2
    assert _AdaptiveConverter.init_options[1].do_table_structure is False
    assert isinstance(_AdaptiveConverter.init_options[1].ocr_options, OcrMacOptions)
    assert _AdaptiveConverter.init_options[1].ocr_options.mode.value == "full_page"
    assert _AdaptiveConverter.init_options[1].ocr_options.lang == ["ru-RU", "en-US"]


def test_alternating_routes_keep_order_and_replace_pages_atomically(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "digital page one", 2: "", 3: "digital page three", 4: ""}, [_picture(2), _picture(4)])
    ocr_pages = {
        2: _ocr_document(2, "scan page two"),
        4: _ocr_document(4, "scan page four"),
    }
    _patch_adaptive_converter(monkeypatch, {
        (False, None): first,
        (True, (2, 2)): ocr_pages[2],
        (True, (4, 4)): ocr_pages[4],
    })

    result = extractor.extract_pdf(pdf_path)

    assert [page.number for page in result.pages] == [1, 2, 3, 4]
    assert [page.route for page in result.pages] == [
        "docling_text", "docling_ocr", "docling_text", "docling_ocr"
    ]
    assert result.markdown == "digital page one\n\nscan page two\n\ndigital page three\n\nscan page four"
    assert "digital page two" not in result.markdown
    assert _AdaptiveConverter.convert_calls == [
        (False, None), (True, (2, 2)), (True, (4, 4))
    ]


def test_same_page_raster_skips_auto_ocr_but_override_forces_it(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "same-page.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "digital text survives raster"}, [_picture(1)])
    ocr = _ocr_document(1, "manual OCR text")

    _patch_adaptive_converter(monkeypatch, {(False, None): first})
    result = extractor.extract_pdf(pdf_path)
    assert result.pages[0].route == "docling_text"
    assert "manual override" in result.pages[0].reason
    assert _AdaptiveConverter.convert_calls == [(False, None)]

    _patch_adaptive_converter(monkeypatch, {
        (False, None): first,
        (True, (1, 1)): ocr,
    })
    overridden = extractor.extract_pdf(
        pdf_path,
        config=extractor.ExtractionConfig(ocr_pages=(1,)),
    )
    assert overridden.pages[0].route == "docling_ocr"
    assert overridden.pages[0].markdown == "manual OCR text"
    assert _AdaptiveConverter.convert_calls == [(False, None), (True, (1, 1))]


def test_embedded_pdf_raster_signal_routes_scans_without_docling_picture_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_synthetic_corpus(tmp_path)
    pdf_path = tmp_path / "synthetic-mixed-tuning.pdf"
    first = _text_document(
        {
            1: "",
            2: "",
            3: "Digital layer: control token ALPHA.",
            4: "",
            5: "",
        }
    )
    _patch_adaptive_converter(
        monkeypatch,
        {
            (False, None): first,
            (True, (1, 1)): _ocr_document(1, "SCANNED RIVER FACT"),
            (True, (2, 2)): _ocr_document(2, "NOISY OCR FACT"),
            (True, (4, 4)): _ocr_document(4, "SCANNED SECOND PAGE"),
        },
    )

    result = extractor.extract_pdf(pdf_path)

    assert [page.route for page in result.pages] == [
        "docling_ocr",
        "docling_ocr",
        "docling_text",
        "docling_ocr",
        "docling_text",
    ]
    assert "manual override" in result.pages[2].reason
    assert result.pages[4].status == "empty"
    assert _AdaptiveConverter.convert_calls == [
        (False, None),
        (True, (1, 1)),
        (True, (2, 2)),
        (True, (4, 4)),
    ]


def test_ocr_text_without_source_regions_fails_closed(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "no-ocr-provenance.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: ""}, [_picture(1)])
    _patch_adaptive_converter(
        monkeypatch,
        {
            (False, None): first,
            (True, (1, 1)): _text_document({1: "OCR without provenance"}),
        },
    )

    result = extractor.extract_pdf(pdf_path)

    assert result.pages[0].route == "docling_ocr"
    assert result.pages[0].status == "failed"
    assert result.pages[0].markdown == ""
    assert result.pages[0].diagnostic is not None
    assert "provenance" in result.pages[0].diagnostic


def test_ocr_converter_initialization_failure_returns_partial_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "ocr-init-failure.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "usable text", 2: ""}, [_picture(1), _picture(2)])

    class InitFailingConverter(_AdaptiveConverter):
        def __init__(self, *, format_options=None) -> None:
            assert format_options is not None
            options = format_options[InputFormat.PDF].pipeline_options
            if options.do_ocr:
                raise RuntimeError("OCR backend unavailable")
            super().__init__(format_options=format_options)

    _AdaptiveConverter.documents = {(False, None): first}
    _AdaptiveConverter.init_options = []
    _AdaptiveConverter.convert_calls = []
    monkeypatch.setattr(extractor, "DocumentConverter", InitFailingConverter)

    result = extractor.extract_pdf(pdf_path)

    assert result.pages[0].route == "docling_text"
    assert result.pages[0].status == "ok"
    assert result.pages[0].markdown == "usable text"
    assert result.pages[1].route == "docling_ocr"
    assert result.pages[1].status == "failed"
    assert len(result.warnings) == 2


def test_manual_ocr_override_of_blank_page_stays_empty(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "blank-override.pdf"
    pdf_path.write_bytes(b"%PDF")
    _patch_adaptive_converter(
        monkeypatch,
        {
            (False, None): _text_document({1: ""}),
            (True, (1, 1)): _text_document({1: ""}),
        },
    )

    result = extractor.extract_pdf(
        pdf_path,
        config=extractor.ExtractionConfig(ocr_pages=(1,)),
    )

    assert result.pages[0].route == "docling_ocr"
    assert result.pages[0].status == "empty"
    assert result.pages[0].diagnostic is None


def test_empty_manual_ocr_result_preserves_usable_first_pass_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "text-override.pdf"
    pdf_path.write_bytes(b"%PDF")
    original = "usable digital text remains authoritative"
    _patch_adaptive_converter(
        monkeypatch,
        {
            (False, None): _text_document({1: original}),
            (True, (1, 1)): _text_document({1: ""}),
        },
    )

    result = extractor.extract_pdf(
        pdf_path,
        config=extractor.ExtractionConfig(ocr_pages=(1,)),
    )

    page = result.pages[0]
    assert page.route == "docling_text"
    assert page.status == "ok"
    assert page.markdown == original
    assert page.diagnostic == "OCR page has no usable text"


def test_invalid_override_is_rejected_before_selective_call(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "invalid-override.pdf"
    pdf_path.write_bytes(b"%PDF")
    _patch_adaptive_converter(
        monkeypatch,
        {(False, None): _text_document({1: "one page of text"})},
    )

    with pytest.raises(ValueError, match="отсутствующие страницы"):
        extractor.extract_pdf(pdf_path, config=extractor.ExtractionConfig(ocr_pages=(2,)))

    assert _AdaptiveConverter.convert_calls == [(False, None)]


def test_sparse_and_blank_pages_without_raster_stay_text_route(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "sparse.pdf"
    pdf_path.write_bytes(b"%PDF")
    _patch_adaptive_converter(monkeypatch, {
        (False, None): _text_document({1: "", 2: "one"}),
    })

    result = extractor.extract_pdf(pdf_path)

    assert [(page.route, page.status) for page in result.pages] == [
        ("docling_text", "empty"),
        ("docling_text", "ok"),
    ]
    assert _AdaptiveConverter.convert_calls == [(False, None)]


def test_ocr_failure_preserves_usable_first_pass_text(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "ocr-failure-with-text.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "short text"}, [_picture(1)])

    class FailingConverter(_AdaptiveConverter):
        def convert(self, source, page_range=None):
            if page_range == (1, 1):
                raise RuntimeError("OCR unavailable")
            return super().convert(source, page_range)

    _AdaptiveConverter.documents = {(False, None): first}
    _AdaptiveConverter.init_options = []
    _AdaptiveConverter.convert_calls = []
    monkeypatch.setattr(extractor, "DocumentConverter", FailingConverter)

    result = extractor.extract_pdf(pdf_path)

    page = result.pages[0]
    assert page.route == "docling_text"
    assert page.status == "ok"
    assert page.markdown == "short text"
    assert page.diagnostic == "OCR unavailable"
    assert result.warnings


def test_blank_failed_ocr_and_usable_text_are_distinct(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "partial.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "usable digital text", 2: "", 3: ""}, [_picture(2), _picture(3)])

    class FailingConverter(_AdaptiveConverter):
        def convert(self, source, page_range=None):
            if page_range == (2, 2):
                raise RuntimeError("OCR unavailable")
            return super().convert(source, page_range)

    _AdaptiveConverter.documents = {(False, None): first}
    _AdaptiveConverter.init_options = []
    _AdaptiveConverter.convert_calls = []
    monkeypatch.setattr(extractor, "DocumentConverter", FailingConverter)

    result = extractor.extract_pdf(pdf_path)

    assert [(page.number, page.status) for page in result.pages] == [
        (1, "ok"), (2, "failed"), (3, "failed")
    ]
    assert result.pages[1].markdown == ""
    assert result.pages[2].markdown == ""
    assert result.pages[0].markdown == "usable digital text"
    assert result.warnings


def test_ocr_identity_mismatch_closes_selected_page(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "mismatch.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "", 2: ""}, [_picture(2)])
    wrong = _PageDocument({9: "wrong page"}, page_keys=(9,))
    _patch_adaptive_converter(monkeypatch, {
        (False, None): first,
        (True, (2, 2)): wrong,
    })

    result = extractor.extract_pdf(pdf_path)

    page = result.pages[1]
    assert page.route == "docling_ocr"
    assert page.status == "failed"
    assert page.markdown == ""
    assert page.diagnostic


def test_ocr_provenance_mismatch_closes_page(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "provenance-mismatch.pdf"
    pdf_path.write_bytes(b"%PDF")
    first = _text_document({1: "", 2: ""}, [_picture(2)])
    wrong_provenance = _PageDocument({2: "OCR text"}, [
        _ProvItem("OCR text", [ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=10, r=10, b=20),
            charspan=(0, 8),
        )]),
    ])
    _patch_adaptive_converter(monkeypatch, {
        (False, None): first,
        (True, (2, 2)): wrong_provenance,
    })

    result = extractor.extract_pdf(pdf_path)

    page = result.pages[1]
    assert page.number == 2
    assert page.status == "failed"
    assert page.markdown == ""
    assert page.diagnostic is not None
    assert "provenance" in page.diagnostic
