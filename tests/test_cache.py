import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from pdf_document_agent import app, cache
from pdf_document_agent.extractor import (
    DEFAULT_EXTRACTION_CONFIG,
    ExtractedDocument,
    ExtractedPage,
    TextRegion,
    ExtractionConfig,
    config_fingerprint,
)
from pdf_document_agent.retrieval import TextChunk

_FINGERPRINT_A = "a" * 64
_FINGERPRINT_B = "b" * 64


def _make_document(
    markdown: str = "hello",
    *,
    fingerprint: str = config_fingerprint(DEFAULT_EXTRACTION_CONFIG),
) -> ExtractedDocument:
    return ExtractedDocument(
        source_name="doc.pdf",
        markdown=markdown,
        page_count=1,
        pages=(
            ExtractedPage(
                number=1,
                markdown=markdown,
                route="docling_ocr",
                status="ok",
                reason="selective OCR",
                diagnostic="test diagnostic",
                regions=(
                    TextRegion(
                        page_number=1,
                        text=markdown,
                        box=(0.1, 0.2, 0.3, 0.4),
                    ),
                ),
            ),
        ),
        config_fingerprint=fingerprint,
        warnings=("warning-a",),
    )


def _make_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            index=0,
            page_number=1,
            text="hello",
            boxes=((0.1, 0.2, 0.3, 0.4),),
            regions=(
                TextRegion(page_number=1, text="hello", box=(0.1, 0.2, 0.3, 0.4)),
            ),
        )
    ]


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    return tmp_path


def _install_fake_extraction(monkeypatch, doc=None, chunks=None):
    doc = doc or _make_document()
    chunks = chunks or _make_chunks()
    calls = {"extract": 0, "chunk": 0, "configs": []}

    def fake_extract(path, *, config):
        calls["extract"] += 1
        calls["configs"].append(config)
        fingerprint = config_fingerprint(config)
        if doc.config_fingerprint == fingerprint:
            return doc
        return replace(doc, config_fingerprint=fingerprint)

    def fake_chunk(document):
        calls["chunk"] += 1
        return chunks

    monkeypatch.setattr(app, "extract_pdf", fake_extract)
    monkeypatch.setattr(app, "chunk_document", fake_chunk)
    return calls, doc, chunks


# --- Pure cache module tests -------------------------------------------------


def test_roundtrip_preserves_dataclasses(cache_dir: Path) -> None:
    doc = _make_document()
    chunks = _make_chunks()
    cache.put(b"%PDF-1.4 fake", doc, chunks)

    result = cache.get(b"%PDF-1.4 fake")

    assert result is not None
    restored_doc, restored_chunks = result
    assert restored_doc == doc
    assert restored_chunks == chunks


def test_get_miss_returns_none(cache_dir: Path) -> None:
    assert cache.get(b"never-seen") is None


def test_hit_updates_mtime(cache_dir: Path) -> None:
    cache.put(b"content", _make_document(), _make_chunks())
    path = cache.cache_path(b"content")
    os.utime(path, (100.0, 100.0))

    assert cache.get(b"content") is not None

    assert path.stat().st_mtime > 100.0


def test_lru_evicts_oldest_after_11_entries(cache_dir: Path) -> None:
    doc = _make_document()
    chunks = _make_chunks()
    for i in range(11):
        content = f"pdf-{i}".encode()
        cache.put(content, doc, chunks)
        os.utime(cache.cache_path(content), (1000 + i, 1000 + i))

    assert len(list(cache.cache_dir().glob("*.json"))) == 10
    assert not cache.cache_path(b"pdf-0").exists()


def test_clear_removes_all_entries(cache_dir: Path) -> None:
    doc = _make_document()
    chunks = _make_chunks()
    for i in range(3):
        cache.put(f"pdf-{i}".encode(), doc, chunks)
    assert len(list(cache.cache_dir().glob("*.json"))) == 3

    cache.clear()

    assert list(cache.cache_dir().glob("*.json")) == []


def test_clear_on_missing_dir_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path / "nope"))

    cache.clear()  # must not raise


def test_get_deletes_corrupt_entry(cache_dir: Path) -> None:
    path = cache.cache_path(b"corrupt")
    path.write_text("not json at all", encoding="utf-8")

    assert cache.get(b"corrupt") is None
    assert not path.exists()


def test_get_ignores_unknown_schema_version(cache_dir: Path) -> None:
    path = cache.cache_path(b"v2")
    path.write_text(json.dumps({"schema_version": cache.SCHEMA_VERSION + 1}))

    assert cache.get(b"v2") is None
    assert not path.exists()


def test_get_ignores_structurally_invalid_entry(cache_dir: Path) -> None:
    path = cache.cache_path(b"bad-types")
    path.write_text(
        json.dumps(
            {
                "schema_version": cache.SCHEMA_VERSION,
                "document": {
                    "source_name": 1,
                    "markdown": "x",
                    "page_count": "1",
                    "pages": [],
                },
                "chunks": [],
            }
        ),
        encoding="utf-8",
    )

    assert cache.get(b"bad-types") is None
    assert not path.exists()


def test_cache_roundtrip_preserves_wave2_fields(cache_dir: Path) -> None:
    document = _make_document(fingerprint=_FINGERPRINT_A)
    cache.put(b"wave2", document, _make_chunks(), config_fingerprint=_FINGERPRINT_A)

    restored = cache.get(b"wave2", config_fingerprint=_FINGERPRINT_A)

    assert restored == (document, _make_chunks())
    assert restored[0].pages[0].route == "docling_ocr"
    assert restored[0].pages[0].diagnostic == "test diagnostic"
    assert restored[0].warnings == ("warning-a",)


def test_cache_schema_and_config_identity_are_distinct(cache_dir: Path) -> None:
    document = _make_document(fingerprint=_FINGERPRINT_A)
    cache.put(b"same-pdf", document, _make_chunks(), config_fingerprint=_FINGERPRINT_A)

    assert cache.get(b"same-pdf", config_fingerprint=_FINGERPRINT_B) is None
    assert cache.cache_path(b"same-pdf", config_fingerprint=_FINGERPRINT_A) != cache.cache_path(
        b"same-pdf", config_fingerprint=_FINGERPRINT_B
    )
    assert cache.cache_path(b"same-pdf", config_fingerprint=_FINGERPRINT_A).exists()

    old_path = cache.cache_path(b"old-schema", config_fingerprint=_FINGERPRINT_A)
    old_path.write_text(
        json.dumps({
            "schema_version": cache.SCHEMA_VERSION - 1,
            "document": {},
            "chunks": [],
        }),
        encoding="utf-8",
    )
    assert cache.get(b"old-schema", config_fingerprint=_FINGERPRINT_A) is None
    assert not old_path.exists()


def test_cache_rejects_invalid_or_mismatched_fingerprint(cache_dir: Path) -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        cache.cache_path(b"pdf", config_fingerprint="../escape")

    document = _make_document(fingerprint=_FINGERPRINT_A)
    with pytest.raises(ValueError, match="fingerprint"):
        cache.put(
            b"pdf",
            document,
            _make_chunks(),
            config_fingerprint=_FINGERPRINT_B,
        )


# --- prepare_pdf integration tests -----------------------------------------


def test_prepare_pdf_returns_extraction_when_cache_write_fails(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, doc, chunks = _install_fake_extraction(monkeypatch)
    monkeypatch.setattr(
        cache,
        "put",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = app.prepare_pdf(b"%PDF-1.4 cache failure", "alpha.pdf")

    assert calls["extract"] == 1
    assert calls["chunk"] == 1
    assert result == (doc, chunks)


def test_prepare_pdf_miss_then_hit_without_extractor(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, doc, chunks = _install_fake_extraction(monkeypatch)
    content = b"%PDF-1.4 fake content"

    first = app.prepare_pdf(content, "alpha.pdf")
    assert calls["extract"] == 1
    assert calls["chunk"] == 1
    assert first == (doc, chunks)

    second = app.prepare_pdf(content, "renamed.pdf")
    assert calls["extract"] == 1
    assert calls["chunk"] == 1
    assert second == (doc, chunks)


def test_prepare_pdf_cache_identity_includes_config(cache_dir: Path, monkeypatch) -> None:
    calls, doc, chunks = _install_fake_extraction(monkeypatch)
    content = b"%PDF-1.4 config-specific"
    override_config = ExtractionConfig(ocr_pages=(1,))

    app.prepare_pdf(content, "alpha.pdf", config=override_config)
    app.prepare_pdf(content, "renamed.pdf", config=override_config)
    app.prepare_pdf(content, "default.pdf", config=DEFAULT_EXTRACTION_CONFIG)

    assert calls["extract"] == 2
    assert calls["configs"] == [override_config, DEFAULT_EXTRACTION_CONFIG]
    assert cache.cache_path(
        content, config_fingerprint=config_fingerprint(override_config)
    ).exists()
    assert cache.cache_path(
        content, config_fingerprint=config_fingerprint(DEFAULT_EXTRACTION_CONFIG)
    ).exists()


def test_prepare_pdf_rejects_empty_before_cache(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _, _ = _install_fake_extraction(monkeypatch)

    with pytest.raises(ValueError):
        app.prepare_pdf(b"", "x.pdf")

    assert calls["extract"] == 0


def test_prepare_pdf_rejects_non_pdf_before_cache(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _, _ = _install_fake_extraction(monkeypatch)

    with pytest.raises(ValueError):
        app.prepare_pdf(b"%PDF", "x.txt")

    assert calls["extract"] == 0


def test_prepare_pdf_never_persists_raw_pdf(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_extraction(monkeypatch)
    content = b"%PDF-1.4 SECRET RAW BYTES"

    app.prepare_pdf(content, "secret.pdf")

    files = list(cache.cache_dir().glob("*.json"))
    assert files
    for file_path in files:
        raw = file_path.read_text(encoding="utf-8")
        assert "%PDF" not in raw
        assert "SECRET RAW BYTES" not in raw


def test_prepare_pdf_recovers_from_corrupt_cache_entry(
    cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, doc, chunks = _install_fake_extraction(monkeypatch)
    content = b"%PDF-1.4 corrupt-entry"
    path = cache.cache_path(content)
    path.write_text("garbage", encoding="utf-8")

    result = app.prepare_pdf(content, "doc.pdf")

    assert calls["extract"] == 1
    assert result == (doc, chunks)
    assert cache.get(content) is not None
