from pathlib import Path

from streamlit.testing.v1 import AppTest

from pdf_document_agent import cache
from pdf_document_agent.extractor import (
    ExtractedDocument,
    ExtractedPage,
    TextRegion,
)
from pdf_document_agent.retrieval import TextChunk


APP_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "pdf_document_agent"
    / "app.py"
)


def test_app_starts_in_russian_with_pdf_atlas_branding() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "PDF Atlas"
    assert app.button_group[0].value == "RU"
    assert len(app.get("file_uploader")) == 1
    assert app.selectbox[0].label == "Хранить вопросов в истории"
    assert app.selectbox[0].value == 10
    assert "Документ" in [item.value for item in app.subheader]
    assert "Диалог" in [item.value for item in app.subheader]
    assert "Загрузи PDF" in app.info[0].value


def test_language_switch_changes_entire_empty_state_ui_to_english() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    app.button_group[0].set_value("EN").run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "PDF Atlas"
    assert app.button_group[0].value == "EN"
    assert app.selectbox[0].label == "Questions kept in history"
    assert "Document" in [item.value for item in app.subheader]
    assert "Conversation" in [item.value for item in app.subheader]
    assert "Upload a PDF" in app.info[0].value


def test_cache_controls_localized_in_russian(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any("кэш" in caption for caption in captions)
    assert any(button.label == "Очистить кэш" for button in app.button)


def test_cache_controls_localized_in_english(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.button_group[0].set_value("EN").run(timeout=30)

    assert not app.exception
    captions = [item.value for item in app.caption]
    assert any("cache" in caption.lower() for caption in captions)
    assert any(button.label == "Clear cache" for button in app.button)


def test_clear_cache_button_removes_disk_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    (tmp_path / "deadbeef.json").write_text('{"schema_version": 1}', encoding="utf-8")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    clear_button = next(button for button in app.button if button.label == "Очистить кэш")
    clear_button.click().run(timeout=30)

    assert not app.exception
    assert list(tmp_path.glob("*.json")) == []
    assert any(
        "кэш" in item.value.lower() and "очищен" in item.value.lower()
        for item in app.success
    )


# --- collapsible upload panel -------------------------------------------------


def _seed_cached_pdf(file_bytes: bytes) -> None:
    """Prime the disk cache so a fake PDF hits the cache without real Docling."""
    region = TextRegion(page_number=1, text="Hello world", box=(0.0, 0.0, 1.0, 1.0))
    page = ExtractedPage(number=1, markdown="# Test\n\nHello world", regions=(region,))
    document = ExtractedDocument(
        source_name="sample.pdf",
        markdown="# Test\n\nHello world",
        page_count=1,
        pages=(page,),
    )
    chunk = TextChunk(
        index=0,
        page_number=1,
        text="Hello world",
        boxes=((0.0, 0.0, 1.0, 1.0),),
        regions=(region,),
    )
    cache.put(file_bytes, document, [chunk])


def _uploader_expander(app, label: str):
    return next(expander for expander in app.get("expander") if expander.label == label)


def test_uploader_expander_open_on_fresh_session() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    expander = _uploader_expander(app, "PDF-документ")
    assert expander.proto.expanded is True
    assert app.session_state["uploader_expander"] is True


def test_uploader_expander_collapses_after_successful_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)

    assert not app.exception
    expander = _uploader_expander(app, "PDF-документ")
    assert expander.proto.expanded is False
    assert app.session_state["uploader_expander"] is False


def test_uploader_expander_reopen_persists_across_reruns(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is False

    # AppTest cannot synthesize a frontend expander click; emulate the documented
    # keyed session-state transition the on_change callback would perform.
    app.session_state["uploader_expander"] = True
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is True

    # A plain rerun (answering, etc.) must not re-collapse the panel.
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is True


def test_failed_upload_keeps_expander_open(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(("empty.pdf", b"", "application/pdf"))
    app.run(timeout=30)

    assert not app.exception
    assert app.session_state["uploader_expander"] is True
    expander = _uploader_expander(app, "PDF-документ")
    assert expander.proto.expanded is True
    assert app.error


def test_language_switch_preserves_expander_state(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_DOCUMENT_AGENT_CACHE_DIR", str(tmp_path))
    file_bytes = b"%PDF-1.4\n1 0 obj\nfake\n"
    _seed_cached_pdf(file_bytes)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("file_uploader")[0].set_value(
        ("sample.pdf", file_bytes, "application/pdf")
    )
    app.run(timeout=30)
    assert app.session_state["uploader_expander"] is False

    app.button_group[0].set_value("EN").run(timeout=30)

    assert not app.exception
    assert app.session_state["uploader_expander"] is False
    expander = _uploader_expander(app, "PDF document")
    assert expander.proto.expanded is False


def test_no_duplicate_uploader_label_inside_expander() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert [e.label for e in app.get("expander")].count("PDF-документ") == 1
    uploader = app.get("file_uploader")[0]
    assert "COLLAPSED" in str(uploader.proto.label_visibility)
    assert uploader.help == "Поддерживаются текстовые PDF и сканы до 50 МБ."
    assert uploader.allowed_type == [".pdf"]
