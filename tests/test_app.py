from pathlib import Path

from streamlit.testing.v1 import AppTest


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
