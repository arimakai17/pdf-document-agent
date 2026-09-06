from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "pdf_document_agent"
    / "app.py"
)


def test_app_starts_and_shows_pdf_uploader() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Спроси свой PDF"
    assert len(app.get("file_uploader")) == 1
    assert "Загрузи PDF" in app.info[0].value
