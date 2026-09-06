import hashlib
import tempfile
from functools import partial
from pathlib import Path

import streamlit as st

from pdf_document_agent.answering import AnswerGenerationError, answer_question
from pdf_document_agent.extractor import (
    ExtractedDocument,
    PdfExtractionError,
    extract_pdf,
)
from pdf_document_agent.ollama import DEFAULT_MODEL, OllamaError, chat_with_ollama
from pdf_document_agent.retrieval import TextChunk, chunk_document


MAX_FILE_BYTES = 50 * 1024 * 1024


@st.cache_data(show_spinner=False)
def prepare_pdf(
    file_bytes: bytes,
    file_name: str,
) -> tuple[ExtractedDocument, list[TextChunk]]:
    """Обработать загруженный PDF один раз и удалить временный оригинал."""
    if not file_bytes:
        raise ValueError("Загруженный PDF пуст.")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise ValueError("PDF превышает ограничение 50 МБ.")

    safe_name = Path(file_name).name
    if Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError("Требуется файл формата PDF.")

    with tempfile.TemporaryDirectory(prefix="pdf-agent-") as directory:
        temporary_pdf = Path(directory) / safe_name
        temporary_pdf.write_bytes(file_bytes)
        document = extract_pdf(temporary_pdf)

    chunks = chunk_document(document)
    if not chunks:
        raise PdfExtractionError("В PDF нет текста, пригодного для поиска.")
    return document, chunks


def run_app() -> None:
    st.set_page_config(
        page_title="PDF Document Agent",
        page_icon="📄",
        layout="centered",
    )
    _apply_styles()

    st.caption("LOCAL DOCUMENT INTELLIGENCE · OLLAMA + DOCLING")
    st.title("Спроси свой PDF")
    st.write(
        "Загрузи книгу, статью или скан. Агент отвечает только по документу "
        "и указывает страницы источников."
    )

    with st.expander("Настройки модели"):
        model = st.text_input("Модель Ollama", value=DEFAULT_MODEL)
        st.caption("PDF обрабатывается локально и не отправляется в облако.")

    uploaded_file = st.file_uploader(
        "PDF-документ",
        type="pdf",
        accept_multiple_files=False,
        help="Поддерживаются text-based PDF и сканы до 50 МБ.",
    )
    if uploaded_file is None:
        st.info("Загрузи PDF, чтобы начать диалог.")
        return

    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("file_id") != file_id:
        try:
            with st.spinner("Извлекаю текст, структуру и страницы…"):
                document, chunks = prepare_pdf(file_bytes, uploaded_file.name)
        except (FileNotFoundError, ValueError, PdfExtractionError) as error:
            st.error(str(error))
            return

        st.session_state.file_id = file_id
        st.session_state.document = document
        st.session_state.chunks = chunks
        st.session_state.messages = []

    document = st.session_state.document
    chunks = st.session_state.chunks
    messages = st.session_state.messages

    st.success(
        f"Готово: {uploaded_file.name} · {document.page_count} стр. · "
        f"{len(chunks)} фрагм."
    )

    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Вопрос по документу",
            placeholder="Например: в чём основная идея второй главы?",
        )
        submitted = st.form_submit_button("Получить ответ", type="primary")

    if not submitted:
        return
    if not question.strip():
        st.warning("Сначала введи вопрос.")
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    chat = partial(chat_with_ollama, model=model.strip() or DEFAULT_MODEL)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ в документе…"):
                answer = answer_question(question, chunks, chat=chat)
            st.write(answer.text)
            if answer.source_pages:
                pages = ", ".join(map(str, answer.source_pages))
                st.caption(f"Источники: стр. {pages}")
    except (ValueError, OllamaError, AnswerGenerationError) as error:
        st.error(str(error))
        return

    messages.append({"role": "assistant", "content": answer.text})


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --agent-bg: #0A0A0B;
            --agent-panel: #121214;
            --agent-border: #2A2A2E;
            --agent-accent: #FF4D00;
        }
        .stApp { background: var(--agent-bg); }
        [data-testid="stFileUploader"] {
            border: 1px solid var(--agent-border);
            border-radius: 8px;
            padding: 0.75rem;
        }
        .stButton > button, .stFormSubmitButton > button {
            border-color: var(--agent-accent);
        }
        code, [data-testid="stCaptionContainer"] {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


run_app()
