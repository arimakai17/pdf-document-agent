import hashlib
import tempfile
from functools import partial
from pathlib import Path

import streamlit as st

from pdf_document_agent.answering import AnswerGenerationError, answer_question
from pdf_document_agent.extractor import (
    ExtractedDocument,
    NormalizedBox,
    PdfExtractionError,
    extract_pdf,
)
from pdf_document_agent.ollama import DEFAULT_MODEL, OllamaError, chat_with_ollama
from pdf_document_agent.retrieval import TextChunk, chunk_document
from pdf_document_agent.viewer import ViewerError, render_page

MAX_FILE_BYTES = 50 * 1024 * 1024
HISTORY_LIMIT_OPTIONS = (5, 10, 25, 50)


@st.cache_data(show_spinner=False)
def prepare_pdf(
    file_bytes: bytes,
    file_name: str,
) -> tuple[ExtractedDocument, list[TextChunk]]:
    """Process an uploaded PDF once and remove the temporary original."""
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


def _select_source(
    page: int,
    boxes: tuple[NormalizedBox, ...],
) -> None:
    st.session_state.active_page = page
    st.session_state.active_boxes = boxes


def _select_history_item(
    message_index: int,
    page_number: int | None,
    boxes: tuple[NormalizedBox, ...],
) -> None:
    st.session_state.selected_history_index = message_index
    if page_number is not None:
        _select_source(page_number, boxes)


def _show_current_dialog() -> None:
    st.session_state.selected_history_index = None


def _trim_history(messages: list[dict], max_questions: int) -> bool:
    """Оставить последние max_questions пар диалога в памяти сессии."""
    question_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if len(question_indexes) <= max_questions:
        return False
    del messages[: question_indexes[-max_questions]]
    return True


def run_app() -> None:
    st.set_page_config(
        page_title="PDF Document Agent",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
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
        history_limit = st.selectbox(
            "Хранить вопросов в истории",
            options=HISTORY_LIMIT_OPTIONS,
            index=1,
        )
        st.caption("PDF обрабатывается локально и не отправляется в облако.")
        st.caption("История хранится только до закрытия текущей сессии.")

    uploaded_file = st.file_uploader(
        "PDF-документ",
        type="pdf",
        accept_multiple_files=False,
        help="Поддерживаются text-based PDF и сканы до 50 МБ.",
    )

    document_panel, conversation_panel = st.columns(2)
    with document_panel:
        st.subheader("Документ")
    with conversation_panel:
        st.subheader("Диалог")

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
        st.session_state.update(
            file_id=file_id,
            file_bytes=file_bytes,
            document=document,
            chunks=chunks,
            messages=[],
            active_page=1,
            active_boxes=(),
            selected_history_index=None,
        )

    document = st.session_state.document
    chunks = st.session_state.chunks
    st.success(
        f"Готово: {uploaded_file.name} · {document.page_count} стр. · "
        f"{len(chunks)} фрагм."
    )

    with document_panel:
        page_number = min(
            max(st.session_state.get("active_page", 1), 1),
            document.page_count,
        )
        try:
            image = render_page(
                st.session_state.file_bytes,
                page_number,
                boxes=st.session_state.get("active_boxes", ()),
                scale=1.5,
            )
        except ViewerError as error:
            st.error(str(error))
            image = None
        if image is not None:
            st.image(image, width="stretch")

        st.caption(f"Страница {page_number}/{document.page_count}")
        previous, next_page = st.columns(2)
        previous.button(
            "Предыдущая",
            disabled=page_number <= 1,
            key="viewer_prev",
            on_click=_select_source,
            args=(page_number - 1, ()),
        )
        next_page.button(
            "Следующая",
            disabled=page_number >= document.page_count,
            key="viewer_next",
            on_click=_select_source,
            args=(page_number + 1, ()),
        )
        if st.session_state.get("active_boxes"):
            st.caption(
                "Выделены области источника; это не точная подсветка слов."
            )

    with conversation_panel:
        messages = st.session_state.messages
        if _trim_history(messages, history_limit):
            st.session_state.selected_history_index = None

        question_indexes = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "user"
        ]
        with st.sidebar:
            st.subheader("История вопросов")
            st.caption("Только текущая сессия и текущий PDF")
            selected_index = st.session_state.get("selected_history_index")
            if selected_index is not None:
                st.button(
                    "Вернуться к текущему диалогу",
                    key="history_current",
                    on_click=_show_current_dialog,
                )
            if not question_indexes:
                st.caption("История пока пуста")
            for message_index in reversed(question_indexes):
                question_text = messages[message_index]["content"]
                label = (
                    question_text
                    if len(question_text) <= 48
                    else f"{question_text[:45]}…"
                )
                answer_message = (
                    messages[message_index + 1]
                    if message_index + 1 < len(messages)
                    and messages[message_index + 1]["role"] == "assistant"
                    else None
                )
                sources = answer_message.get("sources", ()) if answer_message else ()
                first_source = sources[0] if sources else None
                st.button(
                    label,
                    key=f"history_{message_index}",
                    on_click=_select_history_item,
                    args=(
                        message_index,
                        first_source.page_number if first_source else None,
                        first_source.boxes if first_source else (),
                    ),
                )

        selected_index = st.session_state.get("selected_history_index")
        visible_messages = list(enumerate(messages))
        if (
            isinstance(selected_index, int)
            and 0 <= selected_index < len(messages)
            and messages[selected_index]["role"] == "user"
        ):
            end = selected_index + 1
            if end < len(messages) and messages[end]["role"] == "assistant":
                end += 1
            visible_messages = list(enumerate(messages[selected_index:end], selected_index))

        for message_index, message in visible_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] != "assistant":
                    continue
                for source_index, source in enumerate(message.get("sources", ())):
                    st.button(
                        f"Открыть стр. {source.page_number}",
                        key=f"source_{message_index}_{source_index}",
                        on_click=_select_source,
                        args=(source.page_number, source.boxes),
                    )

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

        previous_questions = tuple(
            message["content"]
            for message in messages
            if message["role"] == "user"
        )
        st.session_state.selected_history_index = None
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        try:
            answer = answer_question(
                question,
                chunks,
                chat=partial(
                    chat_with_ollama,
                    model=model.strip() or DEFAULT_MODEL,
                ),
                previous_questions=previous_questions,
            )
        except (ValueError, OllamaError, AnswerGenerationError) as error:
            st.error(str(error))
            return

        messages.append(
            {
                "role": "assistant",
                "content": answer.text,
                "sources": answer.sources,
            }
        )
        _trim_history(messages, history_limit)
        if answer.sources:
            first_source = answer.sources[0]
            _select_source(first_source.page_number, first_source.boxes)
        st.rerun()


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --agent-bg: #0A0A0B;
            --agent-accent: #FF4D00;
        }
        .stApp { background: var(--agent-bg); }
        .stButton > button, .stFormSubmitButton > button {
            border-color: var(--agent-accent);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


run_app()
