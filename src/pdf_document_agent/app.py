import hashlib
import tempfile
from functools import partial
from pathlib import Path

import streamlit as st

from pdf_document_agent.answering import AnswerGenerationError, answer_question
from pdf_document_agent.extractor import ExtractedDocument, PdfExtractionError, extract_pdf
from pdf_document_agent.ollama import DEFAULT_MODEL, OllamaError, chat_with_ollama
from pdf_document_agent.retrieval import TextChunk, chunk_document
from pdf_document_agent.viewer import ViewerError, render_page

MAX_FILE_BYTES = 50 * 1024 * 1024

@st.cache_data(show_spinner=False)
def prepare_pdf(file_bytes: bytes, file_name: str) -> tuple[ExtractedDocument, list[TextChunk]]:
    if not file_bytes: raise ValueError("Загруженный PDF пуст.")
    if len(file_bytes) > MAX_FILE_BYTES: raise ValueError("PDF превышает ограничение 50 МБ.")
    safe_name = Path(file_name).name
    if Path(safe_name).suffix.lower() != ".pdf": raise ValueError("Требуется файл формата PDF.")
    with tempfile.TemporaryDirectory(prefix="pdf-agent-") as directory:
        path = Path(directory) / safe_name
        path.write_bytes(file_bytes)
        document = extract_pdf(path)
    chunks = chunk_document(document)
    if not chunks: raise PdfExtractionError("В PDF нет текста, пригодного для поиска.")
    return document, chunks

def _select_source(page: int, boxes: tuple) -> None:
    st.session_state.active_page = page
    st.session_state.active_boxes = boxes

def run_app() -> None:
    st.set_page_config(page_title="PDF Document Agent", page_icon="📄", layout="wide")
    _apply_styles()
    st.caption("LOCAL DOCUMENT INTELLIGENCE · OLLAMA + DOCLING")
    st.title("Спроси свой PDF")
    st.write("Загрузи книгу, статью или скан. Агент отвечает только по документу и указывает страницы источников.")
    with st.expander("Настройки модели"):
        model = st.text_input("Модель Ollama", value=DEFAULT_MODEL)
        st.caption("PDF обрабатывается локально и не отправляется в облако.")
    uploaded_file = st.file_uploader("PDF-документ", type="pdf", accept_multiple_files=False, help="Поддерживаются text-based PDF и сканы до 50 МБ.")
    left, right = st.columns(2)
    with left:
        st.subheader("Документ")
    with right:
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
            st.error(str(error)); return
        st.session_state.update(file_id=file_id, file_bytes=file_bytes, document=document, chunks=chunks, messages=[], active_page=1, active_boxes=())
    document, chunks = st.session_state.document, st.session_state.chunks
    st.success(f"Готово: {uploaded_file.name} · {document.page_count} стр. · {len(chunks)} фрагм.")
    with left:
        page_number = min(max(st.session_state.get("active_page", 1), 1), document.page_count)
        try: image = render_page(st.session_state.file_bytes, page_number, boxes=st.session_state.get("active_boxes", ()), scale=1.5)
        except ViewerError as error: st.error(str(error)); image = None
        if image is not None: st.image(image, use_container_width=True)
        st.caption(f"Страница {page_number}/{document.page_count}")
        previous, next_page = st.columns(2)
        previous.button("Предыдущая", disabled=page_number <= 1, key="viewer_prev", on_click=_select_source, args=(page_number - 1, ()))
        next_page.button("Следующая", disabled=page_number >= document.page_count, key="viewer_next", on_click=_select_source, args=(page_number + 1, ()))
        if st.session_state.get("active_boxes"): st.caption("Выделены области источника; это не точная подсветка слов.")
    with right:
        messages = st.session_state.messages
        for index, message in enumerate(messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    for source_index, source in enumerate(message.get("sources", ())):
                        st.caption(f"Страница {source.page_number}")
                        st.caption(source.excerpt[:300])
                        st.button(f"Открыть источник · Страница {source.page_number}", key=f"source_{index}_{source_index}", on_click=_select_source, args=(source.page_number, source.boxes))
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input("Вопрос по документу", placeholder="Например: в чём основная идея второй главы?")
            submitted = st.form_submit_button("Получить ответ", type="primary")
        if submitted:
            if not question.strip(): st.warning("Сначала введи вопрос."); return
            messages.append({"role":"user", "content":question})
            try:
                answer = answer_question(question, chunks, chat=partial(chat_with_ollama, model=model.strip() or DEFAULT_MODEL))
            except (ValueError, OllamaError, AnswerGenerationError) as error: st.error(str(error)); return
            messages.append({"role":"assistant", "content":answer.text, "sources":answer.sources})
            if answer.sources: _select_source(answer.sources[0].page_number, answer.sources[0].boxes)
            st.rerun()

def _apply_styles() -> None:
    st.markdown("<style>:root{--agent-bg:#0A0A0B;--agent-accent:#FF4D00}.stApp{background:var(--agent-bg)}.stButton>button,.stFormSubmitButton>button{border-color:var(--agent-accent)}</style>", unsafe_allow_html=True)

run_app()
