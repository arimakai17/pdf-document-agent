import hashlib
import json
import tempfile
from functools import partial
from pathlib import Path

import streamlit as st

from pdf_document_agent import cache
from pdf_document_agent.answering import AnswerGenerationError, answer_question
from pdf_document_agent.extractor import (
    ExtractedDocument,
    NormalizedBox,
    PdfExtractionError,
    extract_pdf,
)
from pdf_document_agent.localization import (
    ANSWER_LANGUAGE,
    DEFAULT_LOCALE,
    LOCALES,
    Locale,
    text,
)
from pdf_document_agent.ollama import DEFAULT_MODEL, OllamaError, chat_with_ollama
from pdf_document_agent.retrieval import TextChunk, chunk_document
from pdf_document_agent.viewer import ViewerError, render_page

MAX_FILE_BYTES = 50 * 1024 * 1024
HISTORY_LIMIT_OPTIONS = (5, 10, 25, 50)

_CAT_IDLE_SVG = """
<svg class="mascot-cat mascot-cat-idle" viewBox="0 0 48 48" aria-hidden="true" focusable="false">
  <g fill="#FF4D00" stroke="#1B1D21" stroke-width="1.5" stroke-linejoin="round">
    <path class="mascot-tail" d="M31 41 C39 41 43 33 39 26 C37 22 33 23 34 28"
          fill="none" stroke="#FF4D00" stroke-width="4" stroke-linecap="round"/>
    <rect x="16" y="23" width="16" height="19" rx="8"/>
    <path d="M18 12 L16 5 L23 10 Z"/>
    <path d="M30 12 L32 5 L25 10 Z"/>
    <circle cx="24" cy="18" r="8"/>
  </g>
  <g fill="#1B1D21">
    <circle cx="21" cy="17.5" r="1.3"/>
    <circle cx="27" cy="17.5" r="1.3"/>
    <path d="M23.4 20.5 L24 21.4 L24.6 20.5 Z"/>
  </g>
  <g stroke="#1B1D21" stroke-width="1" fill="none" stroke-linecap="round">
    <line x1="14.5" y1="17" x2="19" y2="17.5"/>
    <line x1="14.5" y1="20.5" x2="19" y2="19.5"/>
    <line x1="33.5" y1="17" x2="29" y2="17.5"/>
    <line x1="33.5" y1="20.5" x2="29" y2="19.5"/>
  </g>
</svg>
"""

_CAT_BUSY_SVG = """
<svg class="mascot-cat mascot-cat-busy" viewBox="0 0 48 48" aria-hidden="true" focusable="false">
  <g fill="#FF4D00" stroke="#1B1D21" stroke-width="1.5" stroke-linejoin="round">
    <path class="mascot-tail" d="M11 30 C4 27 2 20 6 15 C8 12 12 14 10 18"
          fill="none" stroke="#FF4D00" stroke-width="4" stroke-linecap="round"/>
    <g class="mascot-leg-b">
      <rect x="14" y="33" width="3.5" height="11" rx="1.75"/>
      <rect x="20" y="33" width="3.5" height="11" rx="1.75"/>
    </g>
    <g class="mascot-leg-f">
      <rect x="31" y="33" width="3.5" height="11" rx="1.75"/>
      <rect x="37" y="33" width="3.5" height="11" rx="1.75"/>
    </g>
    <g class="mascot-body">
      <rect x="10" y="22" width="27" height="13" rx="6.5"/>
      <circle cx="38" cy="18" r="6"/>
      <path d="M35 13 L36 7 L40 11 Z"/>
      <path d="M41 12 L42 7 L39 10 Z"/>
    </g>
  </g>
  <g fill="#1B1D21">
    <circle cx="40" cy="17" r="1.2"/>
  </g>
  <g stroke="#1B1D21" stroke-width="1" fill="none" stroke-linecap="round">
    <line x1="41" y1="20" x2="44" y2="19"/>
    <line x1="41" y1="22" x2="44" y2="22"/>
  </g>
</svg>
"""


def _mascot_markup(locale: Locale) -> str:
    """Inline localized cat mascot; the idle/busy pose swap is driven by CSS."""
    idle_label = text(locale, "mascot_idle")
    busy_label = text(locale, "mascot_busy")
    return (
        '<div class="pdf-atlas-mascot">'
        f'<span class="mascot-idle" role="img" aria-label="{idle_label}" '
        f'title="{idle_label}">{_CAT_IDLE_SVG}</span>'
        f'<span class="mascot-busy" role="img" aria-label="{busy_label}" '
        f'title="{busy_label}">{_CAT_BUSY_SVG}</span>'
        "</div>"
    )


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

    cached = cache.get(file_bytes)
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory(prefix="pdf-agent-") as directory:
        temporary_pdf = Path(directory) / safe_name
        temporary_pdf.write_bytes(file_bytes)
        document = extract_pdf(temporary_pdf)

    chunks = chunk_document(document)
    if not chunks:
        raise PdfExtractionError("В PDF нет текста, пригодного для поиска.")
    try:
        cache.put(file_bytes, document, chunks)
    except OSError:
        pass
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
        page_title="PDF Atlas",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if st.session_state.get("ui_locale") not in LOCALES:
        st.session_state.ui_locale = DEFAULT_LOCALE

    brand_panel, language_panel = st.columns([8, 1], vertical_alignment="top")
    with language_panel:
        selected_locale = st.segmented_control(
            text(st.session_state.ui_locale, "language"),
            options=LOCALES,
            key="ui_locale",
            required=True,
            label_visibility="collapsed",
        )
    locale: Locale = (
        selected_locale if selected_locale in LOCALES else DEFAULT_LOCALE
    )
    _apply_styles(locale)
    st.markdown(_mascot_markup(locale), unsafe_allow_html=True)

    with brand_panel:
        st.caption(text(locale, "eyebrow"))
        st.title("PDF Atlas")
        st.write(text(locale, "intro"))

    with st.expander(text(locale, "settings")):
        model = st.text_input(
            text(locale, "model"),
            value=DEFAULT_MODEL,
            key="ollama_model",
        )
        history_limit = st.selectbox(
            text(locale, "history_limit"),
            options=HISTORY_LIMIT_OPTIONS,
            index=1,
            key="history_limit",
        )
        st.caption(text(locale, "local_processing"))
        st.caption(text(locale, "session_history"))
        st.caption(text(locale, "cache_description"))
        if st.button(text(locale, "cache_clear"), key="clear_cache_button"):
            cache.clear()
            st.success(text(locale, "cache_cleared"))

    if st.session_state.pop("uploader_collapse_pending", False):
        st.session_state["uploader_expander"] = False

    with st.expander(
        text(locale, "uploader"),
        expanded=st.session_state.get("uploader_expander", True),
        key="uploader_expander",
        on_change="rerun",
    ):
        uploaded_file = st.file_uploader(
            text(locale, "uploader"),
            type="pdf",
            accept_multiple_files=False,
            help=text(locale, "uploader_help"),
            key="pdf_uploader",
            label_visibility="collapsed",
        )

    document_panel, conversation_panel = st.columns(2)
    with document_panel:
        with st.container(key="document_heading"):
            st.subheader(text(locale, "document"))
    with conversation_panel:
        with st.container(key="conversation_heading"):
            st.subheader(text(locale, "conversation"))

    if uploaded_file is None:
        st.info(text(locale, "upload_prompt"))
        return

    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("file_id") != file_id:
        try:
            with st.spinner(text(locale, "processing")):
                document, chunks = prepare_pdf(file_bytes, uploaded_file.name)
        except (FileNotFoundError, ValueError, PdfExtractionError):
            st.error(text(locale, "pdf_error"))
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
        st.session_state["uploader_collapse_pending"] = True
        st.rerun()

    document = st.session_state.document
    chunks = st.session_state.chunks
    st.success(
        text(
            locale,
            "ready",
            name=uploaded_file.name,
            pages=document.page_count,
            chunks=len(chunks),
        )
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
        except ViewerError:
            st.error(text(locale, "viewer_error"))
            image = None
        if image is not None:
            st.image(image, width="stretch")

        st.caption(
            text(locale, "page", current=page_number, total=document.page_count)
        )
        previous, next_page = st.columns(2)
        previous.button(
            text(locale, "previous"),
            disabled=page_number <= 1,
            key="viewer_prev",
            on_click=_select_source,
            args=(page_number - 1, ()),
        )
        next_page.button(
            text(locale, "next"),
            disabled=page_number >= document.page_count,
            key="viewer_next",
            on_click=_select_source,
            args=(page_number + 1, ()),
        )
        if st.session_state.get("active_boxes"):
            st.caption(text(locale, "highlight_note"))

    with conversation_panel:
        messages = st.session_state.messages
        if _trim_history(messages, history_limit):
            st.session_state.selected_history_index = None

        question_indexes = [
            index
            for index, message in enumerate(messages)
            if message["role"] == "user"
        ]
        with st.popover(text(locale, "history"), icon=":material/history:"):
            st.caption(text(locale, "history_scope"))
            selected_index = st.session_state.get("selected_history_index")
            if selected_index is not None:
                st.button(
                    text(locale, "return_current"),
                    key="history_current",
                    on_click=_show_current_dialog,
                    width="stretch",
                )
            if not question_indexes:
                st.caption(text(locale, "history_empty"))
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
                    width="stretch",
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
                        text(locale, "open_source", page=source.page_number),
                        key=f"source_{message_index}_{source_index}",
                        on_click=_select_source,
                        args=(source.page_number, source.boxes),
                    )

        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                text(locale, "question"),
                placeholder=text(locale, "question_placeholder"),
                key="question",
            )
            submitted = st.form_submit_button(
                text(locale, "submit"),
                type="primary",
            )

        if not submitted:
            return
        if not question.strip():
            st.warning(text(locale, "empty_question"))
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
                answer_language=ANSWER_LANGUAGE[locale],
            )
        except (ValueError, OllamaError, AnswerGenerationError):
            st.error(text(locale, "answer_error"))
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


def _apply_styles(locale: Locale) -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --agent-bg: #1B1D21;
            --agent-surface: #23262A;
            --agent-surface-raised: #2A2D32;
            --agent-border: #3A3E45;
            --agent-border-strong: #4A4F58;
            --agent-text: #ECEEF1;
            --agent-muted: #AEB2BA;
            --agent-accent: #FF4D00;
            --agent-accent-hover: #E64500;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--agent-bg);
            color: var(--agent-text);
        }
        [data-testid="stHeader"] {
            background: var(--agent-bg);
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            background: var(--agent-surface);
            border-right: 1px solid var(--agent-border);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--agent-surface);
        }

        h1, h2, h3, h4, p, label,
        [data-testid="stMarkdownContainer"] {
            color: var(--agent-text);
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        small {
            color: var(--agent-muted) !important;
        }
        [data-testid="stCaptionContainer"] {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: 0.02em;
        }

        [data-testid="stExpander"],
        [data-testid="stForm"],
        [data-testid="stChatMessage"] {
            background: var(--agent-surface);
            border: 1px solid var(--agent-border);
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary:hover {
            color: var(--agent-accent);
        }
        [data-testid="stExpander"] summary svg {
            fill: var(--agent-muted);
        }
        [data-testid="stChatMessage"] {
            padding: 0.85rem 1rem;
        }

        .stTextInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {
            background: var(--agent-surface-raised) !important;
            border-color: var(--agent-border) !important;
            color: var(--agent-text) !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: var(--agent-muted) !important;
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        [data-baseweb="select"] > div:focus-within {
            border-color: var(--agent-accent) !important;
            box-shadow: 0 0 0 1px var(--agent-accent) !important;
        }
        [data-baseweb="popover"],
        [role="listbox"] {
            background: var(--agent-surface-raised) !important;
            color: var(--agent-text) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--agent-surface);
            border: 1px dashed var(--agent-border-strong);
            border-radius: 10px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--agent-accent);
        }

        .stButton > button,
        .stFormSubmitButton > button,
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--agent-surface-raised);
            border: 1px solid var(--agent-border-strong);
            color: var(--agent-text);
            border-radius: 8px;
            transition: background-color 120ms ease, border-color 120ms ease,
                        color 120ms ease;
        }
        .stButton > button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #32363C;
            border-color: var(--agent-accent);
            color: var(--agent-accent);
        }
        .stFormSubmitButton > button {
            background: var(--agent-accent);
            border-color: var(--agent-accent);
            color: #FFFFFF;
            font-weight: 650;
        }
        .stFormSubmitButton > button:hover {
            background: var(--agent-accent-hover);
            border-color: var(--agent-accent-hover);
            color: #FFFFFF;
        }
        .stButton > button:focus,
        .stFormSubmitButton > button:focus,
        [data-testid="stFileUploaderDropzone"] button:focus {
            box-shadow: 0 0 0 2px var(--agent-bg),
                        0 0 0 4px var(--agent-accent) !important;
        }
        button:disabled {
            background: var(--agent-surface) !important;
            border-color: var(--agent-border) !important;
            color: #747982 !important;
        }

        [data-testid="stAlert"] {
            background: var(--agent-surface);
            border: 1px solid var(--agent-border);
            border-left: 3px solid var(--agent-accent);
            color: var(--agent-text);
        }
        [data-testid="stAlertContainer"] {
            background: transparent !important;
            color: var(--agent-accent) !important;
        }
        [data-testid^="stAlertContent"] {
            color: var(--agent-accent) !important;
        }
        [data-testid="stImage"] img {
            border: 1px solid var(--agent-border);
            border-radius: 8px;
        }
        hr {
            border-color: var(--agent-border);
        }
        a {
            color: var(--agent-accent) !important;
        }
        ::selection {
            background: var(--agent-accent);
            color: #FFFFFF;
        }
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: var(--agent-bg);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--agent-border-strong);
            border: 2px solid var(--agent-bg);
            border-radius: 999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--agent-accent);
        }

        /* --- animated status mascot ----------------------------------- */
        .pdf-atlas-mascot {
            position: fixed;
            top: 0.15rem;
            right: 0.75rem;
            width: 44px;
            height: 44px;
            z-index: 1000010;
            pointer-events: none;
            line-height: 0;
        }
        .pdf-atlas-mascot svg {
            width: 44px;
            height: 44px;
            display: block;
            overflow: visible;
        }
        .mascot-idle { display: block; }
        .mascot-busy { display: none; }

        .mascot-idle .mascot-tail {
            transform-box: fill-box;
            transform-origin: 0% 50%;
            animation: mascot-tail-wag 6s ease-in-out infinite;
        }
        @keyframes mascot-tail-wag {
            0%, 88%, 100% { transform: rotate(0deg); }
            92% { transform: rotate(-16deg); }
            96% { transform: rotate(12deg); }
        }
        .mascot-busy .mascot-leg-f {
            transform-box: fill-box;
            transform-origin: 50% 0%;
            animation: mascot-walk-f 0.5s ease-in-out infinite;
        }
        .mascot-busy .mascot-leg-b {
            transform-box: fill-box;
            transform-origin: 50% 0%;
            animation: mascot-walk-b 0.5s ease-in-out infinite;
        }
        .mascot-busy .mascot-body {
            animation: mascot-body-bob 0.5s ease-in-out infinite;
        }
        .mascot-busy .mascot-tail {
            transform-box: fill-box;
            transform-origin: 100% 50%;
            animation: mascot-tail-sway 1s ease-in-out infinite;
        }
        @keyframes mascot-walk-f {
            0%, 100% { transform: rotate(10deg); }
            50% { transform: rotate(-10deg); }
        }
        @keyframes mascot-walk-b {
            0%, 100% { transform: rotate(-10deg); }
            50% { transform: rotate(10deg); }
        }
        @keyframes mascot-body-bob {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-1.5px); }
        }
        @keyframes mascot-tail-sway {
            0%, 100% { transform: rotate(8deg); }
            50% { transform: rotate(-8deg); }
        }

        @supports selector(body:has(*)) {
            body:has([data-testid="stStatusWidgetRunningIcon"]) .pdf-atlas-mascot { right: 4.5rem; }
            body:has([data-testid="stStatusWidgetRunningIcon"]) .mascot-idle { display: none; }
            body:has([data-testid="stStatusWidgetRunningIcon"]) .mascot-busy { display: block; }
            body:has([data-testid="stStatusWidgetRunningIcon"]) [data-testid="stStatusWidgetRunningIcon"] { visibility: hidden; }
        }

        @media (prefers-reduced-motion: reduce) {
            .mascot-idle .mascot-tail,
            .mascot-busy .mascot-leg-f,
            .mascot-busy .mascot-leg-b,
            .mascot-busy .mascot-body,
            .mascot-busy .mascot-tail {
                animation: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    dropzone_copy = json.dumps(text(locale, "dropzone"), ensure_ascii=False)
    dropzone_limit = json.dumps(text(locale, "dropzone_limit"), ensure_ascii=False)
    st.markdown(
        f"""
        <style>
        .st-key-ui_locale {{
            display: flex;
            justify-content: flex-end;
            padding-top: 0.15rem;
        }}
        .st-key-ui_locale [data-baseweb="button-group"] {{
            background: var(--agent-surface);
            border: 1px solid var(--agent-border);
            border-radius: 8px;
        }}
        .st-key-ui_locale button {{
            color: var(--agent-muted);
        }}
        .st-key-ui_locale button[aria-pressed="true"] {{
            background: var(--agent-surface-raised);
            color: var(--agent-accent);
        }}

        .st-key-document_heading h3,
        .st-key-conversation_heading h3 {{
            text-align: center;
        }}

        .st-key-pdf_uploader [data-testid="stFileUploaderDropzone"] {{
            position: relative;
            min-height: 7rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            cursor: pointer;
        }}
        .st-key-pdf_uploader [data-testid="stFileUploaderDropzone"] > * {{
            visibility: hidden;
            position: absolute;
        }}
        .st-key-pdf_uploader [data-testid="stFileUploaderDropzone"]::before {{
            content: {dropzone_copy};
            color: var(--agent-text);
            font-weight: 600;
            pointer-events: none;
        }}
        .st-key-pdf_uploader [data-testid="stFileUploaderDropzone"]::after {{
            content: {dropzone_limit};
            color: var(--agent-muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.78rem;
            pointer-events: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run_app()
