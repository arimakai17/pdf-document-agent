from typing import Final, Literal


Locale = Literal["RU", "EN"]

DEFAULT_LOCALE: Final[Locale] = "RU"
LOCALES: Final[tuple[Locale, ...]] = ("RU", "EN")
ANSWER_LANGUAGE: Final[dict[Locale, Literal["Russian", "English"]]] = {
    "RU": "Russian",
    "EN": "English",
}

_COPY: Final[dict[Locale, dict[str, str]]] = {
    "RU": {
        "language": "Язык интерфейса",
        "eyebrow": "ЛОКАЛЬНЫЙ АНАЛИЗ ДОКУМЕНТОВ · OLLAMA + DOCLING",
        "intro": (
            "Навигатор по документу с ответами из источника. Загрузи книгу, "
            "статью или скан — и сразу переходи к страницам, на которых основан ответ."
        ),
        "settings": "Настройки модели",
        "model": "Модель Ollama",
        "history_limit": "Хранить вопросов в истории",
        "local_processing": "PDF обрабатывается локально и не отправляется в облако.",
        "session_history": "История хранится только до закрытия текущей сессии.",
        "cache_description": "Извлечённые данные до 10 PDF хранятся локально в дисковом кэше.",
        "cache_clear": "Очистить кэш",
        "cache_cleared": "Кэш очищен.",
        "uploader": "PDF-документ",
        "uploader_help": "Поддерживаются текстовые PDF и сканы до 50 МБ.",
        "dropzone": "Перетащи PDF сюда или нажми, чтобы выбрать файл",
        "dropzone_limit": "До 50 МБ · PDF",
        "document": "Документ",
        "conversation": "Диалог",
        "upload_prompt": "Загрузи PDF, чтобы начать диалог.",
        "processing": "Извлекаю текст, структуру и страницы…",
        "pdf_error": "Не удалось обработать PDF. Проверь формат файла и ограничение 50 МБ.",
        "ready": "Готово: {name} · {pages} стр. · {chunks} фрагм.",
        "viewer_error": "Не удалось отобразить страницу PDF.",
        "page": "Страница {current}/{total}",
        "previous": "Предыдущая",
        "next": "Следующая",
        "highlight_note": "Выделены области источника; это не точная подсветка слов.",
        "history": "История вопросов",
        "history_scope": "Только текущая сессия и текущий PDF",
        "return_current": "Вернуться к текущему диалогу",
        "history_empty": "История пока пуста",
        "delete_exchange": "Удалить вопрос и ответ",
        "open_source": "Открыть стр. {page}",
        "question": "Вопрос по документу",
        "question_placeholder": "Например: в чём основная идея второй главы?",
        "submit": "Получить ответ",
        "empty_question": "Сначала введи вопрос.",
        "answer_error": "Не удалось получить подтверждённый документом ответ. Попробуй ещё раз.",
        "mascot_idle": "PDF Atlas: ожидание",
        "mascot_busy": "PDF Atlas: обработка",
    },
    "EN": {
        "language": "Interface language",
        "eyebrow": "LOCAL DOCUMENT INTELLIGENCE · OLLAMA + DOCLING",
        "intro": (
            "Navigate documents with source-grounded answers. Upload a book, article, "
            "or scan and jump straight to the pages supporting each answer."
        ),
        "settings": "Model settings",
        "model": "Ollama model",
        "history_limit": "Questions kept in history",
        "local_processing": "The PDF is processed locally and is never sent to the cloud.",
        "session_history": "History lasts only for the current session.",
        "cache_description": "Extracted data for up to 10 PDFs is kept in a local disk cache.",
        "cache_clear": "Clear cache",
        "cache_cleared": "Cache cleared.",
        "uploader": "PDF document",
        "uploader_help": "Text-based PDFs and scans up to 50 MB are supported.",
        "dropzone": "Drag a PDF here or click to choose a file",
        "dropzone_limit": "Up to 50 MB · PDF",
        "document": "Document",
        "conversation": "Conversation",
        "upload_prompt": "Upload a PDF to start a conversation.",
        "processing": "Extracting text, structure, and pages…",
        "pdf_error": "Could not process the PDF. Check its format and the 50 MB limit.",
        "ready": "Ready: {name} · {pages} pages · {chunks} chunks",
        "viewer_error": "Could not display this PDF page.",
        "page": "Page {current}/{total}",
        "previous": "Previous",
        "next": "Next",
        "highlight_note": "Source regions are highlighted; this is not exact word highlighting.",
        "history": "Question history",
        "history_scope": "Current session and current PDF only",
        "return_current": "Return to the current conversation",
        "history_empty": "No questions yet",
        "delete_exchange": "Delete question and answer",
        "open_source": "Open p. {page}",
        "question": "Question about the document",
        "question_placeholder": "For example: What is the main idea of chapter two?",
        "submit": "Get answer",
        "empty_question": "Enter a question first.",
        "answer_error": "Could not produce an answer grounded in the document. Try again.",
        "mascot_idle": "PDF Atlas: idle",
        "mascot_busy": "PDF Atlas: working",
    },
}

if _COPY["RU"].keys() != _COPY["EN"].keys():
    raise RuntimeError("RU and EN interface dictionaries must contain identical keys.")


def text(locale: Locale, key: str, **values: object) -> str:
    """Return localized interface copy and interpolate named values."""
    template = _COPY[locale][key]
    return template.format(**values) if values else template
