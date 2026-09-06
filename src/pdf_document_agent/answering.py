import re
from collections.abc import Callable
from dataclasses import dataclass

from pdf_document_agent.retrieval import (
    SearchResult,
    TextChunk,
    _tokenize,
    search_chunks,
)


INSUFFICIENT_ANSWER = "В документе недостаточно информации для ответа."
_CITATION_PATTERN = re.compile(r"\[стр\.\s*(\d+)\]")
# Минимальная доля содержательных слов ответа, которые должны встречаться
# дословно в процитированном фрагменте. Это лексическая проверка опоры
# (не семантическая гарантия истинности): корректный свободный перефраз
# может быть отклонён, а совпадение половины слов не доказывает факт.
MIN_CLAIM_SUPPORT = 0.5
_SYSTEM_PROMPT = f"""Ты отвечаешь на вопросы только по предоставленным отрывкам PDF-документа.
Не используй внешние знания и не додумывай отсутствующие факты.
После каждого существенного утверждения указывай страницу в формате [стр. N].
Если отрывков недостаточно, ответь точно: {INSUFFICIENT_ANSWER}"""


class AnswerGenerationError(RuntimeError):
    """Модель недоступна или не вернула пригодный ответ."""


@dataclass(frozen=True)
class GroundedAnswer:
    """Ответ модели и страницы контекста, на которых он основан."""

    text: str
    source_pages: tuple[int, ...]


def answer_question(
    question: str,
    chunks: list[TextChunk],
    *,
    chat: Callable[[str, str], str],
    top_k: int = 5,
) -> GroundedAnswer:
    """Найти контекст и получить ответ, ограниченный содержимым документа."""
    results = search_chunks(question, chunks, top_k=top_k)
    if not results:
        return GroundedAnswer(text=INSUFFICIENT_ANSWER, source_pages=())

    user_prompt = _build_user_prompt(question, results)
    text = chat(_SYSTEM_PROMPT, user_prompt).strip()
    if not text:
        raise AnswerGenerationError("Модель вернула пустой ответ.")

    if text == INSUFFICIENT_ANSWER:
        return GroundedAnswer(text=text, source_pages=())

    context_pages = tuple(
        dict.fromkeys(result.chunk.page_number for result in results)
    )
    cited_pages = tuple(
        dict.fromkeys(int(page) for page in _CITATION_PATTERN.findall(text))
    )
    if not cited_pages:
        raise AnswerGenerationError("Модель не указала страницу источника.")
    if any(page not in context_pages for page in cited_pages):
        raise AnswerGenerationError(
            "Модель сослалась на несуществующую страницу контекста."
        )

    cited_chunks = [
        result.chunk for result in results if result.chunk.page_number in cited_pages
    ]
    if not _claim_supported(text, cited_chunks):
        raise AnswerGenerationError(
            "Ответ имеет недостаточную лексическую опору в процитированном фрагменте."
        )

    return GroundedAnswer(text=text, source_pages=cited_pages)


def _claim_supported(text: str, cited_chunks: list[TextChunk]) -> bool:
    """Проверить, что содержательные слова ответа в основном присутствуют
    дословно в процитированных фрагментах (только cited pages, не весь контекст)."""
    answer_tokens = set(_tokenize(_CITATION_PATTERN.sub("", text)))
    if not answer_tokens:
        return False
    source_tokens: set[str] = set()
    for chunk in cited_chunks:
        source_tokens.update(_tokenize(chunk.text))
    return len(answer_tokens & source_tokens) / len(answer_tokens) >= MIN_CLAIM_SUPPORT


def _build_user_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[Страница {result.chunk.page_number}]\n{result.chunk.text}"
        for result in results
    )
    return f"""Отрывки документа:

{context}

Вопрос пользователя: {question}

Дай краткий осмысленный ответ на языке вопроса."""