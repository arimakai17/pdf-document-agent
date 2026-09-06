import re
from collections.abc import Callable
from dataclasses import dataclass

from pdf_document_agent.retrieval import SearchResult, TextChunk, search_chunks


INSUFFICIENT_ANSWER = "В документе недостаточно информации для ответа."
_CITATION_PATTERN = re.compile(r"\[стр\.\s*(\d+)\]")
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

    return GroundedAnswer(text=text, source_pages=cited_pages)


def _build_user_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[Страница {result.chunk.page_number}]\n{result.chunk.text}"
        for result in results
    )
    return f"""Отрывки документа:

{context}

Вопрос пользователя: {question}

Дай краткий осмысленный ответ на языке вопроса."""