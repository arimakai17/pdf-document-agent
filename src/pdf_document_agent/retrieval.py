import math
import re
from collections import Counter
from dataclasses import dataclass

from pdf_document_agent.extractor import ExtractedDocument


_WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "в",
    "во",
    "вот",
    "где",
    "да",
    "для",
    "до",
    "его",
    "ее",
    "её",
    "если",
    "есть",
    "и",
    "из",
    "или",
    "их",
    "как",
    "какая",
    "какие",
    "какой",
    "когда",
    "кто",
    "ли",
    "на",
    "над",
    "не",
    "но",
    "о",
    "об",
    "он",
    "она",
    "они",
    "от",
    "по",
    "под",
    "при",
    "про",
    "с",
    "со",
    "так",
    "такое",
    "такой",
    "то",
    "у",
    "чем",
    "что",
    "это",
    "the",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class TextChunk:
    """Ограниченный фрагмент текста, связанный с исходной страницей."""

    index: int
    page_number: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """Фрагмент и его лексическая релевантность вопросу."""

    chunk: TextChunk
    score: float


# Минимальная доля содержательных терминов вопроса, которую фрагмент должен
# покрыть, чтобы считаться релевантным. Это лексический coverage-гейт (не
# семантическая гарантия): он отсекает фрагменты, совпавшие по одному общему
# слову из трёх, но пропускает однотерминовые вопросы (1/1).
MIN_QUERY_COVERAGE = 0.5


def chunk_document(
    document: ExtractedDocument,
    *,
    max_chars: int = 1_800,
    overlap_chars: int = 250,
) -> list[TextChunk]:
    """Разбить страницы на ограниченные фрагменты, не теряя provenance."""
    if max_chars <= 0:
        raise ValueError("Размер фрагмента должен быть положительным.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Перекрытие должно быть от 0 до размера фрагмента.")

    chunks: list[TextChunk] = []
    for page in document.pages:
        for text in _split_text(page.markdown, max_chars, overlap_chars):
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    page_number=page.number,
                    text=text,
                )
            )
    return chunks


def search_chunks(
    question: str,
    chunks: list[TextChunk],
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """Найти top-k фрагментов с помощью BM25-подобного ранжирования."""
    if not question.strip():
        raise ValueError("Вопрос не должен быть пустым.")
    if top_k <= 0:
        raise ValueError("Количество результатов должно быть положительным.")
    if not chunks:
        return []

    query_terms = set(_tokenize(question))
    if not query_terms:
        return []

    tokenized_chunks = [_tokenize(chunk.text) for chunk in chunks]
    average_length = sum(map(len, tokenized_chunks)) / len(tokenized_chunks)
    if average_length == 0:
        return []

    document_frequency = {
        term: sum(term in tokens for tokens in tokenized_chunks)
        for term in query_terms
    }
    scored: list[SearchResult] = []

    for chunk, tokens in zip(chunks, tokenized_chunks, strict=True):
        matched_terms = query_terms & set(tokens)
        if len(matched_terms) / len(query_terms) < MIN_QUERY_COVERAGE:
            continue
        frequencies = Counter(tokens)
        score = _bm25_score(
            query_terms,
            frequencies,
            document_frequency,
            document_count=len(chunks),
            document_length=len(tokens),
            average_length=average_length,
        )
        if score > 0:
            scored.append(SearchResult(chunk=chunk, score=score))

    scored.sort(key=lambda result: (-result.score, result.chunk.index))
    return scored[:top_k]


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _WORD_PATTERN.findall(text.casefold())
        if len(token) > 1 and token not in _STOP_WORDS
    ]


def _bm25_score(
    query_terms: set[str],
    frequencies: Counter[str],
    document_frequency: dict[str, int],
    *,
    document_count: int,
    document_length: int,
    average_length: float,
) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0

    for term in query_terms:
        frequency = frequencies[term]
        if frequency == 0:
            continue
        frequency_in_documents = document_frequency[term]
        inverse_document_frequency = math.log(
            1
            + (document_count - frequency_in_documents + 0.5)
            / (frequency_in_documents + 0.5)
        )
        denominator = frequency + k1 * (
            1 - b + b * document_length / average_length
        )
        score += inverse_document_frequency * frequency * (k1 + 1) / denominator

    return score


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            minimum_boundary = start + max_chars // 2
            paragraph_boundary = text.rfind("\n\n", minimum_boundary, end)
            word_boundary = text.rfind(" ", minimum_boundary, end)
            boundary = max(paragraph_boundary, word_boundary)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap_chars)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        start = next_start

    return pieces