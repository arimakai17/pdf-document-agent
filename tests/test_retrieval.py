import pytest

from pdf_document_agent.extractor import ExtractedDocument, ExtractedPage, TextRegion
from pdf_document_agent.retrieval import TextChunk, chunk_document, search_chunks


def make_document(*pages: str) -> ExtractedDocument:
    return ExtractedDocument(
        source_name="book.pdf",
        markdown="\n\n".join(pages),
        page_count=len(pages),
        pages=tuple(
            ExtractedPage(number=number, markdown=text)
            for number, text in enumerate(pages, start=1)
        ),
    )


def test_chunk_document_preserves_source_page() -> None:
    document = make_document("Альфа " * 80, "Бета " * 80)

    chunks = chunk_document(document, max_chars=120, overlap_chars=20)

    assert len(chunks) > 2
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)


def test_search_chunks_ranks_relevant_russian_text_first() -> None:
    document = make_document(
        "Париж является столицей Франции и расположен на реке Сене.",
        "Язык Python создал Гвидо ван Россум. Первая версия вышла в 1991 году.",
    )
    chunks = chunk_document(document)

    results = search_chunks("Кто создал язык Python?", chunks, top_k=2)

    assert results
    assert results[0].chunk.page_number == 2
    assert "Гвидо ван Россум" in results[0].chunk.text
    assert results[0].score > 0


def test_search_chunks_returns_empty_when_document_has_no_query_terms() -> None:
    chunks = chunk_document(make_document("Кошка спит на подоконнике."))

    assert search_chunks("Как устроен ядерный реактор?", chunks) == []


def test_search_chunks_rejects_empty_question() -> None:
    chunks = chunk_document(make_document("Текст документа"))

    with pytest.raises(ValueError, match="Вопрос не должен быть пустым"):
        search_chunks("   ", chunks)


def test_search_chunks_rejects_chunk_with_single_shared_term() -> None:
    chunks = chunk_document(make_document("Python — это простое слово из шести букв."))

    assert search_chunks("Кто создал язык Python?", chunks) == []


def test_search_chunks_keeps_chunk_with_sufficient_coverage() -> None:
    chunks = chunk_document(make_document("Python создал Гвидо ван Россум."))

    results = search_chunks("Кто создал Python?", chunks, top_k=1)

    assert results
    assert results[0].chunk.page_number == 1


def _document_with_regions(*pages: tuple[str, tuple[TextRegion, ...]]) -> ExtractedDocument:
    return ExtractedDocument(
        source_name="book.pdf",
        markdown="\n\n".join(text for text, _ in pages),
        page_count=len(pages),
        pages=tuple(
            ExtractedPage(number=number, markdown=text, regions=regions)
            for number, (text, regions) in enumerate(pages, start=1)
        ),
    )


def test_chunk_document_attaches_boxes_from_same_page_regions() -> None:
    document = _document_with_regions(
        (
            "Париж является столицей Франции.",
            (
                TextRegion(
                    page_number=1,
                    text="Париж является столицей Франции.",
                    box=(0.1, 0.1, 0.5, 0.2),
                ),
            ),
        ),
        (
            "Язык Python создал Гвидо ван Россум.",
            (
                TextRegion(
                    page_number=2,
                    text="Язык Python создал Гвидо ван Россум.",
                    box=(0.2, 0.2, 0.6, 0.3),
                ),
            ),
        ),
    )

    chunks = chunk_document(document)

    page1 = [chunk for chunk in chunks if chunk.page_number == 1]
    page2 = [chunk for chunk in chunks if chunk.page_number == 2]
    assert page1[0].boxes == ((0.1, 0.1, 0.5, 0.2),)
    assert page2[0].boxes == ((0.2, 0.2, 0.6, 0.3),)


def test_chunk_without_lexical_overlap_has_no_boxes() -> None:
    document = _document_with_regions(
        (
            "Совершенно другой текст про квантовую физику.",
            (
                TextRegion(
                    page_number=1,
                    text="Несвязанный регион",
                    box=(0.1, 0.1, 0.2, 0.2),
                ),
            ),
        ),
    )

    chunks = chunk_document(document)

    assert chunks[0].boxes == ()


def test_text_chunk_default_constructor_has_empty_boxes() -> None:
    chunk = TextChunk(index=0, page_number=1, text="текст")

    assert chunk.boxes == ()