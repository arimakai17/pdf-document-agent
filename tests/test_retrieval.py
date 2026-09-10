import pytest

from pdf_document_agent.extractor import ExtractedDocument, ExtractedPage, TextRegion
from pdf_document_agent.retrieval import (
    SearchResult,
    TextChunk,
    chunk_document,
    document_vocabulary,
    search_chunks,
)


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


def test_search_chunks_highlights_only_regions_matching_query_terms() -> None:
    header_box = (0.1, 0.1, 0.9, 0.2)
    target_box = (0.1, 0.4, 0.9, 0.5)
    footer_box = (0.1, 0.8, 0.9, 0.9)
    document = _document_with_regions(
        (
            "Введение в информационный поиск.\n\n"
            "Главное правило RAG: Garbage In, Garbage Out.\n\n"
            "Конец лекции и вопросы для повторения.",
            (
                TextRegion(
                    page_number=1,
                    text="Введение в информационный поиск.",
                    box=header_box,
                ),
                TextRegion(
                    page_number=1,
                    text="Главное правило RAG: Garbage In, Garbage Out.",
                    box=target_box,
                ),
                TextRegion(
                    page_number=1,
                    text="Конец лекции и вопросы для повторения.",
                    box=footer_box,
                ),
            ),
        ),
    )
    chunks = chunk_document(document)

    results = search_chunks("Garbage In Garbage Out", chunks, top_k=1)

    assert chunks[0].boxes == (header_box, target_box, footer_box)
    assert results[0].highlight_boxes == (target_box,)


def test_search_chunks_falls_back_to_manual_chunk_boxes_without_regions() -> None:
    box = (0.1, 0.2, 0.5, 0.3)
    chunk = TextChunk(
        index=0,
        page_number=1,
        text="Python создал Гвидо ван Россум.",
        boxes=(box,),
    )

    results = search_chunks("Кто создал Python?", [chunk], top_k=1)

    assert results[0].highlight_boxes == (box,)


def test_search_chunks_does_not_fall_back_when_regions_do_not_match_query() -> None:
    broad_box = (0.1, 0.1, 0.9, 0.9)
    chunk = TextChunk(
        index=0,
        page_number=1,
        text="Garbage Out находится только в markdown фрагмента.",
        boxes=(broad_box,),
        regions=(
            TextRegion(
                page_number=1,
                text="Несвязанный распознанный регион.",
                box=broad_box,
            ),
        ),
    )

    results = search_chunks("Garbage Out", [chunk], top_k=1)

    assert results[0].highlight_boxes == ()


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
    assert chunk.regions == ()


def test_search_result_default_constructor_has_empty_highlight_boxes() -> None:
    chunk = TextChunk(index=0, page_number=1, text="текст")

    assert SearchResult(chunk=chunk, score=1.0).highlight_boxes == ()


def test_document_vocabulary_contains_only_document_terms() -> None:
    chunks = chunk_document(
        make_document("Software entropy is a measure of disorder in a system.")
    )

    vocabulary = document_vocabulary(chunks)

    assert "software" in vocabulary
    assert "entropy" in vocabulary
    assert "measure" in vocabulary
    # Чуждые языку документа термины не должны появляться в словаре.
    assert "энтропия" not in vocabulary
    assert "программного" not in vocabulary


def test_document_vocabulary_includes_terms_after_first_100k_characters() -> None:
    chunks = chunk_document(
        make_document("обычный текст " * 8_000, "latechapterterm появляется здесь.")
    )
    assert "latechapterterm" in document_vocabulary(chunks)
