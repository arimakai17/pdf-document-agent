from collections.abc import Callable

import pytest

from pdf_document_agent.answering import (
    INSUFFICIENT_ANSWER,
    AnswerGenerationError,
    answer_question,
)
from pdf_document_agent.extractor import ExtractedDocument, ExtractedPage
from pdf_document_agent.retrieval import chunk_document


def make_chunks(*pages: str):
    document = ExtractedDocument(
        source_name="book.pdf",
        markdown="\n\n".join(pages),
        page_count=len(pages),
        pages=tuple(
            ExtractedPage(number=number, markdown=text)
            for number, text in enumerate(pages, start=1)
        ),
    )
    return chunk_document(document)


def test_answer_question_passes_retrieved_page_to_model() -> None:
    chunks = make_chunks(
        "Париж является столицей Франции.",
        "Python создал Гвидо ван Россум в конце 1980-х годов.",
    )
    captured: dict[str, str] = {}

    def fake_chat(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "Python создал Гвидо ван Россум [стр. 2]."

    answer = answer_question(
        "Кто создал Python?",
        chunks,
        chat=fake_chat,
        top_k=1,
    )

    assert answer.text == "Python создал Гвидо ван Россум [стр. 2]."
    assert answer.source_pages == (2,)
    assert "[Страница 2]" in captured["user"]
    assert "только" in captured["system"].lower()


def test_answer_question_does_not_call_model_without_relevant_context() -> None:
    chunks = make_chunks("Кошка спит на подоконнике.")
    called = False

    def fake_chat(_: str, __: str) -> str:
        nonlocal called
        called = True
        return "Этот ответ не должен появиться"

    answer = answer_question("Как устроен ядерный реактор?", chunks, chat=fake_chat)

    assert answer.text == INSUFFICIENT_ANSWER
    assert answer.source_pages == ()
    assert called is False


def test_answer_question_rejects_page_not_present_in_context() -> None:
    chunks = make_chunks("Python создал Гвидо ван Россум.")

    with pytest.raises(AnswerGenerationError, match="несуществующую страницу"):
        answer_question(
            "Кто создал Python?",
            chunks,
            chat=lambda _system, _user: "Его создал Гвидо [стр. 99].",
        )


def test_answer_question_rejects_answer_without_page_citation() -> None:
    chunks = make_chunks("Python создал Гвидо ван Россум.")

    with pytest.raises(AnswerGenerationError, match="не указала страницу"):
        answer_question(
            "Кто создал Python?",
            chunks,
            chat=lambda _system, _user: "Его создал Гвидо ван Россум.",
        )


@pytest.mark.parametrize("model_output", ["", "   "])
def test_answer_question_rejects_empty_model_output(model_output: str) -> None:
    chunks = make_chunks("Python создал Гвидо ван Россум.")
    chat: Callable[[str, str], str] = lambda _system, _user: model_output

    with pytest.raises(AnswerGenerationError, match="пустой ответ"):
        answer_question("Кто создал Python?", chunks, chat=chat)