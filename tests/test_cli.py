import pytest

from pdf_document_agent import cli
from pdf_document_agent.answering import GroundedAnswer
from pdf_document_agent.extractor import ExtractedDocument, ExtractedPage
from pdf_document_agent.retrieval import TextChunk


@pytest.fixture
def document() -> ExtractedDocument:
    return ExtractedDocument(
        source_name="book.pdf",
        markdown="# Книга\n\nТекст",
        page_count=1,
        pages=(ExtractedPage(number=1, markdown="# Книга\n\nТекст"),),
    )


def test_parser_accepts_single_question_and_model() -> None:
    args = cli.build_parser().parse_args(
        ["book.pdf", "--ask", "О чём книга?", "--model", "qwen3:14b"]
    )

    assert args.pdf_path == "book.pdf"
    assert args.ask == "О чём книга?"
    assert args.model == "qwen3:14b"
    assert args.interactive is False


def test_parser_rejects_two_question_modes() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["book.pdf", "--ask", "Вопрос", "--interactive"]
        )


def test_main_prints_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    document: ExtractedDocument,
) -> None:
    monkeypatch.setattr(cli, "extract_pdf", lambda _path: document)
    monkeypatch.setattr(cli, "chunk_document", lambda _document: [object()])
    monkeypatch.setattr(
        cli,
        "answer_question",
        lambda _question, _chunks, *, chat: GroundedAnswer(
            text="Ответ по книге [стр. 1].",
            source_pages=(1,),
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["pdf-document-agent", "book.pdf", "--ask", "О чём книга?"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Ответ по книге [стр. 1]." in output
    assert "Страницы контекста: 1" in output


def test_main_without_question_preserves_markdown_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    document: ExtractedDocument,
) -> None:
    monkeypatch.setattr(cli, "extract_pdf", lambda _path: document)
    monkeypatch.setattr("sys.argv", ["pdf-document-agent", "book.pdf"])

    cli.main()

    output = capsys.readouterr().out
    assert "Документ: book.pdf" in output
    assert "Страниц: 1" in output
    assert "# Книга" in output


def test_interactive_loop_passes_previous_questions_to_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = iter(
        (
            "Что такое энтропия программного обеспечения?",
            "Как с ней бороться?",
            "exit",
        )
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def fake_answer_question(
        question,
        _chunks,
        *,
        chat,
        previous_questions=(),
    ) -> GroundedAnswer:
        calls.append((question, tuple(previous_questions)))
        return GroundedAnswer(text="Ответ [стр. 1].", source_pages=(1,))

    monkeypatch.setattr(cli, "answer_question", fake_answer_question)

    cli._interactive_loop(
        "book.pdf",
        1,
        [TextChunk(index=0, page_number=1, text="Software entropy")],
        lambda _system, _user: "unused",
    )

    assert calls == [
        ("Что такое энтропия программного обеспечения?", ()),
        (
            "Как с ней бороться?",
            ("Что такое энтропия программного обеспечения?",),
        ),
    ]
