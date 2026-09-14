import pytest

from pdf_document_agent import cli
from pdf_document_agent.agent import (
    AgentCounters,
    AgentRun,
    AgentTrace,
    AgentTraceEvent,
)
from pdf_document_agent.answering import GroundedAnswer
from pdf_document_agent.extractor import ExtractionConfig, ExtractedDocument, ExtractedPage
from pdf_document_agent.retrieval import TextChunk


@pytest.fixture
def document() -> ExtractedDocument:
    return ExtractedDocument(
        source_name="book.pdf",
        markdown="# Книга\n\nТекст",
        page_count=1,
        pages=(ExtractedPage(number=1, markdown="# Книга\n\nТекст"),),
    )


def agent_run(status: str = "ok") -> AgentRun:
    trace = AgentTrace(
        events=(AgentTraceEvent("search", "ok", pages=(1,), truncated=True),),
        status=status,
        tool_calls=1,
        llm_calls=2,
        elapsed_s=0.25,
    )
    return AgentRun(
        answer=GroundedAnswer(text="Ответ агента [стр. 1].", source_pages=(1,)),
        final_evidence_packet=(),
        trace=trace,
        status=status,
        counters=AgentCounters(tool_calls=1, llm_calls=2),
    )


def test_parser_accepts_single_question_and_model() -> None:
    args = cli.build_parser().parse_args(
        ["book.pdf", "--ask", "О чём книга?", "--model", "qwen3:14b"]
    )

    assert args.pdf_path == "book.pdf"
    assert args.ask == "О чём книга?"
    assert args.model == "qwen3:14b"
    assert args.interactive is False
    assert args.mode == "fixed"
    assert args.ocr_pages == ""


def test_parser_accepts_agent_mode_and_ocr_pages() -> None:
    args = cli.build_parser().parse_args(
        ["book.pdf", "--mode", "agent", "--ocr-pages", "1, 3-5"]
    )

    assert args.mode == "agent"
    assert args.ocr_pages == "1, 3-5"


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
    configs: list[ExtractionConfig] = []

    def fake_extract(_path, *, config):
        configs.append(config)
        return document

    monkeypatch.setattr(cli, "extract_pdf", fake_extract)
    monkeypatch.setattr(cli, "chunk_document", lambda _document: [object()])
    monkeypatch.setattr(
        cli,
        "answer_question",
        lambda _question, _chunks, *, chat: GroundedAnswer(
            text="Ответ по книге [стр. 1].",
            source_pages=(1,),
        ),
    )

    def fail_agent(*_args, **_kwargs):
        raise AssertionError("fixed mode must not call agent")

    monkeypatch.setattr(cli, "run_agent", fail_agent)
    monkeypatch.setattr(
        "sys.argv",
        ["pdf-document-agent", "book.pdf", "--ask", "О чём книга?"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Ответ по книге [стр. 1]." in output
    assert "Страницы контекста: 1" in output
    assert configs == [ExtractionConfig()]


def test_main_passes_sorted_ocr_override_to_extractor(
    monkeypatch: pytest.MonkeyPatch,
    document: ExtractedDocument,
) -> None:
    configs: list[ExtractionConfig] = []

    def fake_extract(_path, *, config):
        configs.append(config)
        return document

    monkeypatch.setattr(cli, "extract_pdf", fake_extract)
    monkeypatch.setattr("sys.argv", [
        "pdf-document-agent",
        "book.pdf",
        "--ocr-pages",
        "5, 2-3, 3",
    ])

    cli.main()

    assert configs == [ExtractionConfig(ocr_pages=(2, 3, 5))]


def test_main_without_question_preserves_markdown_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    document: ExtractedDocument,
) -> None:
    monkeypatch.setattr(cli, "extract_pdf", lambda _path, *, config: document)
    monkeypatch.setattr("sys.argv", ["pdf-document-agent", "book.pdf"])

    cli.main()

    output = capsys.readouterr().out
    assert "Документ: book.pdf" in output
    assert "Страниц: 1" in output
    assert "# Книга" in output


def test_main_agent_passes_document_chunks_and_bounded_chat(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    document: ExtractedDocument,
) -> None:
    monkeypatch.setattr(cli, "extract_pdf", lambda _path, *, config: document)
    chunks = [TextChunk(index=0, page_number=1, text="evidence")]
    monkeypatch.setattr(cli, "chunk_document", lambda _document: chunks)
    calls = []

    def fake_run_agent(
        question,
        received_document,
        received_chunks,
        *,
        chat,
        previous_questions=(),
    ):
        calls.append((question, received_document, received_chunks, previous_questions))
        assert chat("system", "user", max_output_tokens=123, timeout=7.0) == "unused"
        return agent_run()

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        cli,
        "chat_with_ollama",
        lambda system, user, **kwargs: calls.append((system, user, kwargs)) or "unused",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "pdf-document-agent",
            "book.pdf",
            "--ask",
            "О чём книга?",
            "--mode",
            "agent",
            "--model",
            "test-model",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert calls[0][:3] == ("О чём книга?", document, chunks)
    assert calls[0][3] == ()
    assert calls[1] == (
        "system",
        "user",
        {"model": "test-model", "timeout": 7.0, "num_predict": 123, "num_ctx": 8192},
    )
    assert "Ответ агента [стр. 1]." in output
    assert "status=ok" in output
    assert "tool_calls=1" in output
    assert "search: ok" in output
    assert "pages=1" in output
    assert "truncated=yes" in output


def test_main_non_ok_agent_prints_status_without_fixed_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    document: ExtractedDocument,
) -> None:
    monkeypatch.setattr(cli, "extract_pdf", lambda _path, *, config: document)
    monkeypatch.setattr(cli, "chunk_document", lambda _document: [object()])
    monkeypatch.setattr(cli, "run_agent", lambda *args, **kwargs: agent_run("invalid_action"))
    fixed_calls = []
    monkeypatch.setattr(cli, "answer_question", lambda *args, **kwargs: fixed_calls.append(args))
    monkeypatch.setattr(
        "sys.argv",
        ["pdf-document-agent", "book.pdf", "--ask", "Вопрос", "--mode", "agent"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Ответ агента [стр. 1]." in output
    assert "status=invalid_action" in output
    assert fixed_calls == []


def test_main_prints_extraction_routes_warnings_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = ExtractedDocument(
        source_name="book.pdf",
        markdown="text",
        page_count=2,
        pages=(
            ExtractedPage(number=1, markdown="text", route="docling_text", status="ok"),
            ExtractedPage(
                number=2,
                markdown="",
                route="docling_ocr",
                status="failed",
                diagnostic="OCR unavailable",
            ),
        ),
        warnings=("Page 2: OCR unavailable",),
    )
    monkeypatch.setattr(cli, "extract_pdf", lambda _path, *, config: document)
    monkeypatch.setattr("sys.argv", ["pdf-document-agent", "book.pdf"])

    cli.main()

    output = capsys.readouterr().out
    assert "Page 1: route=docling_text, status=ok" in output
    assert "Page 2: route=docling_ocr, status=failed" in output
    assert "Warning: Page 2: OCR unavailable" in output
    assert "Diagnostic (page 2): OCR unavailable" in output


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


def test_interactive_loop_agent_receives_document_and_previous_questions(
    monkeypatch: pytest.MonkeyPatch,
    document: ExtractedDocument,
) -> None:
    inputs = iter(("Первый вопрос", "Второй вопрос", "exit"))
    calls = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def fake_run_agent(
        question,
        received_document,
        _chunks,
        *,
        chat,
        previous_questions=(),
    ):
        calls.append((question, received_document, tuple(previous_questions)))
        return agent_run()

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)

    cli._interactive_loop(
        "book.pdf",
        1,
        [TextChunk(index=0, page_number=1, text="evidence")],
        lambda _system, _user, **_kwargs: "unused",
        mode="agent",
        document=document,
    )

    assert calls == [
        ("Первый вопрос", document, ()),
        ("Второй вопрос", document, ("Первый вопрос",)),
    ]
