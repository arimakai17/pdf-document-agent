import argparse
import os
from collections.abc import Callable
from functools import partial

from pdf_document_agent.agent import AgentRun, run_agent
from pdf_document_agent.answering import (
    AnswerGenerationError,
    GroundedAnswer,
    answer_question,
)
from pdf_document_agent.extractor import (
    ExtractionConfig,
    ExtractedDocument,
    PdfExtractionError,
    extract_pdf,
    parse_ocr_pages,
)
from pdf_document_agent.ollama import DEFAULT_MODEL, OllamaError, chat_with_ollama
from pdf_document_agent.retrieval import TextChunk, chunk_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Извлекает PDF и отвечает на вопросы по его содержимому."
    )
    parser.add_argument("pdf_path", help="Путь к PDF-документу")
    question_mode = parser.add_mutually_exclusive_group()
    question_mode.add_argument(
        "--ask",
        metavar="ВОПРОС",
        help="Задать один вопрос по документу",
    )
    question_mode.add_argument(
        "--interactive",
        action="store_true",
        help="Задавать несколько вопросов в интерактивном режиме",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PDF_AGENT_MODEL", DEFAULT_MODEL),
        help=f"Модель Ollama (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed", "agent"),
        default="fixed",
        help="Режим вопросов: fixed — кандидат B со статусом HOLD "
        "(выбран по умолчанию только в этой ветке); agent — экспериментальный C "
        "(HOLD, без скрытого fallback)",
    )
    parser.add_argument(
        "--ocr-pages",
        default="",
        metavar="PAGES",
        help="Ручной OCR для страниц: номера и inclusive ranges через запятую, "
        "например 1, 3-5",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = ExtractionConfig(ocr_pages=parse_ocr_pages(args.ocr_pages))
        document = extract_pdf(args.pdf_path, config=config)
    except (FileNotFoundError, ValueError, PdfExtractionError) as error:
        parser.error(str(error))

    _print_extraction_summary(document)

    if args.ask is None and not args.interactive:
        _print_document(document.source_name, document.page_count, document.markdown)
        return

    chunks = chunk_document(document)
    if not chunks:
        parser.error("В документе нет текста для поиска.")

    chat = (
        partial(_bounded_chat, model=args.model)
        if args.mode == "agent"
        else partial(chat_with_ollama, model=args.model)
    )
    if args.ask is not None:
        try:
            if args.mode == "agent":
                run = run_agent(args.ask, document, chunks, chat=chat)
            else:
                answer = answer_question(args.ask, chunks, chat=chat)
        except (ValueError, OllamaError, AnswerGenerationError) as error:
            parser.error(str(error))
        if args.mode == "agent":
            _print_agent_run(run)
        else:
            _print_answer(answer)
        return

    _interactive_loop(
        document.source_name,
        document.page_count,
        chunks,
        chat,
        mode=args.mode,
        document=document,
    )


def _bounded_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    max_output_tokens: int,
    timeout: float,
) -> str:
    return chat_with_ollama(
        system_prompt,
        user_prompt,
        model=model,
        timeout=timeout,
        num_predict=max_output_tokens,
        num_ctx=8192,
    )


def _print_document(source_name: str, page_count: int, markdown: str) -> None:
    """Сохранить прежний режим вывода извлечённого Markdown."""
    print(f"Документ: {source_name}")
    print(f"Страниц: {page_count}")
    print()
    print(markdown)


def _print_extraction_summary(document: ExtractedDocument) -> None:
    print("Извлечение:")
    for page in document.pages:
        print(f"Page {page.number}: route={page.route}, status={page.status}")
    for warning in document.warnings:
        print(f"Warning: {warning}")
    for page in document.pages:
        if page.diagnostic is not None:
            print(f"Diagnostic (page {page.number}): {page.diagnostic}")
    print()


def _print_answer(answer: GroundedAnswer) -> None:
    print("Ответ:")
    print(answer.text)
    if answer.source_pages:
        pages = ", ".join(map(str, answer.source_pages))
        print()
        print(f"Страницы контекста: {pages}")


def _print_agent_run(run: AgentRun) -> None:
    _print_answer(run.answer)
    print()
    print(
        "Agent: "
        f"status={run.status}; tool_calls={run.tool_calls}; "
        f"llm_calls={run.llm_calls}"
    )
    for event in run.trace.events:
        pages = ",".join(map(str, event.pages)) or "-"
        truncated = "yes" if event.truncated else "no"
        print(
            f"Agent action: {event.action}: {event.status}; "
            f"pages={pages}; truncated={truncated}"
        )


def _interactive_loop(
    source_name: str,
    page_count: int,
    chunks: list[TextChunk],
    chat: Callable[..., str],
    *,
    mode: str = "fixed",
    document: ExtractedDocument | None = None,
) -> None:
    print(f"Документ готов: {source_name} ({page_count} стр.)")
    print("Задавай вопросы. Для выхода введи: выход")
    previous_questions: list[str] = []

    while True:
        try:
            question = input("\nВопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.casefold() in {"выход", "exit", "quit"}:
            return
        if not question:
            continue

        try:
            if mode == "agent":
                run = run_agent(
                    question,
                    document,
                    chunks,
                    chat=chat,
                    previous_questions=tuple(previous_questions),
                )
            else:
                answer = answer_question(
                    question,
                    chunks,
                    chat=chat,
                    previous_questions=previous_questions,
                )
        except (ValueError, OllamaError, AnswerGenerationError) as error:
            print(f"Ошибка: {error}")
            continue
        previous_questions.append(question)
        print()
        if mode == "agent":
            _print_agent_run(run)
        else:
            _print_answer(answer)
