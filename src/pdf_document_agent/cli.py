import argparse
import os
from collections.abc import Callable
from functools import partial

from pdf_document_agent.answering import (
    AnswerGenerationError,
    GroundedAnswer,
    answer_question,
)
from pdf_document_agent.extractor import PdfExtractionError, extract_pdf
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        document = extract_pdf(args.pdf_path)
    except (FileNotFoundError, ValueError, PdfExtractionError) as error:
        parser.error(str(error))

    if args.ask is None and not args.interactive:
        _print_document(document.source_name, document.page_count, document.markdown)
        return

    chunks = chunk_document(document)
    if not chunks:
        parser.error("В документе нет текста для поиска.")

    chat = partial(chat_with_ollama, model=args.model)
    if args.ask is not None:
        try:
            answer = answer_question(args.ask, chunks, chat=chat)
        except (ValueError, OllamaError, AnswerGenerationError) as error:
            parser.error(str(error))
        _print_answer(answer)
        return

    _interactive_loop(document.source_name, document.page_count, chunks, chat)


def _print_document(source_name: str, page_count: int, markdown: str) -> None:
    """Сохранить прежний режим вывода извлечённого Markdown."""
    print(f"Документ: {source_name}")
    print(f"Страниц: {page_count}")
    print()
    print(markdown)


def _print_answer(answer: GroundedAnswer) -> None:
    print("Ответ:")
    print(answer.text)
    if answer.source_pages:
        pages = ", ".join(map(str, answer.source_pages))
        print()
        print(f"Страницы контекста: {pages}")


def _interactive_loop(
    source_name: str,
    page_count: int,
    chunks: list[TextChunk],
    chat: Callable[[str, str], str],
) -> None:
    print(f"Документ готов: {source_name} ({page_count} стр.)")
    print("Задавай вопросы. Для выхода введи: выход")

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
            answer = answer_question(question, chunks, chat=chat)
        except (ValueError, OllamaError, AnswerGenerationError) as error:
            print(f"Ошибка: {error}")
            continue
        print()
        _print_answer(answer)
