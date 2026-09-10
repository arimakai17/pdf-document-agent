"""Безопасный дисковый JSON-кэш результатов extraction/chunking PDF.

Кэш хранит только нормализованные результаты (``ExtractedDocument`` и
``TextChunk``), но никогда исходные PDF-байты. Ключом служит SHA-256
содержимого PDF, поэтому переименование файла не ломает повторное
использование. Записи сериализуются в JSON с версией схемы и строго
восстанавливаются в dataclasses без pickle/eval. Повреждённые, неизвестной
версии или структурно невалидные записи трактуются как промах и удаляются.

LRU: максимум ``MAX_ENTRIES`` записей; вытесняется самая давно не
использованная (по mtime). Hit обновляет recency через ``os.utime``.
Запись атомарна (temp + ``os.replace``). Все операции устойчивы к гонкам с
``clear``/prune — ``FileNotFoundError`` не роняет приложение.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pdf_document_agent.extractor import ExtractedDocument, ExtractedPage, TextRegion
from pdf_document_agent.retrieval import TextChunk

SCHEMA_VERSION = 1
MAX_ENTRIES = 10
_CACHE_DIR_ENV = "PDF_DOCUMENT_AGENT_CACHE_DIR"


def cache_dir() -> Path:
    """Каталог кэша: переменная окружения либо ``~/.cache/pdf-document-agent``."""
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "pdf-document-agent"


def cache_path(file_bytes: bytes) -> Path:
    """Путь записи кэша по SHA-256 содержимого (не имени файла)."""
    key = hashlib.sha256(file_bytes).hexdigest()
    return cache_dir() / f"{key}.json"


def get(
    file_bytes: bytes,
) -> tuple[ExtractedDocument, list[TextChunk]] | None:
    """Вернуть кэшированный результат или ``None`` при промахе.

    Повреждённая/невалидная запись удаляется и трактуется как промах. Hit
    обновляет mtime записи (recency), игнорируя гонки с ``clear``.
    """
    path = cache_path(file_bytes)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        data = json.loads(raw)
        document, chunks = _decode(data)
    except Exception:
        _safe_unlink(path)
        return None

    try:
        os.utime(path, None)
    except OSError:
        pass
    return document, chunks


def put(
    file_bytes: bytes,
    document: ExtractedDocument,
    chunks: list[TextChunk],
) -> None:
    """Атомарно сохранить результат и ограничить кэш ``MAX_ENTRIES`` записями."""
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = _encode(document, chunks)
    _atomic_write(
        cache_path(file_bytes),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    _prune()


def clear() -> None:
    """Удалить все записи кэша; устойчиво к отсутствию каталога/гонкам."""
    try:
        entries = list(cache_dir().glob("*.json"))
    except OSError:
        return
    for entry in entries:
        _safe_unlink(entry)


# --- serialization -----------------------------------------------------------


def _region_to_dict(region: TextRegion) -> dict:
    return {
        "page_number": region.page_number,
        "text": region.text,
        "box": list(region.box),
    }


def _document_to_dict(document: ExtractedDocument) -> dict:
    return {
        "source_name": document.source_name,
        "markdown": document.markdown,
        "page_count": document.page_count,
        "pages": [
            {
                "number": page.number,
                "markdown": page.markdown,
                "regions": [_region_to_dict(region) for region in page.regions],
            }
            for page in document.pages
        ],
    }


def _chunk_to_dict(chunk: TextChunk) -> dict:
    return {
        "index": chunk.index,
        "page_number": chunk.page_number,
        "text": chunk.text,
        "boxes": [list(box) for box in chunk.boxes],
        "regions": [_region_to_dict(region) for region in chunk.regions],
    }


def _encode(document: ExtractedDocument, chunks: list[TextChunk]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "document": _document_to_dict(document),
        "chunks": [_chunk_to_dict(chunk) for chunk in chunks],
    }


def _decode(data) -> tuple[ExtractedDocument, list[TextChunk]]:
    if type(data) is not dict:
        raise ValueError("корневой объект не dict")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("неизвестная schema_version")
    document = _document_from_dict(data["document"])
    chunks = [_chunk_from_dict(item) for item in data["chunks"]]
    return document, chunks


def _as_str(value) -> str:
    if type(value) is not str:
        raise ValueError("ожидалась строка")
    return value


def _as_int(value) -> int:
    if type(value) is not int:
        raise ValueError("ожидался int")
    return value


def _as_float(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ожидалось число")
    return float(value)


def _as_box(value) -> tuple[float, float, float, float]:
    if type(value) is not list or len(value) != 4:
        raise ValueError("box должен быть списком из 4 чисел")
    coords = [_as_float(coord) for coord in value]
    return (coords[0], coords[1], coords[2], coords[3])


def _region_from_dict(data) -> TextRegion:
    if type(data) is not dict:
        raise ValueError("region не dict")
    return TextRegion(
        page_number=_as_int(data["page_number"]),
        text=_as_str(data["text"]),
        box=_as_box(data["box"]),
    )


def _page_from_dict(data) -> ExtractedPage:
    if type(data) is not dict:
        raise ValueError("page не dict")
    regions = data["regions"]
    if type(regions) is not list:
        raise ValueError("regions не list")
    return ExtractedPage(
        number=_as_int(data["number"]),
        markdown=_as_str(data["markdown"]),
        regions=tuple(_region_from_dict(region) for region in regions),
    )


def _chunk_from_dict(data) -> TextChunk:
    if type(data) is not dict:
        raise ValueError("chunk не dict")
    boxes = data["boxes"]
    if type(boxes) is not list:
        raise ValueError("boxes не list")
    regions = data["regions"]
    if type(regions) is not list:
        raise ValueError("regions не list")
    return TextChunk(
        index=_as_int(data["index"]),
        page_number=_as_int(data["page_number"]),
        text=_as_str(data["text"]),
        boxes=tuple(_as_box(box) for box in boxes),
        regions=tuple(_region_from_dict(region) for region in regions),
    )


def _document_from_dict(data) -> ExtractedDocument:
    if type(data) is not dict:
        raise ValueError("document не dict")
    pages = data["pages"]
    if type(pages) is not list:
        raise ValueError("pages не list")
    return ExtractedDocument(
        source_name=_as_str(data["source_name"]),
        markdown=_as_str(data["markdown"]),
        page_count=_as_int(data["page_count"]),
        pages=tuple(_page_from_dict(page) for page in pages),
    )


# --- filesystem helpers ------------------------------------------------------


def _atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _prune() -> None:
    directory = cache_dir()
    try:
        entries = list(directory.glob("*.json"))
    except OSError:
        return
    if len(entries) <= MAX_ENTRIES:
        return
    entries.sort(key=_mtime)
    for stale in entries[:-MAX_ENTRIES]:
        _safe_unlink(stale)
