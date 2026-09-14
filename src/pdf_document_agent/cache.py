"""Безопасный дисковый JSON-кэш результатов extraction/chunking PDF.

Кэш хранит только нормализованные результаты (``ExtractedDocument`` и
``TextChunk``), но никогда исходные PDF-байты. Ключом служат SHA-256
содержимого PDF, версия схемы и fingerprint extraction config, поэтому
переименование файла не ломает повторное использование, а смена политики
не переиспользует устаревший результат. Записи сериализуются в JSON с версией схемы и строго
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
import re
import tempfile
from pathlib import Path

from pdf_document_agent.extractor import (
    DEFAULT_EXTRACTION_CONFIG_FINGERPRINT,
    ExtractedDocument,
    ExtractedPage,
    TextRegion,
)
from pdf_document_agent.retrieval import TextChunk

SCHEMA_VERSION = 3
MAX_ENTRIES = 10
_CACHE_DIR_ENV = "PDF_DOCUMENT_AGENT_CACHE_DIR"
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _validate_fingerprint(value: str) -> str:
    if type(value) is not str or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise ValueError("config_fingerprint должен быть SHA-256 в lowercase hex")
    return value


def cache_dir() -> Path:
    """Каталог кэша: переменная окружения либо ``~/.cache/pdf-document-agent``."""
    override = os.environ.get(_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "pdf-document-agent"


def cache_path(
    file_bytes: bytes,
    *,
    config_fingerprint: str = DEFAULT_EXTRACTION_CONFIG_FINGERPRINT,
) -> Path:
    """Путь записи по PDF SHA-256, schema и extraction policy."""
    config_fingerprint = _validate_fingerprint(config_fingerprint)
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()
    return cache_dir() / f"{pdf_hash}.{SCHEMA_VERSION}.{config_fingerprint}.json"


def get(
    file_bytes: bytes,
    *,
    config_fingerprint: str = DEFAULT_EXTRACTION_CONFIG_FINGERPRINT,
) -> tuple[ExtractedDocument, list[TextChunk]] | None:
    """Вернуть кэшированный результат или ``None`` при промахе.

    Повреждённая/невалидная запись удаляется и трактуется как промах. Hit
    обновляет mtime записи (recency), игнорируя гонки с ``clear``.
    """
    path = cache_path(file_bytes, config_fingerprint=config_fingerprint)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        data = json.loads(raw)
        document, chunks = _decode(data, config_fingerprint=config_fingerprint)
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
    *,
    config_fingerprint: str = DEFAULT_EXTRACTION_CONFIG_FINGERPRINT,
) -> None:
    """Атомарно сохранить результат и ограничить кэш ``MAX_ENTRIES`` записями."""
    config_fingerprint = _validate_fingerprint(config_fingerprint)
    if document.config_fingerprint != config_fingerprint:
        raise ValueError("document config_fingerprint не совпадает с cache identity")
    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = _encode(document, chunks, config_fingerprint=config_fingerprint)
    _atomic_write(
        cache_path(file_bytes, config_fingerprint=config_fingerprint),
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
        "config_fingerprint": document.config_fingerprint,
        "warnings": list(document.warnings),
        "pages": [
            {
                "number": page.number,
                "markdown": page.markdown,
                "route": page.route,
                "status": page.status,
                "reason": page.reason,
                "diagnostic": page.diagnostic,
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


def _encode(
    document: ExtractedDocument,
    chunks: list[TextChunk],
    *,
    config_fingerprint: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction_config_fingerprint": config_fingerprint,
        "document": _document_to_dict(document),
        "chunks": [_chunk_to_dict(chunk) for chunk in chunks],
    }


def _decode(
    data,
    *,
    config_fingerprint: str = DEFAULT_EXTRACTION_CONFIG_FINGERPRINT,
) -> tuple[ExtractedDocument, list[TextChunk]]:
    if type(data) is not dict:
        raise ValueError("корневой объект не dict")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("неизвестная schema_version")
    if data.get("extraction_config_fingerprint") != config_fingerprint:
        raise ValueError("неверный extraction_config_fingerprint")
    document = _document_from_dict(data["document"])
    if document.config_fingerprint != config_fingerprint:
        raise ValueError("document config_fingerprint не совпадает с cache identity")
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


def _as_optional_str(value) -> str | None:
    if value is not None and type(value) is not str:
        raise ValueError("ожидалась строка или null")
    return value


def _as_route(value) -> str:
    value = _as_str(value)
    if value not in {"docling_text", "docling_ocr"}:
        raise ValueError("неизвестный route")
    return value


def _as_status(value) -> str:
    value = _as_str(value)
    if value not in {"ok", "empty", "failed"}:
        raise ValueError("неизвестный status")
    return value


def _as_warnings(value) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("warnings должен быть списком строк")
    return tuple(value)


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
        route=_as_route(data["route"]),
        status=_as_status(data["status"]),
        reason=_as_str(data["reason"]),
        diagnostic=_as_optional_str(data["diagnostic"]),
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
        config_fingerprint=_as_str(data["config_fingerprint"]),
        warnings=_as_warnings(data["warnings"]),
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
