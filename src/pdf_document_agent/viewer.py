"""Lazy, byte-backed PDF page rendering for the Streamlit viewer."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from pdf_document_agent.extractor import NormalizedBox


class ViewerError(RuntimeError):
    """A PDF page could not be rendered safely."""


def render_page(
    pdf_bytes: bytes,
    page_number: int,
    *,
    boxes: Iterable[NormalizedBox] = (),
    scale: float = 1.0,
) -> Image.Image:
    """Render one PDF page and overlay valid normalized source regions."""
    if not pdf_bytes:
        raise ViewerError("PDF пуст или повреждён.")
    if not isinstance(page_number, int) or page_number < 1:
        raise ViewerError("Номер страницы вне диапазона.")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ViewerError("Некорректный масштаб страницы.")

    document = None
    page = None
    bitmap = None
    try:
        document = pdfium.PdfDocument(pdf_bytes)
        if page_number > len(document):
            raise ViewerError("Номер страницы вне диапазона.")
        page = document[page_number - 1]
        bitmap = page.render(scale=float(scale))
        image = bitmap.to_pil().convert("RGB")
    except ViewerError:
        raise
    except Exception:
        raise ViewerError("PDF повреждён или не удалось его отобразить.") from None
    finally:
        for resource in (bitmap, page, document):
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:
                pass

    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for box in boxes:
        coordinates = _pixel_coordinates(box, width, height)
        if coordinates is None:
            continue
        draw.rectangle(
            coordinates,
            fill=(255, 220, 0, 70),
            outline=(255, 110, 0, 255),
            width=max(1, round(scale * 2)),
        )
    return image


def _pixel_coordinates(
    box: NormalizedBox,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None

    left, top, right, bottom = (
        max(0.0, min(1.0, value)) for value in values
    )
    if right <= left or bottom <= top:
        return None
    return (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )
