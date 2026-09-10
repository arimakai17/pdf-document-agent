"""Lazy, byte-backed PDF page rendering for the Streamlit viewer."""
from __future__ import annotations

import math
from collections.abc import Iterable

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


class ViewerError(RuntimeError):
    """A PDF page could not be rendered safely."""


def render_page(pdf_bytes: bytes, page_number: int, *, boxes: Iterable = (), scale: float = 1.0) -> Image.Image:
    if not pdf_bytes:
        raise ViewerError("PDF пуст или повреждён.")
    if not isinstance(page_number, int) or page_number < 1:
        raise ViewerError("Номер страницы вне диапазона.")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ViewerError("Некорректный масштаб страницы.")
    document = page = None
    try:
        document = pdfium.PdfDocument(pdf_bytes)
        count = len(document)
        if page_number > count:
            raise ViewerError("Номер страницы вне диапазона.")
        page = document[page_number - 1]
        bitmap = page.render(scale=float(scale))
        image = bitmap.to_pil().convert("RGB")
    except ViewerError:
        raise
    except Exception as exc:
        raise ViewerError("PDF повреждён или не удалось его отобразить.") from None
    finally:
        if page is not None:
            try: page.close()
            except Exception: pass
        if document is not None:
            try: document.close()
            except Exception: pass
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for box in boxes or ():
        try:
            values = tuple(float(v) for v in box)
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                continue
            x0, y0, x1, y1 = (max(0.0, min(1.0, v)) for v in values)
            if x1 <= x0 or y1 <= y0:
                continue
            coords = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
            draw.rectangle(coords, fill=(255, 220, 0, 70), outline=(255, 110, 0, 255), width=max(1, round(scale * 2)))
        except (TypeError, ValueError, OverflowError):
            continue
    return image
