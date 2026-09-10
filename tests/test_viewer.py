import io

import pypdfium2 as pdfium
import pytest

from pdf_document_agent.viewer import ViewerError, render_page


def _make_pdf(width: int = 200, height: int = 200, pages: int = 1) -> bytes:
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(width, height)
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()
    return buffer.getvalue()


def test_render_page_draws_marker_over_requested_box() -> None:
    pdf = _make_pdf()

    image = render_page(pdf, 1, boxes=((0.25, 0.25, 0.75, 0.75),), scale=1.0)

    assert image.mode == "RGB"
    assert image.size == (200, 200)
    # Центр внутри маркированного региона не должен остаться чисто белым.
    assert image.getpixel((100, 100)) != (255, 255, 255)
    # Край имеет ту же заливку: отдельной цветной рамки вокруг marker нет.
    assert image.getpixel((50, 50)) == image.getpixel((100, 100))
    # Угол вне региона остаётся белым.
    assert image.getpixel((5, 5)) == (255, 255, 255)


def test_render_page_without_boxes_is_blank() -> None:
    pdf = _make_pdf()

    image = render_page(pdf, 1, scale=1.0)

    assert image.getpixel((100, 100)) == (255, 255, 255)


def test_render_page_rejects_out_of_range_page() -> None:
    pdf = _make_pdf(pages=1)

    with pytest.raises(ViewerError, match="вне диапазона"):
        render_page(pdf, 2)
    with pytest.raises(ViewerError, match="вне диапазона"):
        render_page(pdf, 0)


def test_render_page_rejects_corrupt_pdf() -> None:
    with pytest.raises(ViewerError, match="повреждён"):
        render_page(b"this is not a pdf", 1)


def test_render_page_rejects_empty_bytes() -> None:
    with pytest.raises(ViewerError):
        render_page(b"", 1)


def test_render_page_supports_second_page() -> None:
    pdf = _make_pdf(pages=2)

    image = render_page(pdf, 2, boxes=((0.0, 0.0, 0.5, 0.5),), scale=1.0)

    assert image.size == (200, 200)
    assert image.getpixel((30, 30)) != (255, 255, 255)
