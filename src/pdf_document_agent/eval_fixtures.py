"""Generate the small deterministic PDF fixtures declared in ``manifest.json``."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_PAGE_WIDTH = 612
_PAGE_HEIGHT = 792
_RASTER_WIDTH = 1200
_RASTER_HEIGHT = 800


class _PdfWriter:
    def __init__(self) -> None:
        self._objects: list[bytes | None] = []

    def reserve(self) -> int:
        self._objects.append(None)
        return len(self._objects)

    def set(self, object_number: int, value: bytes) -> None:
        self._objects[object_number - 1] = value

    def build(self, root_object: int) -> bytes:
        if any(value is None for value in self._objects):
            raise RuntimeError("PDF contains an unset object")
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, value in enumerate(self._objects, start=1):
            assert value is not None
            offsets.append(len(output))
            output.extend(f"{number} 0 obj\n".encode("ascii"))
            output.extend(value)
            output.extend(b"\nendobj\n")
        xref_offset = len(output)
        output.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(offsets)} /Root {root_object} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(output)


def _stream(dictionary: str, payload: bytes) -> bytes:
    prefix = f"<< {dictionary} /Length {len(payload)} >>\nstream\n".encode("ascii")
    return prefix + payload + b"\nendstream"


def _escape_pdf_text(text: str) -> str:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("Synthetic vector text must be ASCII") from error
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _vector_text(lines: tuple[str, ...]) -> bytes:
    if not lines:
        return b""
    commands = ["BT", "/F1 18 Tf", "72 700 Td"]
    for index, line in enumerate(lines):
        if index:
            commands.append("0 -30 Td")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("ascii")


def _raster(lines: tuple[str, ...], *, noisy: bool = False) -> Image.Image:
    image = Image.new("RGB", (_RASTER_WIDTH, _RASTER_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=42)
    draw.rectangle((30, 30, _RASTER_WIDTH - 30, _RASTER_HEIGHT - 30), outline="#FF4D00", width=8)
    if noisy:
        rng = random.Random(20260914)
        for _ in range(450):
            x = rng.randrange(_RASTER_WIDTH)
            y = rng.randrange(_RASTER_HEIGHT)
            shade = rng.randrange(175, 236)
            draw.point((x, y), fill=(shade, shade, shade))
    y = 180
    for line in lines:
        draw.text((90, y), line, fill="black", font=font)
        y += 90
    return image


def _build_pdf(
    pages: tuple[tuple[tuple[str, ...], tuple[str, ...], bool], ...],
) -> bytes:
    writer = _PdfWriter()
    catalog_object = writer.reserve()
    pages_object = writer.reserve()
    font_object = writer.reserve()
    writer.set(font_object, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_objects: list[int] = []
    pending: list[tuple[int, int, int | None, bytes]] = []
    for vector_lines, raster_lines, noisy in pages:
        image_object: int | None = None
        if raster_lines:
            image = _raster(raster_lines, noisy=noisy)
            compressed = zlib.compress(image.tobytes(), level=9)
            image_object = writer.reserve()
            writer.set(
                image_object,
                _stream(
                    (
                        f"/Type /XObject /Subtype /Image /Width {_RASTER_WIDTH} "
                        f"/Height {_RASTER_HEIGHT} /ColorSpace /DeviceRGB "
                        "/BitsPerComponent 8 /Filter /FlateDecode"
                    ),
                    compressed,
                ),
            )
        content = bytearray(_vector_text(vector_lines))
        if image_object is not None:
            content.extend(b"q\n500 0 0 333 56 220 cm\n/Im0 Do\nQ\n")
        content_object = writer.reserve()
        page_object = writer.reserve()
        page_objects.append(page_object)
        pending.append((page_object, content_object, image_object, bytes(content)))

    for page_object, content_object, image_object, content in pending:
        writer.set(content_object, _stream("", content))
        resources = f"/Font << /F1 {font_object} 0 R >>"
        if image_object is not None:
            resources += f" /XObject << /Im0 {image_object} 0 R >>"
        writer.set(
            page_object,
            (
                f"<< /Type /Page /Parent {pages_object} 0 R "
                f"/MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
                f"/Resources << {resources} >> /Contents {content_object} 0 R >>"
            ).encode("ascii"),
        )

    kids = " ".join(f"{page} 0 R" for page in page_objects)
    writer.set(
        pages_object,
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_objects)} >>".encode("ascii"),
    )
    writer.set(catalog_object, f"<< /Type /Catalog /Pages {pages_object} 0 R >>".encode("ascii"))
    return writer.build(catalog_object)


def build_synthetic_corpus(output_dir: str | Path) -> dict[str, str]:
    """Write deterministic mechanics-only fixtures and return basename -> SHA-256."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "synthetic-text-tuning.pdf": _build_pdf(
            (
                (
                    (
                        "PDF ATLAS SYNTHETIC TEXT",
                        "The atlas marker is orange.",
                        "The launch sequence begins with indexing.",
                    ),
                    (),
                    False,
                ),
                (
                    (
                        "The launch sequence ends with citation validation.",
                        "Only the document can support an answer.",
                    ),
                    (),
                    False,
                ),
            )
        ),
        "synthetic-mixed-tuning.pdf": _build_pdf(
            (
                ((), ("SCANNED RIVER FACT", "The river code is ORANGE DELTA."), False),
                ((), ("NOISY OCR FACT", "The noisy signal says COPPER NINE."), True),
                (
                    ("Digital layer: control token ALPHA.",),
                    ("RASTER PANEL FACT", "The panel token is BRAVO."),
                    False,
                ),
                ((), ("SCANNED SECOND PAGE", "Archive key is LANTERN."), False),
                ((), (), False),
            )
        ),
    }
    hashes: dict[str, str] = {}
    for basename, payload in fixtures.items():
        path = output / basename
        path.write_bytes(payload)
        hashes[basename] = hashlib.sha256(payload).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_synthetic_corpus(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
