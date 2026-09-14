# Phase 0 selective-OCR/provenance spike — PDF Atlas V2

Date: 2026-09-14 (Asia/Almaty)

## Scope and environment

Bounded execution probe only. No files under `src/`, `tests/`, `PLAN_V2.md`,
`pyproject.toml`, `uv.lock`, or `README` were changed. No PDF was copied into
the repository; temporary rendered images and the uv cache were in `/tmp`. No
package/model install, web/cloud API, `.env`/credential access, or personal PDF
access was used.

Initial guard check:

```text
pwd                         /Users/arimakai/Downloads/LLM-Engineer/pdf-document-agent
git branch --show-current   feat/pdf-atlas-v2
git rev-parse HEAD          0ca59626a1effccadb467f44287a0df7a4d254fd
git status --short --branch  ## feat/pdf-atlas-v2
```

The expected branch, HEAD prefix `0ca5962`, and clean state matched.

Runtime probe:

```text
UV_CACHE_DIR=/tmp/pdf-atlas-uv-cache uv run python - <<'PY'
import importlib.metadata as m, sys
for name in ('docling','docling-core','pypdfium2','Pillow'):
    try: print(f'{name}={m.version(name)}')
    except m.PackageNotFoundError: print(f'{name}=MISSING')
print(f'python={sys.version.split()[0]}')
PY
```

The first bare `uv run python` was blocked by permissions on `~/.cache/uv`;
the same already-installed runtime then ran with the temporary cache override.
Observed: Python 3.13.0, Docling 2.126.0, docling-core 2.95.0, pypdfium2
5.13.0, Pillow 12.3.0, macOS 26.5.1 arm64. `tesseract` and `ocrmypdf` were
not present; `pytesseract` was not importable.

Input SHA-256:

```text
1b311fdcc5efacd7272fa690ce56c901b66ce2b98807454d05ad82b96251ed69  lecture09-notes.pdf
0307055f5c7c095c0826313d807410c333b42d2f9935d697ae7b526f9522bede  G25_Central_Asia_reading_(1)-1-4.pdf
b6986122600317938b57120adeac239aa3bed62ff07463024c23899d3211cfa3  kaztelecom.pdf
```

PDF inventory via installed `pypdfium2`: `lecture09-notes.pdf` 7 pages;
`G25_Central_Asia_reading_(1)-1-4.pdf` 4 pages; `kaztelecom.pdf` 46 pages.
Scan pages 1 and 2 are different sizes: `(595.4401, 708.9799)` and
`(595.4401, 736.6315)` points.

## Exact probes and observed outputs

Docling construction used for the full and selective calls:

```python
opts = PdfPipelineOptions(); opts.do_ocr = False
doc = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
}).convert(text_pdf).document

opts = PdfPipelineOptions(); opts.do_ocr = True
opts.ocr_options = OcrMacOptions(lang=['ru-RU', 'en-US'])
doc = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
}).convert(scan_pdf, page_range=(2, 2)).document
```

```text
text, do_ocr=False, full pass: doc_pages=7, keys=[1,2,3,4,5,6,7], markdown_chars=6943, elapsed=7.33s
scan, do_ocr=True, page_range=(2,2): doc_pages=1, keys=[2], markdown='<!-- image -->', elapsed=2.87s
```

1. **PARTIAL — full/selective execution.** Both calls complete and the
   selective result retains source key 2. The scan run emitted no OCR text;
   it emitted `PictureItem` provenance only. Trying `OcrMode.DEFAULT`,
   `FULL_PAGE`, `LAYOUT_REGIONS`, and `PDF_AWARE_LAYOUT_REGIONS` on scan page 2
   produced the same two `PictureItem`s and zero non-empty text items. This is
   an unresolved OCR runtime/behavior limitation, not evidence of OCR quality.

2. **PASS — non-adjacent page identity.** Exact loop:

   ```python
   for page_range in ((1, 1), (3, 3)):
       doc = converter.convert(scan_pdf, page_range=page_range).document
       print(sorted(doc.pages), [p.page_no for p in doc.pages.values()],
             sorted({prov.page_no for item in doc.iterate_items()
                     for prov in (item[0] if isinstance(item, tuple) else item).prov}))
   ```

   Observed: `(1,1) -> keys=[1], page.page_no=[1], prov.page_no=[1]` and
   `(3,3) -> keys=[3], page.page_no=[3], prov.page_no=[3]`. No renumbering.

3. **PARTIAL — normalized real provenance boxes.** Applied the existing
   `extractor._normalize_box(prov, doc.pages)` to all provenance from the real
   selective run. Both boxes were finite, within `[0,1]`, and non-zero:

   ```text
   (0.0, 0.0, 0.9161374905, 0.6849014924)
   (0.8149090965, 0.7950065954, 0.9719575204, 0.9583514148)
   ```

   They are real `PictureItem` source boxes on page 2, not OCR text boxes.
   The existing `_collect_text_regions()` returned `{}` because the OCR run
   had zero text items. Normalization and source geometry are validated, but
   the required real OCR-text-box seam remains unvalidated.

4. **PASS — viewer page/geometry probe.** Exact seam:

   ```python
   base = render_page(pdf_bytes, 2, scale=1.0)
   over = render_page(pdf_bytes, 2, boxes=normalized_boxes, scale=1.0)
   ```

   Observed: image size `(596,737)`; pixel rectangles `[(0,0,546,505),
   (486,586,579,706)]`; changed pixels inside expected rectangles `288156`,
   outside `0`; first-box center changed `(210,214,223)` -> `(222,216,162)`.
   The analogous page-1 probe produced `(596,709)`, rectangle `[(295,1,595,480)]`,
   changed inside `144480`, outside `0`. This is an objective overlay result
   on the correct source page using boxes emitted by the selective run.

5. **PASS for different-size; rotated subcase UNKNOWN.** The real scan has
   different-size pages 1 and 2 as above. Both retained page identity and both
   rendered/overlaid correctly. `get_rotation()` was `0` for all four scan
   pages and all 46 `kaztelecom` pages, so no rotated page was found in the
   permitted corpus and no synthetic rotated fixture was created.

6. **PARTIAL — mixed digital text + raster gate.** Real `kaztelecom` page 1
   with `do_ocr=False` had `SectionHeaderItem=1`, `TextItem=2`,
   `PictureItem=1`, three non-empty text items and 118 text characters. Its
   normalized picture area was `0.0105` (1.05%), a small logo rather than
   significant raster content. Signals (digital text presence, picture count,
   normalized area) are observable, but the corpus has no positive real
   significant-raster mixed page on which an automatic OCR decision can be
   validated. The gate must therefore expose a manual per-page OCR override;
   “has digital text” alone is not a safe skip rule.

7. **PARTIAL — replacement invariant at the available seam.** Minimal
   invariant: for a selected page, the OCR representation atomically replaces
   the no-OCR representation; it is not merged with it. Executed:

   ```python
   no = convert(scan_pdf, do_ocr=False, page_range=(2,2)).document
   yes = convert(scan_pdf, do_ocr=True, page_range=(2,2)).document
   ```

   Both had exactly two `PictureItem`s and identical markdown:
   `<!-- image -->\n\n<!-- image -->`; the two placeholders corresponded to
   distinct source boxes and OCR added no item or markdown duplicate. Since
   OCR contained no text, atomic replacement of an actual OCR text result is
   still untested.

Targeted existing tests:

```text
UV_CACHE_DIR=/tmp/pdf-atlas-uv-cache uv run pytest -q tests/test_extractor.py tests/test_viewer.py
12 passed in 5.01s
```

## Verdict

**PARTIAL** — page-range contract, original page identity, one-shot
normalization, different-size handling, and viewer geometry are executable and
safe in this spike. The core selective-OCR value path is not validated because
the permitted real scans produced no OCR text/provenance regions with the
installed `OcrMacOptions` runtime. No production implementation was added.

## Minimum requirements for the next wave

1. Establish an explicit supported OCR backend/runtime preflight and run it on
   a real scan with expected non-empty text items and provenance boxes; a
   silent `PictureItem`-only result must be surfaced as unsupported/failed OCR.
2. Preserve source keys and `page.page_no`/`prov.page_no` through non-adjacent
   ranges, with regression coverage for the single normalization boundary and
   finite/non-zero normalized boxes.
3. Test atomic per-page replacement using a real OCR result: exactly one page
   representation, with no concatenated no-OCR+OCR markdown/items.
4. Add real mixed pages with meaningful digital text and substantial raster
   content; measure false-skip risk and keep a manual per-page OCR override
   mandatory until that evidence is green.
5. Add a real or approved rotated-page fixture and verify coordinate transforms
   against rendered pixels, in addition to the observed different-size case.

## Remaining unknowns

- Why the installed macOS Docling OCR path returns only `PictureItem`s for the
  permitted scan corpus, including with all four supported OCR modes.
- OCR language/quality and text-box provenance on real scans once a supported
  backend is available.
- Safe automatic gate thresholds for significant raster content on a genuinely
  mixed digital/raster page.
- Rotation-specific provenance-to-viewer coordinate behavior.
- Table/image interaction and replacement behavior when OCR produces multiple
  text regions on one page.
