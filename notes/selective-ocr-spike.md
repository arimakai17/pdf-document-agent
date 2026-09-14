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

The first selective probe used `PdfPipelineOptions` with the default
`StandardPdfPipeline`. It completed without an exception but returned only
`PictureItem`s. Inspection of the installed Docling 2.126 API found the
boundary mismatch: the current threaded standard pipeline expects
`ThreadedPdfPipelineOptions`. Direct `ocrmac` recognition of the rendered page
returned 41 text boxes, so the OCR engine itself was healthy.

The decisive current-pipeline probe used:

```python
opts = ThreadedPdfPipelineOptions(); opts.do_ocr = False
doc = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
}).convert(text_pdf).document

opts = ThreadedPdfPipelineOptions(); opts.do_ocr = True
opts.do_table_structure = False
opts.ocr_options = OcrMacOptions(
    mode=OcrMode.FULL_PAGE,
    lang=['ru-RU', 'en-US'],
)
doc = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
}).convert(scan_pdf, page_range=(2, 2)).document
```

```text
text, do_ocr=False, full pass: doc_pages=7, keys=[1,2,3,4,5,6,7], markdown_chars=6943
scan, do_ocr=True, page_range=(2,2): keys=[2], text_items=13, regions=13, markdown_chars=1401
```

1. **PASS — full/selective execution.** The text pass and selective OCR call
   complete, retain source key 2, and produce 13 non-empty OCR-derived document
   items/regions on scan page 2. The earlier `PictureItem`-only output was
   reproduced as an incompatible pipeline-options type, not an OCR-engine
   limitation. Production must use `ThreadedPdfPipelineOptions` with the current
   default `StandardPdfPipeline` and fail visibly if an OCR-selected page still
   contains no usable text.

2. **PASS — non-adjacent page identity.** Selective current-pipeline calls for
   pages 1 and 3 retained source identity:

   ```python
   for page_range in ((1, 1), (3, 3)):
       doc = converter.convert(scan_pdf, page_range=page_range).document
       print(sorted(doc.pages), [p.page_no for p in doc.pages.values()],
             sorted({prov.page_no for item in doc.iterate_items()
                     for prov in (item[0] if isinstance(item, tuple) else item).prov}))
   ```

   Observed: `(1,1) -> keys=[1], page.page_no=[1], prov.page_no=[1]` and
   `(3,3) -> keys=[3], page.page_no=[3], prov.page_no=[3]`. Page 2 likewise
   remained 2. No renumbering.

3. **PASS — normalized real OCR provenance boxes.** Applied the existing
   `extractor._collect_text_regions()`/`_normalize_box()` boundary to real
   selective OCR output. Pages 1, 2, and 3 yielded 6, 13, and 1 text regions;
   every box was finite, within `[0,1]`, and non-zero. First observed boxes:

   ```text
   page 1: (0.1176918303, 0.1279106102, 0.4311542795, 0.2828337926)
   page 2: (0.0128790178, 0.7134466855, 0.1073509628, 0.7352143122)
   page 3: (0.1245540046, 0.8325093239, 0.6634826587, 0.8546493030)
   ```

   These are source-bearing text regions created from the OCR-enabled selective
   conversion, not picture-only placeholders.

4. **PASS — viewer page/geometry probe.** Exact seam:

   ```python
   base = render_page(pdf_bytes, 2, scale=1.0)
   over = render_page(pdf_bytes, 2, boxes=normalized_boxes, scale=1.0)
   ```

   Observed with all OCR text boxes: page 1 image `(596,709)`, changed pixels
   `123507`; page 2 `(596,737)`, changed pixels `67056`; page 3 `(596,764)`,
   changed pixels `5796`. Changed pixels outside the expected OCR rectangles
   were `0` on every page. This objectively ties the overlay to the correct
   source page and OCR geometry.

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

7. **PASS — replacement invariant at the available seam.** Minimal
   invariant: for a selected page, the OCR representation atomically replaces
   the no-OCR representation; it is not merged with it. Executed:

   ```python
   no = convert(scan_pdf, do_ocr=False, page_range=(2,2)).document
   yes = convert(scan_pdf, do_ocr=True, page_range=(2,2)).document
   ```

   The no-OCR page had two `PictureItem`s, zero text items and 30 Markdown
   characters. The OCR representation had the same two pictures plus 13 text
   items, 1401 Markdown characters, and 13/13 unique non-empty texts. The page
   artifact can therefore be replaced atomically; no no-OCR and OCR Markdown
   concatenation is required and no duplicate OCR text was observed.

Targeted existing tests:

```text
UV_CACHE_DIR=/tmp/pdf-atlas-uv-cache uv run pytest -q tests/test_extractor.py tests/test_viewer.py
12 passed in 5.01s
```

## Verdict

**PARTIAL — PROCEED WITH CAVEATS.** The core selective-OCR path is validated on
the permitted real scan: current Docling selective ranges, source page identity,
OCR text/provenance, one-shot normalization, different-size handling, atomic
replacement, and viewer geometry all passed. Wave 2 is not on HOLD. The overall
spike remains `PARTIAL` because the permitted corpus has no rotated page and no
positive real same-page digital-text plus significant-raster example. Manual
per-page OCR override is therefore required. No production implementation was
added.

## Minimum requirements for the next wave

1. Use `ThreadedPdfPipelineOptions` with the current standard pipeline and add
   an explicit OCR preflight/result gate; a selected page with no usable OCR
   text must become `empty`/`failed` with a diagnostic, never silent success.
2. Preserve source keys and `page.page_no`/`prov.page_no` through non-adjacent
   ranges, with regression coverage for the single normalization boundary and
   finite/non-zero normalized boxes.
3. Keep atomic per-page replacement: exactly one page representation, with no
   concatenated no-OCR+OCR Markdown/items.
4. Add real mixed pages with meaningful digital text and substantial raster
   content; measure false-skip risk and keep a manual per-page OCR override
   mandatory until that evidence is green.
5. Add a real or approved rotated-page fixture and verify coordinate transforms
   against rendered pixels, in addition to the observed different-size case.

## Remaining unknowns

- Whether future Docling versions keep the strict
  `StandardPdfPipeline`/`ThreadedPdfPipelineOptions` pairing; this boundary must
  remain covered by a real scan smoke.
- OCR quality thresholds across Russian/English scans beyond this permitted
  course sample.
- Safe automatic gate thresholds for significant raster content on a genuinely
  mixed digital/raster page.
- Rotation-specific provenance-to-viewer coordinate behavior.
- Table/image interaction and replacement behavior when OCR produces multiple
  text regions on one page.
