# PDF Atlas V2 — A → B report

Date: 2026-09-14 (Asia/Almaty)

## Decision

**PROMOTE B EXTRACTION.** Adaptive per-page extraction becomes the default
extraction layer. This promotion does not claim that fixed retrieval is solved:
both A and B still fail the same four source-recall cases and two answerable
questions. The result is sufficient to begin the separately gated C agent
experiment because B changes extraction routes without regressing the measured
answer/citation/refusal vector.

Confidence: **moderate**. Extraction was repeated three times with stable output
hashes. QA used one local-model run and manual labels that were not independently
blinded, so it is regression evidence rather than a strong absolute quality
estimate.

Machine-readable receipt: `evals/results/ab-2026-09-14.json`.

## Baseline and candidate

- A: tag `pdf-atlas-v1-submission`, commit
  `1f7dcc0ee3cd652a3d00688e142d7da33fce802a`.
- B extraction: commit
  `879918b7979f01baa1e00837e22568cbfcbbd569`.
- Lockfile SHA-256:
  `6e790cd37fa13c0e130ea9813d318f608da1640f53edc0ad06b813001a1314eb`.
- Python 3.13.0, macOS 26.5.1 arm64.
- Docling 2.126.0, docling-core 2.95.0, ocrmac 1.0.1,
  pypdfium2 5.13.0, Pillow 12.3.0.
- Local model: `qwen3:14b`, digest
  `bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8`.
- Model options: temperature 0.1, seed 42, context 8192, output limit 600.
- No PDF, extracted text, or raw answer is stored in the committed receipt.

Corpus contained deterministic synthetic text/scan/alternating/same-page/blank
fixtures plus two hash-bound, nonpersonal real course PDFs. A ran in a detached
worktree with its own frozen environment. B ran from the candidate worktree.
Ingestion bypassed the application cache; QA used warmed local Ollama.

## B2 gates

The first run was exploratory because no numeric latency tolerance had been
fixed. Before two repeat runs, the following practical gates were declared:

- no extraction errors or timeouts;
- text-only B median must not exceed A median;
- each mixed-document B/A median ratio must be at most 2.0, reflecting the
  expected first-pass plus selective-OCR cost;
- no tested artifact median may exceed 30 seconds on this machine;
- B must process at most half as many OCR pages as A over the unique declared
  artifacts;
- citation validity, refusal behavior, manual correctness and support must not
  regress.

All these comparative gates passed.

## Extraction evidence

Three uncached runs produced identical candidate-specific output hashes on every
repeat.

Median ingestion by declared artifact:

- synthetic text: A 9.719 s → B 4.137 s (`0.426×`);
- synthetic mixed, default: A 3.330 s → B 4.887 s (`1.468×`);
- synthetic mixed, page-3 override: A 2.716 s → B 4.702 s (`1.731×`);
- real text: A 6.667 s → B 2.559 s (`0.384×`);
- real mixed, pages-1/2 override: A 2.822 s → B 4.511 s (`1.599×`).

Sum of artifact medians: A 25.254 s → B 20.797 s (`−17.6%`). This sum is
informative, not a production benchmark: there were only three runs on one
machine.

OCR work over the five unique declared artifacts fell from 21 pages under A's
always-OCR policy to 9 pages under B (`−57.1%`). Under the seven case-specific
runs, the count is 30 → 9 (`−70%`). B route/status correctness was 19/19; A
matched 6/19 because its frozen policy intentionally OCRs text pages too.

Preservation checks:

- synthetic and real text-only Markdown was identical after normalization;
- synthetic mixed with the explicit page-3 override was identical page by page;
- real mixed similarity was 0.9929 and 0.9986 on pages 1 and 2;
- real OCR regions stayed source-bearing and viewer overlay changed 67,056
  pixels inside expected page-2 boxes and 0 outside;
- no extraction error, timeout, page renumbering or warning occurred.

Trade-off: B is materially faster on text PDFs but 1.47–1.73× slower on mixed
PDFs because it pays for a no-OCR pass and separate selective calls. This is
accepted because it remains below the declared 2× bound and removes unnecessary
OCR from ordinary text documents.

## Fixed QA evidence

A and B used the same frozen retrieval/answering code and the same local model.
Final case configuration separates:

- a default text route for questions answered by the digital layer;
- explicit pages-1/2 OCR override only for the raster-chart question.

This avoids treating full-page OCR as a harmless preprocessing step for a
question that only needs authoritative digital text.

The measured vectors were identical:

- citation validity: 12/12;
- expected source-page recall: 7/11 page units, four question-level failures;
- manual correctness: 19/24 points;
- cited-claim support: 18/24 points;
- false refusals: 2/9 answerable questions;
- correct refusals: 3/3 unanswerable questions;
- LLM calls: 28 each;
- final-config runtime errors/timeouts: 0/0 each.

One-run warmed QA latency was noisy (individual calls ranged from roughly 2 to
32 seconds); B's aggregate was about 9% higher. Since prompts and outputs were
mostly identical and no repeat QA distribution was collected, no latency
superiority is claimed.

## What remains broken

1. The multi-page real question retrieves the workflow but misses page 7, so
   both candidates return the wrong “last two” practice tips and cite pages 5/6
   instead of the expected 6/7.
2. Both candidates refuse a text-answerable mixed-page wording because lexical
   rewrite/search does not bridge the inflected terms.
3. OcrMac full-page override does not recover the numeric labels inside the
   real chart. Both candidates correctly refuse rather than inventing them, but
   the visual question remains unanswered.
4. Page-membership citation validation passes 12/12 yet cannot detect the
   semantically wrong page-5 citation in the multi-page answer. Manual support
   scoring exposes this gap.

These are concrete targets and limits for C. A bounded agent can test query
refinement and adjacent-page reading for items 1–2. It cannot recover chart
numbers absent from extraction; C must refuse that case unless a separately
approved VLM/backend is added.

## Method correction disclosed

The first pilot put the text-only mixed-page question and visual-chart question
under the same forced-OCR configuration. B then exposed a deterministic
rewrite-contract error while A returned a refusal. Before the decision, the
manifest was corrected to separate default-text and visual-override cases. The
questions and reference answers were not changed.

Because this correction followed pilot output, this run is not presented as a
pristine untouched held-out estimate. It is sufficient for the narrower
comparative claim—B extraction did not worsen the finalized QA vector—but a
future external benchmark should remain untouched and independently graded.
