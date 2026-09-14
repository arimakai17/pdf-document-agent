# PDF Atlas V2 — Wave 5 B → C report

Date: 2026-09-14 (Asia/Almaty)

## Decision: HOLD C / A→B prerequisite HOLD

**HOLD C.** This B→C run is historical and conditional. A→B is now
`HOLD B EXTRACTION` after the corrected OCR gate failed, so no C promotion,
merge, or release is authorized. B is the technical fixed default only inside
this candidate branch because A is a frozen external baseline, not an in-app
mode. C remains a visible opt-in experiment with HOLD status. The run shows
measurable value—one mixed-text false refusal is fixed and both expected source
recall and manual correctness/support improve—but the output interface is not
stable enough for branch-default use.

Candidate commit: `5bba39a6d8d66b6b5b846da1cca0c8a343a1f71d`.

Machine receipt: `evals/results/bc-2026-09-14.json`.

## Baseline

- B is the candidate fixed mode and C is the bounded read-only agent over the same saved B extraction artifacts.
- Prerequisite status: A→B is HOLD; the corrected actual-recognition-page gate is 15→9, `B/A = 60%`, versus the required `B/A ≤ 50%` (`B ≤ 7.5`). The separate case-weighted diagnostic is 22→9 and is not the gate.
- This receipt therefore records a conditional historical B→C vector, not authorization to merge or release B or C.
- Model: `qwen3:14b`; digest is recorded in the receipt from the raw run.
- The run used the same saved B artifacts for both candidates and alternating B/C order per question.
- Candidate labels were blinded before grading. Grading was performed by the parent model, not an independent human; confidence is low-to-moderate. The acceptance corpus was already visible during development, so this is not pristine held-out evidence.

## Hard gates

The B→C run's declared experiment checks pass for citation membership, no
uncaught runtime error or timeout, per-run budgets, correct refusals, and no
fabrication of chart values absent from extraction. C has 12/12
citation-membership validity, 3/3 correct refusals, no timeout, and no uncaught
exception. The A→B prerequisite does not pass: its corrected OCR gate is
15→9 actual recognition pages, `B/A = 60%`, against `≤50%`. The malformed-action
and final-contract failures are reported as output-interface failures, not
hidden.

## Objective vector

| Metric | B | C |
|---|---:|---:|
| Source recall | 7/11 | 8/11 |
| Citation membership validity | 12/12 | 12/12 |
| Manual correctness | 19/24 | 21/24 |
| Claim support | 18/24 | 21/24 |
| False refusals | 2/9 | 1/9 |
| Correct refusals | 3/3 | 3/3 |
| LLM calls | 28 | 44 |
| Tool steps | 0 | 23 |
| Total latency | 221.94143254170194 s | 196.2690888340585 s |
| Median latency | 5.908281083684415 s | 10.027464979561046 s |

Source recall is page-unit recall over the 11 expected source pages. Latency totals favor C in this single noisy run because B contains large outliers; C's median is slower. No speed superiority is claimed. The predeclared total-latency ≤3× gate passes.

## B1 experiment

C used only `outline`, `search`, and `read_page` against the saved extraction artifact, with host validation and bounded budgets. The experiment tests query refinement, adjacent-page reading, and multiple evidence sources without changing extraction.

## Observed evidence

- C has 8 `ok`, 3 `invalid_action`, and 1 `answer_error` runs out of 12.
- One malformed action and one final response-contract failure make the interface unreliable for branch-default use.
- C fixes one mixed-text false refusal and improves the aggregate manual vector.
- The multipage target still misses page 7; C does not recover the expected 6/7 coverage.
- The repeated `outline → search → answer_ready` pattern occurs 5/12 times.
- C reaches 6 maximum model calls, 4 maximum tool steps, and 54.428143499884754 s maximum per-question latency.
- Fail-closed refusals remain safe and correct. Chart values absent from extraction are not fabricated.

## Trade-offs

C improves measured retrieval coverage and manual scores, but increases calls from 28 to 44, adds 23 tool steps, and has a slower median despite a lower one-run total. The bounded loop adds interface failure modes without solving the multipage miss. Keeping C opt-in preserves the conditional experiment while B is only the technical branch default.

## Current limitations

- This is one local warmed QA run; latency distributions are noisy and do not establish a performance ranking.
- Manual grading was label-blinded but not independent, and the acceptance corpus was visible during development.
- Citation membership validates page inclusion in the final evidence packet, not semantic factuality.
- The chart case remains unavailable from extraction and must stay a refusal absent a separately approved backend.
- No raw response text, extracted PDF text, local absolute path, or blind key is committed.

## Deferred B2

Defer any C expansion until A→B is separately cleared and a separately
held-out, independently graded comparison demonstrates stable action/final-output
contracts, recovers the multipage source set, and remains within the declared
latency and budget boundaries. No architecture change is made in this wave.
