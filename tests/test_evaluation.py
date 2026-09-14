import json
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from pdf_document_agent.eval_fixtures import build_synthetic_corpus
from pdf_document_agent.evaluation import (
    CaseObservation,
    CitationObservation,
    PageObservation,
    QuestionObservation,
    evaluate_run,
    load_manifest,
)


def _document(kind: str = "synthetic", sha256: str | None = None) -> dict:
    return {
        "id": "doc-1",
        "kind": kind,
        "basename": "fixture.pdf",
        "sha256": sha256,
    }


def _manifest(*, cases: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "baseline": {
            "tag": "pdf-atlas-v1-submission",
            "commit": "1f7dcc0ee3cd652a3d00688e142d7da33fce802a",
        },
        "reproducibility": {
            "git_commit": None,
            "lockfile": None,
            "docling_version": None,
            "ocrmac_version": None,
            "ollama_version": None,
            "model_name": None,
            "model_digest": None,
            "options_prompts": None,
            "extraction_config_fingerprint": None,
            "gate_version": None,
            "warm_cold_protocol": "cold ingestion, warmed QA",
            "seed": None,
        },
        "documents": [_document()],
        "cases": cases
        or [
            {
                "id": "case-1",
                "split": "tuning",
                "categories": [
                    "text_page",
                    "scan_page",
                    "alternating_mixed_pages",
                    "same_page_text_raster",
                    "blank_sparse_page",
                    "ocr_noise",
                    "multi_page_answer",
                    "ambiguous_lexical_matches",
                    "answerable_similar_words",
                    "unanswerable_similar_words",
                ],
                "document_id": "doc-1",
                "extraction": [
                    {"page": 1, "route": "docling_text", "status": "ok"},
                    {"page": 2, "route": "docling_ocr", "status": "ok"},
                ],
                "questions": [
                    {
                        "id": "q-answerable",
                        "text": "What fact is stated?",
                        "answerable": True,
                        "source_page_sets": [[1], [1, 2]],
                    },
                    {
                        "id": "q-unanswerable",
                        "text": "What fact is absent?",
                        "answerable": False,
                        "source_page_sets": [],
                    },
                ],
            },
            {
                "id": "case-2",
                "split": "acceptance",
                "categories": ["text_page"],
                "document_id": "doc-1",
                "extraction": [
                    {"page": 1, "route": "docling_text", "status": "ok"},
                ],
                "questions": [
                    {
                        "id": "q-acceptance",
                        "text": "What acceptance fact is stated?",
                        "answerable": True,
                        "source_page_sets": [[1]],
                    }
                ],
            },
        ],
    }


def _write_manifest(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _complete_observation() -> CaseObservation:
    return CaseObservation(
        case_id="case-1",
        document_sha256=None,
        pdf_page_count=2,
        pages=(
            PageObservation(1, "docling_text", "ok"),
            PageObservation(2, "docling_ocr", "ok"),
        ),
        questions=(
            QuestionObservation(
                question_id="q-answerable",
                source_pages=(1, 2),
                final_evidence_pages=(1, 2),
                citations=(CitationObservation(1), CitationObservation(2)),
                refused=False,
                human_correctness=2,
                claim_support=2,
            ),
            QuestionObservation(
                question_id="q-unanswerable",
                source_pages=(),
                final_evidence_pages=(),
                citations=(),
                refused=True,
                human_correctness=2,
                claim_support=2,
            ),
        ),
        uncached_ingestion_latency_s=1.5,
        warmed_qa_latency_s=0.5,
        ocr_pages_processed=1,
        llm_calls=2,
        tool_steps=1,
        timed_out=False,
        runtime_error=None,
    )


def _acceptance_observation() -> CaseObservation:
    return CaseObservation(
        case_id="case-2",
        document_sha256=None,
        pdf_page_count=1,
        pages=(PageObservation(1, "docling_text", "ok"),),
        questions=(
            QuestionObservation(
                question_id="q-acceptance",
                source_pages=(1,),
                final_evidence_pages=(1,),
                citations=(CitationObservation(1),),
                refused=False,
                human_correctness=2,
                claim_support=2,
            ),
        ),
        uncached_ingestion_latency_s=1.0,
        warmed_qa_latency_s=0.25,
        ocr_pages_processed=0,
        llm_calls=1,
        tool_steps=0,
        timed_out=False,
        runtime_error=None,
    )


def test_manifest_loader_validates_and_evaluator_reports_explicit_metrics(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, _manifest()))

    report = evaluate_run(manifest, (_complete_observation(), _acceptance_observation()))

    assert report.route_correctness.numerator == 3
    assert report.route_correctness.denominator == 3
    assert report.source_recall.numerator == 3
    assert report.source_recall.denominator == 3
    assert report.citation_validity.numerator == 3
    assert report.human_correctness.numerator == 6
    assert report.claim_support.numerator == 6
    assert report.false_refusal.numerator == 0
    assert report.false_refusal.failures == 0
    assert report.correct_refusal.numerator == 1
    assert report.uncached_ingestion_latency.numerator == 2.5
    assert report.warmed_qa_latency.numerator == 0.75
    assert report.ocr_pages_processed.numerator == 1
    assert report.llm_calls.numerator == 3
    assert report.tool_steps.numerator == 1
    assert report.status == "incomplete"


def test_source_recall_uses_best_alternative_and_citation_needs_final_evidence(tmp_path: Path) -> None:
    data = _manifest()
    case = data["cases"][0]
    case["questions"][0]["source_page_sets"] = [[1, 2], [2]]
    manifest = load_manifest(_write_manifest(tmp_path, data))
    observation = _complete_observation()
    observation = CaseObservation(
        **{
            **observation.__dict__,
            "questions": (
                QuestionObservation(
                    question_id="q-answerable",
                    source_pages=(2,),
                    final_evidence_pages=(1,),
                    citations=(CitationObservation(2),),
                    refused=False,
                    human_correctness=None,
                    claim_support=None,
                ),
                observation.questions[1],
            ),
        }
    )

    report = evaluate_run(manifest, (observation, _acceptance_observation()))

    assert report.source_recall.numerator == 2
    assert report.source_recall.denominator == 2
    assert report.citation_validity.numerator == 2
    assert report.citation_validity.failures == 1
    assert report.human_correctness.skipped == 1


def test_partial_source_recall_and_wrong_refusals_are_measured_as_failures(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, _manifest()))
    base = _complete_observation()
    bad = CaseObservation(
        **{
            **base.__dict__,
            "questions": (
                QuestionObservation(
                    question_id="q-answerable",
                    source_pages=(),
                    final_evidence_pages=(),
                    citations=(),
                    refused=True,
                    human_correctness=0,
                    claim_support=0,
                ),
                QuestionObservation(
                    question_id="q-unanswerable",
                    source_pages=(2,),
                    final_evidence_pages=(2,),
                    citations=(CitationObservation(2),),
                    refused=False,
                    human_correctness=0,
                    claim_support=0,
                ),
            ),
        }
    )

    report = evaluate_run(manifest, (bad, _acceptance_observation()))

    assert report.status == "fail"
    assert report.source_recall.failures == 1
    assert report.false_refusal.numerator == 1
    assert report.false_refusal.failures == 1
    assert report.correct_refusal.numerator == 0
    assert report.correct_refusal.failures == 1


def test_missing_question_uses_smallest_alternative_recall_denominator(
    tmp_path: Path,
) -> None:
    data = _manifest()
    case = data["cases"][0]
    case["questions"][0]["source_page_sets"] = [[1, 2], [2]]
    manifest = load_manifest(_write_manifest(tmp_path, data))
    base = _complete_observation()
    partial = CaseObservation(
        **{
            **base.__dict__,
            "questions": (base.questions[1],),
        }
    )

    report = evaluate_run(manifest, (partial, _acceptance_observation()))

    assert report.source_recall.denominator == 2
    assert report.source_recall.skipped == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("route", "not-a-route"),
        ("status", "not-a-status"),
        ("page_number", 0),
    ],
)
def test_invalid_observation_fails_closed(field: str, value: object) -> None:
    kwargs = {"page_number": 1, "route": "docling_text", "status": "ok"}
    kwargs[field] = value

    with pytest.raises(ValueError):
        PageObservation(**kwargs)


def test_missing_observations_and_runtime_failure_are_not_pass_or_zero(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path, _manifest()))
    partial = CaseObservation(
        case_id="case-1",
        document_sha256=None,
        pdf_page_count=2,
        pages=(PageObservation(1, "docling_text", "ok"),),
        questions=(),
        uncached_ingestion_latency_s=None,
        warmed_qa_latency_s=None,
        ocr_pages_processed=None,
        llm_calls=None,
        tool_steps=None,
        timed_out=True,
        runtime_error="timeout",
    )

    report = evaluate_run(manifest, (partial,))

    assert report.status == "fail"
    assert report.route_correctness.skipped == 2
    assert report.source_recall.skipped == 2
    assert report.uncached_ingestion_latency.failures == 1
    assert report.timeouts.numerator == 1
    assert report.errors.numerator == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["documents"].append(data["documents"][0]),
        lambda data: data.update({"schema_version": 99}),
        lambda data: data["cases"][0]["extraction"][0].update({"page": 0}),
        lambda data: data["cases"][0]["questions"][0].update(
            {"source_page_sets": [[]]}
        ),
        lambda data: data["cases"][0]["questions"][1].update(
            {"source_page_sets": [[1]]}
        ),
    ],
)
def test_malformed_manifest_is_rejected(tmp_path: Path, mutate) -> None:
    data = _manifest()
    mutate(data)

    with pytest.raises(ValueError):
        load_manifest(_write_manifest(tmp_path, data))


def test_manifest_requires_category_and_split_coverage(tmp_path: Path) -> None:
    data = _manifest()
    data["cases"][0]["categories"] = ["text_page"]

    with pytest.raises(ValueError, match="coverage"):
        load_manifest(_write_manifest(tmp_path, data))


def test_synthetic_corpus_is_deterministic_and_has_expected_page_layers(
    tmp_path: Path,
) -> None:
    expected_hashes = {
        "synthetic-mixed-tuning.pdf": (
            "335a88b97e9a234299a40f253e2d69c09962a3a2a9009833daf6a88417fda6f4"
        ),
        "synthetic-text-tuning.pdf": (
            "3c109fdc37598cc489f0ee6ef9d4d92414f9df22a676a81bb37c504e315a0a2f"
        ),
    }

    hashes = build_synthetic_corpus(tmp_path)
    assert hashes == expected_hashes
    assert build_synthetic_corpus(tmp_path) == expected_hashes

    manifest_path = Path(__file__).parents[1] / "evals" / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest_hashes = {
        document.basename: document.sha256
        for document in manifest.documents
        if document.kind == "synthetic"
    }
    assert manifest_hashes == hashes

    text_pdf = pdfium.PdfDocument(tmp_path / "synthetic-text-tuning.pdf")
    mixed_pdf = pdfium.PdfDocument(tmp_path / "synthetic-mixed-tuning.pdf")
    try:
        assert len(text_pdf) == 2
        assert len(mixed_pdf) == 5
        assert "atlas marker is orange" in _pdf_page_text(text_pdf, 1)
        assert _pdf_page_text(mixed_pdf, 1) == ""
        assert "control token ALPHA" in _pdf_page_text(mixed_pdf, 3)
        assert _pdf_page_text(mixed_pdf, 5) == ""
    finally:
        text_pdf.close()
        mixed_pdf.close()


def _pdf_page_text(document: pdfium.PdfDocument, page_number: int) -> str:
    page = document[page_number - 1]
    text_page = page.get_textpage()
    try:
        return text_page.get_text_range().strip()
    finally:
        text_page.close()
        page.close()
