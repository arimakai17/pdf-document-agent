"""Deterministic contract and scoring for PDF Atlas evaluation runs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Route = Literal["docling_text", "docling_ocr"]
Status = Literal["ok", "empty", "failed"]
Split = Literal["tuning", "acceptance"]
Kind = Literal["synthetic", "real"]

ROUTES = frozenset(("docling_text", "docling_ocr"))
STATUSES = frozenset(("ok", "empty", "failed"))
CATEGORIES = frozenset(
    (
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
    )
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _require_type(value: object, expected: type, name: str) -> object:
    if type(value) is not expected:
        raise ValueError(f"{name} must be {expected.__name__}")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    _require_type(value, str, name)
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_pages(pages: tuple[int, ...], name: str) -> None:
    if any(type(page) is not int or page < 1 for page in pages):
        raise ValueError(f"{name} must contain page numbers >= 1")
    if len(set(pages)) != len(pages):
        raise ValueError(f"{name} must not contain duplicate pages")


@dataclass(frozen=True)
class DocumentSpec:
    logical_id: str
    kind: Kind
    basename: str
    sha256: str | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.logical_id, "document id")
        if type(self.kind) is not str or self.kind not in {"synthetic", "real"}:
            raise ValueError("document kind must be synthetic or real")
        _require_nonempty_string(self.basename, "document basename")
        if Path(self.basename).name != self.basename:
            raise ValueError("document basename must not contain a path")
        if self.sha256 is not None and (
            type(self.sha256) is not str or not _SHA256.fullmatch(self.sha256)
        ):
            raise ValueError("document sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True)
class ReproducibilitySpec:
    git_commit: str | None
    lockfile: str | None
    docling_version: str | None
    ocrmac_version: str | None
    ollama_version: str | None
    model_name: str | None
    model_digest: str | None
    options_prompts: object | None
    extraction_config_fingerprint: str | None
    gate_version: str | None
    warm_cold_protocol: str
    seed: int | None

    def __post_init__(self) -> None:
        for name in (
            "git_commit",
            "lockfile",
            "docling_version",
            "ocrmac_version",
            "ollama_version",
            "model_name",
            "model_digest",
            "extraction_config_fingerprint",
            "gate_version",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                raise ValueError(f"reproducibility.{name} must be string or null")
        _require_nonempty_string(self.warm_cold_protocol, "warm_cold_protocol")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("reproducibility.seed must be integer or null")


@dataclass(frozen=True)
class ExpectedPage:
    page_number: int
    route: Route
    status: Status

    def __post_init__(self) -> None:
        _validate_pages((self.page_number,), "page")
        if type(self.route) is not str or self.route not in ROUTES:
            raise ValueError("unknown extraction route")
        if type(self.status) is not str or self.status not in STATUSES:
            raise ValueError("unknown extraction status")


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    text: str
    answerable: bool
    source_page_sets: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.question_id, "question id")
        _require_nonempty_string(self.text, "question text")
        if type(self.answerable) is not bool:
            raise ValueError("question answerable must be boolean")
        for page_set in self.source_page_sets:
            _validate_pages(page_set, "source page set")
        if self.answerable and (
            not self.source_page_sets or any(not page_set for page_set in self.source_page_sets)
        ):
            raise ValueError("answerable question needs source page sets")
        if not self.answerable and self.source_page_sets:
            raise ValueError("unanswerable question must have empty source page sets")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    split: Split
    categories: tuple[str, ...]
    document_id: str
    extraction: tuple[ExpectedPage, ...]
    questions: tuple[QuestionSpec, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.case_id, "case id")
        if type(self.split) is not str or self.split not in {"tuning", "acceptance"}:
            raise ValueError("case split must be tuning or acceptance")
        if not self.categories or any(
            type(category) is not str or category not in CATEGORIES for category in self.categories
        ):
            raise ValueError("case has unknown or empty categories")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("case categories must not contain duplicates")
        _require_nonempty_string(self.document_id, "case document id")
        if not self.extraction:
            raise ValueError("case extraction must not be empty")
        if len({page.page_number for page in self.extraction}) != len(self.extraction):
            raise ValueError("case extraction pages must not contain duplicates")
        if not self.questions:
            raise ValueError("case questions must not be empty")
        if len({question.question_id for question in self.questions}) != len(self.questions):
            raise ValueError("case question ids must not contain duplicates")


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    baseline_tag: str
    baseline_commit: str
    reproducibility: ReproducibilitySpec
    documents: tuple[DocumentSpec, ...]
    cases: tuple[CaseSpec, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported manifest schema_version")
        _require_nonempty_string(self.baseline_tag, "baseline tag")
        if type(self.baseline_commit) is not str or not _COMMIT.fullmatch(self.baseline_commit):
            raise ValueError("baseline commit must be a 40-character lowercase SHA")
        if not self.documents or not self.cases:
            raise ValueError("manifest documents and cases must not be empty")
        if len({document.logical_id for document in self.documents}) != len(self.documents):
            raise ValueError("duplicate document ids")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("duplicate case ids")
        document_ids = {document.logical_id for document in self.documents}
        if any(case.document_id not in document_ids for case in self.cases):
            raise ValueError("case references unknown document")
        if {case.split for case in self.cases} != {"tuning", "acceptance"}:
            raise ValueError("manifest must contain tuning and acceptance coverage")
        covered = {category for case in self.cases for category in case.categories}
        if covered != CATEGORIES:
            missing = ", ".join(sorted(CATEGORIES - covered))
            raise ValueError(f"manifest category coverage is incomplete: {missing}")

    def document(self, logical_id: str) -> DocumentSpec:
        for document in self.documents:
            if document.logical_id == logical_id:
                return document
        raise ValueError(f"unknown document id: {logical_id}")


@dataclass(frozen=True)
class CitationObservation:
    page_number: int

    def __post_init__(self) -> None:
        _validate_pages((self.page_number,), "citation page")


@dataclass(frozen=True)
class PageObservation:
    page_number: int
    route: Route
    status: Status

    def __post_init__(self) -> None:
        _validate_pages((self.page_number,), "observed page")
        if type(self.route) is not str or self.route not in ROUTES:
            raise ValueError("unknown observed extraction route")
        if type(self.status) is not str or self.status not in STATUSES:
            raise ValueError("unknown observed extraction status")


@dataclass(frozen=True)
class QuestionObservation:
    question_id: str
    source_pages: tuple[int, ...]
    final_evidence_pages: tuple[int, ...]
    citations: tuple[CitationObservation, ...]
    refused: bool
    human_correctness: int | None
    claim_support: int | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.question_id, "observed question id")
        _validate_pages(self.source_pages, "observed source pages")
        _validate_pages(self.final_evidence_pages, "final evidence pages")
        if type(self.refused) is not bool:
            raise ValueError("refused must be boolean")
        for name in ("human_correctness", "claim_support"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value not in range(3)):
                raise ValueError(f"{name} must be 0, 1, 2, or null")


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    document_sha256: str | None
    pdf_page_count: int
    pages: tuple[PageObservation, ...]
    questions: tuple[QuestionObservation, ...]
    uncached_ingestion_latency_s: float | None
    warmed_qa_latency_s: float | None
    ocr_pages_processed: int | None
    llm_calls: int | None
    tool_steps: int | None
    timed_out: bool
    runtime_error: str | None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.case_id, "observed case id")
        if self.document_sha256 is not None and (
            type(self.document_sha256) is not str or not _SHA256.fullmatch(self.document_sha256)
        ):
            raise ValueError("observed document sha256 must be 64 lowercase hex characters")
        if type(self.pdf_page_count) is not int or self.pdf_page_count < 1:
            raise ValueError("pdf_page_count must be >= 1")
        if len({page.page_number for page in self.pages}) != len(self.pages):
            raise ValueError("duplicate observed page numbers")
        if len({question.question_id for question in self.questions}) != len(self.questions):
            raise ValueError("duplicate observed question ids")
        for page in self.pages:
            if page.page_number > self.pdf_page_count:
                raise ValueError("observed page is outside PDF range")
        for question in self.questions:
            _validate_pages(question.source_pages, "observed source pages")
            _validate_pages(question.final_evidence_pages, "final evidence pages")
            for page in (*question.source_pages, *question.final_evidence_pages):
                if page > self.pdf_page_count:
                    raise ValueError("observed source/evidence page is outside PDF range")
            for citation in question.citations:
                if citation.page_number > self.pdf_page_count:
                    raise ValueError("citation page is outside PDF range")
        for name in (
            "uncached_ingestion_latency_s",
            "warmed_qa_latency_s",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number or null")
        for name in ("ocr_pages_processed", "llm_calls", "tool_steps"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if type(self.timed_out) is not bool:
            raise ValueError("timed_out must be boolean")
        if self.runtime_error is not None:
            _require_nonempty_string(self.runtime_error, "runtime_error")


@dataclass(frozen=True)
class MetricReport:
    numerator: float
    denominator: int
    skipped: int = 0
    failures: int = 0

    def __post_init__(self) -> None:
        if type(self.numerator) not in {int, float} or not math.isfinite(self.numerator):
            raise ValueError("metric numerator must be finite")
        for name in ("denominator", "skipped", "failures"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"metric {name} must be a non-negative integer")


@dataclass(frozen=True)
class EvaluationReport:
    status: Literal["pass", "incomplete", "fail"]
    route_correctness: MetricReport
    source_recall: MetricReport
    citation_validity: MetricReport
    human_correctness: MetricReport
    claim_support: MetricReport
    false_refusal: MetricReport
    correct_refusal: MetricReport
    uncached_ingestion_latency: MetricReport
    warmed_qa_latency: MetricReport
    ocr_pages_processed: MetricReport
    llm_calls: MetricReport
    tool_steps: MetricReport
    timeouts: MetricReport
    errors: MetricReport
    acceptance_readiness: MetricReport


def _strict_object(value: object, name: str, keys: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - keys
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    return value


def _load_reproducibility(value: object) -> ReproducibilitySpec:
    keys = {
        "git_commit",
        "lockfile",
        "docling_version",
        "ocrmac_version",
        "ollama_version",
        "model_name",
        "model_digest",
        "options_prompts",
        "extraction_config_fingerprint",
        "gate_version",
        "warm_cold_protocol",
        "seed",
    }
    data = _strict_object(value, "reproducibility", keys)
    return ReproducibilitySpec(**data)


def load_manifest(path: str | Path) -> Manifest:
    """Load and strictly validate a version-1 evaluation manifest."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("manifest is not valid JSON") from error
    top = _strict_object(
        data,
        "manifest",
        {"schema_version", "baseline", "reproducibility", "documents", "cases"},
    )
    if type(top["schema_version"]) is not int or top["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    baseline = _strict_object(top["baseline"], "baseline", {"tag", "commit"})
    documents_data = top["documents"]
    cases_data = top["cases"]
    if type(documents_data) is not list or type(cases_data) is not list:
        raise ValueError("documents and cases must be arrays")

    documents: list[DocumentSpec] = []
    for index, raw in enumerate(documents_data):
        item = _strict_object(raw, f"documents[{index}]", {"id", "kind", "basename", "sha256"})
        documents.append(
            DocumentSpec(
                logical_id=item["id"],
                kind=item["kind"],
                basename=item["basename"],
                sha256=item["sha256"],
            )
        )

    cases: list[CaseSpec] = []
    for index, raw in enumerate(cases_data):
        item = _strict_object(
            raw,
            f"cases[{index}]",
            {"id", "split", "categories", "document_id", "extraction", "questions"},
        )
        categories = item["categories"]
        extraction_data = item["extraction"]
        questions_data = item["questions"]
        if type(categories) is not list or type(extraction_data) is not list or type(questions_data) is not list:
            raise ValueError(f"cases[{index}] categories, extraction, and questions must be arrays")
        extraction: list[ExpectedPage] = []
        for page_index, raw_page in enumerate(extraction_data):
            page = _strict_object(
                raw_page,
                f"cases[{index}].extraction[{page_index}]",
                {"page", "route", "status"},
            )
            extraction.append(ExpectedPage(page["page"], page["route"], page["status"]))
        questions: list[QuestionSpec] = []
        for question_index, raw_question in enumerate(questions_data):
            question = _strict_object(
                raw_question,
                f"cases[{index}].questions[{question_index}]",
                {"id", "text", "answerable", "source_page_sets"},
            )
            source_sets = question["source_page_sets"]
            if type(source_sets) is not list:
                raise ValueError("source_page_sets must be an array")
            parsed_sets: list[tuple[int, ...]] = []
            for source_set in source_sets:
                if type(source_set) is not list:
                    raise ValueError("each source page set must be an array")
                parsed_sets.append(tuple(source_set))
            question_id = _require_nonempty_string(question["id"], "question id")
            question_text = _require_nonempty_string(question["text"], "question text")
            answerable = question["answerable"]
            if type(answerable) is not bool:
                raise ValueError("question answerable must be boolean")
            questions.append(
                QuestionSpec(
                    question_id,
                    question_text,
                    answerable,
                    tuple(parsed_sets),
                )
            )
        cases.append(
            CaseSpec(
                case_id=item["id"],
                split=item["split"],
                categories=tuple(categories),
                document_id=item["document_id"],
                extraction=tuple(extraction),
                questions=tuple(questions),
            )
        )
    return Manifest(
        schema_version=top["schema_version"],
        baseline_tag=baseline["tag"],
        baseline_commit=baseline["commit"],
        reproducibility=_load_reproducibility(top["reproducibility"]),
        documents=tuple(documents),
        cases=tuple(cases),
    )


def _metric(numerator: float = 0, denominator: int = 0, skipped: int = 0, failures: int = 0) -> MetricReport:
    return MetricReport(numerator, denominator, skipped, failures)


def _page_map(observation: CaseObservation) -> dict[int, PageObservation]:
    return {page.page_number: page for page in observation.pages}


def _question_map(observation: CaseObservation) -> dict[str, QuestionObservation]:
    return {question.question_id: question for question in observation.questions}


def _recall(observed: tuple[int, ...], expected: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
    observed_set = set(observed)
    best = max(
        ((len(observed_set & set(page_set)), len(page_set)) for page_set in expected),
        key=lambda pair: (pair[0] / pair[1], pair[0], -pair[1]),
    )
    return best


def evaluate_run(manifest: Manifest, observations: tuple[CaseObservation, ...]) -> EvaluationReport:
    """Score a typed run without extraction, LLM, or filesystem side effects."""
    if not isinstance(manifest, Manifest):
        raise ValueError("manifest must be a Manifest")
    if type(observations) not in {tuple, list}:
        raise ValueError("observations must be a tuple or list of CaseObservation")
    if any(not isinstance(observation, CaseObservation) for observation in observations):
        raise ValueError("observations must contain CaseObservation values")
    if len({observation.case_id for observation in observations}) != len(observations):
        raise ValueError("duplicate observed case ids")
    expected_cases = {case.case_id: case for case in manifest.cases}
    observed_cases = {observation.case_id: observation for observation in observations}
    unknown = set(observed_cases) - set(expected_cases)
    if unknown:
        raise ValueError(f"observations contain unknown case ids: {sorted(unknown)}")

    route_numerator = route_denominator = route_skipped = route_failures = 0
    recall_numerator = recall_denominator = recall_skipped = recall_failures = 0
    citation_numerator = citation_denominator = citation_skipped = citation_failures = 0
    human_numerator = human_denominator = human_skipped = human_failures = 0
    support_numerator = support_denominator = support_skipped = support_failures = 0
    false_numerator = false_denominator = false_skipped = false_failures = 0
    refusal_numerator = refusal_denominator = refusal_skipped = refusal_failures = 0
    latency_values = {"uncached": [0.0, 0, 0, 0], "warmed": [0.0, 0, 0, 0]}
    operational = {name: [0.0, 0, 0, 0] for name in ("ocr", "llm", "tools")}
    timeout_numerator = timeout_denominator = timeout_skipped = timeout_failures = 0
    error_numerator = error_denominator = error_skipped = error_failures = 0
    readiness_numerator = readiness_denominator = readiness_skipped = readiness_failures = 0

    for case in manifest.cases:
        observation = observed_cases.get(case.case_id)
        document = manifest.document(case.document_id)
        readiness_denominator += 1 if case.split == "acceptance" else 0
        if case.split == "acceptance":
            if document.kind == "real" and document.sha256 is not None and observation is not None:
                if observation.document_sha256 != document.sha256:
                    if observation.document_sha256 is None:
                        readiness_skipped += 1
                    else:
                        raise ValueError(f"document hash mismatch for case {case.case_id}")
                else:
                    readiness_numerator += 1
            else:
                readiness_skipped += 1

        route_denominator += len(case.extraction)
        answerable_questions = [question for question in case.questions if question.answerable]
        unanswerable_questions = [question for question in case.questions if not question.answerable]
        citation_denominator += len(case.questions)
        human_denominator += len(case.questions)
        support_denominator += len(case.questions)
        false_denominator += len(answerable_questions)
        refusal_denominator += len(unanswerable_questions)

        if observation is None:
            route_skipped += len(case.extraction)
            recall_skipped += len(answerable_questions)
            recall_denominator += sum(
                min(len(page_set) for page_set in question.source_page_sets)
                for question in answerable_questions
            )
            citation_skipped += len(case.questions)
            human_skipped += len(case.questions)
            support_skipped += len(case.questions)
            false_skipped += len(answerable_questions)
            refusal_skipped += len(unanswerable_questions)
            for values in latency_values.values():
                values[2] += 1
            for values in operational.values():
                values[2] += 1
            timeout_skipped += 1
            error_skipped += 1
            continue

        pages = _page_map(observation)
        for expected in case.extraction:
            actual = pages.get(expected.page_number)
            if actual is None:
                route_skipped += 1
            elif (actual.route, actual.status) == (expected.route, expected.status):
                route_numerator += 1
            else:
                route_failures += 1

        questions = _question_map(observation)
        expected_question_ids = {question.question_id for question in case.questions}
        if set(questions) - expected_question_ids:
            raise ValueError(f"observations contain unknown question for case {case.case_id}")
        for expected in case.questions:
            actual = questions.get(expected.question_id)
            if actual is None:
                citation_skipped += 1
                human_skipped += 1
                support_skipped += 1
                if expected.answerable:
                    recall_skipped += 1
                    recall_denominator += min(
                        len(page_set) for page_set in expected.source_page_sets
                    )
                    false_skipped += 1
                else:
                    refusal_skipped += 1
                continue
            if expected.answerable:
                matched, total = _recall(actual.source_pages, expected.source_page_sets)
                recall_numerator += matched
                # The selected alternative determines the denominator.
                recall_denominator += total
                if matched < total:
                    recall_failures += 1
                false_numerator += actual.refused
                false_failures += actual.refused
            if expected.answerable:
                if actual.refused:
                    citation_valid = not actual.citations
                else:
                    citation_valid = bool(actual.citations) and all(
                        citation.page_number in actual.final_evidence_pages
                        for citation in actual.citations
                    )
            else:
                citation_valid = actual.refused and not actual.citations
            citation_numerator += citation_valid
            if not citation_valid:
                citation_failures += 1
            if actual.human_correctness is None:
                human_skipped += 1
            else:
                human_numerator += actual.human_correctness
            if actual.claim_support is None:
                support_skipped += 1
            else:
                support_numerator += actual.claim_support
            if not expected.answerable:
                clean_refusal = actual.refused and not actual.citations
                refusal_numerator += clean_refusal
                refusal_failures += not clean_refusal

        for values, value in (
            (latency_values["uncached"], observation.uncached_ingestion_latency_s),
            (latency_values["warmed"], observation.warmed_qa_latency_s),
        ):
            if value is None:
                values[2] += 1
            else:
                values[0] += value
                values[1] += 1
            if observation.runtime_error or observation.timed_out:
                values[3] += 1
        for name, value in (
            ("ocr", observation.ocr_pages_processed),
            ("llm", observation.llm_calls),
            ("tools", observation.tool_steps),
        ):
            if value is None:
                operational[name][2] += 1
            else:
                operational[name][0] += value
                operational[name][1] += 1
            if observation.runtime_error or observation.timed_out:
                operational[name][3] += 1
        timeout_denominator += 1
        timeout_numerator += observation.timed_out
        timeout_failures += observation.timed_out
        error_denominator += 1
        error_numerator += bool(observation.runtime_error)
        error_failures += bool(observation.runtime_error)

    metrics = {
        "route_correctness": _metric(route_numerator, route_denominator, route_skipped, route_failures),
        "source_recall": _metric(recall_numerator, recall_denominator, recall_skipped, recall_failures),
        "citation_validity": _metric(citation_numerator, citation_denominator, citation_skipped, citation_failures),
        "human_correctness": _metric(human_numerator, human_denominator, human_skipped, human_failures),
        "claim_support": _metric(support_numerator, support_denominator, support_skipped, support_failures),
        "false_refusal": _metric(false_numerator, false_denominator, false_skipped, false_failures),
        "correct_refusal": _metric(refusal_numerator, refusal_denominator, refusal_skipped, refusal_failures),
        "uncached_ingestion_latency": _metric(*latency_values["uncached"]),
        "warmed_qa_latency": _metric(*latency_values["warmed"]),
        "ocr_pages_processed": _metric(*operational["ocr"]),
        "llm_calls": _metric(*operational["llm"]),
        "tool_steps": _metric(*operational["tools"]),
        "timeouts": _metric(timeout_numerator, timeout_denominator, timeout_skipped, timeout_failures),
        "errors": _metric(error_numerator, error_denominator, error_skipped, error_failures),
        "acceptance_readiness": _metric(
            readiness_numerator, readiness_denominator, readiness_skipped, readiness_failures
        ),
    }
    has_failures = any(metric.failures for metric in metrics.values())
    has_incomplete = any(metric.skipped for metric in metrics.values())
    status = "fail" if has_failures else "incomplete" if has_incomplete else "pass"
    return EvaluationReport(status=status, **metrics)
